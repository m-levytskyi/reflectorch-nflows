from __future__ import annotations

import copy
from typing import Optional

import torch
from torch import Tensor

from reflectorch.data_generation.priors.parametric_subpriors import BasicParams
from reflectorch.ml.trainers import BasicBatchData, NFlowTrainer


def _importance_weights(log_proposal: Tensor) -> Tensor:
    """Normalize prior/proposal weights in a batch; the uniform prior is constant."""
    log_weights = -log_proposal
    return torch.softmax(log_weights - log_weights.max(), dim=0) * log_weights.numel()


class _ExperimentSNPETrainer(NFlowTrainer):
    def init(self):
        super().init()
        self._round = 1
        self._proposal_model = None
        self._proposal_log_prob = None

    def get_batch_by_idx(self, batch_num: int) -> BasicBatchData:
        count = self.batch_size
        if self._round == 1:
            theta = self._sample_prior(count)
        else:
            theta = self._sample_proposal(count)

        params = BasicParams(
            parameters=theta,
            min_bounds=self._prior_min.expand_as(theta),
            max_bounds=self._prior_max.expand_as(theta),
            max_num_layers=self.loader.prior_sampler.max_num_layers,
            param_model=self.loader.prior_sampler.param_model,
        )
        kwargs = {}
        if self._q_resolution is not None and self.loader.smearing is not None:
            kwargs.update(
                dq=self._q_resolution.expand(count, 1),
                constant_dq=self.loader.smearing.constant_dq,
                gauss_num=self.loader.smearing.gauss_num,
            )
        curves = params.reflectivity(self._q_values.expand(count, -1), **kwargs)
        if self._relative_uncertainty is not None:
            curves = curves * (
                1.0 + torch.randn_like(curves) * self._relative_uncertainty
            )
        curves = curves.clamp_min(1e-10)
        scaled_curves = self.loader.curves_scaler.scale(curves, self._q_values)
        scaled_params = self.loader.prior_sampler.scale_params(params)
        batch = {
            "scaled_params": scaled_params,
            "scaled_noisy_curves": scaled_curves,
            "q_values": self._q_values.expand(count, -1),
        }
        if self._q_resolution is not None:
            batch["q_resolutions"] = self._q_resolution.expand(count, 1)

        if self._round == 2:
            with torch.no_grad():
                self._proposal_log_prob = self._proposal_model.log_prob(
                    inputs=self._scaled_theta(theta),
                    curves=self._observation_scaled.expand(count, -1),
                    bounds=self._scaled_bounds.expand(count, -1)
                    if self.train_with_bounds
                    else None,
                    q_values=self._scaled_q_values.expand(count, -1)
                    if self.train_with_q_input
                    else None,
                    conditioning_params=self._scaled_q_resolution.expand(count, -1)
                    if self.condition_on_q_resolutions
                    else None,
                    unscaled_q_values=self._q_values.expand(count, -1),
                )
        else:
            self._proposal_log_prob = None
        return self._get_batch(batch)

    def _sample_prior(self, count: int) -> Tensor:
        return self._prior_min + torch.rand(
            count, self.loader.prior_sampler.param_dim, device=self._prior_min.device
        ) * (self._prior_max - self._prior_min)

    def _sample_proposal(self, count: int) -> Tensor:
        accepted = []
        accepted_count = 0
        attempts = 0
        while accepted_count < count and attempts < 20:
            n = max(2 * (count - accepted_count), 32)
            with torch.no_grad():
                samples = self._proposal_model.sample(
                    num_samples=n,
                    curves=self._observation_scaled,
                    bounds=self._scaled_bounds if self.train_with_bounds else None,
                    q_values=self._scaled_q_values if self.train_with_q_input else None,
                    conditioning_params=self._scaled_q_resolution
                    if self.condition_on_q_resolutions
                    else None,
                    unscaled_q_values=self._q_values,
                ).squeeze(0)
            samples = samples[torch.all((samples >= -1.0) & (samples <= 1.0), dim=-1)]
            if samples.numel():
                accepted.append(samples[: count - accepted_count])
                accepted_count = sum(part.shape[0] for part in accepted)
            attempts += 1
        if accepted_count != count:
            raise RuntimeError("Posterior proposal produced too few in-prior samples")
        return self.loader.prior_sampler.restore_params(
            torch.cat(
                [torch.cat(accepted), self._scaled_bounds.expand(count, -1)], dim=-1
            )
        ).parameters

    def _scaled_theta(self, theta: Tensor) -> Tensor:
        return self.loader.prior_sampler._scale(theta, self._prior_min, self._prior_max)

    def get_loss_dict(self, batch_data: BasicBatchData):
        log_prob = self.model.log_prob(
            inputs=batch_data.scaled_params,
            curves=batch_data.scaled_curves,
            bounds=batch_data.scaled_bounds,
            q_values=batch_data.scaled_q_values,
            sigmas=batch_data.scaled_sigmas,
            conditioning_params=batch_data.scaled_conditioning_params,
            key_padding_mask=batch_data.key_padding_mask,
            unscaled_q_values=batch_data.unscaled_q_values,
        )
        if self._proposal_log_prob is None:
            loss = -log_prob.mean()
        else:
            loss = -(
                _importance_weights(self._proposal_log_prob).to(log_prob) * log_prob
            ).mean()
        return {"loss": loss}


def fine_tune_nflow(
    trainer: NFlowTrainer,
    q_values,
    reflectivity_curve,
    uncertainty,
    prior_bounds,
    max_steps: int = 15_000,
    *,
    batch_size: int = 2_048,
    learning_rate: float = 1e-5,
    q_resolution: Optional[float] = 0.1,
) -> NFlowTrainer:
    """Fine-tune a separate NF copy with two-round SNPE-C simulation training."""
    if not isinstance(trainer, NFlowTrainer):
        raise TypeError("trainer must be an NFlowTrainer")
    if max_steps < 2 or batch_size < 1:
        raise ValueError("max_steps must be at least 2 and batch_size must be positive")

    device = next(trainer.model.parameters()).device
    dtype = next(trainer.model.parameters()).dtype
    sampler = trainer.loader.prior_sampler
    bounds = torch.as_tensor(prior_bounds, device=device, dtype=dtype)
    if bounds.shape != (sampler.param_dim, 2) or not torch.isfinite(bounds).all():
        raise ValueError(f"prior_bounds must have shape ({sampler.param_dim}, 2)")
    prior_min, prior_max = bounds[:, 0], bounds[:, 1]
    if not torch.all(prior_max > prior_min):
        raise ValueError("Every prior upper bound must exceed its lower bound")

    q = torch.as_tensor(q_values, device=device, dtype=dtype).flatten()
    observed = torch.as_tensor(reflectivity_curve, device=device, dtype=dtype).flatten()
    sigma = torch.as_tensor(uncertainty, device=device, dtype=dtype).flatten()
    if q.numel() < 2 or q.shape != observed.shape or q.shape != sigma.shape:
        raise ValueError(
            "q_values, reflectivity_curve, and uncertainty must have equal lengths >= 2"
        )
    if (
        not torch.isfinite(q).all()
        or not torch.isfinite(observed).all()
        or not torch.isfinite(sigma).all()
    ):
        raise ValueError("measurement arrays must be finite")
    if torch.any(q[1:] <= q[:-1]) or torch.any(observed <= 0) or torch.any(sigma < 0):
        raise ValueError(
            "q must increase, reflectivity must be positive, and uncertainty nonnegative"
        )

    copied_model = copy.deepcopy(trainer.model)
    copied_loader = copy.deepcopy(trainer.loader)
    fine_tuned = _ExperimentSNPETrainer(
        model=copied_model,
        loader=copied_loader,
        lr=learning_rate,
        batch_size=batch_size,
        clip_grad_norm_max=trainer.clip_grad_norm_max,
        optim_cls=torch.optim.AdamW,
        optim_kwargs={"weight_decay": 5e-4},
        train_with_bounds=trainer.train_with_bounds,
        train_with_q_input=trainer.train_with_q_input,
        train_with_sigmas=trainer.train_with_sigmas,
        condition_on_q_resolutions=trainer.condition_on_q_resolutions,
    )
    fine_tuned._prior_min = prior_min
    fine_tuned._prior_max = prior_max
    fine_tuned._q_values = q[None, :]
    fine_tuned._scaled_bounds = torch.cat(
        [
            sampler.scale_bounds(prior_min[None, :]),
            sampler.scale_bounds(prior_max[None, :]),
        ],
        -1,
    ).to(dtype=dtype)
    fine_tuned._observation_scaled = copied_loader.curves_scaler.scale(
        observed[None, :], q[None, :]
    )
    fine_tuned._scaled_q_values = copied_loader.q_generator.scale_q(q[None, :]).to(
        dtype=dtype
    )
    fine_tuned._q_resolution = (
        torch.tensor([[q_resolution]], device=device, dtype=dtype)
        if q_resolution is not None
        else None
    )
    fine_tuned._scaled_q_resolution = (
        copied_loader.smearing.scale_resolutions(fine_tuned._q_resolution)
        if fine_tuned._q_resolution is not None and copied_loader.smearing is not None
        else None
    )
    fine_tuned._relative_uncertainty = (sigma / observed.clamp_min(1e-10))[None, :]

    round_steps = (max_steps // 2, max_steps - max_steps // 2)
    fine_tuned._round = 1
    fine_tuned.train(round_steps[0], disable_tqdm=True)
    fine_tuned._proposal_model = copy.deepcopy(fine_tuned.model).eval()
    fine_tuned._round = 2
    fine_tuned.train(round_steps[1], disable_tqdm=True)
    return fine_tuned
