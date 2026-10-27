from __future__ import annotations

import copy
from typing import Optional

import torch

from reflectorch.data_generation.priors.parametric_subpriors import BasicParams
from reflectorch.ml.trainers import BasicBatchData, NFlowTrainer


class _ExperimentPriorTrainer(NFlowTrainer):
    def get_batch_by_idx(self, batch_num: int) -> BasicBatchData:
        count = self.batch_size
        theta = self._prior_min + torch.rand(
            count, self.loader.prior_sampler.param_dim, device=self._prior_min.device
        ) * (self._prior_max - self._prior_min)
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
        batch = {
            "scaled_params": self.loader.prior_sampler.scale_params(params),
            "scaled_noisy_curves": self.loader.curves_scaler.scale(
                curves.clamp_min(1e-10), self._q_values
            ),
            "q_values": self._q_values.expand(count, -1),
        }
        if self._q_resolution is not None:
            batch["q_resolutions"] = self._q_resolution.expand(count, 1)
        return self._get_batch(batch)


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
    """Continue training a separate NF copy on simulations from the experiment prior."""
    if not isinstance(trainer, NFlowTrainer):
        raise TypeError("trainer must be an NFlowTrainer")
    if max_steps < 1 or batch_size < 1:
        raise ValueError("max_steps and batch_size must be positive")

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

    fine_tuned = _ExperimentPriorTrainer(
        model=copy.deepcopy(trainer.model),
        loader=copy.deepcopy(trainer.loader),
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
    fine_tuned._q_resolution = (
        torch.tensor([[q_resolution]], device=device, dtype=dtype)
        if q_resolution is not None
        else None
    )
    fine_tuned._relative_uncertainty = (sigma / observed.clamp_min(1e-10))[None, :]
    fine_tuned.train(max_steps, disable_tqdm=True)
    return fine_tuned
