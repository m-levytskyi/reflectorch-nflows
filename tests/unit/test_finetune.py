import torch
from torch import nn

from reflectorch import ConstantQ, LogAffineCurvesScaler, SubpriorParametricSampler
from reflectorch.ml.basic_trainer import DataLoader
from reflectorch.ml.trainers import NFlowTrainer
from reflectorch.runs import fine_tune_nflow


class _TinyFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.1))
        self.dim = dim

    def sample(self, num_samples, **kwargs):
        return torch.rand(1, num_samples, self.dim) * 2 - 1 + self.bias * 0

    def log_prob(self, inputs, **kwargs):
        return -0.5 * (inputs - self.bias).square().sum(dim=-1)


class _Loader(DataLoader):
    def __init__(self, prior_sampler):
        self.prior_sampler = prior_sampler
        self.q_generator = ConstantQ((0.01, 0.1, 8), device="cpu")
        self.curves_scaler = LogAffineCurvesScaler()
        self.smearing = None


def test_fine_tune_uses_prior_simulations_and_a_separate_cpu_model():
    sampler = SubpriorParametricSampler(
        param_ranges={
            "thicknesses": (1.0, 100.0),
            "roughnesses": (0.0, 30.0),
            "slds": (-8.0, 16.0),
        },
        bound_width_ranges={
            "thicknesses": (1.0, 99.0),
            "roughnesses": (1.0, 30.0),
            "slds": (1.0, 24.0),
        },
        model_name="standard_model",
        max_num_layers=1,
        constrained_roughness=False,
        device=torch.device("cpu"),
    )
    base_model = _TinyFlow(sampler.param_dim)
    base_trainer = NFlowTrainer(
        model=base_model,
        loader=_Loader(sampler),
        lr=1e-3,
        batch_size=2,
        train_with_bounds=True,
        train_with_q_input=True,
        train_with_sigmas=False,
        condition_on_q_resolutions=False,
    )
    original_weight = base_model.bias.detach().clone()
    bounds = torch.tensor(
        [[10.0, 50.0], [1.0, 10.0], [-4.0, 8.0], [1.0, 10.0], [-4.0, 8.0]]
    )

    adapted = fine_tune_nflow(
        base_trainer,
        q_values=torch.linspace(0.01, 0.1, 8),
        reflectivity_curve=torch.full((8,), 1e-4),
        uncertainty=torch.full((8,), 1e-5),
        prior_bounds=bounds,
        max_steps=2,
        batch_size=2,
    )

    assert adapted is not base_trainer
    assert adapted.model is not base_trainer.model
    assert torch.equal(base_model.bias.detach(), original_weight)
    assert len(adapted.losses[adapted.TOTAL_LOSS_KEY]) == 2
    assert all(
        torch.isfinite(torch.tensor(loss))
        for loss in adapted.losses[adapted.TOTAL_LOSS_KEY]
    )
