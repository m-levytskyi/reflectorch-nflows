import pytest

import torch

from reflectorch import (
    ConstantQ,
    LogAffineCurvesScaler,
    Smearing,
    SubpriorParametricSampler,
    BasicExpIntensityNoise,
)
from reflectorch.paths import TEST_DATA_PATH
from reflectorch.ml.experimental_dataloaders import (
    ExperimentalReflectivityDataLoader,
    MixedReflectivityDataLoader,
)


def _make_sampler_cpu():
    return SubpriorParametricSampler(
        param_ranges={
            "thicknesses": [1.0, 1500.0],
            "roughnesses": [0.0, 60.0],
            "slds": [-8.0, 16.0],
            "r_scale": [0.9, 1.1],
            "log10_background": [-10.0, -4.0],
        },
        bound_width_ranges={
            "thicknesses": [1.0e-2, 1500.0],
            "roughnesses": [1.0e-2, 60.0],
            "slds": [1.0e-2, 24.0],
            "r_scale": [1.0e-3, 0.2],
            "log10_background": [1.0e-2, 6.0],
        },
        shift_param_config={
            "r_scale": True,
            "log10_background": True,
        },
        model_name="standard_model",
        max_num_layers=1,
        constrained_roughness=True,
        max_thickness_share=0.5,
        scale_params_by_ranges=False,
        device="cpu",
    )


def test_experimental_dataloader_batch_shapes():
    prior_sampler = _make_sampler_cpu()

    q_generator = ConstantQ(q=[0.007, 0.284, 256], device="cpu")
    curves_scaler = LogAffineCurvesScaler(weight=0.2, bias=1.0, eps=1.0e-10)
    smearing = Smearing(sigma_range=(0.01, 0.12), constant_dq=False, gauss_num=17, share_smeared=1.0)

    loader = ExperimentalReflectivityDataLoader(
        q_generator=q_generator,
        prior_sampler=prior_sampler,
        curves_scaler=curves_scaler,
        smearing=smearing,
        data_dir=TEST_DATA_PATH / "experimental",
        curve_glob="*_experimental_curve.dat",
        background=5.0e-7,
        q_resolution=0.1,
    )

    batch = loader.get_batch(4)

    assert set(batch.keys()) == {"q_values", "scaled_noisy_curves", "scaled_params", "q_resolutions"}

    assert batch["q_values"].shape == (4, 256)
    assert batch["scaled_noisy_curves"].shape == (4, 256)

    # P = 7 (5 base + 2 nuisance), scaled_params contains params + min_bounds + max_bounds
    assert batch["scaled_params"].shape == (4, 21)
    assert batch["q_resolutions"].shape == (4, 1)


def test_mixed_dataloader_batch_shapes():
    prior_sampler = _make_sampler_cpu()

    q_generator = ConstantQ(q=[0.007, 0.284, 256], device="cpu")
    curves_scaler = LogAffineCurvesScaler(weight=0.2, bias=1.0, eps=1.0e-10)
    smearing = Smearing(sigma_range=(0.01, 0.12), constant_dq=False, gauss_num=17, share_smeared=1.0)
    intensity_noise = BasicExpIntensityNoise(relative_errors=(0.0, 0.05), consistent_rel_err=True)

    loader = MixedReflectivityDataLoader(
        q_generator=q_generator,
        prior_sampler=prior_sampler,
        intensity_noise=intensity_noise,
        curves_scaler=curves_scaler,
        smearing=smearing,
        data_dir=TEST_DATA_PATH / "experimental",
        curve_glob="*_experimental_curve.dat",
        background=5.0e-7,
        q_resolution=0.1,
        mix_fraction=0.5,
    )

    batch = loader.get_batch(6)

    assert batch["q_values"].shape == (6, 256)
    assert batch["scaled_noisy_curves"].shape == (6, 256)
    assert batch["scaled_params"].shape[0] == 6
    assert batch["scaled_params"].shape[1] % 3 == 0

    # Mixed batch should include q_resolutions because experimental always provides it and
    # synthetic provides it if smearing is enabled.
    if "q_resolutions" in batch:
        assert batch["q_resolutions"].shape == (6, 1)
