import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from typing import Any, cast

from reflectorch.inference.inference_model import EasyInferenceModel
from reflectorch.runs.utils import get_trainer_from_config


@pytest.mark.skipif(
    os.environ.get("REFLECTORCH_RUN_TRAIN_TEST", "0") != "1",
    reason="Set REFLECTORCH_RUN_TRAIN_TEST=1 to enable the (slow) train+infer integration test.",
)
def test_nf_train_save_load_and_sample(tmp_path: Path, capsys):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    repo_root = Path(__file__).resolve().parents[2]
    experimental_data_dir = repo_root / "tests" / "data" / "experimental"

    # Create an isolated root_dir so training/inference don't touch repo-level saved_models.
    root_dir = tmp_path / "reflectorch_root"
    (root_dir / "configs").mkdir(parents=True, exist_ok=True)
    (root_dir / "saved_models").mkdir(parents=True, exist_ok=True)
    (root_dir / "saved_losses").mkdir(parents=True, exist_ok=True)

    name = "pytest_nf_train_infer"

    # Tiny config to keep runtime small.
    config = {
        "general": {
            "name": name,
            "root_dir": str(root_dir),
        },
        "dset": {
            "cls": "MixedReflectivityDataLoader",
            "kwargs": {
                "data_dir": str(experimental_data_dir),
                "curve_glob": "*_experimental_curve.dat",
                "curve_suffix": "_experimental_curve.dat",
                "model_suffix": "_model.txt",
                "background": 5.0e-7,
                "q_resolution": 0.1,
                "mix_fraction": 0.5,
            },
            "prior_sampler": {
                "cls": "SubpriorParametricSampler",
                "kwargs": {
                    "param_ranges": {
                        "thicknesses": [1.0, 200.0],
                        "roughnesses": [0.0, 30.0],
                        "slds": [-4.0, 10.0],
                        "r_scale": [0.9, 1.1],
                        "log10_background": [-10.0, -4.0],
                    },
                    "bound_width_ranges": {
                        "thicknesses": [1.0e-2, 200.0],
                        "roughnesses": [1.0e-2, 30.0],
                        "slds": [1.0e-2, 14.0],
                        "r_scale": [1.0e-3, 0.2],
                        "log10_background": [1.0e-2, 6.0],
                    },
                    "shift_param_config": {
                        "r_scale": True,
                        "log10_background": True,
                    },
                    "model_name": "standard_model",
                    "max_num_layers": 1,
                    "max_total_thickness": 200,
                    "constrained_roughness": True,
                    "max_thickness_share": 0.8,
                    "logdist": False,
                    "scale_params_by_ranges": False,
                    "scaled_range": [-1.0, 1.0],
                    "device": device,
                },
            },
            "q_generator": {
                "cls": "ConstantQ",
                "kwargs": {
                    "q": [0.01, 0.25, 64],
                    "device": device,
                },
            },
            "intensity_noise": {
                "cls": "GaussianExpIntensityNoise",
                "kwargs": {
                    "relative_errors": [0.02, 0.05],
                    "add_to_context": True,
                },
            },
            "curves_scaler": {
                "cls": "LogAffineCurvesScaler",
                "kwargs": {
                    "weight": 0.2,
                    "bias": 1.0,
                    "eps": 1.0e-10,
                },
            },
        },
        "model": {
            "network": {
                "cls": "NFNetwork",
                "pretrained_name": None,
                "device": device,
                "kwargs": {
                    "dim_theta": 7,
                    "dim_conditioning_params": 0,
                    "prior_bounds_input": True,
                    "embedding_net_type": "conv",
                    "embedding_net_kwargs": {
                        "in_channels": 2,
                        "hidden_channels": [8, 16],
                        "kernel_size": 3,
                        "dim_embedding": 64,
                        "dim_avpool": 4,
                        "use_batch_norm": False,
                        "use_se": False,
                        "activation": "gelu",
                    },
                    "pretrained_embedding_net": None,
                    "transform_net_kwargs": {
                        "hidden_features": 64,
                        "num_blocks": 1,
                        "activation": "lrelu",
                        "use_batch_norm": False,
                        "use_layer_norm": True,
                    },
                    "flow_kwargs": {
                        "num_layers": 4,
                        "tail_bound": 8.0,
                        "tails": "linear",
                        "num_bins": 4,
                        "use_batch_norm_transform": False,
                        "use_lu": False,
                    },
                },
            }
        },
        "training": {
            "trainer_cls": "NFlowTrainer",
            "trainer_kwargs": {
                "train_with_bounds": True,
                "train_with_q_input": True,
                "train_with_sigmas": False,
                "condition_on_q_resolutions": False,
                "optim_kwargs": {"betas": [0.9, 0.999], "weight_decay": 0.0},
            },
            "num_iterations": 100,
            "batch_size": 8,
            "lr": 1.0e-4,
            "grad_accumulation_steps": 1,
            "clip_grad_norm_max": None,
            "update_tqdm_freq": 200,
            "optimizer": "AdamW",
            "callbacks": {
                "save_best_model": {"enable": False, "freq": 50},
                "lr_scheduler": {"cls": None},
            },
        },
    }

    config_path = root_dir / "configs" / f"{name}.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    with capsys.disabled():
        print(f"[reflectorch][test] device={device}")
        print(f"[reflectorch][test] experimental_data_dir={experimental_data_dir}")
        print(f"[reflectorch][test] wrote config: {config_path}")

    # Train.
    trainer = get_trainer_from_config(config)
    with capsys.disabled():
        print(f"[reflectorch][test] training for {config['training']['num_iterations']} iterations...")
        trainer.train(
            config["training"]["num_iterations"],
            callbacks=(),
            disable_tqdm=False,
            update_tqdm_freq=10,
        )
        print("[reflectorch][test] training done")

    # Save weights in the canonical location EasyInferenceModel expects.
    model_path = root_dir / "saved_models" / f"model_{name}.pt"
    torch.save({"model": trainer.model.state_dict()}, model_path)
    assert model_path.exists()

    with capsys.disabled():
        print(f"[reflectorch][test] saved model: {model_path}")

    # Reload + run NF sampling inference.
    infer = EasyInferenceModel(
        config_name=name,
        root_dir=str(root_dir),
        weights_format="pt",
        repo_id="",
        device=device,
    )

    with capsys.disabled():
        print("[reflectorch][test] loaded model via EasyInferenceModel")

    batch = cast(dict[str, Any], trainer.loader.get_batch(1))

    scaled_noisy_curves = cast(torch.Tensor, batch["scaled_noisy_curves"])
    q_values_t = cast(torch.Tensor, batch["q_values"])

    curve_unscaled = trainer.loader.curves_scaler.restore(scaled_noisy_curves[0]).detach().cpu().numpy()
    q_values = q_values_t[0].detach().cpu().numpy()

    params_obj = cast(Any, batch["params"])
    min_b = params_obj.min_bounds[0].detach().cpu().numpy()
    max_b = params_obj.max_bounds[0].detach().cpu().numpy()
    prior_bounds = np.stack([min_b, max_b], axis=-1)

    pred = infer.sample(
        num_samples=16,
        reflectivity_curve=curve_unscaled,
        q_values=q_values,
        prior_bounds=prior_bounds,
        clip_prediction=True,
        q_resolution=0.1,
        calc_sampled_curves=False,
        enable_importance_sampling=False,
    )

    assert "predicted_params_array" in pred
    arr = np.asarray(pred["predicted_params_array"])
    assert arr.shape == (16, 7)
    assert np.isfinite(arr).all()

    # `clip_prediction=True` clamps to the sampler's global bounds.
    eps = 1e-6
    global_min = trainer.loader.prior_sampler.min_bounds.squeeze(-1).detach().cpu().numpy()
    global_max = trainer.loader.prior_sampler.max_bounds.squeeze(-1).detach().cpu().numpy()
    assert np.all(arr >= global_min[None, :] - eps)
    assert np.all(arr <= global_max[None, :] + eps)
