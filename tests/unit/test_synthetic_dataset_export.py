import json
from pathlib import Path

import numpy as np
import torch
import yaml
from reflectorch.export_synthetic_dataset import export_synthetic_dataset
from reflectorch.inference.inference_model import EasyInferenceModel
from reflectorch.runs.config import load_config
from reflectorch.runs.utils import get_trainer_from_config


def _write_config(path: Path, config: dict) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return path


def _example_config_on_cpu() -> dict:
    config = load_config("example_nf_config_reflectorch")
    config["dset"]["prior_sampler"]["kwargs"]["device"] = "cpu"
    config["dset"]["q_generator"]["kwargs"]["device"] = "cpu"
    config["model"]["network"]["device"] = "cpu"
    return config


def _tiny_export_config(root_dir: Path) -> dict:
    return {
        "general": {
            "name": "pytest_synthetic_export",
            "root_dir": str(root_dir),
        },
        "dset": {
            "cls": "ReflectivityDataLoader",
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
                    "device": "cpu",
                },
            },
            "q_generator": {
                "cls": "VariableQ",
                "kwargs": {
                    "q_min_range": [0.001, 0.02],
                    "q_max_range": [0.05, 0.4],
                    "n_q_range": [32, 32],
                    "device": "cpu",
                },
            },
            "intensity_noise": {
                "cls": "GaussianExpIntensityNoise",
                "kwargs": {
                    "relative_errors": [0.02, 0.05],
                    "add_to_context": True,
                },
            },
            "smearing": {
                "cls": "Smearing",
                "kwargs": {
                    "sigma_range": [0.01, 0.12],
                    "gauss_num": 9,
                    "share_smeared": 1.0,
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
                "device": "cpu",
                "kwargs": {
                    "dim_theta": 7,
                    "dim_conditioning_params": 1,
                    "prior_bounds_input": True,
                    "embedding_net_type": "conv",
                    "embedding_net_kwargs": {
                        "in_channels": 2,
                        "hidden_channels": [8, 16],
                        "kernel_size": 3,
                        "dim_embedding": 32,
                        "dim_avpool": 4,
                        "use_batch_norm": False,
                        "use_se": False,
                        "activation": "gelu",
                    },
                    "pretrained_embedding_net": None,
                    "transform_net_kwargs": {
                        "hidden_features": 32,
                        "num_blocks": 1,
                        "activation": "lrelu",
                        "use_batch_norm": False,
                        "use_layer_norm": True,
                    },
                    "flow_kwargs": {
                        "num_layers": 2,
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
                "condition_on_q_resolutions": True,
                "optim_kwargs": {"betas": [0.9, 0.999], "weight_decay": 0.0},
            },
            "num_iterations": 1,
            "batch_size": 2,
            "lr": 1.0e-4,
            "grad_accumulation_steps": 1,
            "clip_grad_norm_max": None,
            "update_tqdm_freq": 1,
            "optimizer": "AdamW",
            "callbacks": {
                "save_best_model": {"enable": False, "freq": 50},
                "lr_scheduler": {"cls": None},
            },
        },
    }


def test_export_synthetic_dataset_matches_dataset_layout(tmp_path: Path):
    config = _example_config_on_cpu()
    config_path = _write_config(tmp_path / "example_cpu.yaml", config)

    output_dir = export_synthetic_dataset(
        config_name_or_path=str(config_path),
        output_dir=tmp_path / "synthetic_test",
        num_samples=8,
        seed=7,
    )

    summary_path = output_dir / "dataset_summary.json"
    assert summary_path.exists()

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["num_samples"] == 8
    assert summary["paired_curves"] is True
    assert len(summary["samples"]) == 8

    theoretical_files = sorted(output_dir.glob("*_theoretical_curve.dat"))
    experimental_files = sorted(output_dir.glob("*_experimental_curve.dat"))
    model_files = sorted(output_dir.glob("*_model.txt"))

    assert len(theoretical_files) == 8
    assert len(experimental_files) == 8
    assert len(model_files) == 8

    theoretical = np.loadtxt(theoretical_files[0], comments="#")
    experimental = np.loadtxt(experimental_files[0], comments="#")
    model_text = model_files[0].read_text(encoding="utf-8")

    assert theoretical.shape[1] == 3
    assert experimental.shape[1] == 4
    assert np.allclose(theoretical[:, 0], experimental[:, 0])
    assert np.allclose(theoretical[:, 2], theoretical[:, 1] * theoretical[:, 0] ** 4)
    assert np.all(np.isfinite(experimental))
    assert np.all(experimental[:, 1:] >= 0.0)
    assert "#layer" in model_text
    assert "fronting" in model_text
    assert "layer1" in model_text
    assert "backing" in model_text


def test_exported_noisy_curve_runs_through_inference(tmp_path: Path):
    torch.manual_seed(1)
    np.random.seed(1)

    config = _tiny_export_config(tmp_path)
    config_path = _write_config(tmp_path / "tiny_export.yaml", config)

    output_dir = export_synthetic_dataset(
        config_name_or_path=str(config_path),
        output_dir=tmp_path / "synthetic_infer",
        num_samples=4,
        seed=11,
    )

    with open(output_dir / "dataset_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    first_sample = summary["samples"][0]
    sample_id = first_sample["sample_id"]

    noisy_curve = np.loadtxt(
        output_dir / f"{sample_id}_experimental_curve.dat", comments="#"
    )
    q_values = noisy_curve[:, 0]
    reflectivity = noisy_curve[:, 1]

    prior_bounds = np.array(
        [first_sample["prior_bounds"][label] for label in summary["param_labels"]],
        dtype=np.float32,
    )

    trainer = get_trainer_from_config(config)
    infer = EasyInferenceModel(trainer=trainer, device="cpu")

    pred = infer.sample(
        num_samples=4,
        reflectivity_curve=reflectivity,
        q_values=q_values,
        prior_bounds=prior_bounds,
        q_resolution=first_sample["q_resolution"],
        clip_prediction=True,
        calc_sampled_curves=False,
        enable_importance_sampling=False,
    )

    arr = np.asarray(pred["predicted_params_array"])
    assert arr.shape == (4, 7)
    assert np.isfinite(arr).all()
