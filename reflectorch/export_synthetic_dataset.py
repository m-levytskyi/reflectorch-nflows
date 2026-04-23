from __future__ import annotations

import copy
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch
import yaml

from reflectorch.data_generation.priors.parametric_models import NuisanceParamsWrapper
from reflectorch.paths import ROOT_DIR
from reflectorch.runs.config import load_config
from reflectorch.runs.utils import init_dset

EXPERIMENTAL_HEADER = "#  Q(A^-1)        R           dR         dQ(A^-1)"
THEORETICAL_HEADER = "# Q (A^-1),  R, RQ^4 (A^-4)"
MODEL_HEADER = "#layer        sld(A^-2)   thickness(A) roughness(A)"


def _load_config_from_name_or_path(config_name_or_path: str) -> dict:
    """Load a config from a YAML path or the package config directory."""
    config_path = Path(config_name_or_path)
    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config["config_path"] = str(config_path.resolve())
        return config

    return load_config(config_name_or_path)


def _set_seed(seed: int | None) -> None:
    """Seed Python, NumPy, and Torch RNGs when reproducible export is requested."""
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_empty_output_dir(output_dir: Path) -> None:
    """Create the output directory and refuse to write into a non-empty target."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory {output_dir} already exists and is not empty."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _to_numpy(tensor_or_array: Any) -> np.ndarray:
    """Convert tensors or array-like inputs to detached CPU NumPy arrays."""
    if isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array
    if torch.is_tensor(tensor_or_array):
        return tensor_or_array.detach().cpu().numpy()
    return np.asarray(tensor_or_array)


def _restore_noisy_curves(batch_data: dict, loader) -> np.ndarray:
    """Recover noisy physical-space curves from a sampled batch."""
    noisy_curves = batch_data.get("noisy_curves")
    if noisy_curves is not None:
        return _to_numpy(noisy_curves)

    scaled_noisy_curves = batch_data["scaled_noisy_curves"]
    q_values = batch_data["q_values"]
    restored = loader.curves_scaler.restore(scaled_noisy_curves, q_values)
    return _to_numpy(restored)


def _write_model_file(
    path: Path,
    *,
    thicknesses: np.ndarray,
    roughnesses: np.ndarray,
    slds_internal: np.ndarray,
    nuisance_params: dict[str, float],
) -> None:
    """Write one model file matching the conventions used in `dataset/test`."""
    slds = slds_internal * 1.0e-6
    lines = [MODEL_HEADER]

    if nuisance_params:
        nuisance_comment = ", ".join(
            f"{key}={value:.8g}" for key, value in nuisance_params.items()
        )
        lines.append(f"# nuisance: {nuisance_comment}")

    lines.append(f"{'fronting':<12} {0.0:>12.5e} {'inf':>10} {roughnesses[0]:>10.2f}")
    lines.extend(
        f"{f'layer{i}':<12} {sld:>12.5e} {thickness:>10.2f} {roughnesses[i]:>10.2f}"
        for i, (thickness, sld) in enumerate(zip(thicknesses, slds[:-1]), start=1)
    )
    lines.append(f"{'backing':<12} {slds[-1]:>12.5e} {'inf':>10} {'none':>10}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sample_record(
    *,
    sample_id: str,
    q_values: np.ndarray,
    q_resolution: float,
    parameters: np.ndarray,
    min_bounds: np.ndarray,
    max_bounds: np.ndarray,
    thicknesses: np.ndarray,
    roughnesses: np.ndarray,
    slds: np.ndarray,
    param_labels: list[str],
    nuisance_params: dict[str, float],
) -> dict[str, Any]:
    """Build the per-sample metadata record stored in the dataset summary JSON."""
    return {
        "sample_id": sample_id,
        "n_q": int(q_values.shape[0]),
        "q_min": float(q_values.min()),
        "q_max": float(q_values.max()),
        "q_resolution": float(q_resolution),
        "parameters": {
            label: float(value)
            for label, value in zip(param_labels, parameters.tolist())
        },
        "prior_bounds": {
            label: [float(lo), float(hi)]
            for label, lo, hi in zip(
                param_labels, min_bounds.tolist(), max_bounds.tolist()
            )
        },
        "standard_model": {
            "thicknesses": [float(x) for x in thicknesses.tolist()],
            "roughnesses": [float(x) for x in roughnesses.tolist()],
            "slds_internal": [float(x) for x in slds.tolist()],
            "slds_a_minus_2": [float(x * 1.0e-6) for x in slds.tolist()],
        },
        "nuisance_params": nuisance_params,
    }


def _write_curve_data(path: Path, data: np.ndarray, header: str) -> None:
    """Write a curve table with a single header line and scientific notation columns."""
    np.savetxt(path, data, fmt="%.8e", header=header, comments="")


def _nuisance_dict(param_model, parameter_values: np.ndarray) -> dict[str, float]:
    """Extract enabled nuisance params from a sampled parameter vector."""
    if not isinstance(param_model, NuisanceParamsWrapper):
        return {}

    base_dim = param_model.base_model.param_dim
    return {
        name: float(parameter_values[base_dim + i])
        for i, name in enumerate(param_model.enabled_nuisance_params)
    }


def _batch_arrays(batch_data: dict, loader) -> dict[str, np.ndarray | None]:
    """Convert the exported batch fields needed by the writer into NumPy arrays."""
    return {
        "q_values": _to_numpy(batch_data["q_values"]),
        "clean_curves": _to_numpy(batch_data["curves"]),
        "noisy_curves": _restore_noisy_curves(batch_data, loader),
        "sigmas": _to_numpy(batch_data["sigmas"]) if "sigmas" in batch_data else None,
        "q_resolutions": (
            _to_numpy(batch_data["q_resolutions"])
            if "q_resolutions" in batch_data
            else None
        ),
    }


def export_synthetic_dataset(
    config_name_or_path: str = "example_nf_config_reflectorch.yaml",
    output_dir: str | Path | None = None,
    num_samples: int = 2048,
    seed: int | None = None,
) -> Path:
    """Export paired clean/noisy synthetic curves and metadata for inference testing."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    _set_seed(seed)

    config = _load_config_from_name_or_path(config_name_or_path)
    dset_config = copy.deepcopy(config["dset"])
    dset_config.setdefault("kwargs", {})
    dset_config["kwargs"]["calc_denoised_curves"] = True

    loader = init_dset(dset_config)

    output_path = (
        Path(output_dir)
        if output_dir is not None
        else Path(config["general"].get("root_dir") or ROOT_DIR)
        / "dataset"
        / "synthetic_inference_test"
    )
    _ensure_empty_output_dir(output_path)

    param_model = loader.prior_sampler.param_model
    param_labels = param_model.get_param_labels()
    smearing = loader.smearing

    sample_records: list[dict[str, Any]] = []
    generated = 0

    while generated < num_samples:
        batch_size = min(256, num_samples - generated)
        batch_data = loader.get_batch(batch_size)
        params = batch_data["params"]
        arrays = _batch_arrays(batch_data, loader)
        q_values = arrays["q_values"]
        clean_curves = arrays["clean_curves"]
        noisy_curves = arrays["noisy_curves"]
        sigmas = arrays["sigmas"]
        q_resolutions = arrays["q_resolutions"]
        parameters = _to_numpy(params.parameters)
        min_bounds = _to_numpy(params.min_bounds)
        max_bounds = _to_numpy(params.max_bounds)
        thicknesses = _to_numpy(params.thicknesses)
        roughnesses = _to_numpy(params.roughnesses)
        slds = _to_numpy(params.slds)

        for i in range(batch_size):
            sample_num = generated + i + 1
            sample_id = f"s{sample_num:06d}"
            q_row = q_values[i]
            clean_curve = clean_curves[i]
            noisy_curve = noisy_curves[i]
            sigma_row = (
                sigmas[i]
                if sigmas is not None
                else np.full_like(noisy_curve, np.nan, dtype=np.float64)
            )
            q_resolution = (
                float(q_resolutions[i, 0]) if q_resolutions is not None else 0.0
            )
            dq_row = (
                np.full_like(q_row, q_resolution)
                if smearing is not None and smearing.constant_dq
                else q_row * q_resolution
            )

            theoretical_data = np.column_stack(
                [q_row, clean_curve, clean_curve * q_row**4]
            )
            experimental_data = np.column_stack([q_row, noisy_curve, sigma_row, dq_row])

            _write_curve_data(
                output_path / f"{sample_id}_theoretical_curve.dat",
                theoretical_data,
                header=THEORETICAL_HEADER,
            )
            _write_curve_data(
                output_path / f"{sample_id}_experimental_curve.dat",
                experimental_data,
                header=EXPERIMENTAL_HEADER,
            )

            nuisance_params = _nuisance_dict(param_model, parameters[i])
            _write_model_file(
                output_path / f"{sample_id}_model.txt",
                thicknesses=thicknesses[i],
                roughnesses=roughnesses[i],
                slds_internal=slds[i],
                nuisance_params=nuisance_params,
            )

            sample_records.append(
                _sample_record(
                    sample_id=sample_id,
                    q_values=q_row,
                    q_resolution=q_resolution,
                    parameters=parameters[i],
                    min_bounds=min_bounds[i],
                    max_bounds=max_bounds[i],
                    thicknesses=thicknesses[i],
                    roughnesses=roughnesses[i],
                    slds=slds[i],
                    param_labels=param_labels,
                    nuisance_params=nuisance_params,
                )
            )

        generated += batch_size

    summary = {
        "source_config": str(Path(config["config_path"]).resolve()),
        "output_dir": str(output_path.resolve()),
        "num_samples": num_samples,
        "paired_curves": True,
        "seed": seed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "param_labels": param_labels,
        "global_param_ranges": loader.prior_sampler.param_ranges,
        "global_bound_width_ranges": loader.prior_sampler.bound_width_ranges,
        "q_generator": {
            "class": loader.q_generator.__class__.__name__,
            "settings": {
                "q_min_range": getattr(loader.q_generator, "q_min_range", None),
                "q_max_range": getattr(loader.q_generator, "q_max_range", None),
                "n_q_range": getattr(loader.q_generator, "n_q_range", None),
                "mode": getattr(loader.q_generator, "mode", None),
            },
        },
        "noise": {
            "intensity_noise_class": (
                loader.intensity_noise.__class__.__name__
                if loader.intensity_noise is not None
                else None
            ),
            "smearing_class": smearing.__class__.__name__
            if smearing is not None
            else None,
            "smearing_settings": (
                {
                    "sigma_range": [smearing.sigma_min, smearing.sigma_max],
                    "gauss_num": smearing.gauss_num,
                    "share_smeared": smearing.share_smeared,
                    "constant_dq": smearing.constant_dq,
                }
                if smearing is not None
                else None
            ),
        },
        "samples": sample_records,
    }

    with open(output_path / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return output_path


@click.command()
@click.option(
    "--config",
    "config_name_or_path",
    default="example_nf_config_reflectorch.yaml",
    show_default=True,
    help="Config name in configs/ or a direct YAML path.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory where the synthetic dataset will be written.",
)
@click.option(
    "--num-samples",
    type=int,
    default=2048,
    show_default=True,
    help="Number of paired clean/noisy samples to export.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Optional random seed for reproducible export.",
)
def run_export_synthetic_dataset(
    config_name_or_path: str,
    output_dir: Path | None,
    num_samples: int,
    seed: int | None,
) -> None:
    """CLI wrapper around `export_synthetic_dataset`."""
    output_path = export_synthetic_dataset(
        config_name_or_path=config_name_or_path,
        output_dir=output_dir,
        num_samples=num_samples,
        seed=seed,
    )
    click.echo(f"Exported {num_samples} samples to {output_path}")


if __name__ == "__main__":
    run_export_synthetic_dataset()
