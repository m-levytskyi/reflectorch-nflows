from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from reflectorch.data_generation.priors.parametric_subpriors import BasicParams, SubpriorParametricSampler
from reflectorch.inference.preprocess_exp.interpolation import interp_reflectivity
from reflectorch.ml.basic_trainer import DataLoader
from reflectorch.ml.dataloaders import ReflectivityDataLoader, QWeightedReflectivityDataLoader

__all__ = [
    "ExperimentalReflectivityDataLoader",
    "MixedReflectivityDataLoader",
]


@dataclass(frozen=True)
class ExperimentalSample:
    q: np.ndarray  # shape [N]
    r: np.ndarray  # shape [N]
    dr: Optional[np.ndarray]  # shape [N], may be None if not in file
    params_true: torch.Tensor  # shape [P] on CPU


def _safe_float(x: str) -> Optional[float]:
    x = x.strip()
    if x.lower() in {"inf", "+inf", "-inf", "none", "nan", ""}:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _read_curve_dat(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    # Expected columns: q, R, dR, dQ (headers allowed)
    data = np.loadtxt(str(path), comments="#")
    if data.ndim == 1:
        data = data[None, :]

    if data.shape[1] < 2:
        raise ValueError(f"{path} must have at least 2 columns (q, R)")

    q = data[:, 0].astype(np.float64)
    r = data[:, 1].astype(np.float64)
    dr = data[:, 2].astype(np.float64) if data.shape[1] >= 3 else None
    dq = data[:, 3].astype(np.float64) if data.shape[1] >= 4 else None

    # sort + dedupe q (keep first occurrence)
    order = np.argsort(q)
    q = q[order]
    r = r[order]
    if dr is not None:
        dr = dr[order]
    if dq is not None:
        dq = dq[order]

    _, unique_idx = np.unique(q, return_index=True)
    unique_idx.sort()
    q = q[unique_idx]
    r = r[unique_idx]
    if dr is not None:
        dr = dr[unique_idx]
    if dq is not None:
        dq = dq[unique_idx]

    return q, r, dr, dq


def _read_standard_model_txt(
    path: Path,
    *,
    max_num_layers: int,
    subtract_fronting_sld: bool,
    sld_units_scale: float,
) -> Tuple[List[float], List[float], List[float]]:
    """Parse a Refl1D-like model file with rows: fronting, layer1..layerN, backing.

    Returns (thicknesses, roughnesses, slds) in the *internal* units used by reflectorch models:
    - thicknesses: Angstrom
    - roughnesses: Angstrom (n_layers+1 values)
    - slds: in units of 1e-6 A^-2 (n_layers+1 values: layers + backing; ambient assumed 0)
    """

    rows: List[Tuple[str, Optional[float], Optional[float], Optional[float]]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            layer = parts[0]
            sld = _safe_float(parts[1])
            thickness = _safe_float(parts[2])
            roughness = _safe_float(parts[3])
            rows.append((layer, sld, thickness, roughness))

    if not rows:
        raise ValueError(f"No layer rows found in {path}")

    front = next((r for r in rows if r[0].lower().startswith("front")), None)
    backing = next((r for r in rows if r[0].lower().startswith("back")), None)
    layers = [r for r in rows if r is not front and r is not backing]

    if front is None or backing is None:
        raise ValueError(f"Expected 'fronting' and 'backing' rows in {path}")

    if len(layers) < max_num_layers:
        raise ValueError(
            f"{path} has {len(layers)} layers but config expects max_num_layers={max_num_layers}"
        )

    layers = layers[:max_num_layers]

    # Convert SLD to internal units (1e-6 A^-2)
    # Files typically store in A^-2 (e.g. 3.5e-06); scale by (1e-6)^-1 = 1e6.
    def to_internal_sld(val: float) -> float:
        return val * sld_units_scale

    front_sld = to_internal_sld(front[1] if front[1] is not None else 0.0)

    thicknesses: List[float] = []
    roughnesses: List[float] = []
    slds: List[float] = []

    # Roughnesses are (n_layers + 1): interface fronting-layer1, layer1-layer2, ..., lastlayer-backing
    rough0 = front[3] if front[3] is not None else 0.0
    roughnesses.append(float(rough0))

    for layer in layers:
        layer_sld = to_internal_sld(layer[1] if layer[1] is not None else 0.0)
        if subtract_fronting_sld:
            layer_sld -= front_sld
        slds.append(float(layer_sld))

        thickness = layer[2]
        if thickness is None:
            raise ValueError(f"Layer thickness missing/invalid in {path}: {layer}")
        thicknesses.append(float(thickness))

        # Each layer row in these files typically stores the roughness of the interface below it.
        rough = layer[3] if layer[3] is not None else 0.0
        roughnesses.append(float(rough))

    backing_sld = to_internal_sld(backing[1] if backing[1] is not None else 0.0)
    if subtract_fronting_sld:
        backing_sld -= front_sld
    slds.append(float(backing_sld))

    if len(thicknesses) != max_num_layers:
        raise AssertionError("Unexpected thickness vector length")
    if len(roughnesses) != max_num_layers + 1:
        raise AssertionError("Unexpected roughness vector length")
    if len(slds) != max_num_layers + 1:
        raise AssertionError("Unexpected sld vector length")

    return thicknesses, roughnesses, slds


class ExperimentalReflectivityDataLoader(DataLoader):
    """Loads experimental reflectivity curves from `.dat` files and ground-truth parameters from `_model.txt`.

    It produces batches compatible with `NFlowTrainer` / `PointEstimatorTrainer`:
    - `q_values`: [B, Nq]
    - `scaled_noisy_curves`: [B, Nq]
    - `scaled_params`: [B, 3*P] (params + min_bounds + max_bounds, scaled like `SubpriorParametricSampler`)
    - `q_resolutions`: [B, 1] (dq/q or dq depending on `Smearing.constant_dq`)

    Notes:
    - For performance, this loader assumes a *fixed* q grid (e.g. `ConstantQ`) and pre-interpolates all curves.
    - Bounds are resampled every `get_batch` call to match the on-the-fly subprior setup.
    """

    def __init__(
        self,
        *,
        q_generator,
        prior_sampler: SubpriorParametricSampler,
        intensity_noise=None,
        curves_scaler=None,
        sigma_scaler=None,
        smearing=None,
        q_noise=None,
        data_dir: str | Path,
        curve_glob: str = "*_experimental_curve.dat",
        model_suffix: str = "_model.txt",
        curve_suffix: str = "_experimental_curve.dat",
        background: float = 0.5e-6,
        q_resolution: float = 0.1,
        estimate_r_scale: bool = True,
        subtract_fronting_sld: bool = True,
        sld_units_scale: float = 1e6,
        interpolation_logspace_q: bool = False,
        min_curve_value: float = 1e-10,
        seed: Optional[int] = None,
    ):
        self.q_generator = q_generator
        self.prior_sampler = prior_sampler
        self.intensity_noise = intensity_noise
        self.curves_scaler = curves_scaler
        self.sigma_scaler = sigma_scaler
        self.smearing = smearing
        self.q_noise = q_noise

        self.data_dir = Path(data_dir)
        self.curve_glob = curve_glob
        self.model_suffix = model_suffix
        self.curve_suffix = curve_suffix

        self.background = float(background)
        self.q_resolution = float(q_resolution)
        self.estimate_r_scale = bool(estimate_r_scale)
        self.subtract_fronting_sld = bool(subtract_fronting_sld)
        self.sld_units_scale = float(sld_units_scale)
        self.interpolation_logspace_q = bool(interpolation_logspace_q)
        self.min_curve_value = float(min_curve_value)

        self._rng = np.random.default_rng(seed)

        # Prefer fixed q grid (ConstantQ)
        if hasattr(self.q_generator, "q"):
            q_grid = self.q_generator.q.detach().cpu().numpy().astype(np.float64)
        else:
            # Fallback: pick a single grid from generator once (not ideal but functional)
            q_grid = self.q_generator.get_batch(1).detach().cpu().numpy()[0].astype(np.float64)
        self._q_grid_np = q_grid
        self._q_grid = torch.tensor(q_grid, dtype=torch.float32)

        self._samples: List[ExperimentalSample] = []
        self._interp_curves: Optional[torch.Tensor] = None  # [N, Nq] on CPU
        self._interp_sigmas: Optional[torch.Tensor] = None  # [N, Nq] on CPU, may be None
        self._params_true: Optional[torch.Tensor] = None  # [N, P] on CPU

        self._load_all()
        self._precompute_interpolated_curves()

    @property
    def num_samples(self) -> int:
        return len(self._samples)

    def _load_all(self) -> None:
        curve_paths = sorted(self.data_dir.glob(self.curve_glob))
        if not curve_paths:
            raise FileNotFoundError(f"No experimental curves found in {self.data_dir} with glob {self.curve_glob}")

        # Determine expected model dimensionality
        if not isinstance(self.prior_sampler, SubpriorParametricSampler):
            raise TypeError("ExperimentalReflectivityDataLoader currently requires SubpriorParametricSampler")

        max_num_layers = getattr(self.prior_sampler, "max_num_layers", None) or getattr(self.prior_sampler, "num_layers", None)
        if max_num_layers is None:
            raise ValueError("Could not determine max_num_layers from prior_sampler")

        # Build param vectors
        params_list: List[torch.Tensor] = []

        for curve_path in curve_paths:
            base = curve_path.name
            if not base.endswith(self.curve_suffix):
                continue
            stem = base[: -len(self.curve_suffix)]
            model_path = curve_path.with_name(stem + self.model_suffix)
            if not model_path.is_file():
                raise FileNotFoundError(f"Missing model file for {curve_path}: expected {model_path}")

            q, r, dr, _ = _read_curve_dat(curve_path)

            thicknesses, roughnesses, slds = _read_standard_model_txt(
                model_path,
                max_num_layers=max_num_layers,
                subtract_fronting_sld=self.subtract_fronting_sld,
                sld_units_scale=self.sld_units_scale,
            )

            # Build base StandardModel parameter vector: [d..., sigma...(n+1), sld...(n+1)]
            base_params = np.array(thicknesses + roughnesses + slds, dtype=np.float32)

            # Add nuisance params if enabled by prior_sampler.param_model
            # Wrapper appends nuisance params in the insertion order of shift_param_config.
            nuisance = []
            enabled = getattr(self.prior_sampler.param_model, "enabled_nuisance_params", [])

            # Estimate r_scale from low-q points if requested, else 1.0
            r_scale = 1.0
            if "r_scale" in enabled:
                if self.estimate_r_scale:
                    k = min(10, r.shape[0])
                    # background is additive in linear space
                    r0 = np.clip(r[:k] - self.background, 1e-12, None)
                    r_scale = float(np.median(r0))
                r_scale = float(np.clip(r_scale, 0.0, 10.0))
                nuisance.append(r_scale)

            if "log10_background" in enabled:
                nuisance.append(float(np.log10(self.background)))

            full_params = torch.tensor(
                np.concatenate([base_params, np.asarray(nuisance, dtype=np.float32)], axis=0),
                dtype=torch.float32,
            )

            params_list.append(full_params)
            self._samples.append(
                ExperimentalSample(q=q, r=r, dr=dr, params_true=full_params)
            )

        self._params_true = torch.stack(params_list, dim=0)

        # Basic consistency check against sampler
        expected_dim = int(self.prior_sampler.param_dim)
        if self._params_true.shape[1] != expected_dim:
            raise ValueError(
                f"Experimental params dim mismatch: got {self._params_true.shape[1]}, expected {expected_dim}. "
                f"Check max_num_layers and enabled nuisance params."
            )

    def _precompute_interpolated_curves(self) -> None:
        curves: List[np.ndarray] = []
        sigmas: List[np.ndarray] = []
        has_any_dr = False
        
        for s in self._samples:
            r_interp = interp_reflectivity(
                self._q_grid_np,
                s.q,
                s.r,
                min_value=self.min_curve_value,
                logspace=self.interpolation_logspace_q,
            ).astype(np.float32)
            curves.append(r_interp)
            
            # Interpolate dr values if present
            if s.dr is not None:
                has_any_dr = True
                dr_interp = interp_reflectivity(
                    self._q_grid_np,
                    s.q,
                    s.dr,
                    min_value=self.min_curve_value,
                    logspace=self.interpolation_logspace_q,
                ).astype(np.float32)
                sigmas.append(dr_interp)
            else:
                # Placeholder for samples without dR
                sigmas.append(np.zeros_like(r_interp))

        self._interp_curves = torch.tensor(np.stack(curves, axis=0), dtype=torch.float32)
        
        # Only store sigmas if at least one sample has dR values
        if has_any_dr:
            self._interp_sigmas = torch.tensor(np.stack(sigmas, axis=0), dtype=torch.float32)
        else:
            self._interp_sigmas = None

    def _sample_bounds(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample subprior bounds around provided params.

        Uses the same underlying per-dimension delta ranges as `SubpriorParametricSampler`.
        """
        # prior_sampler.{min_bounds,max_bounds,min_delta,max_delta} shapes: [P, 1]
        min_bounds_global = self.prior_sampler.min_bounds.squeeze(-1).to(params)
        max_bounds_global = self.prior_sampler.max_bounds.squeeze(-1).to(params)
        min_delta = self.prior_sampler.min_delta.squeeze(-1).to(params)
        max_delta = self.prior_sampler.max_delta.squeeze(-1).to(params)

        width = torch.rand_like(params) * (max_delta - min_delta) + min_delta
        u = torch.rand_like(params)

        min_b = params - u * width
        max_b = min_b + width

        min_b = torch.maximum(min_b, min_bounds_global)
        max_b = torch.minimum(max_b, max_bounds_global)

        # Ensure truth is inside the interval after clamping
        min_b = torch.minimum(min_b, params)
        max_b = torch.maximum(max_b, params)

        return min_b, max_b

    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if self.num_samples <= 0:
            raise RuntimeError("No experimental samples loaded")

        idx = torch.randint(0, self.num_samples, (batch_size,), dtype=torch.int64)

        device = None
        if hasattr(self.q_generator, "q"):
            device = self.q_generator.q.device
        elif hasattr(self.prior_sampler, "device"):
            device = torch.device(self.prior_sampler.device)
        else:
            device = torch.device("cpu")

        q_values = self._q_grid.to(device=device)
        q_values = q_values[None, :].expand(batch_size, q_values.shape[0])

        curves = self._interp_curves[idx].to(device=device)

        if self.curves_scaler is not None:
            scaled_noisy_curves = self.curves_scaler.scale(curves, q_values)
        else:
            scaled_noisy_curves = curves
        
        # Handle sigmas if available
        scaled_sigmas = None
        if self._interp_sigmas is not None and self.sigma_scaler is not None:
            sigmas = self._interp_sigmas[idx].to(device=device)
            # Apply Q-weighted sigma transformation: dR' = dR * Q^(-beta) / (R * ln(10))
            scaled_sigmas = self.sigma_scaler.scale(sigmas, curves, q_values)

        params = self._params_true[idx].to(device=device)
        min_b, max_b = self._sample_bounds(params)

        params_obj = BasicParams(
            parameters=params,
            min_bounds=min_b,
            max_bounds=max_b,
            max_num_layers=self.prior_sampler.max_num_layers,
            param_model=self.prior_sampler.param_model,
        )

        scaled_params = self.prior_sampler.scale_params(params_obj)

        q_resolutions = torch.full((batch_size, 1), self.q_resolution, device=device, dtype=torch.float32)

        batch_dict = {
            "q_values": q_values,
            "scaled_noisy_curves": scaled_noisy_curves,
            "scaled_params": scaled_params,
            "q_resolutions": q_resolutions,
        }
        
        # Add scaled_sigmas if available
        if scaled_sigmas is not None:
            batch_dict["scaled_sigmas"] = scaled_sigmas
        
        return batch_dict


class MixedReflectivityDataLoader(DataLoader):
    """Mixes synthetic on-the-fly curves (via `ReflectivityDataLoader`) with experimental curves.

    This is a thin wrapper that returns the concatenated batch along the batch dimension.
    """

    def __init__(
        self,
        *,
        q_generator,
        prior_sampler,
        intensity_noise=None,
        curves_scaler=None,
        sigma_scaler=None,
        smearing=None,
        q_noise=None,
        mix_fraction: float = 0.5,
        # Experimental loader params
        data_dir: str | Path,
        curve_glob: str = "*_experimental_curve.dat",
        model_suffix: str = "_model.txt",
        curve_suffix: str = "_experimental_curve.dat",
        background: float = 0.5e-6,
        q_resolution: float = 0.1,
        estimate_r_scale: bool = True,
        subtract_fronting_sld: bool = True,
        sld_units_scale: float = 1e6,
        interpolation_logspace_q: bool = False,
        min_curve_value: float = 1e-10,
        seed: Optional[int] = None,
        # Synthetic loader passthrough
        synthetic_kwargs: Optional[dict] = None,
    ):
        self.q_generator = q_generator
        self.prior_sampler = prior_sampler
        self.intensity_noise = intensity_noise
        self.curves_scaler = curves_scaler
        self.sigma_scaler = sigma_scaler
        self.smearing = smearing
        self.q_noise = q_noise

        self.mix_fraction = float(mix_fraction)
        if not (0.0 <= self.mix_fraction <= 1.0):
            raise ValueError("mix_fraction must be in [0,1]")

        self.experimental = ExperimentalReflectivityDataLoader(
            q_generator=q_generator,
            prior_sampler=prior_sampler,
            intensity_noise=intensity_noise,
            curves_scaler=curves_scaler,
            sigma_scaler=sigma_scaler,
            smearing=smearing,
            q_noise=q_noise,
            data_dir=data_dir,
            curve_glob=curve_glob,
            model_suffix=model_suffix,
            curve_suffix=curve_suffix,
            background=background,
            q_resolution=q_resolution,
            estimate_r_scale=estimate_r_scale,
            subtract_fronting_sld=subtract_fronting_sld,
            sld_units_scale=sld_units_scale,
            interpolation_logspace_q=interpolation_logspace_q,
            min_curve_value=min_curve_value,
            seed=seed,
        )

        synthetic_kwargs = synthetic_kwargs or {}
        
        # Use QWeightedReflectivityDataLoader if sigma_scaler is provided for consistency
        if sigma_scaler is not None:
            self.synthetic = QWeightedReflectivityDataLoader(
                q_generator=q_generator,
                prior_sampler=prior_sampler,
                intensity_noise=intensity_noise,
                curves_scaler=curves_scaler,
                sigma_scaler=sigma_scaler,
                smearing=smearing,
                q_noise=q_noise,
                **synthetic_kwargs,
            )
        else:
            self.synthetic = ReflectivityDataLoader(
                q_generator=q_generator,
                prior_sampler=prior_sampler,
                intensity_noise=intensity_noise,
                curves_scaler=curves_scaler,
                smearing=smearing,
                q_noise=q_noise,
                **synthetic_kwargs,
            )

    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        n_exp = int(round(batch_size * self.mix_fraction))
        n_syn = batch_size - n_exp

        batch_parts: List[Dict[str, torch.Tensor]] = []
        if n_syn > 0:
            batch_parts.append(self.synthetic.get_batch(n_syn))
        if n_exp > 0:
            batch_parts.append(self.experimental.get_batch(n_exp))

        if len(batch_parts) == 1:
            return batch_parts[0]

        keys = set().union(*(b.keys() for b in batch_parts))
        out: Dict[str, torch.Tensor] = {}
        for k in keys:
            vals = [b.get(k, None) for b in batch_parts]
            if any(v is None for v in vals):
                # Only include keys present in all parts
                continue
            out[k] = torch.cat(vals, dim=0)
        return out
