# Blending experimental `.dat` curves into NF training (reflectorch)

This document explains the changes made to support training Normalizing Flows (NF) on a mixture of:

- **Synthetic, on-the-fly simulated reflectivity curves** (the existing `ReflectivityDataLoader` path), and
- **Real experimental curves** provided as `*.dat` files, paired with a `*_model.txt` containing the “true” parameters.

It also documents a small inference fix and the test coverage added to validate end-to-end training → save → load → inference.

## What was the goal?

You trained an NF using `configs/example_nf_config_reflectorch.yaml` and wanted to “blend” real data into training.

Constraints you provided:

- Experimental curves are in `.dat`.
- There is a corresponding `_model.txt` with the true parameters, but **no prior bounds**.
- The bounds should be generated **exactly like the existing synthetic data generation** (subprior bounds sampled per curve).
- Experimental curves must be interpolated onto the model q-grid.
- Fixed experimental settings:
  - Resolution: **$dQ/Q = 0.1$** (10%).
  - Background: around **$5\times 10^{-7}$**.
- Training must still work on your GPU (GTX 1080 / sm_61).

## Critical vs optional changes

### Critical changes (required for the new workflow)

1) **Experimental + mixed dataloaders**

- Added `reflectorch/ml/experimental_dataloaders.py` with:
  - `ExperimentalReflectivityDataLoader`: loads your experimental files, interpolates curves to the training q-grid, constructs truth parameters, and generates subprior bounds in the same style as synthetic generation.
  - `MixedReflectivityDataLoader`: wraps a synthetic `ReflectivityDataLoader` and the experimental loader, and mixes batches by concatenation.

2) **Export for YAML config resolution**

- Updated `reflectorch/ml/__init__.py` to export:
  - `ExperimentalReflectivityDataLoader`
  - `MixedReflectivityDataLoader`

This is important because your training pipeline is YAML-driven (class names as strings).

3) **More robust class resolution for config instantiation**

- Updated `reflectorch/runs/utils.py`:
  - `init_from_conf()` and `init_dset()` now resolve `cls` names via the fully imported `reflectorch` package first, with a fallback to local `globals()`.

This avoids intermittent “class not found” issues that can appear when the import graph is partially initialized (circular-import timing).

4) **Example YAML config for mixed training**

- Added `configs/example_nf_config_experimental_mixed.yaml` showing how to train an NF with a mixed loader and fixed q-grid.

### Optional changes (not strictly required, but improve correctness / maintainability)

5) **Inference fix for `ConstantQ`**

- Updated `reflectorch/inference/inference_model.py` (`EasyInferenceModel.predict()` and `.sample()`):
  - If the trainer uses a `ConstantQ` generator and `q` is stored as a 1D tensor, it is expanded to shape `[1, Nq]`.

This prevents dimension mismatches when the NF network concatenates q-values as part of its conditioning input.

6) **Tests + fixtures**

- Added minimal fixtures under `tests/data/experimental/`.
- Added unit tests validating loader batch shape/contracts.
- Added a gated integration test that trains briefly and runs NF inference.

These tests aren’t required to *use* the feature, but they prevent regression.

## File-by-file inventory

### 1) `reflectorch/ml/experimental_dataloaders.py` (new)

#### `ExperimentalReflectivityDataLoader`

**What it reads**

- Experimental curves: `*_experimental_curve.dat` (configurable).
- Ground truth params: matching `*_model.txt` (configurable).

**Expected `.dat` format**

The loader accepts a numeric whitespace-separated table. Only the first two columns are required:

- Column 1: $q$ (Å$^{-1}$)
- Column 2: $R$ (reflectivity)
- Column 3 (optional): $dR$ (ignored by this loader right now)
- Column 4 (optional): $dQ$ (ignored by this loader right now)

Notes:

- It sorts by q and removes duplicate q entries.
- It interpolates your reflectivity curve onto the model q-grid using `reflectorch.inference.preprocess_exp.interpolation.interp_reflectivity`.

**Expected `_model.txt` format**

`_read_standard_model_txt()` expects a Refl1D-like table with at least these rows:

- A row whose first token starts with `front` (fronting)
- One or more layer rows (e.g. `layer1`)
- A row whose first token starts with `back` (backing)

Each row is expected to have (at least) four columns:

1. name
2. SLD in Å$^{-2}$ (often written as e.g. `3.5e-06`)
3. thickness in Å (layers only)
4. roughness in Å

The loader:

- Converts SLD values into reflectorch “internal units” (default scale factor `1e6`, i.e. Å$^{-2}$ → $10^{-6}$ Å$^{-2}$ units).
- Optionally subtracts the fronting SLD from all SLDs (default `subtract_fronting_sld=True`) to match common “relative-to-ambient/fronting” conventions.

**How it builds the parameter vector**

It constructs parameters for `standard_model` in this order:

- thicknesses: `max_num_layers` values
- roughnesses: `max_num_layers + 1` values
- slds: `max_num_layers + 1` values (layers + backing; ambient is not included)

Then, if the `SubpriorParametricSampler` enables nuisance params (e.g. via `shift_param_config`), it appends nuisance parameters:

- `r_scale`: estimated from the low-q points (simple heuristic) if enabled and `estimate_r_scale=True`.
- `log10_background`: set to `log10(background)` if enabled.

**How bounds are generated (important)**

Your `_model.txt` contains truth parameters but no bounds. For NF training, the trainer expects per-sample subprior bounds.

This loader generates bounds per batch using the sampler’s configured global ranges:

- Uses the sampler’s `min_bounds`, `max_bounds`, `min_delta`, `max_delta`.
- Samples a random width per parameter in `[min_delta, max_delta]`.
- Samples a random placement around the truth value.
- Clamps bounds to global `[min_bounds, max_bounds]`.
- Ensures the truth stays inside `[min_b, max_b]` after clamping.

This mirrors the *style* of subprior bound generation used during synthetic sampling: “truth + random window”, bounded by global ranges.

**Batch contract output**

`get_batch(batch_size)` returns a dict compatible with `NFlowTrainer`:

- `q_values`: `[B, Nq]` (from a fixed q grid)
- `scaled_noisy_curves`: `[B, Nq]` (currently the experimental curve; scaled via `curves_scaler` if configured)
- `scaled_params`: `[B, 3*P]` (params + min_bounds + max_bounds, scaled by `SubpriorParametricSampler.scale_params()`)
- `q_resolutions`: `[B, 1]` (constant `q_resolution`, your $dQ/Q=0.1$)

#### `MixedReflectivityDataLoader`

- Produces a batch by calling:
  - `ReflectivityDataLoader.get_batch(n_syn)`
  - `ExperimentalReflectivityDataLoader.get_batch(n_exp)`
- Concatenates tensors along the batch dimension.
- The split is determined by `mix_fraction` (experimental share), with rounding.

### 2) `reflectorch/runs/utils.py` (modified)

Why this was necessary:

- The repo uses YAML configs with `cls` strings.
- `init_from_conf()` / `init_dset()` originally used `globals()` only.
- With circular imports, new classes (like `MixedReflectivityDataLoader`) can fail to resolve depending on import timing.

What changed:

- Both now try `getattr(reflectorch, cls_name)` first, then fall back to `globals()`.
- Clear `ValueError` if resolution fails.

### 3) `reflectorch/ml/__init__.py` (modified)

- Imports `reflectorch.ml.experimental_dataloaders` and includes both new loader classes in `__all__`.

This makes them discoverable for the config system.

### 4) `configs/example_nf_config_experimental_mixed.yaml` (new)

This config demonstrates:

- `dset.cls: MixedReflectivityDataLoader`
- `dset.kwargs.data_dir`: points to `tests/data/experimental` by default
- Fixed values you requested:
  - `q_resolution: 0.1`
  - `background: 5e-7`
- A fixed q grid via `ConstantQ` (recommended for experimental pre-interpolation)

GPU note included in the file:

- GTX 1080 (sm_61) requires a PyTorch build that includes sm_61.
- Newer CUDA wheels (e.g. many `cu128` builds) may not include sm_61; in that case, use a `cu118` wheel.

### 5) `reflectorch/inference/inference_model.py` (modified)

- For `ConstantQ`, ensures `q_values` is 2D: `[1, Nq]`.
- Prevents errors like: “Tensors must have same number of dimensions” during NF sampling/prediction.

### 6) Tests and fixtures (new)

- `tests/data/experimental/sample_experimental_curve.dat`
- `tests/data/experimental/sample_model.txt`
- `tests/unit/test_experimental_dataloader.py`
  - Verifies keys and shapes for experimental and mixed loaders.

- `tests/integration/test_nf_train_and_infer.py`
  - **Gated** (skipped unless enabled) to avoid slow default test runs.
  - Trains a very small NF for 100 iterations.
  - Saves weights to `model_<name>.pt`.
  - Reloads with `EasyInferenceModel(weights_format="pt", repo_id=None)`.
  - Runs `infer.sample(...)` and checks finiteness + bounds (with `clip_prediction=True`).

## How to use this for your own data

### Naming convention

By default, the experimental loader expects paired files:

- `<id>_experimental_curve.dat`
- `<id>_model.txt`

You can change this in YAML:

- `curve_glob`
- `curve_suffix`
- `model_suffix`

### Required YAML pieces

Minimum required pieces for mixed training:

- `dset.cls: MixedReflectivityDataLoader`
- `dset.kwargs.data_dir: /path/to/your/experimental/files`
- `dset.q_generator.cls: ConstantQ` (strongly recommended)
- `dset.prior_sampler.cls: SubpriorParametricSampler`

### Fixed experimental parameters

You requested fixed values:

- `q_resolution = 0.1` (interpreted as $dQ/Q$)
- `background ≈ 5e-7`

These are fed into:

- The loader’s returned `q_resolutions` tensor.
- The nuisance param `log10_background` (if enabled) and `r_scale` estimation (if enabled).

## Challenges / obstacles encountered (and what we did)

### 1) GPU compatibility (GTX 1080 / sm_61)

Problem:

- Your GPU is sm_61.
- A PyTorch build compiled without sm_61 will fail at runtime (CUDA kernel image not available).

Resolution:

- Use a PyTorch wheel that includes sm_61 (commonly CUDA 11.8 builds do).
- The example config includes a reminder comment because the exact working wheel depends on your environment.

### 2) Config class resolution issues

Problem:

- YAML instantiation occasionally couldn’t resolve newly added classes due to import timing.

Resolution:

- `reflectorch/runs/utils.py` now resolves classes from the imported `reflectorch` package first.

### 3) `ConstantQ` inference shape mismatch

Problem:

- `ConstantQ.q` was a 1D tensor in some code paths.
- The NF network expects q-values with a batch dimension.

Resolution:

- Expand `q` to `[1, Nq]` in `EasyInferenceModel.predict()` / `.sample()`.

## How to run tests

From the repo root:

- Unit tests (fast):

  - `pytest -q tests/unit/test_experimental_dataloader.py`

- Integration test (slow; trains 100 iters):

  - `REFLECTORCH_RUN_TRAIN_TEST=1 pytest -q tests/integration/test_nf_train_and_infer.py`

## Known limitations / caveats

- The experimental loader currently **ignores** per-point `dR` / `dQ` columns from the `.dat` file and uses a fixed `q_resolution` as requested.
- `r_scale` estimation is a simple heuristic (median of low-q reflectivity after subtracting background). If you want a more principled estimate (e.g. quick fit), that would be a separate change.
- The `_model.txt` parser is intentionally strict about having fronting/backing rows and `max_num_layers` layers; if your format differs, we’ll need to adapt `_read_standard_model_txt()`.

## Next steps (if you want)

- Extend the experimental loader to optionally consume `dR` (for weighting) and/or `dQ` (per-point resolution) when present.
- Support alternative `_model.txt` formats by making the parser pluggable.
- Improve nuisance estimation (background/r_scale) via an initial fast fit.
