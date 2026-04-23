# Synthetic Inference Test Dataset

This directory is meant to be reproducible rather than versioned as raw data. The generated curves, model files, and `dataset_summary.json` can be recreated with `reflectorch/export_synthetic_dataset.py` and seed `42`, so the intended checked-in artifact here is this README.

## Regeneration

Regenerate the local dataset with:

```bash
python -m reflectorch.export_synthetic_dataset \
  --config example_nf_config_reflectorch.yaml \
  --output-dir dataset/synthetic_inference_test \
  --num-samples 2048 \
  --seed 42
```

The exporter can write into an existing directory. Regenerated sample files and
`dataset_summary.json` are overwritten by path, while unrelated files such as
this README are left in place.

## What Gets Generated

The exporter loads the `dset` section of `configs/example_nf_config_reflectorch.yaml`, forces `calc_denoised_curves=True`, builds the dataset with `init_dset(...)`, samples batches until the requested size is reached, and writes one group of files per sample:

- `sXXXXXX_theoretical_curve.dat`
- `sXXXXXX_experimental_curve.dat`
- `sXXXXXX_model.txt`

It also writes `dataset_summary.json` with dataset-level provenance and per-sample metadata.

## Generation Process

For each exported sample, the script:

1. Samples parameters and subprior bounds from the configured prior sampler.
2. Samples the per-curve `q` grid with `VariableQ`.
3. Simulates the reflectivity curve from the sampled parameters.
4. Applies resolution smearing when smearing is enabled.
5. Stores the clean smeared curve as `*_theoretical_curve.dat`.
6. Applies intensity noise and stores the noisy result as `*_experimental_curve.dat`.
7. Writes the layered ground-truth model to `*_model.txt`.
8. Appends the sample metadata record to `dataset_summary.json`.

## Current Dataset Snapshot

The current local export summarized by `dataset_summary.json` was generated on `2026-04-21T11:30:07.290340+00:00` with:

- source config: `configs/example_nf_config_reflectorch.yaml`
- output directory: `dataset/synthetic_inference_test`
- number of samples: `2048`
- paired curves: `true`
- seed: `42`
- parameter labels: `Thickness L1`, `Roughness L1`, `Roughness sub`, `SLD L1`, `SLD sub`, `r_scale`, `log10_background`

Global sampled ranges recorded in the summary:

- thicknesses: `[1.0, 1500.0]`
- roughnesses: `[0.0, 60.0]`
- slds: `[-8.0, 16.0]`
- `r_scale`: `[0.9, 1.1]`
- `log10_background`: `[-10.0, -4.0]`

Global bound-width ranges recorded in the summary:

- thicknesses: `[0.01, 1500.0]`
- roughnesses: `[0.01, 60.0]`
- slds: `[0.01, 24.0]`
- `r_scale`: `[0.001, 0.2]`
- `log10_background`: `[0.01, 6.0]`

Generator settings from the summary:

- `q_generator`: `VariableQ`
- `q_min_range`: `[0.001, 0.02]`
- `q_max_range`: `[0.05, 0.4]`
- `n_q_range`: `[256, 256]`
- `mode`: `equidistant`
- `intensity_noise_class`: `GaussianExpIntensityNoise`
- `smearing_class`: `Smearing`
- `sigma_range`: `[0.01, 0.12]`
- `gauss_num`: `17`
- `share_smeared`: `1.0`
- `constant_dq`: `false`

## File Meanings

`*_theoretical_curve.dat` contains the clean synthetic curve on the exported `q` grid with columns:

- `Q (A^-1)`
- `R`
- `RQ^4 (A^-4)`

`*_experimental_curve.dat` contains the noisy synthetic curve on the same `q` grid with columns:

- `Q (A^-1)`
- `R`
- `dR`
- `dQ (A^-1)`

`*_model.txt` stores the sampled layered structure in a Refl1D-like text format. When nuisance parameters are enabled, they are written as a comment line at the top of the file.

`dataset_summary.json` stores:

- dataset provenance
- generator settings
- parameter labels and global ranges
- per-sample `q` limits and `q_resolution`
- per-sample sampled parameters, prior bounds, standard-model values, and nuisance parameters

## Important Semantics

In this directory, `theoretical` and `experimental` are a paired clean/noisy view of the same sampled synthetic curve:

- `theoretical`: clean, already smeared, not yet intensity-noised
- `experimental`: the same smeared curve after intensity noise

Both files for a sample share the same `q` grid and the same underlying parameters.

`dR` is the uncertainty scale used by the noise generator, not the realized perturbation itself. The realized noise is:

```text
R_noisy - R_clean
```

In this fork, `GaussianNoiseGenerator` samples `torch.normal(mean=0., std=sigmas).clamp_min_(0.0)`, so the Gaussian perturbation is clipped to be nonnegative before it is added to the clean curve.

## Relation To `dataset/test`

This directory is not equivalent to the bundled `dataset/test` layout. Here, `theoretical` means the clean partner of the exported noisy sample. In `dataset/test`, `theoretical` is used more like a dense reference/model curve and `experimental` can live on a different, sparser grid.

## Intended Use

Use this dataset when you want a fixed synthetic inference set that can be regenerated exactly, compared against known ground truth, and reused for repeated evaluation without sampling fresh curves during every run.
