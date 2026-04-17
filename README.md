# Reflectorch — Normalizing Flows for Reflectometry

This repository extends the [reflectorch](https://github.com/schreiber-lab/reflectorch) package — a machine learning Python package for the analysis of X-ray and neutron reflectometry data — with support for normalizing flows models and several custom input transformations.

> **Note:** This package is designed to be installed by the [evaluation pipeline repository](https://github.com/m-levytskyi/inverse-eval). It is dedicated to **training** the models. For evaluation and benchmarking, refer to the second repository.

## What Was Changed

The following modifications and extensions were made on top of the original reflectorch package:

### 1. Normalizing Flows Support

Added support for training and inference with normalizing flows (NF) models, enabling posterior estimation for reflectometry parameter recovery.

### 2. Mixed Data Loader

Introduced `MixedReflectivityDataLoader`, which combines synthetically generated reflectivity curves with real experimental data during training. This loader concatenates batches from a synthetic data generator and an experimental data loader (which reads pre-processed `.dat` files with Q, R, and dR columns), allowing the model to learn from both simulated physics and measured data simultaneously.

### 3. Input Transformations

#### Q-Weighted Inputs

Applied Q-dependent scaling to reflectivity curves and uncertainty (sigma) values:

- **Curves:** $R' = R \times Q^{-\alpha}$ — emphasizes high-Q features (thin layers, interface roughness)
- **Sigmas:** $dR' = dR \times Q^{-\beta} / (R \times \ln(10))$ — proper uncertainty weighting accounting for Q-dependence and log scaling

Implemented via `QWeightedCurvesScaler` and `QWeightedSigmaScaler` classes in `reflectorch/data_generation/scale_curves.py`. The $\alpha$ and $\beta$ exponents are configurable via YAML.

#### Mean Conditioning

A simple per-sample normalization that centers each curve around zero:

- **Curves:** $R' = R - \text{mean}(R)$

This makes the model invariant to absolute intensity offsets while preserving relative curve shape. Implemented via `MeanConditionedCurvesScaler` — a drop-in replacement for any existing curve scaler, requiring no changes to data loaders or datasets.

## Trained Models

| Model | Config | Description |
|-------|--------|-------------|
| **NF Baseline** | [`example_nf_config_reflectorch.yaml`](configs/example_nf_config_reflectorch.yaml) | No transformations, default data loader |
| **NF + Mixed Data** | [`nf_config_mixed.yaml`](configs/nf_config_mixed.yaml) | Mixed synthetic + experimental data |
| **NF + Mixed + dR** | [`nf_config_mixed_sigmas.yaml`](configs/nf_config_mixed_sigmas.yaml) | Mixed data with dR as additional input |
| **NF + Mixed + dR + Q-Weighted** | [`nf_config_mixed_sigmas_qweighted.yaml`](configs/nf_config_mixed_sigmas_qweighted.yaml) | Mixed data + dR + Q-weighted transformations (α=2, β=3) |
| ↳ Exp1 (α=2, β=2) | [`nf_config_mixed_sigmas_qweighted_exp1.yaml`](configs/nf_config_mixed_sigmas_qweighted_exp1.yaml) | Matched standard Fresnel weighting |
| ↳ Exp2 (α=4, β=4) | [`nf_config_mixed_sigmas_qweighted_exp2.yaml`](configs/nf_config_mixed_sigmas_qweighted_exp2.yaml) | Strong Fresnel matching |
| **NF + Mean Conditioned** | [`nf_config_mixed_mean_conditioned.yaml`](configs/nf_config_mixed_mean_conditioned.yaml) | Mean-centered curve scaling |

Saved model weights are stored in the `saved_models/` directory.

## Quick Start

### Installation

Clone the repository and install in editable mode:

```bash
git clone <repo-url>
cd nflows_reflectorch
pip install -e .
```

Users with Nvidia GPUs should additionally install PyTorch with CUDA support from the [PyTorch website](https://pytorch.org/get-started/locally/).

### Training

A quick example of the training process can be found in [nf_training.ipynb](nf_training.ipynb).

To train from the command line:

```bash
python -m reflectorch.train configs/<config_name>.yaml
```

### Inference

Some inference examples are provided in [nf_inference.ipynb](nf_inference.ipynb).
