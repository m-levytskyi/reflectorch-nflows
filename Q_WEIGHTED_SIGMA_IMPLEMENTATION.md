# Q-Weighted Sigma and Curve Transformation Implementation

**Date:** December 22, 2025  
**Status:** Completed

## Overview

This document describes the implementation of Q-weighted transformations for reflectivity curves and sigma (error) values in the reflectorch package. The implementation applies the following transformations:

- **Curves:** `R' = R * Q^(-α)` where α = 2
- **Sigmas:** `dR' = dR * Q^(-β) / (R * ln(10))` where β = 3

## Motivation

The original implementation used Q-independent scaling for both curves and sigmas. This created asymmetry in how synthetic and experimental data were handled:
- Synthetic data generated sigmas but didn't scale them consistently
- Experimental data completely ignored dR error values from files
- Mixed dataloaders concatenated inconsistent batches

This implementation provides a unified Q-dependent scaling approach that can be applied consistently across all data sources.

## Implementation Details

### 1. New Scaler Classes

**File:** `reflectorch/data_generation/scale_curves.py`

#### QWeightedCurvesScaler
```python
class QWeightedCurvesScaler(CurvesScaler):
    """Curve scaler: R' = R * Q^(-alpha)"""
    def __init__(self, alpha: float = 2.0)
    def scale(self, curves: Tensor, q_values: Tensor) -> Tensor
    def restore(self, scaled_curves: Tensor, q_values: Tensor) -> Tensor
```

**Parameters:**
- `alpha`: Exponent for Q-weighting (default: 2.0)

**Usage:**
```python
from reflectorch import QWeightedCurvesScaler

scaler = QWeightedCurvesScaler(alpha=2.0)
scaled_R = scaler.scale(R, Q)  # R * Q^(-2)
original_R = scaler.restore(scaled_R, Q)  # Inverse transformation
```

#### QWeightedSigmaScaler
```python
class QWeightedSigmaScaler:
    """Sigma scaler: dR' = dR * Q^(-beta) / (R * ln(10))"""
    def __init__(self, beta: float = 3.0)
    def scale(self, sigmas: Tensor, curves: Tensor, q_values: Tensor) -> Tensor
    def restore(self, scaled_sigmas: Tensor, curves: Tensor, q_values: Tensor) -> Tensor
```

**Parameters:**
- `beta`: Exponent for Q-weighting of sigmas (default: 3.0)

**Usage:**
```python
from reflectorch import QWeightedSigmaScaler

scaler = QWeightedSigmaScaler(beta=3.0)
scaled_dR = scaler.scale(dR, R, Q)  # dR * Q^(-3) / (R * ln(10))
original_dR = scaler.restore(scaled_dR, R, Q)  # Inverse transformation
```

### 2. Base Scaler Interface Updates

**File:** `reflectorch/data_generation/scale_curves.py`

Updated the base `CurvesScaler` class and existing scalers to accept optional `q_values` parameter:

```python
class CurvesScaler(object):
    def scale(self, curves: Tensor, q_values: Tensor = None):
        raise NotImplementedError
    
    def restore(self, curves: Tensor, q_values: Tensor = None):
        raise NotImplementedError
```

**Modified Scalers:**
- `LogAffineCurvesScaler`: Now accepts but ignores `q_values` (backward compatible)
- `MeanNormalizationCurvesScaler`: Now accepts but ignores `q_values` (backward compatible)

**Backward Compatibility:** Existing code continues to work as the `q_values` parameter is optional.

### 3. Custom Dataset with Sigma Transformation

**File:** `reflectorch/data_generation/dataset.py`

#### QWeightedDataset
```python
class QWeightedDataset(BasicDataset):
    """Dataset with Q-weighted sigma transformation applied in update_batch_data()"""
    def __init__(self, sigma_scaler=None, **kwargs)
    def update_batch_data(self, batch_data: BATCH_DATA_TYPE) -> None
```

**Features:**
- Inherits from `BasicDataset`
- Adds `sigma_scaler` attribute
- Overrides `update_batch_data()` to transform `context['sigmas']` → `batch_data['scaled_sigmas']`
- Requires `calc_denoised_curves=True` to access unscaled R values

**Usage:**
```python
from reflectorch import QWeightedDataset, QWeightedSigmaScaler

sigma_scaler = QWeightedSigmaScaler(beta=3.0)

dataset = QWeightedDataset(
    q_generator=q_gen,
    prior_sampler=prior,
    intensity_noise=noise,
    curves_scaler=curves_scaler,
    sigma_scaler=sigma_scaler,
    calc_denoised_curves=True,  # REQUIRED for sigma transformation
)
```

### 4. Experimental Data Loader Updates

**File:** `reflectorch/ml/experimental_dataloaders.py`

#### ExperimentalSample Dataclass
Added `dr` field to store error values:
```python
@dataclass(frozen=True)
class ExperimentalSample:
    q: np.ndarray
    r: np.ndarray
    dr: Optional[np.ndarray]  # NEW: sigma values from .dat files
    params_true: torch.Tensor
```

#### ExperimentalReflectivityDataLoader
**New Parameter:**
- `sigma_scaler`: Optional QWeightedSigmaScaler instance

**Changes:**
1. Reads dR values from column 3 of `.dat` files (via `_read_curve_dat`)
2. Stores dR values in `ExperimentalSample` objects
3. Precomputes interpolated sigmas in `_precompute_interpolated_curves()`
4. Applies sigma transformation in `get_batch()` if `sigma_scaler` is provided
5. Includes `scaled_sigmas` in batch output

**Batch Output:**
```python
{
    "q_values": torch.Tensor,           # [B, Nq]
    "scaled_noisy_curves": torch.Tensor, # [B, Nq]
    "scaled_params": torch.Tensor,       # [B, 3*P]
    "q_resolutions": torch.Tensor,       # [B, 1]
    "scaled_sigmas": torch.Tensor,       # [B, Nq] - NEW (if sigma_scaler provided)
}
```

### 5. Scaler Call Site Updates

All `curves_scaler.scale()` calls updated to pass `q_values` parameter:

#### Dataset.py
- Line 98: `scaled_noisy_curves = self.curves_scaler.scale(noisy_curves, q_values)`
- Line 230: Test code updated with Q-values

#### Inference Model
**File:** `reflectorch/inference/inference_model.py`

Updated multiple locations:
- `_scale_curve()` methods in both `EasyInferenceModel` and `InferenceModel` classes
- `predict()` method: Reordered to get Q-values before scaling
- `sample()` method: Reordered to get Q-values before scaling
- `_qshift_prediction()` methods: Expanded Q-values to match batch size
- Sigma scaling: Added check for `sigma_scaler` attribute

### 6. Mixed Data Loader

**File:** `reflectorch/ml/experimental_dataloaders.py`

#### MixedReflectivityDataLoader
**New Parameter:**
- `sigma_scaler`: Optional QWeightedSigmaScaler instance

**Changes:**
- Passes `sigma_scaler` to both synthetic and experimental loaders
- Automatic concatenation of `scaled_sigmas` from both sources (existing logic handles this)

**Usage:**
```python
from reflectorch import MixedReflectivityDataLoader

mixed_loader = MixedReflectivityDataLoader(
    q_generator=q_gen,
    prior_sampler=prior,
    curves_scaler=curves_scaler,
    sigma_scaler=sigma_scaler,  # NEW parameter
    mix_fraction=0.5,
    data_dir="exp_data/",
    # ... other params
)
```

### 7. Q-Weighted Dataloader Wrapper

**File:** `reflectorch/ml/dataloaders.py`

#### QWeightedReflectivityDataLoader
```python
class QWeightedReflectivityDataLoader(QWeightedDataset, DataLoader):
    """Dataloader combining QWeightedDataset with DataLoader interface"""
    pass
```

**Usage:**
Drop-in replacement for `ReflectivityDataLoader` when Q-weighted transformations are needed:

```python
from reflectorch import QWeightedReflectivityDataLoader

loader = QWeightedReflectivityDataLoader(
    q_generator=q_gen,
    prior_sampler=prior,
    curves_scaler=curves_scaler,
    sigma_scaler=sigma_scaler,
    calc_denoised_curves=True,
)
```

## Complete Usage Example

### Synthetic Data with Q-Weighted Transformations

```python
import torch
from reflectorch import (
    QWeightedCurvesScaler,
    QWeightedSigmaScaler,
    QWeightedReflectivityDataLoader,
    ConstantQ,
    SubpriorParametricSampler,
    PoissonNoiseGenerator,
)

# Create scalers
curves_scaler = QWeightedCurvesScaler(alpha=2.0)
sigma_scaler = QWeightedSigmaScaler(beta=3.0)

# Setup data generation components
q_generator = ConstantQ((0.01, 0.3, 256), device='cuda')
prior_sampler = SubpriorParametricSampler(
    thickness_range=(1, 500),
    roughness_range=(0, 60),
    sld_range=(0, 150),
    max_num_layers=5,
    device='cuda',
)
noise_generator = PoissonNoiseGenerator(relative_error=(0.01, 0.08))

# Create Q-weighted dataloader
loader = QWeightedReflectivityDataLoader(
    q_generator=q_generator,
    prior_sampler=prior_sampler,
    intensity_noise=noise_generator,
    curves_scaler=curves_scaler,
    sigma_scaler=sigma_scaler,
    calc_denoised_curves=True,  # Required for sigma transformation
)

# Get batch
batch = loader.get_batch(32)

# Batch contains:
# - q_values: [32, 256]
# - scaled_noisy_curves: [32, 256] - R * Q^(-2)
# - scaled_sigmas: [32, 256] - dR * Q^(-3) / (R * ln(10))
# - scaled_params: [32, P]
```

### Experimental Data with Q-Weighted Transformations

```python
from reflectorch import ExperimentalReflectivityDataLoader

# Create experimental loader with sigma transformation
exp_loader = ExperimentalReflectivityDataLoader(
    q_generator=q_generator,
    prior_sampler=prior_sampler,
    curves_scaler=curves_scaler,
    sigma_scaler=sigma_scaler,  # NEW: Will use dR from .dat files
    data_dir="exp_data/",
    curve_glob="*_experimental_curve.dat",
    model_suffix="_model.txt",
)

# Get batch (will include scaled_sigmas if .dat files have dR column)
batch = exp_loader.get_batch(16)
```

### Mixed Synthetic + Experimental Data

```python
from reflectorch import MixedReflectivityDataLoader

# Create mixed loader
mixed_loader = MixedReflectivityDataLoader(
    q_generator=q_generator,
    prior_sampler=prior_sampler,
    curves_scaler=curves_scaler,
    sigma_scaler=sigma_scaler,
    mix_fraction=0.5,  # 50% experimental, 50% synthetic
    data_dir="exp_data/",
    synthetic_kwargs={
        'calc_denoised_curves': True,  # Required for synthetic sigma transformation
    },
)

# Get mixed batch
batch = mixed_loader.get_batch(64)
# Contains 32 experimental + 32 synthetic samples
# Both with consistent Q-weighted transformations
```

## Data File Format

For experimental data with error values, `.dat` files should have at least 3 columns:

```
# Q(Å^-1)    R          dR         dQ(optional)
0.010        0.95       0.05       0.001
0.015        0.89       0.04       0.0015
0.020        0.81       0.038      0.002
...
```

- Column 1: Q values
- Column 2: R (reflectivity)
- Column 3: dR (error/sigma) - **NEW: Now used if sigma_scaler provided**
- Column 4: dQ (optional Q resolution)

## Key Design Decisions

### 1. Separate Scalers
- **Curves Scaler:** `QWeightedCurvesScaler` for R transformations
- **Sigma Scaler:** `QWeightedSigmaScaler` for dR transformations
- **Rationale:** Different transformation formulas, clearer separation of concerns

### 2. No Epsilon Stabilization
- Negative R values removed during preprocessing
- No need for `R_safe = max(R, eps)` approach
- Cleaner implementation

### 3. Backward Compatibility
- Existing scalers accept but ignore `q_values` parameter
- Existing code continues to work without modifications
- Q-weighted features opt-in via new classes

### 4. Sigma Transformation Location
- Applied in `update_batch_data()` for synthetic data (QWeightedDataset)
- Applied in `get_batch()` for experimental data (ExperimentalReflectivityDataLoader)
- Ensures transformation happens after noise generation but before training

### 5. Inverse Transformation
- `restore()` methods implemented for completeness
- **Not used in practice** - sigmas only forward-transformed during training
- May be useful for future debugging or visualization

## Testing

Basic functionality verified:

```python
# Test curve scaler
curve_scaler = QWeightedCurvesScaler(alpha=2.0)
R = torch.tensor([[1.0, 0.5, 0.1]])
Q = torch.tensor([[0.01, 0.05, 0.1]])
R_scaled = curve_scaler.scale(R, Q)
# Result: R * Q^(-2) = [[10000, 200, 10]]

# Test sigma scaler
sigma_scaler = QWeightedSigmaScaler(beta=3.0)
dR = torch.tensor([[0.1, 0.05, 0.01]])
dR_scaled = sigma_scaler.scale(dR, R, Q)
# Result: dR * Q^(-3) / (R * ln(10))
```

✅ All files compile without syntax errors  
✅ Scalers produce expected numerical results  
✅ Backward compatibility maintained

## Migration Guide

### For Existing Code (No Changes Needed)
Existing code continues to work as-is. The `q_values` parameter is optional and existing scalers ignore it.

### To Enable Q-Weighted Transformations

**Step 1:** Create scalers
```python
from reflectorch import QWeightedCurvesScaler, QWeightedSigmaScaler

curves_scaler = QWeightedCurvesScaler(alpha=2.0)
sigma_scaler = QWeightedSigmaScaler(beta=3.0)
```

**Step 2:** Use Q-weighted dataset/loader
```python
from reflectorch import QWeightedReflectivityDataLoader

loader = QWeightedReflectivityDataLoader(
    # ... standard params ...
    curves_scaler=curves_scaler,
    sigma_scaler=sigma_scaler,
    calc_denoised_curves=True,  # Don't forget this!
)
```

**Step 3:** Update experimental loader (if using)
```python
exp_loader = ExperimentalReflectivityDataLoader(
    # ... standard params ...
    sigma_scaler=sigma_scaler,  # Add this parameter
)
```

## Files Modified

1. `reflectorch/data_generation/scale_curves.py` - New scalers + interface updates
2. `reflectorch/data_generation/dataset.py` - QWeightedDataset class
3. `reflectorch/data_generation/__init__.py` - Export new classes
4. `reflectorch/ml/experimental_dataloaders.py` - Sigma handling in experimental loader
5. `reflectorch/ml/dataloaders.py` - QWeightedReflectivityDataLoader wrapper
6. `reflectorch/inference/inference_model.py` - Q-values passed to scalers

## Known Limitations

1. **Requires calc_denoised_curves=True** for synthetic data with sigma transformation (slight performance overhead)
2. **Sigma transformation requires R values** - not applicable to certain edge cases
3. **Inverse transformation not tested in production** - only forward scaling used during training

## Future Enhancements

- [ ] Add alpha/beta parameters to YAML config files
- [ ] Create tutorial notebook demonstrating Q-weighted transformations
- [ ] Add visualization tools for comparing scaled vs unscaled curves
- [ ] Performance optimization for sigma transformation
- [ ] Unit tests for edge cases (zero Q values, etc.)

## References

- Based on codemap analysis of sigma handling in reflectorch
- Implements formula: `dR = dR/(R * ln(10))` with Q-weighting
- Designed for compatibility with existing reflectorch training pipelines
