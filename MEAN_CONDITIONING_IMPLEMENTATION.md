# Mean Conditioning Implementation Plan

**Date:** February 3, 2026  
**Status:** Planning  
**Related:** Q_WEIGHTED_SIGMA_IMPLEMENTATION.md

## Overview

This document describes the implementation plan for mean conditioning of reflectivity curves. This is a simple normalization technique that centers each curve around zero by subtracting its per-sample mean value.

**Transformation:**
- **Curves:** `R' = R - mean(R)` where mean is computed per-sample along Q dimension

**Key Features:**
- ✅ Simple per-sample transformation (no batch dependencies)
- ✅ Independent of Q-weighting (can be used separately or combined)
- ✅ No sigma transformation (keeps implementation minimal)
- ✅ Makes model invariant to absolute intensity offsets

## Motivation

Mean conditioning addresses several training challenges:

1. **Scale Invariance:** Different experimental setups may have different absolute intensity scales. Centering removes this variability.
2. **Noise Distribution:** When noise is additive, centering helps the model focus on shape rather than absolute values.
3. **Batch Stability:** Each sample normalized independently, avoiding batch-size dependencies.
4. **Simplicity:** Single arithmetic operation with clear interpretation.

This is a complementary technique to Q-weighting, addressing different aspects of data preprocessing.

## Implementation Details

### 1. New Scaler Class

**File:** `reflectorch/data_generation/scale_curves.py`

#### MeanConditionedCurvesScaler

```python
class MeanConditionedCurvesScaler(CurvesScaler):
    """Center reflectivity curves by subtracting per-sample mean.
    
    Transformation: R' = R - mean(R)
    
    This makes the model invariant to absolute intensity offsets while
    preserving the relative shape of the curve. Each sample is centered
    independently based on its own mean value.
    
    The transformation is Q-independent and can be combined with other
    scalers (e.g., QWeightedCurvesScaler) in sequence.
    
    Example:
        >>> scaler = MeanConditionedCurvesScaler()
        >>> R = torch.tensor([[1.0, 0.8, 0.6, 0.4]])  # Mean = 0.7
        >>> R_centered = scaler.scale(R)
        >>> # Result: [[0.3, 0.1, -0.1, -0.3]]
    """
    
    def __init__(self, eps: float = 1e-10):
        """Initialize mean conditioning scaler.
        
        Args:
            eps: Small constant for numerical stability (not currently used,
                 kept for API consistency with other scalers)
        """
        self.eps = eps
    
    def scale(self, curves: Tensor, q_values: Tensor = None) -> Tensor:
        """Center curves by subtracting per-curve mean.
        
        Args:
            curves: [B, Nq] reflectivity curves
            q_values: Optional [B, Nq] Q values (not used, kept for API compatibility)
            
        Returns:
            [B, Nq] mean-centered curves where each row has mean ≈ 0
            
        Note:
            The q_values parameter is ignored but kept for compatibility with
            the CurvesScaler interface and Q-weighted scalers.
        """
        # Compute mean along Q dimension (dim=-1)
        curve_means = curves.mean(dim=-1, keepdim=True)  # [B, 1]
        
        # Subtract mean from each curve
        return curves - curve_means
    
    def restore(self, scaled_curves: Tensor, q_values: Tensor = None, 
                original_mean: Tensor = None) -> Tensor:
        """Restore original curves by adding back the mean.
        
        Args:
            scaled_curves: [B, Nq] mean-centered curves
            q_values: Optional [B, Nq] Q values (not used)
            original_mean: [B, 1] original mean values (required for restoration)
            
        Returns:
            [B, Nq] restored curves
            
        Note:
            Exact restoration requires storing the original mean values during
            the scale() operation. If original_mean is not provided, the curves
            cannot be restored to their original scale.
            
            In practice, this method is rarely used as the model predicts in
            the mean-centered space and predictions are evaluated there.
        """
        if original_mean is not None:
            return scaled_curves + original_mean
        else:
            # Cannot restore without original mean - return as-is
            return scaled_curves
```

**Design Decisions:**
- **Per-sample normalization:** Each curve centered by its own mean (no batch coupling)
- **Q-independent:** Does not use Q values (unlike Q-weighted scaling)
- **Optional restoration:** `restore()` implemented but requires storing original means
- **API compatibility:** Accepts `q_values` parameter for consistency with other scalers

### 2. Export Updates

**File:** `reflectorch/data_generation/__init__.py`

Add new scaler to public API:

```python
from .scale_curves import (
    CurvesScaler,
    LogAffineCurvesScaler,
    MeanNormalizationCurvesScaler,
    QWeightedCurvesScaler,
    QWeightedSigmaScaler,
    MeanConditionedCurvesScaler,  # NEW
)

__all__ = [
    # ...existing exports...
    'MeanConditionedCurvesScaler',  # NEW
]
```

### 3. No Dataset Changes Required

**Key Insight:** Unlike Q-weighted sigma transformation, mean conditioning does NOT require:
- ❌ New dataset class
- ❌ Changes to `update_batch_data()`
- ❌ `calc_denoised_curves=True` flag

**Reason:** Mean conditioning is a pure curve transformation that works identically to existing scalers like `LogAffineCurvesScaler`. It operates on noisy curves directly without needing access to denoised curves or sigma values.

**Usage Pattern:**

```python
# Just swap the scaler - no other changes needed
from reflectorch import ReflectivityDataLoader, MeanConditionedCurvesScaler

# OLD: Q-weighted scaling
# curves_scaler = QWeightedCurvesScaler(alpha=2.0)

# NEW: Mean conditioning
curves_scaler = MeanConditionedCurvesScaler()

# Same loader works with new scaler
loader = ReflectivityDataLoader(
    q_generator=q_gen,
    prior_sampler=prior,
    curves_scaler=curves_scaler,  # <-- Only change
    # ...other params unchanged...
)
```

### 4. Experimental Data Loader

**File:** `reflectorch/ml/experimental_dataloaders.py`

**No changes required.** The experimental data loader already supports arbitrary curve scalers through the `curves_scaler` parameter.

**Usage:**

```python
from reflectorch import ExperimentalReflectivityDataLoader, MeanConditionedCurvesScaler

exp_loader = ExperimentalReflectivityDataLoader(
    q_generator=q_generator,
    prior_sampler=prior_sampler,
    curves_scaler=MeanConditionedCurvesScaler(),  # <-- Works out of the box
    data_dir="exp_data/",
    curve_glob="*_experimental_curve.dat",
)
```

### 5. Mixed Data Loader

**File:** `reflectorch/ml/experimental_dataloaders.py`

**No changes required.** Same as experimental loader.

```python
from reflectorch import MixedReflectivityDataLoader, MeanConditionedCurvesScaler

mixed_loader = MixedReflectivityDataLoader(
    q_generator=q_generator,
    prior_sampler=prior_sampler,
    curves_scaler=MeanConditionedCurvesScaler(),  # <-- Works for both sources
    mix_fraction=0.5,
    data_dir="exp_data/",
)
```

## Complete Usage Examples

### Example 1: Synthetic Data Only

```python
import torch
from reflectorch import (
    MeanConditionedCurvesScaler,
    ReflectivityDataLoader,
    ConstantQ,
    SubpriorParametricSampler,
    PoissonNoiseGenerator,
)

# Create mean conditioning scaler
curves_scaler = MeanConditionedCurvesScaler()

# Setup data generation (standard)
q_generator = ConstantQ((0.01, 0.3, 256), device='cuda')
prior_sampler = SubpriorParametricSampler(
    thickness_range=(1, 500),
    roughness_range=(0, 60),
    sld_range=(0, 150),
    max_num_layers=5,
    device='cuda',
)
noise_generator = PoissonNoiseGenerator(relative_error=(0.01, 0.08))

# Create standard dataloader with mean conditioning
loader = ReflectivityDataLoader(
    q_generator=q_generator,
    prior_sampler=prior_sampler,
    intensity_noise=noise_generator,
    curves_scaler=curves_scaler,  # <-- Mean conditioning applied
)

# Get batch
batch = loader.get_batch(32)

# Batch contains:
# - q_values: [32, 256]
# - scaled_noisy_curves: [32, 256] - mean-centered R
# - scaled_params: [32, P]
#
# Each curve in batch has mean ≈ 0
```

### Example 2: Combined with Q-Weighting

Mean conditioning can be combined with Q-weighting by applying transformations sequentially:

```python
from reflectorch import (
    MeanConditionedCurvesScaler,
    QWeightedCurvesScaler,
    CurvesScaler,
)

class SequentialScaler(CurvesScaler):
    """Apply multiple scalers in sequence."""
    
    def __init__(self, *scalers):
        """
        Args:
            *scalers: Variable number of CurvesScaler instances to apply in order
        """
        self.scalers = scalers
    
    def scale(self, curves: Tensor, q_values: Tensor = None) -> Tensor:
        """Apply each scaler in sequence."""
        result = curves
        for scaler in self.scalers:
            result = scaler.scale(result, q_values)
        return result
    
    def restore(self, scaled_curves: Tensor, q_values: Tensor = None) -> Tensor:
        """Apply inverse transformations in reverse order."""
        result = scaled_curves
        for scaler in reversed(self.scalers):
            result = scaler.restore(result, q_values)
        return result

# Create combined scaler
# Option A: Q-weight first, then mean-center
combined_scaler = SequentialScaler(
    QWeightedCurvesScaler(alpha=2.0),
    MeanConditionedCurvesScaler(),
)

# Option B: Mean-center first, then Q-weight
combined_scaler = SequentialScaler(
    MeanConditionedCurvesScaler(),
    QWeightedCurvesScaler(alpha=2.0),
)

# Use with any loader
loader = ReflectivityDataLoader(
    curves_scaler=combined_scaler,
    # ...other params...
)
```

### Example 3: Experimental Data

```python
from reflectorch import (
    ExperimentalReflectivityDataLoader,
    MeanConditionedCurvesScaler,
)

exp_loader = ExperimentalReflectivityDataLoader(
    q_generator=q_generator,
    prior_sampler=prior_sampler,
    curves_scaler=MeanConditionedCurvesScaler(),
    data_dir="exp_data/",
    curve_glob="*_experimental_curve.dat",
    model_suffix="_model.txt",
)

batch = exp_loader.get_batch(16)
# Experimental curves are mean-centered
```

### Example 4: Mixed Synthetic + Experimental

```python
from reflectorch import MixedReflectivityDataLoader, MeanConditionedCurvesScaler

mixed_loader = MixedReflectivityDataLoader(
    q_generator=q_generator,
    prior_sampler=prior_sampler,
    curves_scaler=MeanConditionedCurvesScaler(),
    mix_fraction=0.5,
    data_dir="exp_data/",
)

batch = mixed_loader.get_batch(64)
# Both synthetic and experimental curves mean-centered consistently
```

## Testing Plan

### Unit Tests

Create `tests/test_mean_conditioning.py`:

```python
import torch
import pytest
from reflectorch import MeanConditionedCurvesScaler

def test_mean_conditioning_basic():
    """Test that mean conditioning produces zero-mean curves."""
    scaler = MeanConditionedCurvesScaler()
    
    # Create test data
    curves = torch.tensor([
        [1.0, 0.8, 0.6, 0.4],  # Mean = 0.7
        [2.0, 1.5, 1.0, 0.5],  # Mean = 1.25
    ])
    
    # Apply scaling
    scaled = scaler.scale(curves)
    
    # Check means are approximately zero
    means = scaled.mean(dim=-1)
    assert torch.allclose(means, torch.zeros(2), atol=1e-6)
    
    # Check values are correct
    expected = torch.tensor([
        [0.3, 0.1, -0.1, -0.3],
        [0.75, 0.25, -0.25, -0.75],
    ])
    assert torch.allclose(scaled, expected, atol=1e-6)

def test_mean_conditioning_restore():
    """Test restoration with original mean."""
    scaler = MeanConditionedCurvesScaler()
    
    curves = torch.tensor([[1.0, 0.8, 0.6, 0.4]])
    original_mean = curves.mean(dim=-1, keepdim=True)
    
    scaled = scaler.scale(curves)
    restored = scaler.restore(scaled, original_mean=original_mean)
    
    assert torch.allclose(restored, curves, atol=1e-6)

def test_mean_conditioning_qvalues_ignored():
    """Test that Q values are ignored."""
    scaler = MeanConditionedCurvesScaler()
    
    curves = torch.tensor([[1.0, 0.8, 0.6, 0.4]])
    q_values = torch.tensor([[0.01, 0.05, 0.1, 0.15]])
    
    # Should produce same result with or without Q values
    scaled_with_q = scaler.scale(curves, q_values)
    scaled_without_q = scaler.scale(curves)
    
    assert torch.allclose(scaled_with_q, scaled_without_q)

def test_mean_conditioning_batch():
    """Test batch processing."""
    scaler = MeanConditionedCurvesScaler()
    
    # Random batch
    torch.manual_seed(42)
    curves = torch.randn(16, 128)
    
    scaled = scaler.scale(curves)
    
    # Each sample should have zero mean
    means = scaled.mean(dim=-1)
    assert torch.allclose(means, torch.zeros(16), atol=1e-5)
```

### Integration Tests

Test with actual data loaders:

```python
def test_mean_conditioning_with_dataloader():
    """Test mean conditioning in actual training pipeline."""
    from reflectorch import (
        ReflectivityDataLoader,
        MeanConditionedCurvesScaler,
        ConstantQ,
        SubpriorParametricSampler,
    )
    
    scaler = MeanConditionedCurvesScaler()
    
    loader = ReflectivityDataLoader(
        q_generator=ConstantQ((0.01, 0.3, 128)),
        prior_sampler=SubpriorParametricSampler(
            thickness_range=(1, 100),
            roughness_range=(0, 20),
            sld_range=(0, 50),
            max_num_layers=3,
        ),
        curves_scaler=scaler,
    )
    
    batch = loader.get_batch(8)
    curves = batch['scaled_noisy_curves']
    
    # Check curves are mean-centered
    means = curves.mean(dim=-1)
    assert torch.allclose(means, torch.zeros(8), atol=1e-4)
```

## Key Design Decisions

### 1. No Batch-Level Statistics
**Decision:** Use per-sample mean, not batch-wide mean.

**Rationale:**
- ✅ Deterministic: Same input always produces same output
- ✅ No batch size dependency
- ✅ Works with batch_size=1
- ✅ Simpler implementation
- ✅ Each sample processed independently

**Alternative considered:** Batch normalization-style with running statistics → Rejected for added complexity.

### 2. No Standard Deviation Scaling
**Decision:** Only subtract mean, don't divide by std.

**Rationale:**
- ✅ Simpler as requested by supervisor
- ✅ Preserves relative magnitudes within each curve
- ✅ Avoids division by near-zero values
- ✅ Clear single-purpose transformation

**Alternative considered:** Full z-score normalization `(R - mean) / std` → Can be added later if needed.

### 3. No Sigma Transformation
**Decision:** Do not transform sigma values.

**Rationale:**
- ✅ Keeps implementation minimal
- ✅ Mean shift is additive, so sigmas in absolute units remain valid
- ✅ Avoids complexity of Q-weighted sigma implementation

**Note:** If sigmas need transformation later, it would be: `dR' = dR` (unchanged, since mean shift doesn't affect errors).

### 4. Restoration Optional
**Decision:** Implement `restore()` but don't require it in practice.

**Rationale:**
- ✅ API completeness
- ✅ May be useful for visualization
- ✅ Not critical for training (predictions evaluated in scaled space)

### 5. Reuse Existing Infrastructure
**Decision:** No new dataset or loader classes needed.

**Rationale:**
- ✅ Works with existing `CurvesScaler` interface
- ✅ Compatible with all current loaders
- ✅ Minimal code changes
- ✅ Easy to test and deploy

## Comparison with Q-Weighting

| Feature | Q-Weighting | Mean Conditioning |
|---------|-------------|-------------------|
| **Formula** | `R * Q^(-α)` | `R - mean(R)` |
| **Depends on Q** | Yes | No |
| **Sigma transform** | Yes (complex) | No |
| **New dataset class** | Yes (`QWeightedDataset`) | No |
| **Requires denoised curves** | Yes | No |
| **Purpose** | Emphasize high-Q features | Remove DC offset |
| **Complexity** | Higher | Lower |
| **Can combine** | Yes (sequential) | Yes (sequential) |

**Recommendation:** Start with mean conditioning alone, then experiment with combining both if needed.

## Implementation Checklist

- [ ] Create `MeanConditionedCurvesScaler` class in `scale_curves.py`
- [ ] Add exports to `__init__.py`
- [ ] Write unit tests (`test_mean_conditioning.py`)
- [ ] Write integration tests (with real loaders)
- [ ] Test with experimental data
- [ ] Test with mixed loader
- [ ] Test sequential combination with Q-weighting
- [ ] Update documentation/examples
- [ ] Verify backward compatibility
- [ ] Performance benchmarking

## Files to Modify

**New Files:**
- `tests/test_mean_conditioning.py` - Unit and integration tests

**Modified Files:**
1. `reflectorch/data_generation/scale_curves.py` - Add `MeanConditionedCurvesScaler` class
2. `reflectorch/data_generation/__init__.py` - Export new scaler

**No Changes Required:**
- ✅ `dataset.py` - Existing infrastructure works
- ✅ `experimental_dataloaders.py` - Already supports arbitrary scalers
- ✅ `dataloaders.py` - No new loader needed
- ✅ `inference_model.py` - Q-values already passed to scalers

**Total:** 2 files modified, 1 file created

## Migration Guide

### For New Users

Simply use `MeanConditionedCurvesScaler` instead of other scalers:

```python
from reflectorch import MeanConditionedCurvesScaler, ReflectivityDataLoader

curves_scaler = MeanConditionedCurvesScaler()

loader = ReflectivityDataLoader(
    curves_scaler=curves_scaler,
    # ...other params...
)
```

### For Existing Users

**Option 1:** Replace existing scaler
```python
# OLD
# curves_scaler = LogAffineCurvesScaler(a=1.0, b=0.0)

# NEW
curves_scaler = MeanConditionedCurvesScaler()
```

**Option 2:** Combine with existing scaler
```python
from reflectorch import SequentialScaler

curves_scaler = SequentialScaler(
    LogAffineCurvesScaler(a=1.0, b=0.0),
    MeanConditionedCurvesScaler(),
)
```

## Expected Benefits

1. **Training Stability:** Centering inputs can improve gradient flow
2. **Generalization:** Model learns shape features rather than absolute scales
3. **Experimental Compatibility:** Different experimental setups normalized consistently
4. **Noise Robustness:** Additive noise centered around zero
5. **Simplicity:** Easy to understand and debug

## Potential Issues and Mitigations

### Issue 1: Information Loss
**Problem:** Absolute intensity scale is lost.

**Mitigation:** This is intentional. If absolute scale is needed, store original means and use `restore()`.

### Issue 2: Near-Constant Curves
**Problem:** Curves with very small variation might become noisy after centering.

**Mitigation:** This is rare for reflectivity data. Can add optional std check if needed.

### Issue 3: Combined with Log Scaling
**Problem:** Mean of log-curves may behave differently.

**Mitigation:** Apply in correct order: `log() → mean_center()` or vice versa depending on desired effect.

## Future Enhancements

- [ ] Add `SequentialScaler` utility to package (currently in examples)
- [ ] Create notebook tutorial comparing different scalers
- [ ] Add visualization tools for scaled vs unscaled curves
- [ ] Consider adding std normalization variant: `MeanStdConditionedCurvesScaler`
- [ ] Batch-level statistics variant if needed: `BatchNormalizedCurvesScaler`

## References

- Related to Q_WEIGHTED_SIGMA_IMPLEMENTATION.md (complementary technique)
- Standard preprocessing in many ML pipelines (mean centering)
- Similar to batch normalization but per-sample instead of per-batch

---

**Estimated Implementation Time:** 2-3 hours
- Scaler class: 30 min
- Tests: 1 hour
- Documentation: 1 hour
- Validation: 30 min