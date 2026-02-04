# Q-Weighted Transformation Experiments

**Date:** February 4, 2026  
**Status:** Experimental  
**Purpose:** Systematic evaluation of Q-weighting exponents for curves and sigma transformations

## Overview

This document describes a systematic experimental study to determine optimal Q-weighting parameters for reflectivity curve and uncertainty transformations. Two key parameters control the weighting:

- **α (alpha)**: Curve scaling exponent in `R' = R × Q^(-α)`
- **β (beta)**: Sigma scaling exponent in `dR' = dR × Q^(-β) / (R × ln(10))`

## Experimental Configurations

### Experiment 1: Matched Standard Weighting
**Config:** [nf_config_mixed_sigmas_qweighted_exp1.yaml](configs/nf_config_mixed_sigmas_qweighted_exp1.yaml)

```yaml
curves_scaler:
  cls: QWeightedCurvesScaler
  kwargs:
    alpha: 2.0

sigma_scaler:
  cls: QWeightedSigmaScaler
  kwargs:
    beta: 2.0
```

**Motivation:**
- **Physical basis**: Q^(-2) weighting matches Fresnel reflectivity decay from smooth interfaces
- **Consistent weighting**: β = α ensures relative uncertainties σ_R'/R' remain approximately constant across Q range
- **Standard practice**: Widely used in reflectometry analysis software (e.g., Motofit, GenX, Refl1D)
- **Numerical stability**: Matched exponents prevent over/under-weighting at different Q scales
- **Error propagation**: Maintains proper balance between low-Q and high-Q contributions to loss

**Expected behavior:**
- Balanced emphasis across Q range
- Good performance on both thick films (low-Q) and thin layers (high-Q)
- Stable training dynamics
- Conservative baseline for comparison

---

### Experiment 2: Strong Fresnel Matching
**Config:** [nf_config_mixed_sigmas_qweighted_exp2.yaml](configs/nf_config_mixed_sigmas_qweighted_exp2.yaml)

```yaml
curves_scaler:
  cls: QWeightedCurvesScaler
  kwargs:
    alpha: 4.0

sigma_scaler:
  cls: QWeightedSigmaScaler
  kwargs:
    beta: 4.0
```

**Motivation:**
- **Physical basis**: Q^(-4) matches Fresnel reflectivity from rough interfaces and Born approximation scattering
- **High-Q emphasis**: Strongly weights thin layer features, interface roughness, and fine structure
- **Kinematic scattering**: At high Q, reflectivity decays as Q^(-4) due to rough surface scattering
- **Thin film sensitivity**: Better suited when dataset contains many multilayer structures with sharp interfaces
- **Consistent weighting**: β = α maintains relative uncertainty balance (like Exp1)

**Expected behavior:**
- Superior performance on thin films and multilayers
- Enhanced sensitivity to interface roughness
- May struggle with thick films if low-Q signal is weak
- Potential for gradient instability if high-Q noise is significant

---

## Comparison with Original Configuration

### Original (for reference only)
**Config:** [nf_config_mixed_sigmas_qweighted.yaml](configs/nf_config_mixed_sigmas_qweighted.yaml)

```yaml
curves_scaler:
  kwargs:
    alpha: 2.0

sigma_scaler:
  kwargs:
    beta: 3.0  # β > α
```

**Issues identified:**
- **Asymmetric weighting**: β - α = 1 creates Q^(-1) mismatch
- **Over-emphasis on high-Q uncertainties**: May give too much importance to noisy high-Q data
- **Non-standard**: Not commonly used in reflectometry analysis
- **Potential instability**: Different scaling for curves vs uncertainties can affect training

**Note:** This configuration is retained for historical comparison but not recommended for production use.

---

## Running the Experiments

### Experiment 1 (α=2, β=2)
```bash
cd /home/levytskyi/Documents/reflectorch_devvm/reflectorch
python -m reflectorch.train configs/nf_config_mixed_sigmas_qweighted_exp1.yaml
```

**Output directory:** `saved_models/nf_config_mixed_sigmas_qweighted_exp1/`

### Experiment 2 (α=4, β=4)
```bash
cd /home/levytskyi/Documents/reflectorch_devvm/reflectorch
python -m reflectorch.train configs/nf_config_mixed_sigmas_qweighted_exp2.yaml
```

**Output directory:** `saved_models/nf_config_mixed_sigmas_qweighted_exp2/`

### Training Duration
Both experiments use identical training settings:
- **Iterations:** 300,000
- **Batch size:** 2,048
- **Learning rate:** 1.0e-4 → 1.0e-6 (cosine annealing)
- **Estimated time:** ~8-12 hours on single GPU (depends on hardware)

---

## Evaluation Metrics

### 1. Training Loss
Monitor convergence behavior:
```bash
# Check training logs
tail -f saved_models/nf_config_mixed_sigmas_qweighted_exp1/training.log
tail -f saved_models/nf_config_mixed_sigmas_qweighted_exp2/training.log
```

**Compare:**
- Final loss values
- Convergence speed (iterations to plateau)
- Loss stability (variance in final 10k iterations)

### 2. Reconstruction Accuracy
Test on held-out experimental data:

```python
from reflectorch.inference import InferenceModel
import numpy as np

# Load models
model_exp1 = InferenceModel.from_checkpoint('saved_models/nf_config_mixed_sigmas_qweighted_exp1/best_model.pt')
model_exp2 = InferenceModel.from_checkpoint('saved_models/nf_config_mixed_sigmas_qweighted_exp2/best_model.pt')

# Run predictions on test set
test_files = ['dataset/test/s000001_experimental_curve.dat', ...]
for curve_file in test_files:
    pred_exp1 = model_exp1.predict(curve_file)
    pred_exp2 = model_exp2.predict(curve_file)
    # Compare χ² values, parameter accuracy, etc.
```

**Key metrics:**
- **χ² (chi-squared):** Goodness of fit to experimental data
- **Parameter accuracy:** RMSE for thickness, roughness, SLD recovery
- **Posterior quality:** Width and shape of uncertainty distributions

### 3. Q-Range Performance
Evaluate separately for different Q regions:

| Q Range | Physical Features | Exp1 (α=2) Expected | Exp2 (α=4) Expected |
|---------|------------------|---------------------|---------------------|
| 0.007-0.03 | Total film thickness, substrate SLD | Good | Fair |
| 0.03-0.10 | Layer thicknesses, average SLD | Good | Good |
| 0.10-0.284 | Interface roughness, thin layers | Good | Excellent |

### 4. Film Type Performance

**Thick films (>500 Å):**
- Low-Q features dominate
- **Hypothesis:** Exp1 should perform better
- **Test:** Measure RMSE for thickness >500 Å

**Thin films (<100 Å):**
- High-Q features critical
- **Hypothesis:** Exp2 should perform better
- **Test:** Measure RMSE for thickness <100 Å

**Multilayers (2+ layers):**
- Interface sharpness critical
- **Hypothesis:** Exp2 should excel at roughness recovery
- **Test:** Compare roughness parameter accuracy

---

## Decision Criteria

### Choose Experiment 1 (α=2, β=2) if:
✓ Dataset contains mix of thick and thin films  
✓ Low-Q data quality is comparable to high-Q  
✓ Training stability is priority  
✓ Standard, reproducible approach needed  
✓ Total film thickness is important parameter  

### Choose Experiment 2 (α=4, β=4) if:
✓ Dataset dominated by thin films and multilayers  
✓ High-Q data has excellent signal-to-noise  
✓ Interface roughness is critical parameter  
✓ Willing to trade some thick-film performance  
✓ Sample structures have sharp interfaces  

### Consider Alternative Values if:
⚠ Both experiments show poor convergence → Try α=0, β=0 (no weighting)  
⚠ High-Q performance inadequate → Try α=3, β=3 (intermediate)  
⚠ Training unstable with α=4 → Reduce to α=3, β=3  
⚠ Low-Q features missed → Reduce to α=1, β=1  

---

## Theoretical Background

### Why β = α (Matched Exponents)?

The transformation pair:
- R' = R × Q^(-α)
- σ_R' = σ_R × Q^(-β) / (R × ln(10))

For relative uncertainty to be Q-independent:
```
σ_R' / R' = [σ_R × Q^(-β) / (R × ln(10))] / [R × Q^(-α)]
          = [σ_R / R] × Q^(α-β) / ln(10)
```

**When β = α:** Relative uncertainty becomes `(σ_R / R) / ln(10)`, independent of Q.

**When β ≠ α:** Relative uncertainty varies as Q^(α-β), creating non-physical Q-dependent weighting.

### Why α = 2?

Fresnel reflectivity from sharp interface:
```
R(Q) ∝ (Δρ/Q²)²  →  R ∝ Q^(-4) for sharp interfaces
R(Q) ∝ 1/Q²      for smooth interfaces (with roughness damping)
```

In practice:
- **α = 2:** Emphasizes all features proportionally
- **α = 4:** Strong emphasis on high-Q (assumes sharp interfaces)

### Why α = 4?

Born approximation for surface scattering:
```
R(Q) = R_Fresnel × PSD(Q)
     ∝ Q^(-4) × PSD(Q)
```

For rough surfaces with power-law PSD, total decay is often Q^(-4) or steeper.

---

## Validation Checklist

After running both experiments, verify:

- [ ] Training completed without errors
- [ ] Final loss values recorded and compared
- [ ] Best models saved in respective directories
- [ ] Test set χ² computed for both models
- [ ] Parameter recovery accuracy quantified
- [ ] Q-range breakdown analysis completed
- [ ] Film type performance compared
- [ ] Gradient norms checked for stability
- [ ] Posterior distributions visualized
- [ ] Decision made on production configuration

---

## Next Steps

1. **Run both experiments** in parallel (if GPU resources allow) or sequentially
2. **Monitor training** logs for convergence and stability
3. **Evaluate on test set** using metrics above
4. **Visualize results:**
   - Plot loss curves
   - Compare χ² distributions
   - Show parameter recovery scatter plots
   - Display posterior widths
5. **Document findings** in results section below
6. **Select best configuration** for production deployment

---

## Results (To Be Filled)

### Experiment 1 (α=2, β=2)
**Training:**
- Final loss: _TBD_
- Convergence iteration: _TBD_
- Training stability: _TBD_

**Test Performance:**
- Mean χ²: _TBD_
- Thickness RMSE: _TBD_
- Roughness RMSE: _TBD_
- SLD RMSE: _TBD_

**Observations:**
- _TBD_

---

### Experiment 2 (α=4, β=4)
**Training:**
- Final loss: _TBD_
- Convergence iteration: _TBD_
- Training stability: _TBD_

**Test Performance:**
- Mean χ²: _TBD_
- Thickness RMSE: _TBD_
- Roughness RMSE: _TBD_
- SLD RMSE: _TBD_

**Observations:**
- _TBD_

---

### Comparison Summary
| Metric | Exp1 (α=2,β=2) | Exp2 (α=4,β=4) | Winner |
|--------|----------------|----------------|---------|
| Training Loss | _TBD_ | _TBD_ | _TBD_ |
| Test χ² | _TBD_ | _TBD_ | _TBD_ |
| Thick Films | _TBD_ | _TBD_ | _TBD_ |
| Thin Films | _TBD_ | _TBD_ | _TBD_ |
| Roughness | _TBD_ | _TBD_ | _TBD_ |
| Stability | _TBD_ | _TBD_ | _TBD_ |

**Recommendation:** _TBD after analysis_

---

## References

1. **Fresnel Reflectivity Theory:**
   - Born, M. & Wolf, E. "Principles of Optics" (1999)
   - Parratt, L.G. "Surface Studies of Solids by Total Reflection of X-Rays" Phys. Rev. 95, 359 (1954)

2. **Q-Weighting in Reflectometry:**
   - Nelson, A. "Co-refinement of multiple-contrast neutron/X-ray reflectivity data using MOTOFIT" J. Appl. Cryst. 39, 273-276 (2006)
   - Björck, M. & Andersson, G. "GenX: an extensible X-ray reflectivity refinement program utilizing differential evolution" J. Appl. Cryst. 40, 1174-1178 (2007)

3. **Error Weighting:**
   - Press, W.H. et al. "Numerical Recipes" Chapter 15: Modeling of Data (2007)

---

## Related Documentation

- [Q_WEIGHTED_SIGMA_IMPLEMENTATION.md](Q_WEIGHTED_SIGMA_IMPLEMENTATION.md) - Implementation details
- [TRAINING_WITH_QWEIGHTED_SIGMAS.md](TRAINING_WITH_QWEIGHTED_SIGMAS.md) - Training guide
- [configs/nf_config_mixed_sigmas_qweighted_exp1.yaml](configs/nf_config_mixed_sigmas_qweighted_exp1.yaml) - Exp1 configuration
- [configs/nf_config_mixed_sigmas_qweighted_exp2.yaml](configs/nf_config_mixed_sigmas_qweighted_exp2.yaml) - Exp2 configuration
