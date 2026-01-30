# Training with Q-Weighted Sigma Transformations

## Quick Start

### 1. Verify Your Data Structure

Your experimental data files must contain Q, R, and dR columns:

```
dataset/train/
├── s000001_experimental_curve.dat  # Contains: Q R dR [optional_dR]
├── s000001_model.txt
├── s000002_experimental_curve.dat
├── s000002_model.txt
└── ...
```

**Important**: `.dat` files must have at least 3 columns: Q, R, dR

Example `.dat` file format:
```
0.007  1.0      0.02
0.010  0.95     0.019
0.015  0.88     0.018
...
```

### 2. Run Training

```bash
cd /home/levytskyi/Documents/reflectorch_devvm/reflectorch
python -m reflectorch.train configs/nf_config_mixed_sigmas_qweighted.yaml
```

### 3. Monitor Progress

Training will:
- Save checkpoints to `saved_models/nf_config_mixed_sigmas_qweighted/`
- Display loss metrics in terminal
- Save best model based on validation loss

## Configuration Details

The config uses these Q-weighted transformations:

### Curves Transformation
- **Scaler**: `QWeightedCurvesScaler`
- **Formula**: `R' = R × Q^(-2)`
- **Purpose**: Emphasize high-Q features (thin layers, interfaces)

### Sigma Transformation
- **Scaler**: `QWeightedSigmaScaler`
- **Formula**: `dR' = dR × Q^(-3) / (R × ln(10))`
- **Purpose**: Proper uncertainty weighting accounting for Q-dependence and log scaling

### Key Settings
```yaml
curves_scaler:
  cls: QWeightedCurvesScaler
  kwargs:
    alpha: 2.0

sigma_scaler:
  cls: QWeightedSigmaScaler
  kwargs:
    beta: 3.0

synthetic_kwargs:
  calc_denoised_curves: true  # Required for sigma transformation

training:
  trainer_kwargs:
    train_with_sigmas: true   # Use sigmas for loss weighting
```

## Customizing the Config

### Change Q-weighting exponents

```yaml
curves_scaler:
  cls: QWeightedCurvesScaler
  kwargs:
    alpha: 2.5  # Change from 2.0 to 2.5

sigma_scaler:
  cls: QWeightedSigmaScaler
  kwargs:
    beta: 3.5   # Change from 3.0 to 3.5
```

### Adjust experimental/synthetic mix ratio

```yaml
dset:
  kwargs:
    mix_fraction: 0.3  # 30% experimental, 70% synthetic
```

### Change data directory

```yaml
dset:
  kwargs:
    data_dir: "path/to/your/dataset"
```

## Troubleshooting

### Error: "Expected 4 columns in .dat file"
**Solution**: Your experimental `.dat` files must have at least 3 columns (Q, R, dR)

### Error: "KeyError: 'sigma_scaler'"
**Solution**: Make sure you're using the updated config with `sigma_scaler` section

### Error: "'ExperimentalReflectivityDataLoader' has no attribute 'sigma_scaler'"
**Solution**: Check that:
1. You're using updated code with modified experimental_dataloaders.py
2. Config includes sigma_scaler section
3. reflectorch/runs/utils.py has been updated

### Warning: "calc_denoised_curves not enabled"
**Solution**: Add to config:
```yaml
dset:
  kwargs:
    synthetic_kwargs:
      calc_denoised_curves: true
```

## Resume Training

To continue from a checkpoint:

```bash
python -m reflectorch.train configs/nf_config_mixed_sigmas_qweighted.yaml \
  --resume saved_models/nf_config_mixed_sigmas_qweighted/checkpoint_latest.pt
```

## Inference After Training

Use trained model for predictions:

```python
from reflectorch.inference import InferenceModel

# Load trained model
model = InferenceModel.from_checkpoint(
    'saved_models/nf_config_mixed_sigmas_qweighted/best_model.pt'
)

# Run inference on experimental data
predictions = model.predict('path/to/experimental_curve.dat')
```

The model will automatically use the Q-weighted scalers saved during training.

## Validation

To verify Q-weighted transformations are active, check training logs for:
- `Using sigma_scaler: QWeightedSigmaScaler`
- `Using curves_scaler: QWeightedCurvesScaler`
- `calc_denoised_curves: True`
- `train_with_sigmas: True`

## Performance Tips

1. **GPU Memory**: Reduce `batch_size` if you encounter OOM errors
2. **Training Speed**: Increase `batch_size` if GPU memory allows
3. **Convergence**: Monitor loss - should decrease steadily for first 10k iterations
4. **Mix Ratio**: Start with 0.5, adjust based on which data type is more abundant

## Next Steps

After training completes:
1. Check `saved_models/nf_config_mixed_sigmas_qweighted/best_model.pt`
2. Run inference on test data
3. Compare predictions with/without Q-weighting
4. Adjust alpha/beta parameters if needed

For detailed implementation info, see [Q_WEIGHTED_SIGMA_IMPLEMENTATION.md](Q_WEIGHTED_SIGMA_IMPLEMENTATION.md)
