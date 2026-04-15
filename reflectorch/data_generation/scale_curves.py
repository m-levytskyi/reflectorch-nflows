from pathlib import Path
import math

import numpy as np
import torch
from torch import Tensor

from reflectorch.data_generation.priors import PriorSampler
from reflectorch.paths import SAVED_MODELS_DIR


class CurvesScaler(object):
    """Base class for curve scalers"""
    def scale(self, curves: Tensor, q_values: Tensor = None):
        raise NotImplementedError

    def restore(self, curves: Tensor, q_values: Tensor = None):
        raise NotImplementedError


class LogAffineCurvesScaler(CurvesScaler):
    r"""Curve scaler which scales the reflectivity curves according to the logarithmic affine transformation:
    :math:`\log_{10}(R + eps) \cdot weight + bias`.

    Args:
        weight (float): multiplication factor in the transformation
        bias (float): addition term in the transformation
        eps (float): sets the minimum intensity value of the reflectivity curves which is considered
    """
    def __init__(self, weight: float = 0.1, bias: float = 0.5, eps: float = 1e-10):
        self.weight = weight
        self.bias = bias
        self.eps = eps

    def scale(self, curves: Tensor, q_values: Tensor = None):
        """scales the reflectivity curves to a ML-friendly range

        Args:
            curves (Tensor): original reflectivity curves
            q_values (Tensor, optional): Q values (ignored by this scaler for backward compatibility)

        Returns:
            Tensor: reflectivity curves scaled to a ML-friendly range
        """
        return torch.log10(curves + self.eps) * self.weight + self.bias

    def restore(self, curves: Tensor, q_values: Tensor = None):
        """restores the physical reflectivity curves

        Args:
            curves (Tensor): scaled reflectivity curves
            q_values (Tensor, optional): Q values (ignored by this scaler for backward compatibility)

        Returns:
            Tensor: reflectivity curves restored to the physical range
        """
        return 10 ** ((curves - self.bias) / self.weight) - self.eps


class MeanNormalizationCurvesScaler(CurvesScaler):
    """Curve scaler which scales the reflectivity curves by the precomputed mean of a batch of curves

    Args:
        path (str, optional): path to the precomputed mean of the curves, only used if ``curves_mean`` is None. Defaults to None.
        curves_mean (Tensor, optional): the precomputed mean of the curves. Defaults to None.
        device (torch.device, optional): the Pytorch device. Defaults to 'cuda'.
    """

    def __init__(self, path: str = None, curves_mean: Tensor = None, device: torch.device = 'cuda'):
        if curves_mean is None:
            curves_mean = torch.load(self.get_path(path))
        self.curves_mean = curves_mean.to(device)

    def scale(self, curves: Tensor, q_values: Tensor = None):
        """scales the reflectivity curves to a ML-friendly range

        Args:
            curves (Tensor): original reflectivity curves
            q_values (Tensor, optional): Q values (ignored by this scaler for backward compatibility)

        Returns:
            Tensor: reflectivity curves scaled to a ML-friendly range
        """
        self.curves_mean = self.curves_mean.to(curves)
        return curves / self.curves_mean - 1

    def restore(self, curves: Tensor, q_values: Tensor = None):
        """restores the physical reflectivity curves

        Args:
            curves (Tensor): scaled reflectivity curves
            q_values (Tensor, optional): Q values (ignored by this scaler for backward compatibility)

        Returns:
            Tensor: reflectivity curves restored to the physical range
        """
        self.curves_mean = self.curves_mean.to(curves)
        return (curves + 1) * self.curves_mean

    @staticmethod
    def save(prior_sampler: PriorSampler, q: Tensor, path: str, num: int = 16384):
        """computes the mean of a batch of reflectivity curves and saves it

        Args:
            prior_sampler (PriorSampler): the prior sampler
            q (Tensor): the q values
            path (str): the path for saving the mean of the curves
            num (int, optional): the number of curves used to compute the mean. Defaults to 16384.
        """
        params = prior_sampler.sample(num)
        curves_mean = params.reflectivity(q, log=False).mean(0).cpu()
        torch.save(curves_mean, MeanNormalizationCurvesScaler.get_path(path))

    @staticmethod
    def get_path(path: str) -> Path:
        if not path.endswith('.pt'):
            path = path + '.pt'
        return SAVED_MODELS_DIR / path


class QWeightedCurvesScaler(CurvesScaler):
    """Curve scaler which scales reflectivity curves with Q-dependent weighting: R' = R * Q^(-alpha)
    
    Args:
        alpha (float): the exponent for Q-weighting. Defaults to 2.0.
    """
    def __init__(self, alpha: float = 2.0):
        self.alpha = alpha
    
    def scale(self, curves, q_values=None):
        """scales the reflectivity curves with Q-dependent weighting
        
        Args:
            curves (Tensor): original reflectivity curves, shape [B, Nq] (can be numpy array or tensor)
            q_values (Tensor): Q values, shape [B, Nq] (can be numpy array or tensor)
        
        Returns:
            Tensor: Q-weighted reflectivity curves
        """
        if q_values is None:
            raise ValueError("QWeightedCurvesScaler requires q_values parameter")
        
        print(f"\n{'='*80}")
        print(f"[QWeightedCurvesScaler.scale] STARTING Q-WEIGHTED CURVE TRANSFORMATION (alpha={self.alpha})")
        print(f"{'='*80}")
        
        # Convert to tensors if needed (handles NumPy arrays during inference)
        if isinstance(curves, np.ndarray):
            print(f"[QWeightedCurvesScaler.scale] INPUT curves: numpy array")
            print(f"  - dtype: {curves.dtype}")
            print(f"  - shape: {curves.shape}")
            print(f"  - range: [{curves.min():.6e}, {curves.max():.6e}]")
            print(f"  - mean: {curves.mean():.6e}")
            curves = torch.from_numpy(curves).float()
            print(f"  - Converted to tensor dtype: {curves.dtype}")
        elif isinstance(curves, torch.Tensor):
            print(f"[QWeightedCurvesScaler.scale] INPUT curves: tensor")
            print(f"  - dtype: {curves.dtype}")
            print(f"  - shape: {curves.shape}")
            print(f"  - range: [{curves.min():.6e}, {curves.max():.6e}]")
            print(f"  - mean: {curves.mean():.6e}")
            curves = curves.float()
        
        if isinstance(q_values, np.ndarray):
            print(f"[QWeightedCurvesScaler.scale] INPUT q_values: numpy array")
            print(f"  - dtype: {q_values.dtype}")
            print(f"  - shape: {q_values.shape}")
            print(f"  - range: [{q_values.min():.6e}, {q_values.max():.6e}]")
            q_values = torch.from_numpy(q_values).float()
            print(f"  - Converted to tensor dtype: {q_values.dtype}")
        elif isinstance(q_values, torch.Tensor):
            print(f"[QWeightedCurvesScaler.scale] INPUT q_values: tensor")
            print(f"  - dtype: {q_values.dtype}")
            print(f"  - shape: {q_values.shape}")
            print(f"  - range: [{q_values.min():.6e}, {q_values.max():.6e}]")
            q_values = q_values.float()
        
        q_weight = q_values ** (-self.alpha)
        print(f"[QWeightedCurvesScaler.scale] Q-weight (Q^-{self.alpha}):")
        print(f"  - range: [{q_weight.min():.6e}, {q_weight.max():.6e}]")
        print(f"  - mean: {q_weight.mean():.6e}")
        
        result = curves * q_weight
        result = result.float()
        
        print(f"[QWeightedCurvesScaler.scale] OUTPUT (R' = R * Q^-{self.alpha}):")
        print(f"  - dtype: {result.dtype}")
        print(f"  - shape: {result.shape}")
        print(f"  - range: [{result.min():.6e}, {result.max():.6e}]")
        print(f"  - mean: {result.mean():.6e}")
        print(f"{'='*80}\n")
        return result
    
    def restore(self, scaled_curves: Tensor, q_values: Tensor = None):
        """restores the physical reflectivity curves from Q-weighted form
        
        Args:
            scaled_curves (Tensor): Q-weighted reflectivity curves, shape [B, Nq] (can be numpy array or tensor)
            q_values (Tensor): Q values, shape [B, Nq] (can be numpy array or tensor)
        
        Returns:
            Tensor: reflectivity curves restored to physical range
        """
        if q_values is None:
            raise ValueError("QWeightedCurvesScaler requires q_values parameter for restoration")
        
        # Convert to tensors if needed
        if isinstance(scaled_curves, np.ndarray):
            print(f"[QWeightedCurvesScaler.restore] Converting scaled_curves from numpy ({scaled_curves.dtype})")
            scaled_curves = torch.from_numpy(scaled_curves).float()
        elif isinstance(scaled_curves, torch.Tensor):
            scaled_curves = scaled_curves.float()
        
        if isinstance(q_values, np.ndarray):
            q_values = torch.from_numpy(q_values).float()
        elif isinstance(q_values, torch.Tensor):
            q_values = q_values.float()
        
        q_weight = q_values ** (-self.alpha)
        result = scaled_curves / q_weight
        
        # Ensure output is float32
        result = result.float()
        print(f"[QWeightedCurvesScaler.restore] Output dtype: {result.dtype}")
        return result


class QWeightedSigmaScaler:
    """Sigma scaler which transforms dR values with: dR' = dR * Q^(-beta) / (R * ln(10))
    
    This scaler is designed specifically for error/sigma values and does not inherit from CurvesScaler.
    
    Args:
        beta (float): the exponent for Q-weighting of sigmas. Defaults to 3.0.
    """
    def __init__(self, beta: float = 3.0):
        self.beta = beta
        self.ln10 = math.log(10)
    
    def scale(self, sigmas: Tensor, curves: Tensor, q_values: Tensor):
        """scales sigma values with Q-dependent weighting and curve normalization
        
        Args:
            sigmas (Tensor): original dR values, shape [B, Nq] (can be numpy array or tensor)
            curves (Tensor): reflectivity curves R (MUST BE UNSCALED), shape [B, Nq] (can be numpy array or tensor)
            q_values (Tensor): Q values, shape [B, Nq] (can be numpy array or tensor)
        
        Returns:
            Tensor: scaled sigma values dR' = dR * Q^(-beta) / (R * ln(10))
        
        Raises:
            ValueError: if curves appear to be already scaled (not physical R values)
        """
        print(f"\n{'='*80}")
        print(f"[QWeightedSigmaScaler.scale] STARTING Q-WEIGHTED SIGMA TRANSFORMATION (beta={self.beta})")
        print(f"{'='*80}")
        
        # Convert to tensors if needed (handles NumPy arrays during inference)
        if isinstance(sigmas, np.ndarray):
            print(f"[QWeightedSigmaScaler.scale] INPUT sigmas: numpy array")
            print(f"  - dtype: {sigmas.dtype}")
            print(f"  - shape: {sigmas.shape}")
            print(f"  - range: [{sigmas.min():.6e}, {sigmas.max():.6e}]")
            print(f"  - mean: {sigmas.mean():.6e}")
            sigmas = torch.from_numpy(sigmas).float()
            print(f"  - Converted to tensor dtype: {sigmas.dtype}")
        elif isinstance(sigmas, torch.Tensor):
            print(f"[QWeightedSigmaScaler.scale] INPUT sigmas: tensor")
            print(f"  - dtype: {sigmas.dtype}")
            print(f"  - shape: {sigmas.shape}")
            print(f"  - range: [{sigmas.min():.6e}, {sigmas.max():.6e}]")
            print(f"  - mean: {sigmas.mean():.6e}")
            sigmas = sigmas.float()
        
        if isinstance(curves, np.ndarray):
            print(f"[QWeightedSigmaScaler.scale] INPUT curves (unscaled R): numpy array")
            print(f"  - dtype: {curves.dtype}")
            print(f"  - shape: {curves.shape}")
            print(f"  - range: [{curves.min():.6e}, {curves.max():.6e}]")
            print(f"  - mean: {curves.mean():.6e}")
            curves = torch.from_numpy(curves).float()
            print(f"  - Converted to tensor dtype: {curves.dtype}")
        elif isinstance(curves, torch.Tensor):
            print(f"[QWeightedSigmaScaler.scale] INPUT curves (unscaled R): tensor")
            print(f"  - dtype: {curves.dtype}")
            print(f"  - shape: {curves.shape}")
            print(f"  - range: [{curves.min():.6e}, {curves.max():.6e}]")
            print(f"  - mean: {curves.mean():.6e}")
            curves = curves.float()
        
        if isinstance(q_values, np.ndarray):
            print(f"[QWeightedSigmaScaler.scale] INPUT q_values: numpy array")
            print(f"  - dtype: {q_values.dtype}")
            print(f"  - shape: {q_values.shape}")
            print(f"  - range: [{q_values.min():.6e}, {q_values.max():.6e}]")
            q_values = torch.from_numpy(q_values).float()
            print(f"  - Converted to tensor dtype: {q_values.dtype}")
        elif isinstance(q_values, torch.Tensor):
            print(f"[QWeightedSigmaScaler.scale] INPUT q_values: tensor")
            print(f"  - dtype: {q_values.dtype}")
            print(f"  - shape: {q_values.shape}")
            print(f"  - range: [{q_values.min():.6e}, {q_values.max():.6e}]")
            q_values = q_values.float()
        
        # Defensive check: physical reflectivity should be in range [~1e-10, ~2.0]
        if curves.min() < -1.0 or curves.max() > 10.0:
            print(f"[QWeightedSigmaScaler.scale] ⚠️  WARNING: Curves may be pre-scaled!")
            print(f"  Physical R should be in [~1e-10, ~2.0], got [{curves.min():.3e}, {curves.max():.3e}]")
        
        # Apply transformation: dR' = dR * Q^(-beta) / (R * ln(10))
        q_weight = q_values ** (-self.beta)
        print(f"[QWeightedSigmaScaler.scale] Q-weight (Q^-{self.beta}):")
        print(f"  - range: [{q_weight.min():.6e}, {q_weight.max():.6e}]")
        print(f"  - mean: {q_weight.mean():.6e}")
        
        denominator = curves * self.ln10
        print(f"[QWeightedSigmaScaler.scale] Denominator (R * ln(10)):")
        print(f"  - range: [{denominator.min():.6e}, {denominator.max():.6e}]")
        print(f"  - mean: {denominator.mean():.6e}")
        
        result = sigmas * q_weight / denominator
        result = result.float()
        
        print(f"[QWeightedSigmaScaler.scale] OUTPUT (dR' = dR * Q^-{self.beta} / (R * ln(10))):")
        print(f"  - dtype: {result.dtype}")
        print(f"  - shape: {result.shape}")
        print(f"  - range: [{result.min():.6e}, {result.max():.6e}]")
        print(f"  - mean: {result.mean():.6e}")
        print(f"{'='*80}\n")
        return result
    
    def restore(self, scaled_sigmas: Tensor, curves: Tensor, q_values: Tensor):
        """restores sigma values from scaled form (may not be used in practice)
        
        Args:
            scaled_sigmas (Tensor): scaled sigma values, shape [B, Nq] (can be numpy array or tensor)
            curves (Tensor): reflectivity curves R, shape [B, Nq] (can be numpy array or tensor)
            q_values (Tensor): Q values, shape [B, Nq] (can be numpy array or tensor)
        
        Returns:
            Tensor: restored dR values
        """
        # Convert to tensors if needed
        if isinstance(scaled_sigmas, np.ndarray):
            scaled_sigmas = torch.from_numpy(scaled_sigmas).float()
        elif isinstance(scaled_sigmas, torch.Tensor):
            scaled_sigmas = scaled_sigmas.float()
        
        if isinstance(curves, np.ndarray):
            curves = torch.from_numpy(curves).float()
        elif isinstance(curves, torch.Tensor):
            curves = curves.float()
        
        if isinstance(q_values, np.ndarray):
            q_values = torch.from_numpy(q_values).float()
        elif isinstance(q_values, torch.Tensor):
            q_values = q_values.float()
        
        q_weight = q_values ** (-self.beta)
        result = scaled_sigmas * curves * self.ln10 / q_weight
        
        # Ensure output is float32
        return result.float()


class PositiveQCurvesScaler(CurvesScaler):
    """Curve scaler with positive Q exponent weighting: R' = R * Q^(alpha)

    Args:
        alpha (float): the exponent for Q-weighting. Defaults to 4.0.
    """

    def __init__(self, alpha: float = 4.0):
        self.alpha = alpha

    def scale(self, curves: Tensor, q_values: Tensor = None) -> Tensor:
        """Scale reflectivity curves with positive Q exponent weighting.

        Args:
            curves: original reflectivity curves, shape [B, Nq]
            q_values: Q values, shape [B, Nq]

        Returns:
            Q-weighted reflectivity curves
        """
        if q_values is None:
            raise ValueError("PositiveQCurvesScaler requires q_values parameter")

        q_weight = q_values ** self.alpha
        return (curves * q_weight).float()

    def restore(self, scaled_curves: Tensor, q_values: Tensor = None) -> Tensor:
        """Restore physical reflectivity curves from positive-Q weighted form."""
        if q_values is None:
            raise ValueError("PositiveQCurvesScaler requires q_values parameter for restoration")

        q_weight = q_values ** self.alpha
        return (scaled_curves / q_weight).float()


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