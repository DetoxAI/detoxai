import numpy as np
import torch

from ...metrics.bias_metrics import (
    BiasMetrics,
    calculate_bias_metric_torch,
    calculate_bias_metric_np,
)
from ...metrics.metrics import balanced_accuracy_torch, balanced_accuracy_np


def phi_torch(
    Y_true: torch.Tensor,
    Y_pred: torch.Tensor,
    ProtAttr: torch.Tensor,
    epsilon: float = 0.05,
    bias_metric: BiasMetrics | str = BiasMetrics.TPR_GAP,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate phi as in the paper

    phi = balanced_accuracy(Y_true, Y_pred) if bias < epsilon else 0
    """

    assert (
        Y_true.shape == Y_pred.shape == ProtAttr.shape
    ), f"Y_true {Y_true.shape}, Y_pred {Y_pred.shape}, ProtAttr {ProtAttr.shape} must have the same shape"

    # Compute the bias metric
    bias = calculate_bias_metric_torch(bias_metric, Y_pred, Y_true, ProtAttr)

    # Compute phi
    phi = (
        balanced_accuracy_torch(Y_true, Y_pred)
        if bias < epsilon
        else torch.tensor(0, dtype=torch.float32, device=Y_true.device)
    )

    return phi, bias


def phi_np(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    ProtAttr: np.ndarray,
    epsilon: float = 0.05,
    bias_metric: BiasMetrics | str = BiasMetrics.EO_GAP,
) -> tuple[float, float]:
    """
    Calculate phi as in the paper

    phi = balanced_accuracy(Y_true, Y_pred) if bias < epsilon else 0
    """

    assert (
        Y_true.shape == Y_pred.shape == ProtAttr.shape
    ), f"Y_true {Y_true.shape}, Y_pred {Y_pred.shape}, ProtAttr {ProtAttr.shape} must have the same shape"

    # Compute the bias metric
    bias = calculate_bias_metric_np(bias_metric, Y_pred, Y_true, ProtAttr)

    # Compute phi
    phi = balanced_accuracy_np(Y_true, Y_pred) if bias < epsilon else 0

    return phi, bias
