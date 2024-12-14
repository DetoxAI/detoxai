import numpy as np
import torch
import enum


class BiasMetrics(enum.Enum):
    TPR_GAP = "TPR-GAP"
    FPR_GAP = "FPR-GAP"
    TNR_GAP = "TNR-GAP"
    FNR_GAP = "FNR-GAP"


def calculate_bias_metric(
    metric: BiasMetrics | str,
    Y_pred: torch.Tensor,
    ProtAttr: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the bias metric

    Args:
        metric: Bias metric to calculate
        Y_true: True labels
        Y_pred: Predicted labels
        ProtAttr: Protected attribute

    Returns:
        Bias metric value
    """

    if isinstance(metric, str):
        metric = BiasMetrics(metric)

    if metric == BiasMetrics.TPR_GAP:
        bias = torch.abs(
            (Y_pred[ProtAttr == 1] == 1).float().mean()
            - (Y_pred[ProtAttr == 0] == 1).float().mean()
        )
    elif metric == BiasMetrics.FPR_GAP:
        bias = torch.abs(
            (Y_pred[ProtAttr == 1] == 1).float().mean()
            - (Y_pred[ProtAttr == 0] == 1).float().mean()
        )
    elif metric == BiasMetrics.TNR_GAP:
        bias = torch.abs(
            (Y_pred[ProtAttr == 1] == 0).float().mean()
            - (Y_pred[ProtAttr == 0] == 0).float().mean()
        )
    elif metric == BiasMetrics.FNR_GAP:
        bias = torch.abs(
            (Y_pred[ProtAttr == 1] == 0).float().mean()
            - (Y_pred[ProtAttr == 0] == 0).float().mean()
        )
    else:
        raise ValueError(f"Unknown bias metric: {metric}")

    return bias


def balanced_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Calculate the balanced accuracy metric
    """

    y_true = y_true.int()
    y_pred = y_pred.int()

    # Compute confusion matrix
    n_classes = len(torch.unique(y_true))
    confusion_matrix = torch.zeros(n_classes, n_classes)
    for t, p in zip(y_true, y_pred):
        confusion_matrix[t, p] += 1

    # Compute balanced accuracy
    balanced_acc = 0
    for i in range(n_classes):
        tp = confusion_matrix[i, i]
        fn = confusion_matrix[i, :].sum() - tp
        fp = confusion_matrix[:, i].sum() - tp
        tn = confusion_matrix.sum() - tp - fn - fp
        balanced_acc += tp / (tp + fn) + tn / (tn + fp)
    balanced_acc /= 2 * n_classes

    return balanced_acc


def phi(
    Y_true: torch.Tensor,
    Y_pred: torch.Tensor,
    ProtAttr: torch.Tensor,
    epsilon: float = 0.05,
    bias_metric: BiasMetrics | str = BiasMetrics.TPR_GAP,
) -> torch.Tensor:
    """
    Calculate phi as in the paper

    phi = balanced_accuracy(Y_true, Y_pred) if bias < epsilon else 0
    """

    assert (
        Y_true.shape == Y_pred.shape == ProtAttr.shape
    ), f"Y_true {Y_true.shape}, Y_pred {Y_pred.shape}, ProtAttr {ProtAttr.shape} must have the same shape"

    # Compute the bias metric
    bias = calculate_bias_metric(bias_metric, Y_pred, ProtAttr)

    # Compute phi
    phi = balanced_accuracy(Y_true, Y_pred) if bias < epsilon else 0

    return phi


def calculate_bias_metric_np(
    metric: BiasMetrics | str,
    Y_pred: np.ndarray,
    ProtAttr: np.ndarray,
) -> float:
    """
    Calculate the bias metric

    Args:
        metric: Bias metric to calculate
        Y_true: True labels
        Y_pred: Predicted labels
        ProtAttr: Protected attribute

    Returns:
        Bias metric value
    """

    if isinstance(metric, str):
        metric = BiasMetrics(metric)

    if metric == BiasMetrics.TPR_GAP:
        bias = np.abs(
            (Y_pred[ProtAttr == 1] == 1).mean() - (Y_pred[ProtAttr == 0] == 1).mean()
        )
    elif metric == BiasMetrics.FPR_GAP:
        bias = np.abs(
            (Y_pred[ProtAttr == 1] == 1).mean() - (Y_pred[ProtAttr == 0] == 1).mean()
        )
    elif metric == BiasMetrics.TNR_GAP:
        bias = np.abs(
            (Y_pred[ProtAttr == 1] == 0).mean() - (Y_pred[ProtAttr == 0] == 0).mean()
        )
    elif metric == BiasMetrics.FNR_GAP:
        bias = np.abs(
            (Y_pred[ProtAttr == 1] == 0).mean() - (Y_pred[ProtAttr == 0] == 0).mean()
        )
    else:
        raise ValueError(f"Unknown bias metric: {metric}")

    return bias


def balanced_accuracy_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the balanced accuracy metric
    """

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    # Compute confusion matrix
    n_classes = len(np.unique(y_true))
    confusion_matrix = np.zeros((n_classes, n_classes))
    for t, p in zip(y_true, y_pred):
        confusion_matrix[t, p] += 1

    # Compute balanced accuracy
    balanced_acc = 0
    for i in range(n_classes):
        tp = confusion_matrix[i, i]
        fn = confusion_matrix[i, :].sum() - tp
        fp = confusion_matrix[:, i].sum() - tp
        tn = confusion_matrix.sum() - tp - fn - fp
        balanced_acc += tp / (tp + fn) + tn / (tn + fp)
    balanced_acc /= 2 * n_classes

    return balanced_acc


def phi_np(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    ProtAttr: np.ndarray,
    epsilon: float = 0.05,
    bias_metric: BiasMetrics | str = BiasMetrics.TPR_GAP,
) -> float:
    """
    Calculate phi as in the paper

    phi = balanced_accuracy(Y_true, Y_pred) if bias < epsilon else 0
    """

    assert (
        Y_true.shape == Y_pred.shape == ProtAttr.shape
    ), f"Y_true {Y_true.shape}, Y_pred {Y_pred.shape}, ProtAttr {ProtAttr.shape} must have the same shape"

    # Compute the bias metric
    bias = calculate_bias_metric_np(bias_metric, Y_pred, ProtAttr)

    # Compute phi
    phi = balanced_accuracy_np(Y_true, Y_pred) if bias < epsilon else 0

    return phi
