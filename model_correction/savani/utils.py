import numpy as np
import torch
import enum


class BiasMetrics(enum.Enum):
    TPR_GAP = "TPR_GAP"
    FPR_GAP = "FPR_GAP"
    TNR_GAP = "TNR_GAP"
    FNR_GAP = "FNR_GAP"
    EO_GAP = "EO_GAP"
    DP_GAP = "DP_GAP"


def calculate_bias_metric_torch(
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

    tp = (Y_pred[ProtAttr == 1] == 1).sum()
    fp = (Y_pred[ProtAttr == 0] == 1).sum()
    tn = (Y_pred[ProtAttr == 1] == 0).sum()
    fn = (Y_pred[ProtAttr == 0] == 0).sum()

    tpr = tp / stabilize(tp + fn)
    fpr = fp / stabilize(fp + tn)
    tnr = tn / stabilize(tn + fp)
    fnr = fn / stabilize(fn + tp)

    if metric == BiasMetrics.TPR_GAP:
        bias = torch.abs(tpr - fpr)
    elif metric == BiasMetrics.FPR_GAP:
        bias = torch.abs(fpr - tpr)
    elif metric == BiasMetrics.TNR_GAP:
        bias = torch.abs(tnr - fnr)
    elif metric == BiasMetrics.FNR_GAP:
        bias = torch.abs(fnr - tnr)
    elif metric == BiasMetrics.EO_GAP or metric == BiasMetrics.DP_GAP:
        tp_a = (Y_pred[ProtAttr == 1] == 1).sum()
        fp_a = (Y_pred[ProtAttr == 1] == 0).sum()
        tn_a = (Y_pred[ProtAttr == 0] == 0).sum()
        fn_a = (Y_pred[ProtAttr == 0] == 1).sum()

        tpr_a = tp_a / stabilize(tp_a + fn_a)
        fpr_a = fp_a / stabilize(fp_a + tn_a)

        tp_b = (Y_pred[ProtAttr == 0] == 1).sum()
        fp_b = (Y_pred[ProtAttr == 0] == 0).sum()
        tn_b = (Y_pred[ProtAttr == 1] == 0).sum()
        fn_b = (Y_pred[ProtAttr == 1] == 1).sum()

        tpr_b = tp_b / stabilize(tp_b + fn_b)
        fpr_b = fp_b / stabilize(fp_b + tn_b)

        if metric == BiasMetrics.EO_GAP:
            bias = 0.5 * (torch.abs(tpr_a - tpr_b) + torch.abs(fpr_a - fpr_b))
        elif metric == BiasMetrics.DP_GAP:
            bias = torch.abs(tpr_a - tpr_b)
    else:
        raise ValueError(f"Unknown bias metric: {metric}")

    return bias


def balanced_accuracy_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
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
    bias = calculate_bias_metric_torch(bias_metric, Y_pred, ProtAttr)

    # Compute phi
    phi = (
        balanced_accuracy_torch(Y_true, Y_pred)
        if bias < epsilon
        else torch.tensor(0, dtype=torch.float32, device=Y_true.device)
    )

    return phi, bias


def stabilize(x, epsilon=1e-6):
    return x + epsilon


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

    tp = (Y_pred[ProtAttr == 1] == 1).sum()
    fp = (Y_pred[ProtAttr == 0] == 1).sum()
    tn = (Y_pred[ProtAttr == 1] == 0).sum()
    fn = (Y_pred[ProtAttr == 0] == 0).sum()

    tpr = tp / stabilize(tp + fn)
    fpr = fp / stabilize(fp + tn)
    tnr = tn / stabilize(tn + fp)
    fnr = fn / stabilize(fn + tp)

    if metric == BiasMetrics.TPR_GAP:
        bias = abs(tpr - fpr)
    elif metric == BiasMetrics.FPR_GAP:
        bias = abs(fpr - tpr)
    elif metric == BiasMetrics.TNR_GAP:
        bias = abs(tnr - fnr)
    elif metric == BiasMetrics.FNR_GAP:
        bias = abs(fnr - tnr)
    elif metric == BiasMetrics.EO_GAP or metric == BiasMetrics.DP_GAP:
        tp_a = (Y_pred[ProtAttr == 1] == 1).sum()
        fp_a = (Y_pred[ProtAttr == 1] == 0).sum()
        tn_a = (Y_pred[ProtAttr == 0] == 0).sum()
        fn_a = (Y_pred[ProtAttr == 0] == 1).sum()

        tpr_a = tp_a / stabilize(tp_a + fn_a)
        fpr_a = fp_a / stabilize(fp_a + tn_a)

        tp_b = (Y_pred[ProtAttr == 0] == 1).sum()
        fp_b = (Y_pred[ProtAttr == 0] == 0).sum()
        tn_b = (Y_pred[ProtAttr == 1] == 0).sum()
        fn_b = (Y_pred[ProtAttr == 1] == 1).sum()

        tpr_b = tp_b / stabilize(tp_b + fn_b)
        fpr_b = fp_b / stabilize(fp_b + tn_b)

        if metric == BiasMetrics.EO_GAP:
            bias = 0.5 * (abs(tpr_a - tpr_b) + abs(fpr_a - fpr_b))
        elif metric == BiasMetrics.DP_GAP:
            bias = abs(tpr_a - tpr_b)
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
    bias = calculate_bias_metric_np(bias_metric, Y_pred, ProtAttr)

    # Compute phi
    phi = balanced_accuracy_np(Y_true, Y_pred) if bias < epsilon else 0

    return phi, bias


def flatten_with_map(arr, indices):
    return arr[indices].flatten()


def unflatten_with_map(original, flat_arr, indices):
    original[indices] = flat_arr.reshape(original[indices].shape)
    return original
