import enum
import torch
import numpy as np


class BiasMetrics(enum.Enum):
    TPR_GAP = "TPR_GAP"
    FPR_GAP = "FPR_GAP"
    TNR_GAP = "TNR_GAP"
    FNR_GAP = "FNR_GAP"
    EO_GAP = "EO_GAP"
    DP_GAP = "DP_GAP"


def stabilize(x, epsilon=1e-6):
    return x + epsilon


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

    if isinstance(metric, BiasMetrics):
        metric = metric.value

    tp = (Y_pred[ProtAttr == 1] == 1).sum()
    fp = (Y_pred[ProtAttr == 0] == 1).sum()
    tn = (Y_pred[ProtAttr == 1] == 0).sum()
    fn = (Y_pred[ProtAttr == 0] == 0).sum()

    tpr = tp / stabilize(tp + fn)
    fpr = fp / stabilize(fp + tn)
    tnr = tn / stabilize(tn + fp)
    fnr = fn / stabilize(fn + tp)

    if metric == BiasMetrics.TPR_GAP.value:
        bias = torch.abs(tpr - fpr)
    elif metric == BiasMetrics.FPR_GAP.value:
        bias = torch.abs(fpr - tpr)
    elif metric == BiasMetrics.TNR_GAP.value:
        bias = torch.abs(tnr - fnr)
    elif metric == BiasMetrics.FNR_GAP.value:
        bias = torch.abs(fnr - tnr)
    elif metric == BiasMetrics.EO_GAP.value or metric == BiasMetrics.DP_GAP.value:
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

        if metric == BiasMetrics.EO_GAP.value:
            bias = 0.5 * (torch.abs(tpr_a - tpr_b) + torch.abs(fpr_a - fpr_b))
        elif metric == BiasMetrics.DP_GAP.value:
            bias = torch.abs(tpr_a - tpr_b)
    else:
        raise ValueError(f"Unknown bias metric: {metric}")

    return bias


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

    if isinstance(metric, BiasMetrics):
        metric = metric.value

    tp = (Y_pred[ProtAttr == 1] == 1).sum()
    fp = (Y_pred[ProtAttr == 0] == 1).sum()
    tn = (Y_pred[ProtAttr == 1] == 0).sum()
    fn = (Y_pred[ProtAttr == 0] == 0).sum()

    tpr = tp / stabilize(tp + fn)
    fpr = fp / stabilize(fp + tn)
    tnr = tn / stabilize(tn + fp)
    fnr = fn / stabilize(fn + tp)

    if metric == BiasMetrics.TPR_GAP.value:
        bias = abs(tpr - fpr)
    elif metric == BiasMetrics.FPR_GAP.value:
        bias = abs(fpr - tpr)
    elif metric == BiasMetrics.TNR_GAP.value:
        bias = abs(tnr - fnr)
    elif metric == BiasMetrics.FNR_GAP.value:
        bias = abs(fnr - tnr)
    elif metric == BiasMetrics.EO_GAP.value or metric == BiasMetrics.DP_GAP.value:
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

        if metric == BiasMetrics.EO_GAP.value:
            bias = 0.5 * (abs(tpr_a - tpr_b) + abs(fpr_a - fpr_b))
        elif metric == BiasMetrics.DP_GAP.value:
            bias = abs(tpr_a - tpr_b)
    else:
        raise ValueError(f"Unknown bias metric: {metric}")

    return bias
