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
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    protected_attribute: torch.Tensor,
) -> torch.Tensor:
    if isinstance(metric, BiasMetrics):
        metric = metric.value

    # Make sure proper data types are used
    protected_attribute = protected_attribute.to(dtype=torch.bool)

    # Calculate confusion matrix for group A (protected_attribute == 1)
    tp_a = (
        (y_pred[protected_attribute] == 1) & (y_true[protected_attribute] == 1)
    ).sum()
    fp_a = (
        (y_pred[protected_attribute] == 1) & (y_true[protected_attribute] == 0)
    ).sum()
    tn_a = (
        (y_pred[protected_attribute] == 0) & (y_true[protected_attribute] == 0)
    ).sum()
    fn_a = (
        (y_pred[protected_attribute] == 0) & (y_true[protected_attribute] == 1)
    ).sum()

    # Calculate rates for group A
    tpr_a = tp_a / stabilize(tp_a + fn_a)
    fpr_a = fp_a / stabilize(fp_a + tn_a)
    tnr_a = tn_a / stabilize(tn_a + fp_a)
    fnr_a = fn_a / stabilize(fn_a + tp_a)

    # Calculate confusion matrix for group B (protected_attribute == 0)
    tp_b = (
        (y_pred[~protected_attribute] == 1) & (y_true[~protected_attribute] == 1)
    ).sum()
    fp_b = (
        (y_pred[~protected_attribute] == 1) & (y_true[~protected_attribute] == 0)
    ).sum()
    tn_b = (
        (y_pred[~protected_attribute] == 0) & (y_true[~protected_attribute] == 0)
    ).sum()
    fn_b = (
        (y_pred[~protected_attribute] == 0) & (y_true[~protected_attribute] == 1)
    ).sum()

    tpr_b = tp_b / stabilize(tp_b + fn_b)
    fpr_b = fp_b / stabilize(fp_b + tn_b)
    tnr_b = tn_b / stabilize(tn_b + fp_b)
    fnr_b = fn_b / stabilize(fn_b + tp_b)

    ppr_a = (y_pred[protected_attribute] == 1).sum() / protected_attribute.sum()
    ppr_b = (y_pred[~protected_attribute] == 1).sum() / (~protected_attribute).sum()

    if metric == BiasMetrics.TPR_GAP.value:
        bias = torch.abs(tpr_a - tpr_b)
    elif metric == BiasMetrics.FPR_GAP.value:
        bias = torch.abs(fpr_a - fpr_b)
    elif metric == BiasMetrics.TNR_GAP.value:
        bias = torch.abs(tnr_a - tnr_b)
    elif metric == BiasMetrics.FNR_GAP.value:
        bias = torch.abs(fnr_a - fnr_b)
    elif metric == BiasMetrics.EO_GAP.value:
        bias = torch.max(torch.abs(tpr_a - tpr_b), torch.abs(fpr_a - fpr_b))
    elif metric == BiasMetrics.DP_GAP.value:
        bias = torch.abs(ppr_a - ppr_b)
    else:
        raise ValueError(f"Unknown bias metric: {metric}")

    return bias


def calculate_bias_metric_np(
    metric: BiasMetrics | str,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    protected_attribute: np.ndarray,
) -> float:
    if isinstance(metric, BiasMetrics):
        metric = metric.value

    # Calculate confusion matrix for group A (protected_attribute == 1)
    tp_a = (
        (y_pred[protected_attribute] == 1) & (y_true[protected_attribute] == 1)
    ).sum()
    fp_a = (
        (y_pred[protected_attribute] == 1) & (y_true[protected_attribute] == 0)
    ).sum()
    tn_a = (
        (y_pred[protected_attribute] == 0) & (y_true[protected_attribute] == 0)
    ).sum()
    fn_a = (
        (y_pred[protected_attribute] == 0) & (y_true[protected_attribute] == 1)
    ).sum()

    # Calculate rates for group A
    tpr_a = tp_a / stabilize(tp_a + fn_a)
    fpr_a = fp_a / stabilize(fp_a + tn_a)
    tnr_a = tn_a / stabilize(tn_a + fp_a)
    fnr_a = fn_a / stabilize(fn_a + tp_a)

    # Calculate confusion matrix for group B (protected_attribute == 0)
    tp_b = (
        (y_pred[~protected_attribute] == 1) & (y_true[~protected_attribute] == 1)
    ).sum()
    fp_b = (
        (y_pred[~protected_attribute] == 1) & (y_true[~protected_attribute] == 0)
    ).sum()
    tn_b = (
        (y_pred[~protected_attribute] == 0) & (y_true[~protected_attribute] == 0)
    ).sum()
    fn_b = (
        (y_pred[~protected_attribute] == 0) & (y_true[~protected_attribute] == 1)
    ).sum()

    # Calculate rates for group B
    tpr_b = tp_b / stabilize(tp_b + fn_b)
    fpr_b = fp_b / stabilize(fp_b + tn_b)
    tnr_b = tn_b / stabilize(tn_b + fp_b)
    fnr_b = fn_b / stabilize(fn_b + tp_b)

    if metric == BiasMetrics.TPR_GAP.value:
        bias = abs(tpr_a - tpr_b)
    elif metric == BiasMetrics.FPR_GAP.value:
        bias = abs(fpr_a - fpr_b)
    elif metric == BiasMetrics.TNR_GAP.value:
        bias = abs(tnr_a - tnr_b)
    elif metric == BiasMetrics.FNR_GAP.value:
        bias = abs(fnr_a - fnr_b)
    elif metric == BiasMetrics.EO_GAP.value:
        bias = max(abs(tpr_a - tpr_b), abs(fpr_a - fpr_b))
    elif metric == BiasMetrics.DP_GAP.value:
        ppr_a = (y_pred[protected_attribute] == 1).sum() / protected_attribute.sum()
        ppr_b = (y_pred[~protected_attribute] == 1).sum() / (~protected_attribute).sum()
        bias = abs(ppr_a - ppr_b)
    else:
        raise ValueError(f"Unknown bias metric: {metric}")

    return float(bias)
