import enum

import torch


class BiasMetrics(enum.Enum):
    """ """

    TPR_GAP = "TPR_GAP"
    FPR_GAP = "FPR_GAP"
    TNR_GAP = "TNR_GAP"
    FNR_GAP = "FNR_GAP"
    EO_GAP = "EO_GAP"
    DP_GAP = "DP_GAP"


def stabilize(x, epsilon=1e-4):
    """

    Args:
      x:
      epsilon:  (Default value = 1e-4)

    Returns:

    """
    return torch.max(x, torch.tensor(epsilon, dtype=x.dtype, device=x.device))


def calculate_bias_metric_torch(
    metric: BiasMetrics | str,
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    protected_attribute: torch.Tensor,
) -> torch.Tensor:
    """

    Args:
      metric: BiasMetrics | str:
      y_pred: torch.Tensor:
      y_true: torch.Tensor:
      protected_attribute: torch.Tensor:

    Returns:

    """
    if isinstance(metric, BiasMetrics):
        metric = metric.value

    # Make sure proper data types are used
    prot_attr = protected_attribute.bool()
    y_true = y_true.int()
    y_pred = y_pred.int()

    # Calculate confusion matrix for group A (prot_attr == 1)
    tp_a = ((y_pred[prot_attr] == 1) & (y_true[prot_attr] == 1)).sum().float()
    fp_a = ((y_pred[prot_attr] == 1) & (y_true[prot_attr] == 0)).sum().float()
    tn_a = ((y_pred[prot_attr] == 0) & (y_true[prot_attr] == 0)).sum().float()
    fn_a = ((y_pred[prot_attr] == 0) & (y_true[prot_attr] == 1)).sum().float()

    # Calculate rates for group A
    tpr_a = tp_a / stabilize(tp_a + fn_a)
    fpr_a = fp_a / stabilize(fp_a + tn_a)
    tnr_a = tn_a / stabilize(tn_a + fp_a)
    fnr_a = fn_a / stabilize(fn_a + tp_a)

    # Calculate confusion matrix for group B (prot_attr == 0)
    tp_b = ((y_pred[~prot_attr] == 1) & (y_true[~prot_attr] == 1)).sum().float()
    fp_b = ((y_pred[~prot_attr] == 1) & (y_true[~prot_attr] == 0)).sum().float()
    tn_b = ((y_pred[~prot_attr] == 0) & (y_true[~prot_attr] == 0)).sum().float()
    fn_b = ((y_pred[~prot_attr] == 0) & (y_true[~prot_attr] == 1)).sum().float()

    tpr_b = tp_b / stabilize(tp_b + fn_b)
    fpr_b = fp_b / stabilize(fp_b + tn_b)
    tnr_b = tn_b / stabilize(tn_b + fp_b)
    fnr_b = fn_b / stabilize(fn_b + tp_b)

    ppr_a = (y_pred[prot_attr] == 1).sum().float() / stabilize(
        prot_attr.int().sum()
    ).float()
    ppr_b = (y_pred[~prot_attr] == 1).sum().float() / stabilize(
        (~prot_attr).int().sum()
    ).float()

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
