import torch


def stabilize_torch(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Stabilize a tensor by adding a small epsilon

    Args:
      x: torch.Tensor:
      eps: float:  (Default value = 1e-4)

    Returns:

    """
    eps = torch.tensor(eps, dtype=x.dtype, device=x.device)
    return torch.max(x, eps)


def balanced_accuracy_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Calculate the balanced accuracy metric

    Args:
      y_true: torch.Tensor:
      y_pred: torch.Tensor:

    Returns:

    """

    y_true = y_true.int()
    y_pred = y_pred.int()

    # Compute confusion matrix
    n_classes = 2  # Assuming binary
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
        balanced_acc += tp / stabilize_torch(tp + fn) + tn / stabilize_torch(tn + fp)
    balanced_acc /= 2 * n_classes

    return balanced_acc


def comprehensive_metrics_torch(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    prot_attr: torch.Tensor | None = None,
    return_torch: bool = True,
) -> dict[str, torch.Tensor | float]:
    """Calculate a comprehensive set of metrics

    Args:
      y_true: torch.Tensor:
      y_pred: torch.Tensor:
      prot_attr: torch.Tensor | None:  (Default value = None)
      return_torch: bool:  (Default value = True)

    Returns:

    """

    y_true = y_true.int()
    y_pred = y_pred.int()

    # Compute confusion matrix
    n_classes = 2  # Assuming binary
    confusion_matrix = torch.zeros(n_classes, n_classes)
    for t, p in zip(y_true, y_pred):
        confusion_matrix[t, p] += 1

    # Compute metrics
    tp = confusion_matrix[1, 1].float()
    fn = confusion_matrix[1, 0].float()
    fp = confusion_matrix[0, 1].float()
    tn = confusion_matrix[0, 0].float()

    tpr = tp / stabilize_torch(tp + fn)
    fpr = fp / stabilize_torch(fp + tn)  # noqa
    all_pos = stabilize_torch(tp + fn)  # noqa
    all_neg = stabilize_torch(fp + tn)  # noqa

    # Performance
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / stabilize_torch(tp + fp)
    recall = tpr
    specificity = tn / stabilize_torch(tn + fp)
    f1 = 2 * (precision * recall) / stabilize_torch(precision + recall)
    geometric_mean = (recall * specificity) ** 0.5
    balanced_accuracy = (recall + specificity) / 2

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1": f1,
        "GMean": geometric_mean,
        "Balanced_accuracy": balanced_accuracy,
    }

    # Fairness GAP metrics
    if prot_attr is not None:
        prot_attr = prot_attr.to(dtype=torch.bool)

        # Group-wise
        tp_0 = ((y_pred[prot_attr] == 1) & (y_true[prot_attr] == 1)).sum().float()
        fp_0 = ((y_pred[prot_attr] == 1) & (y_true[prot_attr] == 0)).sum().float()
        tn_0 = ((y_pred[prot_attr] == 0) & (y_true[prot_attr] == 0)).sum().float()
        fn_0 = ((y_pred[prot_attr] == 0) & (y_true[prot_attr] == 1)).sum().float()

        tp_1 = ((y_pred[~prot_attr] == 1) & (y_true[~prot_attr] == 1)).sum().float()
        fp_1 = ((y_pred[~prot_attr] == 1) & (y_true[~prot_attr] == 0)).sum().float()
        tn_1 = ((y_pred[~prot_attr] == 0) & (y_true[~prot_attr] == 0)).sum().float()
        fn_1 = ((y_pred[~prot_attr] == 0) & (y_true[~prot_attr] == 1)).sum().float()

        tpr_0 = tp_0 / stabilize_torch(tp_0 + fn_0)
        tnr_0 = tn_0 / stabilize_torch(fp_0 + tn_0)  # noqa
        fpr_0 = fp_0 / stabilize_torch(fp_0 + tn_0)
        fnr_0 = fn_0 / stabilize_torch(tp_0 + fn_0)  # noqa

        tpr_1 = tp_1 / stabilize_torch(tp_1 + fn_1)
        tnr_1 = tn_1 / stabilize_torch(fp_1 + tn_1)  # noqa
        fpr_1 = fp_1 / stabilize_torch(fp_1 + tn_1)
        fnr_1 = fn_1 / stabilize_torch(tp_1 + fn_1)  # noqa

        ppr_0 = (tp_0 + fp_0) / stabilize_torch(tp_0 + fp_0 + tn_0 + fn_0)
        ppr_1 = (tp_1 + fp_1) / stabilize_torch(tp_1 + fp_1 + tn_1 + fn_1)

        accuracy_0 = (tp_0 + tn_0) / stabilize_torch(tp_0 + tn_0 + fp_0 + fn_0)
        accuracy_1 = (tp_1 + tn_1) / stabilize_torch(tp_1 + tn_1 + fp_1 + fn_1)

        # Fairness
        equal_opportunity = torch.abs(tpr_0 - tpr_1)
        equalized_odds = torch.max(torch.abs(tpr_0 - tpr_1), torch.abs(fpr_0 - fpr_1))
        demographic_parity = torch.abs(ppr_0 - ppr_1)
        accuracy_gap = torch.abs(accuracy_0 - accuracy_1)

        metrics["Equal_opportunity"] = equal_opportunity
        metrics["Equalized_odds"] = equalized_odds
        metrics["Demographic_parity"] = demographic_parity
        metrics["Accuracy_parity"] = accuracy_gap

    if not return_torch:
        metrics = {k: v.cpu().detach().item() for k, v in metrics.items()}

    return metrics
