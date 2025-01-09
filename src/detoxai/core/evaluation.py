import torch
import torch.nn as nn
import logging
from torch.utils.data import DataLoader

from ..metrics.metrics import comprehensive_metrics_torch


logger = logging.getLogger(__name__)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    pareto_metrics: list[str] | None = None,
    verbose: bool = False,
    device: str = None,
) -> dict:
    """
    Evaluate the model on various metrics

    Args:
        - model: Model to evaluate
        - dataloader: DataLoader for the dataset
        - class_labels: List of class labels (usually taken from your collator to ensure consistency)
        - prot_attr_arity: Arity of the protected attribute (e.g. 2 for binary)

    ***
    `TEMPLATE FOR METRICS DICT`
    ***

    metrics_dict_template = {
        "pareto": {
            "balanced_accuracy": 0.0,
            "equal_opportunity": 0.0,
        },
        "all": {
            "balanced_accuracy": 0.0,
            "equal_opportunity": 0.0,
            "equalized_odds": 0.0,
            "demographic_parity": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        },
    }

    Args:
        model: Model to evaluate
    """

    device = str(device)

    logger.debug(f"Evaluating model on device: {device}")

    model.eval()
    preds = []
    targets = []
    protected_attributes = []
    for batch in dataloader:
        x, y, prot_attr = batch
        x = x.to(device)
        y = y.to(device)
        prot_attr = prot_attr.to(device)
        with torch.no_grad():
            pred = model(x).argmax(dim=1)
        preds.append(pred)
        targets.append(y)
        protected_attributes.append(prot_attr)

    preds = torch.cat(preds).to(device)
    targets = torch.cat(targets).to(device)
    protected_attributes = torch.cat(protected_attributes).to(device)

    raw_results = comprehensive_metrics_torch(targets, preds, protected_attributes)

    logger.debug(f"Raw results: {raw_results}")

    metrics = {"pareto": {}, "all": {}}

    for metric in raw_results:
        if pareto_metrics and metric in pareto_metrics:
            metrics["pareto"][metric] = raw_results[metric].cpu().detach().numpy()

        # Collect all metrics
        metrics["all"][metric] = raw_results[metric].cpu().detach().numpy()

    return metrics
