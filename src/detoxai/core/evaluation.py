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
    device: str | None = None,
) -> dict:
    """
    Evaluate the model on various metrics

    Args:
        - model: Model to evaluate
        - dataloader: DataLoader for the dataset
        - pareto_metrics: List of metrics to include in the pareto front
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

    if device is not None:
        model_device = device
        model.to(device)
    else:
        model_device = next(model.parameters()).device

    logger.debug(f"Evaluating model on device: {model_device}")

    model.eval()
    preds = []
    targets = []
    protected_attributes = []
    for batch in dataloader:
        x, y, prot_attr = batch
        x = x.to(model_device)
        y = y.to(model_device)
        prot_attr = prot_attr.to(model_device)
        with torch.no_grad():
            pred = model(x).argmax(dim=1)
        preds.append(pred)
        targets.append(y)
        protected_attributes.append(prot_attr)

    preds = torch.cat(preds).to(model_device)
    targets = torch.cat(targets).to(model_device)
    protected_attributes = torch.cat(protected_attributes).to(model_device)

    raw_results = comprehensive_metrics_torch(targets, preds, protected_attributes)

    logger.debug(f"Raw results: {raw_results}")

    metrics = {"pareto": {}, "all": {}}

    for metric in raw_results:
        if pareto_metrics and metric in pareto_metrics:
            metrics["pareto"][metric] = raw_results[metric].cpu().detach().item()

        # Collect all metrics
        metrics["all"][metric] = raw_results[metric].cpu().detach().item()

    return metrics
