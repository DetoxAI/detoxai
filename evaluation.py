from torch.utils.data import DataLoader

from .model_wrappers import FairnessLightningWrapper
from .metrics.fairness_metrics import AllMetrics


def evaluate_model(
    model: FairnessLightningWrapper,
    dataloader: DataLoader,
    metrics_calculator: AllMetrics,
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
