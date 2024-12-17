from ..src.models.LightningWrapper import BaseLightningWrapper


def evaluate_model(model: BaseLightningWrapper) -> dict:
    """
    Evaluate the model on various metrics


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
    raise NotImplementedError
