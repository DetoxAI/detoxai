from torch.utils.data import DataLoader
import lightning as L
import logging


# Project imports
from .model_wrappers import FairnessLightningWrapper

logger = logging.getLogger(__name__)


def evaluate_model(
    model: FairnessLightningWrapper,
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
    if device is not None and "cuda" in device and ":" in device:
        devices_id = [int(device.split(":")[1])]
    else:
        devices_id = "auto"

    logger.debug(f"Evaluating model on device: {device}")

    trainer = L.Trainer(
        logger=False,
        enable_model_summary=verbose,
        enable_progress_bar=verbose,
        devices=devices_id,
    )
    model.eval()
    raw_results = trainer.test(model, dataloader, verbose=verbose)[0]

    logger.debug(f"Raw results: {raw_results}")

    metrics = {"pareto": {}, "all": {}}

    if pareto_metrics:
        # Options
        # test_{metric}_macro for performance metrics
        # test_{metric}_difference for fairness metrics
        # test_{metric}_ratio for fairness metrics
        for metric in raw_results.keys():
            accept = (
                metric.endswith("macro")
                or metric.endswith("difference")
                or metric.endswith("ratio")
            )
            # We only collect the metrics we care about
            if accept:
                cleaned_metric = metric.split("_")[1]

                for pareto_metric in pareto_metrics:
                    if pareto_metric in metric:
                        metrics["pareto"][cleaned_metric] = raw_results[metric]

                # Collect all metrics
                metrics["all"][cleaned_metric] = raw_results[metric]

    return metrics
