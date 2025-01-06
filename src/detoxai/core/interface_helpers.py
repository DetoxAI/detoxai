import os
import yaml


def load_supported_tags() -> dict:
    """
    From ./datasets/catalog/<dataset_name>/labels_mapping.yaml, load the dicts

    ***
    STRUCTURE:
    {
        "datasets": [a, b, c],
        "attributes": [x, y, z],
        "mapping": {
            "dataset_name": labels_mapping...
            }
    }
    """
    mapping = {}
    datasets = []
    attributes = set()
    for dataset in os.listdir("./datasets/catalog"):
        labels_mapping_path = f"./datasets/catalog/{dataset}/labels_mapping.yaml"
        if os.path.exists(labels_mapping_path):
            with open(labels_mapping_path, "r") as f:
                labels_mapping = yaml.safe_load(f)

                # It's a dict of dicts, we need to reverse the inner dict
                labels_mapping = {}
                for attribute in labels_mapping:
                    for idx, value in labels_mapping[attribute].items():
                        labels_mapping[value] = idx

                mapping[dataset] = labels_mapping
                datasets.append(dataset)
                attributes.update(list(map(lambda s: s.lower(), labels_mapping.keys())))

    d = {
        "datasets": datasets,
        "attributes": list(attributes),
        "mapping": mapping,
    }

    return d


def construct_metrics_config(
    metrics: list[str] | str = "all", types: str = "GAP"
) -> dict:
    """
    Construct the metrics configuration for the fairness and performance metrics

    Args:
        metrics: List of metrics to include in the configuration
        types: Type of metric to use. Options are "GAP" or "RATIO"
    """
    if types == "GAP":
        f_reduce = "difference"
    elif types == "RATIO":
        f_reduce = "ratio"
    else:
        raise ValueError(f"Invalid type {types}")

    dl_metrics_config = {
        "performance": {"metrics": {}},
        "fairness": {"metrics": {}},
    }

    # Default to all metrics
    if metrics == "all":
        fair_metrics = ["ACC", "EO", "DP", "EOO"]
        perf_metrics = ["GMean", "F1Score", "Accuracy", "Precision", "Recall"]
    else:
        raise NotImplementedError("Custom metrics not yet supported")

    for metric in fair_metrics:
        dl_metrics_config["fairness"]["metrics"][metric] = {"reduce": [f"{f_reduce}"]}

    for metric in perf_metrics:
        a = {"reduce": ["macro", "per_class"]}
        dl_metrics_config["performance"]["metrics"][metric] = a

    return dl_metrics_config
