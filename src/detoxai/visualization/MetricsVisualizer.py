from abc import ABC

import pandas as pd

from .Visualizer import Visualizer

CANNONICAL_COLUMNS = [
    "metric_name",
    "metric_class",
    "is_per_class",
    "class_name",
    "is_per_group",
    "group_name",
    "metrics_type",
    "reducer",
    "value",
]


class MetricsVisualizer(Visualizer, ABC):
    """ """

    metrics_config: dict | None = None

    @classmethod
    def canonize_results(cls, results: dict, metrics_config: dict) -> pd.DataFrame:
        """

        Args:
          results: dict:
          metrics_config: dict:

        Returns:

        """
        joint_df = pd.DataFrame()

        for method_name, res in results.items():
            # Check if already cannonical

            if isinstance(res, pd.DataFrame) and all(
                [i in res.columns for i in CANNONICAL_COLUMNS]
            ):
                new_df = res.copy()
            # Else convert to canonical
            else:
                new_df = results_to_tidy_df(res, metrics_config)

            new_df["method_name"] = method_name

            joint_df = pd.concat([joint_df, new_df], ignore_index=True)

        joint_df["value"] = joint_df["value"].astype(float)

        return joint_df.reset_index(drop=True)


def results_to_tidy_df(results, metrics_config):
    """

    Args:
      results:
      metrics_config:

    Returns:

    """
    tidy_results = {}
    for key, value in results.items():
        splitted_key = key.split("_")

        metrics_type = splitted_key[0]

        raw_metric_name = key

        metric_name = splitted_key[1]

        metric_class = (
            "performance"
            if metric_name in metrics_config["performance"]["metrics"]
            else "fairness"
        )

        metric_config = metrics_config[metric_class]["metrics"].get(metric_name)

        if metric_config is None:
            raise ValueError(f"Metric config for {metric_name} not found")

        value = format(value * 100, ".2f")

        reducer_name = splitted_key[-1]
        is_per_class = splitted_key[-1] == "class"
        is_per_group = splitted_key[-1] == "group"
        merged_name = "_".join(splitted_key[2:-1])
        class_name = merged_name if is_per_class else None
        group_name = merged_name if is_per_group else None

        tidy_results[raw_metric_name] = [
            metric_name,
            metric_class,
            is_per_class,
            class_name,
            is_per_group,
            group_name,
            metrics_type,
            reducer_name,
            value,
        ]

    df = pd.DataFrame(tidy_results).T
    df.columns = [
        "metric_name",
        "metric_class",
        "is_per_class",
        "class_name",
        "is_per_group",
        "group_name",
        "metrics_type",
        "reducer",
        "value",
    ]
    return df
