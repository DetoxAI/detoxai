from typing import Dict, List, Optional, Union

import pandas as pd
import torch
from torchmetrics import ClasswiseWrapper, MetricCollection
from torchmetrics.classification import (
    BinaryGroupStatRates,
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

# Assumption class 1 is the positive decision that we are interested in
# Assumption group_0 is the privileged group, group_1 is the unprivileged group

# LB - Lower Better
# HB - Higher Better
# NOV - No Optimal Value

# Fairness Glossary
# Binary Group Stat Rates returns Confusion Matrix for each group
# TP: True Positive = Hits | HB
# FP: False Positive = False Alarm = Overestimation | LB
# TN: True Negative = Correct Rejection | HB
# FN: False Negative = Misses = Underestimation | LB
# You can mannually calculate
# PP: Predicted Positive = TP + FP | NOV
# PN: Predicted Negative = TN + FN | NOV
# AP: Actual Positive = TP + FN | NOV
# AN: Actual Negative = TN + FP | NOV

# Derived Metrics
# TPR: True Positive Rate = TP / (TP + FN) = Recall = Sensitivity = Hit Rate | HB
# FPR: False Positive Rate = FP / (FP + TN) = Fall-Out = False Alarm Rate = Type I Error Rate | LB
# FNR: False Negative Rate = FN / (TP + FN) = Miss Rate = Type II Error Rate | LB
# TNR: True Negative Rate = TN / (FP + TN) = Specificity = Selectivity = Correct Rejection Rate | HB
# ER: Error Rate = (FP + FN) / (TP + FP + TN + FN) | LB
# ACC: Accuracy = (TP + TN) / (TP + FP + TN + FN) | HB
# FOR: False Omission Rate = FN / (FN + TN) | LB
# PPV: Positive Predictive Value = TP / (TP + FP) = Precision | HB
# NPV: Negative Predictive Value = TN / (TN + FN) | HB
# FDR: False Discovery Rate = FP / (TP + FP) | LB
# PPR: Predicted Positive Rate = PP / (PP + PN) | NOV

# Advanced Derived Metrics
# F1: F1 Score = 2 * (PPV * TPR) / (PPV + TPR) = 2TP / (2TP + FP + FN) | HB
# MCC: Matthews Correlation Coefficient = sqrt(PPV * TPR * TNR * NPV) - sqrt(FPR * FNR * FNR * FNR) | ?
# GMean: Geometric Mean = root of the product of class-wise sensitivity = sqrt(TPR * TNR) | HB

# Advanced Fairness Metrics
# EO = return the worst score of TPR difference/ratio and FPR difference/ratio between groups | Difference - LB (0 is best) | Ratio - HB (1 is best)
# DP = requires equal proportion of positive predictions in each group (PPR difference/ratio) | Difference - LB (0 is best) | Ratio - HB (1 is best)
# EOO = requires equal TPR in each group | Difference - LB (0 is best) | Ratio - HB (1 is best)
# TreatmentEquality = requires FN/FP ratio between groups to be equal  | Difference - LB (0 is best, CAN BE HIGHER THAN 1) | Ratio - 1 is best

# You can also use reducers to calculate the ratio or difference of metrics between groups
# Ratio = min(group_0_metric, group_1_metric) / max(group_0_metric, group_1_metric)
# Difference = max(group_0_metric, group_1_metric) - min(group_0_metric, group_1_metric)


#  tp, fp, tn, fn = stats[group]


DEFAULT_METRICS_CONFIG = {
    "performance": {
        # Will be calculated per class
        # reduce: Literal["micro", "macro", "none"]
        "metrics": {
            "Accuracy": {"reduce": ["macro", "per_class"]},
            "F1Score": {"reduce": ["macro", "per_class"]},
        }
    },
    "fairness": {
        # Will be calculated per group
        # reduce: Literal["difference", "ratio", "none"]
        "metrics": {
            "EO": {"reduce": ["difference"]},
            "DP": {"reduce": ["difference"]},
            "EOO": {"reduce": ["difference"]},
        }
    },
}


def stabilize(x, eps=1e-8):
    """

    Args:
      x:
      eps:  (Default value = 1e-8)

    Returns:

    """
    return x + eps


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


class PerformanceMetrics:
    """ """

    @classmethod
    def infer_metrics(
        self, metrics_config: Dict[str, Dict[str, List[str]]], class_labels: List[str]
    ):
        """

        Args:
          metrics_config: Dict[str:
          Dict[str:
          List[str]]]:
          class_labels: List[str]:

        Returns:

        """
        num_classes = len(class_labels)
        metrics = {}
        for metric, config in metrics_config.items():
            for reduce in config["reduce"]:
                if metric == "Accuracy":
                    if reduce == "per_class":
                        metrics[metric] = ClasswiseWrapper(
                            MulticlassAccuracy(num_classes, average=None),
                            labels=class_labels,
                            postfix="_class",
                        )
                    else:
                        metrics[metric + "_" + reduce] = MulticlassAccuracy(
                            num_classes, average=reduce
                        )
                elif metric == "Precision":
                    if reduce == "per_class":
                        metrics[metric] = ClasswiseWrapper(
                            MulticlassPrecision(num_classes, average=None),
                            labels=class_labels,
                            postfix="_class",
                        )
                    else:
                        metrics[metric + "_" + reduce] = MulticlassPrecision(
                            num_classes, average=reduce
                        )
                elif metric == "Recall":
                    if reduce == "per_class":
                        metrics[metric] = ClasswiseWrapper(
                            MulticlassRecall(num_classes, average=None),
                            labels=class_labels,
                            postfix="_class",
                        )
                    else:
                        metrics[metric + "_" + reduce] = MulticlassRecall(
                            num_classes, average=reduce
                        )
                elif metric == "F1Score":
                    if reduce == "per_class":
                        metrics[metric] = ClasswiseWrapper(
                            MulticlassF1Score(num_classes, average=None),
                            labels=class_labels,
                            postfix="_class",
                        )
                    else:
                        metrics[metric + "_" + reduce] = MulticlassF1Score(
                            num_classes, average=reduce
                        )
        return MetricCollection(metrics)


class AllMetrics:
    """ """

    def __init__(
        self,
        metrics_config: Dict[str, Dict[str, Dict[str, List[str]]]],
        class_labels: List[str],
        num_groups: Optional[int] = None,
    ):
        self.performance_metrics = PerformanceMetrics.infer_metrics(
            metrics_config["performance"]["metrics"], class_labels
        )
        self.fairness_metrics = (
            MetricCollection(
                {
                    "custom_fairness": FairnessMetrics(
                        num_groups=num_groups,
                        metrics_spec=metrics_config["fairness"]["metrics"],
                    )
                }
            )
            if num_groups is not None
            else None
        )

    def __repr__(self):
        return f"AllMetrics(performance_metrics={self.performance_metrics}, fairness_metrics={self.fairness_metrics})"

    def get_performance_metrics(self) -> PerformanceMetrics:
        """ """
        return self.performance_metrics

    def get_fairness_metrics(self) -> MetricCollection:
        """ """
        return self.fairness_metrics


class BinaryGroupStatRatesUnwrapped(BinaryGroupStatRates):
    """ """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        sensitive_features: torch.Tensor,
    ):
        """

        Args:
          preds: torch.Tensor:
          target: torch.Tensor:
          sensitive_features: torch.Tensor:

        Returns:

        """
        super().update(preds, target, sensitive_features)

    def compute(self):
        """ """
        stats = super().compute()
        unwrapped_stats = {}
        for group in stats.keys():
            group_name = group.split("_")[0]
            group_number = group.split("_")[1]
            unwrapped_stats["TP_" + group_number + "_" + group_name] = stats[group][0]
            unwrapped_stats["FP_" + group_number + "_" + group_name] = stats[group][1]
            unwrapped_stats["TN_" + group_number + "_" + group_name] = stats[group][2]
            unwrapped_stats["FN_" + group_number + "_" + group_name] = stats[group][3]
        return unwrapped_stats


metrics_spec_dict_type = Dict[str, Dict[str, List[Union[str, None]]]]


class FairnessMetrics(BinaryGroupStatRatesUnwrapped):
    """Computes derived metrics (like TPR, FPR, etc.) for each group and allows for ratio and difference
    comparisons between groups to assess fairness. Supports various fairness metrics such as Equalized Odds.

    Args:
      metrics_spec(metrics_spec_dict_type): Dictionary specifying metrics to calculate.
      args, **kwargs: Additional arguments for the base class initialization.
    Supported metrics:
    - True Positive Rate (TPR), False Positive Rate (FPR), Error Rate (ER), etc.
    - Equalized Odds can be calculated with either a 'ratio' or 'difference' reduction.
    Example usage:

    Returns:

    >>> preds = torch.tensor([1, 1, 1, 0, 1, 0, 1, 0])
        >>> target = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])
        >>> sensitive_features = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])

        >>> metrics_spec = {
        ...     "TPR": {"reduce": ["ratio", "difference", "per_group"]},
        ...     "FPR": {"reduce": ["ratio", "difference"]},
        ...     "EO": {"reduce": ["difference"]},
        ...     "ACC": {"reduce":  ["per_group"]},
        ... }
        >>> derived_metrics = DerivedMetrics(metrics_spec, num_groups=2)
        >>> results = derived_metrics(preds, target, sensitive_features) #forward calls update and compute internally
        >>> results
        {
            'TPR_0_group': tensor(1.),
            'TPR_1_group': tensor(0.6667),
            'TPR_ratio': tensor(nan),
            'TPR_difference': tensor(nan),
            'FPR_ratio': tensor(1.),
            'FPR_difference': tensor(0.),
            'EqualizedOdds_difference': tensor(nan),
            'ACC_0_group': tensor(0.5000),
            'ACC_1_group': tensor(0.5000),
        }
    """

    _metrics_options = ["reduce"]
    _reduce_options = ["ratio", "difference", "per_group"]
    _selected_metrics_options_dict = {
        "TPR": "True_Positive_Rate",
        "FPR": "False_Positive_Rate",
        "FNR": "False_Negative_Rate",
        "TNR": "True_Negative_Rate",
        "ER": "Error_Rate",
        "ACC": "Accuracy",
        "FOR": "False_Omission_Rate",
        "PPV": "Positive_Predictive_Value",
        "NPV": "Negative_Predictive_Value",
        "FDR": "False_Discovery_Rate",
        "F1": "F1_Score",
        "GMean": "Geometric_Mean",
        "PPR": "Predicted_Positive_Rate",
        "EO": "Equalize_Odds",
        "DP": "Demographic_Parity",
        "EOO": "Equality_of_Opportunity",
        "TreatmentEquality": "Treatment_Equality",
    }

    def __init__(
        self,
        metrics_spec: metrics_spec_dict_type,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert set(metrics_spec.keys()).issubset(
            set(self._selected_metrics_options_dict.keys())
        ), "Invalid metric specified"
        for metric in metrics_spec.keys():
            assert set(metrics_spec[metric].keys()).issubset(
                set(self._metrics_options)
            ), "Invalid reduce option specified"
            assert set(metrics_spec[metric]["reduce"]).issubset(
                set(self._reduce_options)
            ), "Invalid reduce option specified"
        # Assert that if there is EO then it must have reduce option specified
        assert (
            "EO" not in metrics_spec.keys()
            or not metrics_spec["EO"]["reduce"] == "per_group"
        ), "EO must have a reduce option 'ratio' or 'difference' specified"
        assert (
            "DP" not in metrics_spec.keys()
            or not metrics_spec["DP"]["reduce"] == "per_group"
        ), "DP must have a reduce option 'ratio' or 'difference' specified"
        assert (
            "EOO" not in metrics_spec.keys()
            or not metrics_spec["EOO"]["reduce"] == "per_group"
        ), "EOO must have a reduce option 'ratio' or 'difference' specified"
        assert (
            "TreatmentEquality" not in metrics_spec.keys()
            or not metrics_spec["TreatmentEquality"]["reduce"] == "per_group"
        ), (
            "TreatmentEquality must have a reduce option 'ratio' or 'difference' specified"
        )
        self.metrics_spec = metrics_spec

    def compute(self):
        """ """
        stats = super().compute()

        derived_metrics = self._calculate_derived_metrics(stats)

        out_metrics = {}
        for metric in self.metrics_spec.keys():
            reduce_options = self.metrics_spec[metric]["reduce"]
            if "per_group" in reduce_options:
                for group in range(self.num_groups):
                    group_name = str(group) + "_group"
                    out_metrics.update(
                        {
                            metric + "_" + group_name: derived_metrics[
                                metric + "_" + group_name
                            ]
                        }
                    )
            if "ratio" or "difference" in reduce_options:
                if metric == "EO":
                    min_group_metric, max_group_metric = self.calculate_equalized_odds(
                        derived_metrics
                    )
                elif metric == "DP":
                    min_group_metric, max_group_metric = (
                        self.calculate_demographic_parity(derived_metrics)
                    )
                elif metric == "EOO":
                    min_group_metric, max_group_metric = (
                        self.calculate_equality_of_opportunity(derived_metrics)
                    )
                elif metric == "TreatmentEquality":
                    min_group_metric, max_group_metric = (
                        self.calculate_treatment_equality(stats)
                    )
                else:
                    min_group_metric = min(
                        [
                            derived_metrics[metric + "_" + str(group) + "_group"]
                            for group in range(self.num_groups)
                        ]
                    )
                    max_group_metric = max(
                        [
                            derived_metrics[metric + "_" + str(group) + "_group"]
                            for group in range(self.num_groups)
                        ]
                    )
                if "ratio" in reduce_options:
                    out_metrics[metric + "_ratio"] = min_group_metric / max_group_metric
                if "difference" in reduce_options:
                    out_metrics[metric + "_difference"] = (
                        max_group_metric - min_group_metric
                    )
        return out_metrics

    def _calculate_derived_metrics(self, stats):
        """

        Args:
          stats:

        Returns:

        """
        derived_metrics = {}
        for group in range(self.num_groups):
            group_name = str(group) + "_group"
            tp = stats["TP_" + group_name]
            fp = stats["FP_" + group_name]
            tn = stats["TN_" + group_name]
            fn = stats["FN_" + group_name]
            pp = tp + fp
            pn = tn + fn
            ap = tp + fn
            an = tn + fp
            derived_metrics["TPR_" + group_name] = tp / stabilize(ap)
            derived_metrics["FPR_" + group_name] = fp / stabilize(an)
            derived_metrics["FNR_" + group_name] = fn / stabilize(ap)
            derived_metrics["TNR_" + group_name] = tn / stabilize(an)
            derived_metrics["ER_" + group_name] = (fp + fn) / (tp + fp + tn + fn)
            derived_metrics["ACC_" + group_name] = (tp + tn) / (tp + fp + tn + fn)
            derived_metrics["FOR_" + group_name] = fn / (fn + tn)
            derived_metrics["PPV_" + group_name] = tp / stabilize(pp)
            derived_metrics["NPV_" + group_name] = tn / stabilize(pn)
            derived_metrics["FDR_" + group_name] = fp / (tp + fp)
            derived_metrics["F1_" + group_name] = (
                2
                * (
                    derived_metrics["PPV_" + group_name]
                    * derived_metrics["TPR_" + group_name]
                )
                / (
                    derived_metrics["PPV_" + group_name]
                    + derived_metrics["TPR_" + group_name]
                )
            )
            derived_metrics["GMean_" + group_name] = (
                derived_metrics["TPR_" + group_name]
                * derived_metrics["TNR_" + group_name]
            ) ** 0.5
            derived_metrics["PPR_" + group_name] = pp / (pp + pn)
        return derived_metrics

    def calculate_equalized_odds(self, derived_metrics):
        """

        Args:
          derived_metrics:

        Returns:

        """
        min_group_tpr = min(
            [
                derived_metrics["TPR" + "_" + str(group) + "_group"]
                for group in range(self.num_groups)
            ]
        )
        max_group_tpr = max(
            [
                derived_metrics["TPR" + "_" + str(group) + "_group"]
                for group in range(self.num_groups)
            ]
        )
        min_group_fpr = min(
            [
                derived_metrics["FPR" + "_" + str(group) + "_group"]
                for group in range(self.num_groups)
            ]
        )
        max_group_fpr = max(
            [
                derived_metrics["FPR" + "_" + str(group) + "_group"]
                for group in range(self.num_groups)
            ]
        )
        # return fpr if fpr is greater than tpr
        if max_group_fpr - min_group_fpr > max_group_tpr - min_group_tpr:
            return min_group_fpr, max_group_fpr
        else:
            return min_group_tpr, max_group_tpr

    def calculate_demographic_parity(self, derived_metrics):
        """

        Args:
          derived_metrics:

        Returns:

        """
        min_group_ppr = min(
            [
                derived_metrics["PPR" + "_" + str(group) + "_group"]
                for group in range(self.num_groups)
            ]
        )
        max_group_ppr = max(
            [
                derived_metrics["PPR" + "_" + str(group) + "_group"]
                for group in range(self.num_groups)
            ]
        )
        return min_group_ppr, max_group_ppr

    def calculate_equality_of_opportunity(self, derived_metrics):
        """

        Args:
          derived_metrics:

        Returns:

        """
        min_group_tpr = min(
            [
                derived_metrics["TPR" + "_" + str(group) + "_group"]
                for group in range(self.num_groups)
            ]
        )
        max_group_tpr = max(
            [
                derived_metrics["TPR" + "_" + str(group) + "_group"]
                for group in range(self.num_groups)
            ]
        )
        return min_group_tpr, max_group_tpr

    def calculate_treatment_equality(self, stats):
        """

        Args:
          stats:

        Returns:

        """
        min_group_fn_fp_ratio = min(
            [
                stats["FN" + "_" + str(group) + "_group"]
                / stats["FP" + "_" + str(group) + "_group"]
                for group in range(self.num_groups)
            ]
        )
        max_group_fn_fp_ratio = max(
            [
                stats["FN" + "_" + str(group) + "_group"]
                / stats["FP" + "_" + str(group) + "_group"]
                for group in range(self.num_groups)
            ]
        )
        return min_group_fn_fp_ratio, max_group_fn_fp_ratio
