import pandas as pd

from ..utils.decorators import ensure_metrics_config_not_empty
from .MetricsVisualizer import MetricsVisualizer


class TableVisualizer(MetricsVisualizer):
    """ """

    def __init__(self, results: dict, metrics_config: dict) -> None:
        self.raw_results: pd.DataFrame = self.canonize_results(results, metrics_config)
        self.metrics_config = metrics_config

        self.results: pd.DataFrame | None = None

        self.create_table()

    @ensure_metrics_config_not_empty
    def create_table(self) -> None:
        """ """
        self.results = self.__organize_metrics(self.raw_results)

    def get_table(self) -> pd.DataFrame:
        """Get formated table aggregated for all methods"""
        return self.results

    def get_latex_table(self) -> str:
        """Get formatted table in LaTeX format with enhanced styling"""
        latex = self.results.to_latex(
            index=True,
            multirow=True,
            multicolumn=True,
            bold_rows=True,
            column_format="|l" + "c" * (len(self.results.columns)) + "|",
            float_format="%.2f",
            caption="Aggregated Metrics Results Table",
            label="tab:metrics_results",
        )

        # Replace all "_" with "\_"
        latex = latex.replace("_", "\\_")

        return latex

    def get_raw_results(self) -> pd.DataFrame:
        """Get raw results, i.e., the results as a melted DataFrame"""
        return self.raw_results

    def save_to_csv(self, path: str) -> None:
        """Save the table to a CSV file

        Args:
          path: str:

        Returns:

        """
        self.results.to_csv(path, index=True)

    @classmethod
    def __organize_metrics(cls, df: pd.DataFrame) -> pd.DataFrame:
        # Pivot performance metrics based on method, class_name, and reducer

        performance_df = df[df["metric_class"] == "performance"]
        performance_df.loc[performance_df["class_name"].isna(), "class_name"] = "all"
        performance_df.loc[performance_df["group_name"].isna(), "group_name"] = "-"

        performance_pivot = performance_df.pivot_table(
            index="method_name",
            columns=[
                "metric_name",
                "metric_class",
                "reducer",
                "is_per_class",
                "group_name",
                "class_name",
            ],
            values="value",
            # dropna=False,
        )

        fairness_df = df[df["metric_class"] == "fairness"]
        fairness_df.loc[fairness_df["class_name"].isna(), "class_name"] = "-"
        fairness_df.loc[fairness_df["group_name"].isna(), "group_name"] = "all"

        fairness_pivot = fairness_df.pivot_table(
            index="method_name",
            columns=[
                "metric_name",
                "metric_class",
                "reducer",
                "is_per_class",
                "group_name",
                "class_name",
            ],
            values="value",
            # dropna=False,
        )

        # Combine performance and fairness metrics back into a single DataFrame
        combined_pivot = pd.concat([performance_pivot, fairness_pivot], axis=1)

        return combined_pivot
