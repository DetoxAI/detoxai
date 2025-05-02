import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .Visualizer import Visualizer
from ..core.results_class import CorrectionResult


class ScatterVisualizer(Visualizer):
    """ """

    def __init__(self, plots_config: dict = {}) -> None:
        self.set_up_plots_configuration(plots_config)

    def create_plot(
        self,
        results: pd.DataFrame | dict[str, CorrectionResult],
        rows: list[str] = ["Accuracy", "GMean", "F1"],
        cols: list[str] = [
            "Equalized_odds",
            "Demographic_parity",
            "Equal_opportunity",
            "Accuracy_parity",
        ],
        ticks_x: int = 4,
        ticks_y: int = 4,
        round_to: int = 3,
    ):
        """

        Args:
          results:  pd.DataFrame | dict[str, CorrectionResult]
          rows: list[str]:  (Default value = ["Accuracy")
          "GMean":
          "F1"]:
          cols: list[str]:  (Default value = ["Equalized_odds")
          "Demographic_parity":
          "Equal_opportunity":
          "Accuracy_parity":
          ]:
          ticks_x: int:  (Default value = 4)
          ticks_y: int:  (Default value = 4)
          round_to: int:  (Default value = 3)

        Returns:

        """
        n_rows = len(rows)
        n_cols = len(cols)
        fig, axes = self.get_canvas(n_rows, n_cols, shape=(n_cols * 4, n_rows * 3))

        if isinstance(results, pd.DataFrame):
            metrics = results
        else:
            metrics = ScatterVisualizer._parse_results(results)

        colors = [
            "#a6cee3",
            "#1f78b4",
            "#b2df8a",
            "#33a02c",
            "#fb9a99",
            "#e31a1c",
            "#fdbf6f",
            "#ff7f00",
            "#cab2d6",
            "#6a3d9a",
            "#ffff99",
            "#b15928",
            "#000000",
            "#b7b7b7",
            "#e000ff",
        ]
        shapes = [
            "o",
            "s",
            "^",
            "x",
            "+",
            "D",
            "v",
            "p",
            "*",
            "D",
            "X",
            "H",
            "1",
            "<",
            ">",
            "d",
        ]

        # Adjust to add one legend below the plot
        plt.subplots_adjust(bottom=0.1)

        methods = metrics["method"].unique()

        for row_idx, row in enumerate(rows):
            for col_idx, col in enumerate(cols):
                for i, method in enumerate(methods):
                    data = metrics[metrics["method"] == method]
                    ax = axes[row_idx, col_idx]
                    ax.scatter(
                        data[col],
                        data[row],
                        c=colors[i],
                        marker=shapes[i],
                        label=method,
                    )

                    # If it's the first column, add the y label
                    if col_idx == 0:
                        ax.set_ylabel(row, fontsize=self.fontsize)

                    # If it's the last row, add the x label
                    if row_idx == n_rows - 1:
                        ax.set_xlabel(col, fontsize=self.fontsize)

        # Turn on axis
        for ax in axes.flat:
            ax.axis("on")

        # Find the max and min values for each metric and set the limits
        for row_idx, row in enumerate(rows):
            for col_idx, col in enumerate(cols):
                ax = axes[row_idx, col_idx]
                ticksx = np.linspace(metrics[col].min(), metrics[col].max(), ticks_x)
                ticksy = np.linspace(metrics[row].min(), metrics[row].max(), ticks_x)

                # ax.set_xlim(metrics[col].min() * 0.9, metrics[col].max() * 1.1)
                # ax.set_ylim(metrics[row].min() * 0.9, metrics[row].max() * 1.1)
                ax.set_xticks(ticksx.round(round_to))
                ax.set_yticks(ticksy.round(round_to))

        # create a legend
        handles, labels = ax.get_legend_handles_labels()
        handles, labels = handles[: len(methods)], labels[: len(methods)]
        fig.legend(
            handles,
            labels,
            ncol=len(methods),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
        )
        fig.tight_layout()

    @classmethod
    def _parse_results(cls, results: dict[str, CorrectionResult]) -> pd.DataFrame:
        metrics = []
        for result in results.values():
            _metrics = result.get_all_metrics()["all"]
            _method_name = result.get_method()
            _metrics["method"] = _method_name
            metrics.append(_metrics)
        df = pd.DataFrame(metrics)

        return df
