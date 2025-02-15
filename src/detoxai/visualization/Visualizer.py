from abc import ABC

import matplotlib.pyplot as plt
import numpy as np

from ..utils.experiment_logger import ExperimentLogger


class Visualizer(ABC):
    # logger: ExperimentLogger | None = None
    # plt_save_kwargs: dict = {}
    # plot_style: str = "ggplot"

    # figure: plt.Figure | None = None
    # ax: plt.Axes | None = None

    def set_up_plots_configuration(self, config: dict) -> None:
        self.plt_save_kwargs = config.get("plt_save_kwargs", {})
        self.plot_style = config.get("plot_style", "ggplot")
        self.plot_shape_multiplier = config.get("shape_multiplier", 1)
        self.fontsize = config.get("fontsize", 12)
        plt.style.use(self.plot_style)

    def attach_logger(self, logger: ExperimentLogger) -> None:
        """
        Attach a logger to the visualizer, enabling logging of plots
        """
        self.logger = logger

    def show_plot(self) -> None:
        self.figure.show()

    def show(self) -> None:
        plt.show()

    def close_plot(self) -> None:
        plt.close(self.figure)

    def get_canvas(
        self,
        rows: int = 1,
        cols: int = 1,
        shape: tuple[int, int] = (10, 10),
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Get an empty canvas for plotting
        """
        self.figure, self.ax = plt.subplots(rows, cols, figsize=shape)

        if isinstance(self.ax, (list, np.ndarray)):
            for a in self.ax:
                if isinstance(a, (list, np.ndarray)):
                    for a_ in a:
                        a_.axis("off")
                else:
                    a.axis("off")
        else:
            self.ax.axis("off")

        self.figure.tight_layout()

        return self.figure, self.ax

    def local_save(self, path: str):
        """
        Save the figure to a local path
        """
        self.figure.savefig(path, **self.plt_save_kwargs)

    def log(self, name: str, step: int = None):
        assert self.logger is not None, "A logger must be provided in self.logger"
        self.logger.log_image(self.figure, name, step)

    def log_image(self, name: str, step: int = None):
        assert self.logger is not None, "A logger must be provided in self.logger"
        self.logger.log_image(self.figure, name, step)

    def log_table(self, table: dict, name: str, step: int = None):
        assert self.logger is not None, "A logger must be provided in self.logger"
        self.logger.log_table(table, name, step)
