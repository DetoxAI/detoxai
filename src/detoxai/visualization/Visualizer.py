from abc import ABC

import matplotlib.pyplot as plt
import numpy as np

from ..utils.experiment_logger import ExperimentLogger


class Visualizer(ABC):
    """Base class for visualizers with plotting and logging capabilities.

    This class provides basic functionality for creating, displaying, saving, and logging
    plots using matplotlib. It can be used with an ExperimentLogger for experiment tracking.
    """

    def set_up_plots_configuration(self, config: dict) -> None:
        """Configure plotting parameters from a configuration dictionary.

        Args:
            config: Dictionary containing plot configuration parameters:
                - plt_save_kwargs: Dict of kwargs for plt.savefig (default: {})
                - plot_style: Matplotlib style to use (default: 'ggplot')
                - shape_multiplier: Multiplier for plot dimensions (default: 1)
                - fontsize: Base font size for plots (default: 12)
        """
        self.plt_save_kwargs = config.get("plt_save_kwargs", {})
        self.plot_style = config.get("plot_style", "ggplot")
        self.plot_shape_multiplier = config.get("shape_multiplier", 1)
        self.fontsize = config.get("fontsize", 12)
        plt.style.use(self.plot_style)

    def attach_logger(self, logger: ExperimentLogger) -> None:
        """Attach a logger to the visualizer, enabling logging of plots

        Args:
          logger: ExperimentLogger:
        """

        self.logger = logger

    def show_plot(self) -> None:
        """Display the current figure using matplotlib's show method.

        This method displays the figure stored in self.figure.
        """
        self.figure.show()

    def show(self) -> None:
        """Display all current matplotlib figures.

        This method shows all active matplotlib figures using plt.show().
        """
        plt.show()

    def close_plot(self) -> None:
        """Close the current figure.

        This method closes the figure stored in self.figure to free up memory.
        """
        plt.close(self.figure)

    def get_canvas(
        self,
        rows: int = 1,
        cols: int = 1,
        shape: tuple[int, int] = (10, 10),
    ) -> tuple[plt.Figure, plt.Axes]:
        """Create and return an empty matplotlib canvas for plotting.

        Args:
            rows: Number of subplot rows (default: 1)
            cols: Number of subplot columns (default: 1)
            shape: Tuple of (width, height) in inches for the figure size (default: (10, 10))

        Returns:
            tuple: A tuple containing:
                - matplotlib.figure.Figure: The created figure
                - matplotlib.axes.Axes or array of Axes: The created subplot(s)
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

        self.figure.tight_layout(h_pad=-0.1, w_pad=-0.5)

        return self.figure, self.ax

    def local_save(self, path: str):
        """Save the figure to a local path

        Args:
          path: str:
        """
        self.figure.savefig(path, **self.plt_save_kwargs)

    def log(self, name: str, step: int = None):
        """Log the current figure using the attached logger.

        Args:
            name: Name of the logged figure
            step: Optional step or iteration number for the logged figure

        Raises:
            AssertionError: If no logger is attached to the visualizer
        """
        assert self.logger is not None, "A logger must be provided in self.logger"
        self.logger.log_image(self.figure, name, step)

    def log_image(self, name: str, step: int = None):
        """Log the current figure as an image using the attached logger.

        This method is an alias for the log() method.

        Args:
            name: Name of the logged image
            step: Optional step or iteration number for the logged image

        Raises:
            AssertionError: If no logger is attached to the visualizer
        """
        assert self.logger is not None, "A logger must be provided in self.logger"
        self.logger.log_image(self.figure, name, step)

    def log_table(self, table: dict, name: str, step: int = None):
        """Log a dictionary as a table using the attached logger.

        Args:
            table: Dictionary containing the table data
            name: Name of the logged table
            step: Optional step or iteration number for the logged table

        Raises:
            AssertionError: If no logger is attached to the visualizer
        """
        assert self.logger is not None, "A logger must be provided in self.logger"
        self.logger.log_table(table, name, step)
