import matplotlib.pyplot as plt

try:
    from clearml import Logger as ClearMLLogger
except ImportError:
    ClearMLLogger = None

from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger


class ExperimentLogger:
    """ """

    def __init__(self, logger: TensorBoardLogger | WandbLogger | ClearMLLogger) -> None:
        self.logger = logger

    def log_metric(self, metric: float, name: str, step: int = None):
        """

        Args:
          metric: float:
          name: str:
          step: int:  (Default value = None)

        Returns:

        """
        if isinstance(self.logger, TensorBoardLogger):
            if step is None:
                raise ValueError("Step must be provided for TensorBoardLogger")
            self.logger.experiment.add_scalar(name, metric, global_step=step)
        elif isinstance(self.logger, WandbLogger):
            self.logger.log({name: metric})
        elif isinstance(self.logger, ClearMLLogger):
            self.logger.report_scalar(
                title=name, series=name, value=metric, iteration=step
            )
        else:
            raise ValueError(
                f"Logger must be either a TensorBoardLogger or a WandbLogger. Is {type(self.logger)}"
            )

    def log_image(self, figure: plt.Figure, name: str, step: int = None):
        """

        Args:
          figure: plt.Figure:
          name: str:
          step: int:  (Default value = None)

        Returns:

        """
        if isinstance(self.logger, TensorBoardLogger):
            if step is None:
                raise ValueError("Step must be provided for TensorBoardLogger")
            self.logger.experiment.add_figure(name, figure, global_step=step)
        elif isinstance(self.logger, WandbLogger):
            self.logger.log_image(key=name, images=[figure])
        elif isinstance(self.logger, ClearMLLogger):
            self.logger.report_matplotlib_figure(
                title=name,
                series=name,
                iteration=step,
                figure=figure,
                report_image=True,
            )
        else:
            raise ValueError(
                f"Logger must be either a TensorBoardLogger or a WandbLogger. Is {type(self.logger)}"
            )

    def log_table(self, table: dict, name: str, step: int = None):
        """

        Args:
          table: dict:
          name: str:
          step: int:  (Default value = None)

        Returns:

        """
        if isinstance(self.logger, TensorBoardLogger):
            if step is None:
                raise ValueError("Step must be provided for TensorBoardLogger")
            self.logger.experiment.add_text(name, table, global_step=step)
        elif isinstance(self.logger, WandbLogger):
            self.logger.log(table)
        elif isinstance(self.logger, ClearMLLogger):
            self.logger.report_table(
                title=name, series=name, iteration=step, table_plot=table
            )
        else:
            raise ValueError(
                f"Logger must be either a TensorBoardLogger or a WandbLogger. Is {type(self.logger)}"
            )
