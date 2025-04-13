from .model_wrappers import BaseLightningWrapper


class CorrectionResult:
    """ """

    def __init__(self, method: str, model: BaseLightningWrapper, metrics: dict) -> None:
        self.method = method
        self.model = model
        self.metrics = metrics

    def __str__(self):
        return f"Results for: {self.method}"

    def __repr__(self):
        return self.__str__()

    def get_all_metrics(self) -> dict:
        """ """
        return self.metrics

    def get_metric(self, metric: str) -> float:
        """

        Args:
          metric: str:

        Returns:

        """
        return self.metrics["all"][metric]

    def get_model(self) -> BaseLightningWrapper:
        """ """
        return self.model

    def get_method(self) -> str:
        """ """
        return self.method
