from .model_wrappers import BaseLightningWrapper


class CorrectionResult:
    def __init__(self):
        self.method: str
        self.model: BaseLightningWrapper
        self.metrics: dict

    def __str__(self):
        return f"Results for: {self.method}"

    def __repr__(self):
        return self.__str__()

    def get_all_metrics(self) -> dict:
        return self.metrics

    def get_metric(self, metric: str) -> float:
        return self.metrics[metric]

    def get_model(self) -> BaseLightningWrapper:
        return self.model

    def get_method(self) -> str:
        return self.method
