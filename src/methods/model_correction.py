from abc import ABC, abstractmethod

from torch import nn
import lightning as L

from src.models.SklearnWrapper import SklearnWrapper


class ModelCorrectionMethod(ABC):
    def __init__(
        self, model: nn.Module | L.LightningModule, experiment_name: str, device: str
    ) -> None:
        # Unwrap LightningModule
        if isinstance(model, L.LightningModule):
            model = model.model

        self.model = model
        self.experiment_name = experiment_name
        self.device = device

    @abstractmethod
    def apply_model_correction(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_corrected_model(self) -> SklearnWrapper | L.LightningModule:
        raise NotImplementedError

    def remove_hooks(self) -> None:
        if hasattr(self, "hooks"):
            self.hooks = list()
