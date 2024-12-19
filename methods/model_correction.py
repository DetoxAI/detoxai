from abc import ABC, abstractmethod

from torch import nn
import lightning as L


class ModelCorrectionMethod(ABC):
    def __init__(
        self, model: nn.Module | L.LightningModule, experiment_name: str, device: str
    ) -> None:
        # Unwrap LightningModule
        if isinstance(model, L.LightningModule):
            self.lightning_model = model
            self.model = model.model

        self.experiment_name = experiment_name
        self.device = device

        self.requires_cav: bool = False
        self.requires_acts: bool = False

    @abstractmethod
    def apply_model_correction(self) -> None:
        raise NotImplementedError

    def get_model(self) -> nn.Module:
        return self.model

    def get_lightning_model(self) -> L.LightningModule:
        if hasattr(self, "lightning_model"):
            return self.lightning_model
        else:
            raise AttributeError("No Lightning model found")

    def remove_hooks(self) -> None:
        if hasattr(self, "hooks"):
            self.hooks = list()
