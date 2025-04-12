from abc import ABC, abstractmethod

import lightning as L
from torch import nn


class ModelCorrectionMethod(ABC):
    """ """

    def __init__(
        self, model: nn.Module | L.LightningModule, experiment_name: str, device: str
    ) -> None:
        # Unwrap LightningModule
        if isinstance(model, L.LightningModule):
            self.lightning_model = model.to(device)
            self.model = model.model.to(device)
        else:
            self.model = model.to(device)

        self.experiment_name = experiment_name
        self.device = str(device)
        if "cuda" in self.device and ":" in self.device:
            self.devices_indices = [int(str(self.device).split(":")[1])]
        else:
            self.devices_indices = "auto"

        self.requires_cav: bool = False
        self.requires_acts: bool = False

    @abstractmethod
    def apply_model_correction(self) -> None:
        """ """
        raise NotImplementedError

    def get_model(self) -> nn.Module:
        """ """
        return self.model

    def get_lightning_model(self) -> L.LightningModule:
        """ """
        if hasattr(self, "lightning_model"):
            return self.lightning_model
        else:
            raise AttributeError("No Lightning model found")

    def remove_hooks(self) -> None:
        """ """
        if hasattr(self, "hooks"):
            self.hooks = list()
