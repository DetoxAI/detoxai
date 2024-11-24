from torch import nn
import lightning as L

from .base_model_correction import ModelCorrectionMethod
from .hooks import add_clarc_hook


class PCLARC(ModelCorrectionMethod):
    def __init__(
        self, model: nn.Module | L.LightningModule, experiment_name: str, device: str
    ):
        # Unwrap LightningModule
        if isinstance(model, L.LightningModule):
            model = model.model

        super().__init__(model, experiment_name, device)

    def apply_model_correction(self, cav_layer: str, alpha: float = 1.0) -> None:
        hook = add_clarc_hook(
            self.model, self.cav, self.mean_act_na, [cav_layer], alpha
        )
        self.hooks.append(hook)
