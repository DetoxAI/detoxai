from .base_model_correction import ModelCorrectionMethod
from .hooks import add_clarc_hook


class PCLARC(ModelCorrectionMethod):
    def __init__(self, model, dataloader, experiment_name, layers, device):
        super().__init__(model, dataloader, experiment_name, layers, device)

    def apply_model_correction(self, cav_layer: str, alpha: float = 1.0) -> None:
        self.hooks = add_clarc_hook(
            self.model, self.cav, self.mean_act, [cav_layer], alpha
        )
