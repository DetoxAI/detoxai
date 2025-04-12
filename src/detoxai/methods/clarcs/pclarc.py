import lightning as L

from .clarc import CLARC
from .hooks import add_clarc_hook


class PCLARC(CLARC):
    """ """

    def __init__(
        self, model: L.LightningModule, experiment_name: str, device: str, **kwargs
    ) -> None:
        super().__init__(model, experiment_name, device)

    def apply_model_correction(
        self, cav_layers: list[str], alpha: float = 1.0, **kwargs
    ) -> None:
        """

        Args:
          cav_layers: list[str]:
          alpha: float:  (Default value = 1.0)
          **kwargs:

        Returns:

        """
        for cav_layer in cav_layers:
            hook = add_clarc_hook(
                self.model,
                self.cav[cav_layer],
                self.mean_act_na[cav_layer],
                cav_layer,
                alpha,
            )
            self.hooks.append(hook)
