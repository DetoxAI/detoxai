import lightning as L
import torch

from .clarc import CLARC
from .hooks import add_clarc_hook


class ACLARC(CLARC):
    """ """

    def __init__(
        self, model: L.LightningModule, experiment_name: str, device: str, **kwargs
    ) -> None:
        super().__init__(model, experiment_name, device)

    def apply_model_correction(
        self,
        cav_layers: list[str],
        dataloader: torch.utils.data.DataLoader,
        logger: object | bool = False,
        fine_tune_epochs: int = 1,
        alpha: float = 1.0,
        **kwargs,
    ) -> None:
        """

        Args:
          cav_layers: list[str]:
          dataloader: torch.utils.data.DataLoader:
          logger: object | bool:  (Default value = False)
          fine_tune_epochs: int:  (Default value = 1)
          alpha: float:  (Default value = 1.0)
          **kwargs:

        Returns:

        """
        for cav_layer in cav_layers:
            hook = add_clarc_hook(
                self.model,
                self.cav[cav_layer],
                self.mean_act_a[cav_layer],
                cav_layer,
                alpha,
            )
            self.hooks.append(hook)

        # Make sure model is in training mode
        self.model.train()

        trainer = L.Trainer(
            max_epochs=fine_tune_epochs,
            logger=logger,
            log_every_n_steps=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            devices=self.devices_indices,
        )
        trainer.fit(self.lightning_model, dataloader)

        # Go back to eval mode
        self.model.eval()

        # Remove hooks
        self.remove_hooks()
