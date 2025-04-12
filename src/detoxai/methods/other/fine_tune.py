import types

import lightning as L
import torch

from ..model_correction import ModelCorrectionMethod


class FineTune(ModelCorrectionMethod):
    """This is kind-of a dummy correction method that is a baseline in form of further fine tuning of the model"""

    def __init__(
        self, model: L.LightningModule, experiment_name: str, device: str, **kwargs
    ) -> None:
        super().__init__(model, experiment_name, device)

    def apply_model_correction(
        self,
        dataloader: torch.utils.data.DataLoader,
        logger: object | bool = False,
        fine_tune_epochs: int = 1,
        lr: float = 1e-4,
        **kwargs,
    ) -> None:
        """

        Args:
          dataloader: torch.utils.data.DataLoader:
          logger: object | bool:  (Default value = False)
          fine_tune_epochs: int:  (Default value = 1)
          lr: float:  (Default value = 1e-4)
          **kwargs:

        Returns:

        """

        def configure_optimizers(self):
            """ """
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            return optimizer

        self.lightning_model.configure_optimizers = types.MethodType(
            configure_optimizers, self.lightning_model
        )

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
