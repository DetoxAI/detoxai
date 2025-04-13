import logging
from abc import ABC, abstractmethod

import lightning as L
import torch
from torch import nn

from ..model_correction import ModelCorrectionMethod

logger = logging.getLogger(__name__)


class PosthocBase(ModelCorrectionMethod, ABC):
    """Abstract base class for binary post-hoc debiasing methods."""

    def __init__(
        self,
        model: nn.Module | L.LightningModule,
        experiment_name: str,
        device: str,
        **kwargs,
    ) -> None:
        super().__init__(model, experiment_name, device)
        self.hooks = []

    @abstractmethod
    def apply_model_correction(self) -> None:
        """ """
        raise NotImplementedError

    def _get_model_predictions(
        self, dataloader: torch.utils.data.DataLoader
    ) -> torch.Tensor:
        """Get model predictions on dataloader

        Args:
          dataloader: torch.utils.data.DataLoader:

        Returns:

        """
        self.model.eval()
        predictions, labels, protected_attribute = [], [], []
        with torch.no_grad():
            for batch in dataloader:
                inputs, _labels, _protected_attribute = batch
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                predictions.append(outputs)
                labels.append(_labels)
                protected_attribute.append(_protected_attribute)
        return torch.cat(predictions), torch.cat(labels), torch.cat(protected_attribute)
