from abc import ABC, abstractmethod

import lightning as L
import torch

from ...cavs import compute_cav, compute_mass_mean_probe, extract_activations
from ..model_correction import ModelCorrectionMethod
from ..utils import ACTIVATIONS_DIR


# Wrapper for requiring activations and CAVs to be computed before applying model correction
def require_activations_and_cav(func):
    """

    Args:
      func:

    Returns:

    """

    def wrapped(self, cav_layers: list[str], *args, **kwargs):
        """

        Args:
          cav_layers: list[str]:
          *args:
          **kwargs:

        Returns:

        """
        if not hasattr(self, "activations"):
            raise ValueError(
                "Activations must be computed before applying model correction"
            )

        if not hasattr(self, "cav"):
            raise ValueError("CAVs must be computed before applying model correction")

        return func(self, cav_layers, *args, **kwargs)

    return wrapped


class CLARC(ModelCorrectionMethod, ABC):
    """ """

    def __init__(
        self, model: L.LightningModule, experiment_name: str, device: str
    ) -> None:
        super().__init__(model, experiment_name, device)
        self.hooks = list()
        self.requires_cav = True
        self.requires_acts = True

    def __init_subclass__(cls) -> None:
        """
        Adds a decorator to the apply_model_correction method to require activations and CAVs to be computed
        """
        cls.apply_model_correction = require_activations_and_cav(
            cls.apply_model_correction
        )

    def extract_activations(
        self,
        dataloader: torch.utils.data.DataLoader,
        layers: list,
        use_cache: bool = True,
        save_dir: str = ACTIVATIONS_DIR,
    ) -> None:
        """

        Args:
          dataloader: torch.utils.data.DataLoader:
          layers: list:
          use_cache: bool:  (Default value = True)
          save_dir: str:  (Default value = ACTIVATIONS_DIR)

        Returns:

        """
        # Freeze the model
        self.model.eval()

        self.activations = extract_activations(
            self.model,
            dataloader,
            self.experiment_name,
            save_dir,
            layers,
            self.device,
            use_cache,
        )

    def compute_cavs(self, cav_type: str, cav_layers: list[str]) -> None:
        """

        Args:
          cav_type: str:
          cav_layers: list[str]:

        Returns:

        """
        labels = self.activations["labels"][:, 1]

        self.cav = dict()
        self.mean_act_na = dict()
        self.mean_act_a = dict()

        for cav_layer in cav_layers:
            layer_acts = self.activations[cav_layer].reshape(
                self.activations[cav_layer].shape[0], -1
            )

            match cav_type:
                case "mmp":
                    cav, mean_na, mean_a = compute_mass_mean_probe(layer_acts, labels)
                case _:
                    cav, mean_na, mean_a = compute_cav(layer_acts, labels, cav_type)

            # Move cav and mean_act to proper torch dtype
            self.cav[cav_layer] = cav.float().to(self.device)
            # mean activation over non-artifact samples
            self.mean_act_na[cav_layer] = mean_na.float().to(self.device)
            # mean activation over artifact samples
            self.mean_act_a[cav_layer] = mean_a.float().to(self.device)

        self.cav_type = cav_type

        self.activations = None

    @abstractmethod
    def apply_model_correction(self, cav_layer: str) -> None:
        """

        Args:
          cav_layer: str:

        Returns:

        """
        raise NotImplementedError
