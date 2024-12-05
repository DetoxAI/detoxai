from abc import ABC, abstractmethod
import torch
from torch import nn

from ..cavs import extract_activations, compute_mass_mean_probe, compute_cav
from .helpers import require_activations_and_cav


class ModelCorrectionMethod(ABC):
    def __init__(self, model: nn.Module, experiment_name: str, device: str) -> None:
        self.model = model
        self.experiment_name = experiment_name
        self.device = device
        self.hooks = list()

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
        save_dir: str = './activations'
    ) -> None:
        # Freeze the model
        self.model.eval()

        self.activations = extract_activations(
            self.model,
            dataloader,
            self.experiment_name,
            layers,
            self.device,
            use_cache,
            save_dir
        )

    def compute_cav(self, cav_type: str, cav_layer: str) -> None:
        labels = self.activations["labels"][:, 1]
        layer_acts = self.activations[cav_layer].reshape(
            self.activations[cav_layer].shape[0], -1
        )

        match cav_type:
            case "mmp":
                cav, mean_na, mean_a = compute_mass_mean_probe(layer_acts, labels)
            case _:
                cav, mean_na, mean_a = compute_cav(layer_acts, labels, cav_type)

        # Move cav and mean_act to proper torch dtype
        self.cav = cav.float().to(self.device)
        # mean activation over non-artifact samples
        self.mean_act_na = mean_na.float().to(self.device)
        # mean activation over artifact samples
        self.mean_act_a = mean_a.float().to(self.device)
        self.cav_type = cav_type

        self.activations = None

    def remove_hooks(self) -> None:
        if hasattr(self, "hooks"):
            self.hooks = list()

    @abstractmethod
    def apply_model_correction(self, cav_layer: str) -> None:
        raise NotImplementedError
