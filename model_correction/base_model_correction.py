from abc import ABC, abstractmethod
import torch
from torch import nn

from ..cavs import extract_activations, compute_mass_mean_probe, compute_cav


# Wrapper for requiring activations and CAVs to be computed before applying model correction
def require_activations_and_cav(func):
    def wrapped(self, cav_layer: str):
        if not hasattr(self, "activations"):
            raise ValueError("Activations have not been computed yet")
        if not hasattr(self, "cav"):
            raise ValueError("CAV has not been computed yet")
        if not hasattr(self, "mean_act"):
            raise ValueError("Mean activations have not been computed yet")
        return func(self, cav_layer)

    return wrapped


class ModelCorrectionMethod(ABC):
    def __init__(self, model: nn.Module, experiment_name: str, device: str) -> None:
        self.model = model
        self.experiment_name = experiment_name
        self.device = device

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
        use_cache: bool = False,
    ) -> None:
        self.activations = extract_activations(
            self.model,
            dataloader,
            self.experiment_name,
            layers,
            self.device,
            use_cache,
        )

    def compute_cav(self, cav_type: str, cav_layer: str) -> None:
        labels = self.activations["labels"][:, 1]
        layer_acts = self.activations[cav_layer].reshape(
            self.activations[cav_layer].shape[0], -1
        )

        match cav_type:
            case "mmp":
                cav, mean_act = compute_mass_mean_probe(layer_acts, labels)
            case _:
                cav, mean_act = compute_cav(layer_acts, labels, cav_type=cav_type)

        # Move cav and mean_act to proper torch dtype
        self.cav = cav.float().to(self.device)
        self.mean_act = mean_act.float().to(self.device)
        self.cav_type = cav_type

    def remove_hooks(self) -> None:
        if hasattr(self, "hooks"):
            for hook in self.hooks:
                hook.remove()

    @abstractmethod
    def apply_model_correction(self, cav_layer: str) -> None:
        raise NotImplementedError
