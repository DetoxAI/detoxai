import logging

import lightning as L
import torch
from concept_erasure import LeaceEraser
from torch import nn

from ...cavs import extract_activations
from ..model_correction import ModelCorrectionMethod
from ..utils import ACTIVATIONS_DIR

logger = logging.getLogger(__name__)


class LEACE(ModelCorrectionMethod):
    """ """

    def __init__(
        self,
        model: nn.Module | L.LightningModule,
        experiment_name: str,
        device: str,
        **kwargs,
    ) -> None:
        super().__init__(model, experiment_name, device)

        self.hooks = list()
        self.requires_acts = True

    def extract_activations(
        self,
        dataloader: torch.utils.data.DataLoader,
        intervention_layers: list[str],
        use_cache: bool = True,
        save_dir: str = ACTIVATIONS_DIR,
    ) -> None:
        """

        Args:
          dataloader: torch.utils.data.DataLoader:
          intervention_layers: list[str]:
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
            intervention_layers,
            self.device,
            use_cache,
        )

    def apply_model_correction(
        self, intervention_layers: list[str], use_n_examples: int = 15_000, **kwargs
    ) -> None:
        """Apply the LEACE eraser to the specified layers of the model.

        Args:
          intervention_layers: list[str]:
          use_n_examples: int:  (Default value = 15_000)
          **kwargs:

        Returns:

        """
        assert hasattr(self, "activations"), "Activations must be extracted first."
        assert self.activations is not None, "Activations must be extracted first."

        logger.debug(f"LEACE will be applied to layers: {intervention_layers}")

        for lay in intervention_layers:
            logger.debug(f"Applying LEACE to layer: {lay}")

            labels = self.activations["labels"][:, 1]
            layer_acts = self.activations[lay].reshape(
                self.activations[lay].shape[0], -1
            )

            X_torch = torch.from_numpy(layer_acts[:use_n_examples]).to(self.device)
            y_torch = torch.from_numpy(labels[:use_n_examples]).to(self.device)

            eraser = LeaceEraser.fit(X_torch, y_torch)

            self.add_clarc_hook(eraser, [lay])

    def add_clarc_hook(
        self,
        eraser: LeaceEraser,
        layer_names: list,
    ) -> None:
        """Applies debiasing to the specified layers of a PyTorch model using the provided CAV.

        Args:
          model(nn.Module): The PyTorch model to be debiased.
          cav(torch.Tensor): The Concept Activation Vector, shape (channels,).
          mean_length(torch.Tensor): Mean activation length of the unaffected activations.
          layer_names(list): List of layer names (strings) to apply the hook on.
          alpha(float): Scaling factor for the debiasing.
          eraser: LeaceEraser:
          layer_names: list:

        Returns:
          list: A list of hook handles. Keep them to remove hooks later if needed.

        """

        def __leace_hook(eraser: LeaceEraser) -> callable:
            def hook(
                module: nn.Module, input: tuple, output: torch.Tensor
            ) -> torch.Tensor:
                """

                Args:
                  module: nn.Module:
                  input: tuple:
                  output: torch.Tensor:

                Returns:

                """
                nonlocal eraser

                output = eraser(output.flatten(start_dim=1)).reshape(output.shape)

                # logger.debug(f"LEACE hook fired in layer: {module}")

                return output

            return hook

        for name, module in self.model.named_modules():
            if name in layer_names:
                hook_fn = __leace_hook(eraser)
                handle = module.register_forward_hook(hook_fn)
                self.hooks.append(handle)
                logger.debug(f"Added hook to layer: {name}")
