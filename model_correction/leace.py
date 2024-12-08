import torch
from torch import nn
from concept_erasure import LeaceEraser

from ..cavs import extract_activations
from .base_model_correction import ModelCorrectionMethod


class LEACE(ModelCorrectionMethod):
    def __init__(self, model: nn.Module, experiment_name: str, device: str) -> None:
        super().__init__(model, experiment_name, device)
        self.hooks = list()

    def extract_activations(
        self,
        dataloader: torch.utils.data.DataLoader,
        layers: list,
        use_cache: bool = True,
        save_dir: str = "./activations",
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
            save_dir,
        )

    def apply_model_correction(self, layers: list[str]) -> None:
        """
        Apply the LEACE eraser to the specified layers of the model.
        """
        assert hasattr(self, "activations"), "Activations must be extracted first."
        assert self.activations is not None, "Activations must be extracted first."

        for lay in layers:
            labels = self.activations["labels"][:, 1]
            layer_acts = self.activations[lay].reshape(
                self.activations[lay].shape[0], -1
            )

            X_torch = torch.from_numpy(layer_acts).to(self.device)
            y_torch = torch.from_numpy(labels).to(self.device)

            print(X_torch.shape)

            eraser = LeaceEraser.fit(X_torch, y_torch)

            self.add_clarc_hook(eraser, [lay])

    def add_clarc_hook(
        self,
        eraser: LeaceEraser,
        layer_names: list,
    ) -> None:
        """
        Applies debiasing to the specified layers of a PyTorch model using the provided CAV.

        Args:
            model (nn.Module): The PyTorch model to be debiased.
            cav (torch.Tensor): The Concept Activation Vector, shape (channels,).
            mean_length (torch.Tensor): Mean activation length of the unaffected activations.
            layer_names (list): List of layer names (strings) to apply the hook on.
            alpha (float): Scaling factor for the debiasing.

        Returns:
            list: A list of hook handles. Keep them to remove hooks later if needed.
        """

        def __leace_hook(eraser: LeaceEraser) -> callable:
            def hook(
                module: nn.Module, input: tuple, output: torch.Tensor
            ) -> torch.Tensor:
                nonlocal eraser
                print(output.shape)
                output = eraser(output.flatten(start_dim=1)).reshape(output.shape)
                return output

            return hook

        for name, module in self.model.named_modules():
            if name in layer_names:
                hook_fn = __leace_hook(eraser)
                handle = module.register_forward_hook(hook_fn)
                self.hooks.append(handle)
                print(f"DEBUG: Added hook to layer: {name}")

    def remove_hooks(self) -> None:
        if hasattr(self, "hooks"):
            self.hooks = list()

    def get_corrected_model(self) -> nn.Module:
        return self.model
