import logging
import torch
from torch import nn

logger = logging.getLogger(__name__)


def stabilize(x: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    return x + epsilon


def mass_mean_probe_hook(probe: torch.Tensor, alpha: float):
    def hook(module: nn.Module, input: tuple, output: torch.Tensor):
        nonlocal probe, alpha
        o = output.clone().flatten(start_dim=1)
        perturbed = o - probe * alpha
        perturbed = perturbed.reshape(output.shape)
        # print("DEBUG: mass mean probe hook applied")
        return perturbed

    return hook


def add_mass_mean_probe_hook(
    model: nn.Module, probe: torch.Tensor, layer_names: list, alpha: float = 1.0
) -> list:
    """
    Adds a probe to the specified layers of a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to be probed.
        probe (torch.Tensor): The probe tensor to be added to the output.
        layer_names (list): List of layer names (strings) to apply the hook on.
        alpha (float): Scaling factor for the probe.

    Returns:
        list: A list of hook handles. Keep them to remove hooks later if needed.
    """
    hooks = []
    for name, module in model.named_modules():
        if name in layer_names:
            hook_fn = mass_mean_probe_hook(probe, alpha)
            handle = module.register_forward_hook(hook_fn)
            hooks.append(handle)
            # print(f"DEBUG: Added probe to layer: {name}")
    return hooks


def clarc_hook(cav: torch.Tensor, mean_length: torch.Tensor, alpha: float):
    """
    Creates a forward hook to adjust layer activations based on the CAV.

    Args:
        cav (torch.Tensor): Concept Activation Vector of shape (channels,).
        mean_length (float): Desired mean alignment length.

    Returns:
        function: A hook function to be registered with a PyTorch module.

    """

    def hook(module: nn.Module, input: tuple, output: torch.Tensor) -> torch.Tensor:
        nonlocal alpha, cav, mean_length
        output_shapes = output.shape

        v = stabilize(cav).squeeze(0)
        z = stabilize(mean_length).unsqueeze(0)
        x_copy_detached = output.clone().flatten(start_dim=1).detach()
        output = output.flatten(start_dim=1)

        vvt = torch.outer(v, v)

        A = torch.matmul(vvt, (x_copy_detached - z).T).T  # (N, batch_size)

        results = output - A * alpha

        adjusted_output = results.reshape(output_shapes)

        # logger.debug(f"CLARC hook fired in layer: {module}")

        return adjusted_output

    return hook


def add_clarc_hook(
    model: nn.Module,
    cav: torch.Tensor,
    mean_length: torch.Tensor,
    layer_name: str,
    alpha: float = 1.0,
) -> list:
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
    hooks = []
    for name, module in model.named_modules():
        if name == layer_name:
            hook_fn = clarc_hook(cav, mean_length, alpha)
            handle = module.register_forward_hook(hook_fn)
            hooks.append(handle)
            logger.debug(f"Added CLARC hook to layer: {name}")
    return hooks
