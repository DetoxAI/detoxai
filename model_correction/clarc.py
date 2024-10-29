import torch
from torch import nn

def stabilize(x: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    return x + epsilon

def clarc_hook(cav: torch.Tensor, mean_length: float):
    """
    Creates a forward hook to adjust layer activations based on the CAV.

    Args:
        cav (torch.Tensor): Concept Activation Vector of shape (channels,).
        mean_length (float): Desired mean alignment length.

    Returns:
        function: A hook function to be registered with a PyTorch module.

    """
    def hook(module: nn.Module, input: tuple, output: torch.Tensor) -> torch.Tensor:
        output_shapes = output.shape
        flat_output = output.flatten(start_dim=1).detach()
        cav_dot = cav @ cav.T
        shift = flat_output - mean_length  # (B, F), (F,)
        correction = cav_dot * shift
        output = flat_output - correction
        output = output.reshape(output_shapes)
        return output

    return hook

def add_clarc_hook(model: nn.Module, cav: torch.Tensor, mean_length: float, layer_names: list) -> list:
    """
    Applies debiasing to the specified layers of a PyTorch model using the provided CAV.

    Args:
        model (nn.Module): The PyTorch model to be debiased.
        cav (torch.Tensor): The Concept Activation Vector, shape (channels,).
        mean_length (float): The desired mean alignment length.
        layer_names (list): List of layer names (strings) to apply the hook on.

    Returns:
        list: A list of hook handles. Keep them to remove hooks later if needed.
    """
    hooks = []
    model_device = next(model.parameters()).device
    cav = cav.to(model_device)
    mean_length = torch.tensor(mean_length).to(model_device)
    for name, module in model.named_modules():
        if name in layer_names:
            hook_fn = clarc_hook(cav, mean_length)
            handle = module.register_forward_hook(hook_fn)
            hooks.append(handle)
            print(f"Added hook to layer: {name}")
    return hooks
