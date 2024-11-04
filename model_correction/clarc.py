import torch
from torch import nn


def stabilize(x: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    return x + epsilon


def clarc_hook(cav: torch.Tensor, mean_length: torch.Tensor, alpha: float = 1.0):
    """
    Creates a forward hook to adjust layer activations based on the CAV.

    Args:
        cav (torch.Tensor): Concept Activation Vector of shape (channels,).
        mean_length (float): Desired mean alignment length.

    Returns:
        function: A hook function to be registered with a PyTorch module.

    """

    def hook(module: nn.Module, input: tuple, output: torch.Tensor) -> torch.Tensor:
        device = output.device

        cav_local = stabilize(cav.to(device))
        mean_length_local = stabilize(mean_length.to(device))
        output_shapes = output.shape
        flat_output = output.clone().flatten(start_dim=1).detach()
        cav_dot = torch.matmul(cav_local, cav_local.T)
        shift = flat_output - mean_length_local
        correction = cav_dot * shift
        nonlocal alpha
        adjusted_flat_output = output.flatten(start_dim=1) - correction * alpha
        adjusted_output = adjusted_flat_output.reshape(output_shapes)
        return adjusted_output

    return hook


def mass_mean_probe_hook(probe: torch.Tensor):
    def hook(module: nn.Module, input: tuple, output: torch.Tensor):
        nonlocal probe
        probe = probe.to(output)
        o = output.clone().flatten(start_dim=1)
        perturbed = o - probe
        perturbed = perturbed.reshape(output.shape)
        return perturbed

    return hook


def add_clarc_hook(
    model: nn.Module, cav: torch.Tensor, mean_length: torch.Tensor, layer_names: list
) -> list:
    """
    Applies debiasing to the specified layers of a PyTorch model using the provided CAV.

    Args:
        model (nn.Module): The PyTorch model to be debiased.
        cav (torch.Tensor): The Concept Activation Vector, shape (channels,).
        mean_length (torch.Tensor): Mean activation length of the unaffected activations.
        layer_names (list): List of layer names (strings) to apply the hook on.

    Returns:
        list: A list of hook handles. Keep them to remove hooks later if needed.
    """
    hooks = []
    for name, module in model.named_modules():
        if name in layer_names:
            hook_fn = clarc_hook(cav, mean_length)
            handle = module.register_forward_hook(hook_fn)
            hooks.append(handle)
            print(f"Added hook to layer: {name}")
    return hooks


def add_mass_mean_probe_hook(
    model: nn.Module, probe: torch.Tensor, layer_names: list
) -> list:
    """
    Adds a probe to the specified layers of a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to be probed.
        probe (torch.Tensor): The probe tensor to be added to the output.
        layer_names (list): List of layer names (strings) to apply the hook on.

    Returns:
        list: A list of hook handles. Keep them to remove hooks later if needed.
    """
    hooks = []
    for name, module in model.named_modules():
        if name in layer_names:
            hook_fn = mass_mean_probe_hook(probe)
            handle = module.register_forward_hook(hook_fn)
            hooks.append(handle)
            print(f"Added probe to layer: {name}")
    return hooks


def outer_clarc_hook(cav: torch.Tensor, mean_length: torch.Tensor, alpha: float = 1.0):
    """
    Creates a forward hook to adjust layer activations based on the CAV.

    Args:
        cav (torch.Tensor): Concept Activation Vector of shape (channels,).
        mean_length (float): Desired mean alignment length.

    Returns:
        function: A hook function to be registered with a PyTorch module.

    """

    def hook(module: nn.Module, input: tuple, output: torch.Tensor) -> torch.Tensor:
        nonlocal alpha
        device = output.device
        output_shapes = output.shape

        v = stabilize(cav.to(device)).squeeze(0)
        z = stabilize(mean_length.to(device)).unsqueeze(0)
        x_copy_detached = output.clone().flatten(start_dim=1).detach()
        output = output.flatten(start_dim=1)

        vvt = torch.outer(v, v)

        A = torch.matmul(vvt, (x_copy_detached - z).T).T  # (N, batch_size)

        results = output - A

        adjusted_output = results.reshape(output_shapes)
        return adjusted_output

    return hook


def add_outer_clarc_hook(
    model: nn.Module, cav: torch.Tensor, mean_length: torch.Tensor, layer_names: list
) -> list:
    """
    Applies debiasing to the specified layers of a PyTorch model using the provided CAV.

    Args:
        model (nn.Module): The PyTorch model to be debiased.
        cav (torch.Tensor): The Concept Activation Vector, shape (channels,).
        mean_length (torch.Tensor): Mean activation length of the unaffected activations.
        layer_names (list): List of layer names (strings) to apply the hook on.

    Returns:
        list: A list of hook handles. Keep them to remove hooks later if needed.
    """
    hooks = []
    for name, module in model.named_modules():
        if name in layer_names:
            hook_fn = outer_clarc_hook(cav, mean_length)
            handle = module.register_forward_hook(hook_fn)
            hooks.append(handle)
            print(f"Added hook to layer: {name}")
    return hooks
