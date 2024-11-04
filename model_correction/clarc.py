import torch
from torch import nn

def stabilize(x: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    return x + epsilon

def clarc_hook(cav: torch.Tensor, mean_length: torch.Tensor):
    """
    Creates a forward hook to adjust layer activations based on the CAV.

    Args:
        cav (torch.Tensor): Concept Activation Vector of shape (channels,).
        mean_length (float): Desired mean alignment length.

    Returns:
        function: A hook function to be registered with a PyTorch module.

    """
    def hook_forward(self, m, i, o):
        outs = o + 0
        cav = self.cav.to(outs)
        nonlocal mean_length
        mean_length = mean_length.to(outs)
        length = (outs.flatten(start_dim=2).max(2).values * cav).sum(1)
        v = self.cav.to(outs)
        beta = (cav * v).sum(1)
        mag = (self.mean_length - length).to(outs) / stabilize(beta)
        addition = (mag[:, None, None, None] * v[..., None, None])
        acts = outs + addition
        return acts
    return hook_forward



def add_clarc_hook(model: nn.Module, cav: torch.Tensor, mean_length: torch.Tensor, layer_names: list) -> list:
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
