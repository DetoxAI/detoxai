import numpy as np
import torch
from torch import nn
from tqdm import tqdm


def prepare_model(model: nn.Module, device: str) -> nn.Module:
    """
    Prepare the model for evaluation by setting it to eval mode and moving it to the specified device.

    Args:
        model (nn.Module): The PyTorch model to prepare.
        device (str): Device to move the model to ('cuda' or 'cpu').

    Returns:
        nn.Module: The prepared model.
    """
    model.eval()
    model.to(device)
    return model


def process_cav(cav: np.ndarray or torch.Tensor, device: str) -> torch.Tensor:
    """
    Convert the CAV to a torch tensor, move it to the specified device, and normalize it.

    Args:
        cav (np.ndarray or torch.Tensor): The Concept Activation Vector.
        device (str): Device to move the CAV to ('cuda' or 'cpu').

    Returns:
        torch.Tensor: The processed and normalized CAV.
    """
    if isinstance(cav, np.ndarray):
        cav = torch.from_numpy(cav).float().to(device)
    elif isinstance(cav, torch.Tensor):
        cav = cav.float().to(device)
    else:
        raise TypeError("CAV must be a NumPy array or a PyTorch tensor.")
    
    cav_norm = torch.norm(cav)
    if cav_norm == 0:
        raise ValueError("CAV has zero norm and cannot be normalized.")
    
    return cav / cav_norm


def register_activation_hook(model: nn.Module, target_layer: str, activations: dict) -> torch.utils.hooks.RemovableHandle:
    """
    Register a forward hook to capture activations from the specified target layer.

    Args:
        model (nn.Module): The PyTorch model.
        target_layer (str): The name of the layer to hook.
        activations (dict): Dictionary to store the activations.

    Returns:
        torch.utils.hooks.RemovableHandle: The hook handle for later removal.
    """
    def forward_hook(module, input, output):
        output.retain_grad()  # Retain gradients for non-leaf tensors
        activations['activations'] = output

    for name, module in model.named_modules():
        if name == target_layer:
            return module.register_forward_hook(forward_hook)
    
    raise ValueError(f"Layer '{target_layer}' not found in the model.")


def compute_gradients(
    model: nn.Module, 
    inputs: torch.Tensor, 
    target_class: int
) -> torch.Tensor:
    """
    Perform a forward and backward pass to compute gradients with respect to the target class.

    Args:
        model (nn.Module): The PyTorch model.
        inputs (torch.Tensor): Input tensor batch.
        target_class (int): The target class index.

    Returns:
        torch.Tensor: Gradients with respect to the inputs.
    """
    inputs.requires_grad = True
    model.zero_grad()
    outputs = model(inputs)
    
    if outputs.ndimension() == 1 or outputs.size(1) <= target_class:
        raise IndexError(f"Target class {target_class} is out of bounds for the model output.")

    target_output = outputs[:, target_class]
    target_output.backward(target_output)
    
    if inputs.grad is None:
        raise ValueError("Gradients with respect to inputs could not be computed.")
    
    return inputs.grad


def extract_activation_gradients(activations: dict) -> torch.Tensor:
    """
    Extract gradients from the stored activations.

    Args:
        activations (dict): Dictionary containing the activations.

    Returns:
        torch.Tensor: Gradients with respect to the activations.
    """
    act = activations.get('activations')
    if act is None:
        raise ValueError("Activations have not been captured.")
    
    if act.grad is None:
        raise ValueError("Gradients with respect to activations could not be computed.")
    
    return act.grad


def normalize_tensor(tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Normalize a tensor along the specified dimension.

    Args:
        tensor (torch.Tensor): The tensor to normalize.
        dim (int): The dimension along which to normalize.

    Returns:
        torch.Tensor: The normalized tensor.
    """
    norm = torch.norm(tensor, dim=dim, keepdim=True)
    norm = torch.where(norm == 0, torch.tensor(1.0, device=tensor.device), norm)
    return tensor / norm


def compute_alignment(grad: torch.Tensor, cav: torch.Tensor) -> torch.Tensor:
    """
    Compute the alignment between gradients and the CAV using cosine similarity.

    Args:
        grad (torch.Tensor): Normalized gradients.
        cav (torch.Tensor): Normalized Concept Activation Vector.

    Returns:
        torch.Tensor: Alignment scores.
    """
    # Ensure that grad and cav have compatible dimensions
    if grad.size(1) != cav.size(0):
        raise ValueError("Dimension mismatch between gradients and CAV.")
    
    return torch.sum(grad * cav, dim=1)


def get_tcav_scores(
    model: nn.Module, 
    cav: np.ndarray or torch.Tensor, 
    dataloader: torch.utils.data.DataLoader, 
    target_layer: str, 
    target_class: int, 
    device: str = 'cuda', 
    num_samples: int = 100
) -> float:
    """
    Compute TCAV scores for a given model, CAV, and dataset.

    Args:
        model (nn.Module): The PyTorch model to analyze.
        cav (np.ndarray or torch.Tensor): The Concept Activation Vector.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the dataset to evaluate.
        target_layer (str): The name of the layer from which to extract activations.
        target_class (int): The index of the target class for which to compute TCAV scores.
        device (str): Device to perform computations on ('cuda' or 'cpu').
        num_samples (int): Number of samples to use for TCAV computation.

    Returns:
        float: The TCAV score ranging from 0 to 1.
    """
    # Prepare model and CAV
    model = prepare_model(model, device)
    cav = process_cav(cav, device)

    # Register activation hook
    activations = {}
    hook_handle = register_activation_hook(model, target_layer, activations)

    tcav_positive = 0
    tcav_total = 0

    try:
        # Iterate through the DataLoader
        for batch in tqdm(dataloader, desc="Computing TCAV Scores"):
            if tcav_total >= num_samples:
                break

            # Handle different batch structures
            if isinstance(batch, (list, tuple)):
                inputs, _ = batch[:2]
            else:
                inputs = batch  # Assuming batch is just inputs

            inputs = inputs.to(device)

            # Compute gradients with respect to inputs
            try:
                _ = compute_gradients(model, inputs, target_class)
            except Exception as e:
                print(f"Skipping batch due to error in gradient computation: {e}")
                continue

            # Extract activation gradients
            try:
                grad = extract_activation_gradients(activations)
            except Exception as e:
                print(f"Skipping batch due to error in extracting activation gradients: {e}")
                continue

            # Flatten gradients and activations
            grad = grad.view(grad.size(0), -1)
            act = activations['activations']
            act = act.view(act.size(0), -1)

            # Normalize gradients
            grad = normalize_tensor(grad)

            # Compute alignment with CAV
            try:
                alignment = compute_alignment(grad, cav)
            except Exception as e:
                print(f"Skipping batch due to alignment computation error: {e}")
                continue

            # Count positive alignments
            tcav_positive += (alignment > 0).sum().item()
            tcav_total += inputs.size(0)

    finally:
        # Ensure that the hook is removed even if an error occurs
        hook_handle.remove()

    # Compute TCAV score
    tcav_score = tcav_positive / tcav_total if tcav_total > 0 else 0.0
    return tcav_score, tcav_positive, tcav_total