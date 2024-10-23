import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm

def get_all_layers(model, prefix=''):
    """
    Recursively get all layers from the model.
    
    Args:
        model (nn.Module): The PyTorch model.
        prefix (str): Prefix for the layer names (used during recursion).
    
    Returns:
        dict: Dictionary mapping layer names to layer modules.
    """
    layers = {}
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        layers[full_name] = module
        # Recursively get child layers
        child_layers = get_all_layers(module, full_name)
        layers.update(child_layers)
    return layers

def get_layer_by_name(model, layer_name):
    """
    Retrieve a layer from the model by its name.
    
    Args:
        model (nn.Module): The PyTorch model.
        layer_name (str): Dot-separated name of the layer.
    
    Returns:
        nn.Module: The layer module.
    """
    components = layer_name.split('.')
    module = model
    for comp in components:
        module = getattr(module, comp)
    return module

def extract_activations(model, dataloader, save_path, layers=None, device='cuda', save_per_layer=False):
    """
    Extract activations from all layers of a model for data from a dataloader.
    
    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): DataLoader providing the input data.
        save_path (str): Directory path to save the activations.
        layers (list or dict, optional): Layers to extract activations from. 
                                         If None, all layers are used.
                                         Can be a list of layer names or a dict of {name: module}.
        device (str, optional): Device to run the model on ('cuda' or 'cpu').
        save_per_layer (bool, optional): Whether to save activations per layer in separate files.
    """
    model.eval()
    model.to(device)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Set up hooks
    activations = {}

    if layers is None:
        # Get all layers
        layers = get_all_layers(model)
    elif isinstance(layers, list):
        # layers is a list of layer names
        layers_dict = {}
        for name in layers:
            try:
                layer = get_layer_by_name(model, name)
                layers_dict[name] = layer
            except AttributeError:
                raise ValueError(f"Layer '{name}' not found in the model.")
        layers = layers_dict
    elif isinstance(layers, dict):
        # layers is already a dict of {name: module}
        pass
    else:
        raise ValueError("layers must be None, a list of layer names, or a dict of {name: module}")

    handles = []
    for name, layer in layers.items():
        def get_activation(name):
            def hook(model, input, output):
                if name not in activations:
                    activations[name] = []
                activations[name].append(output.detach().cpu())
            return hook
        handle = layer.register_forward_hook(get_activation(name))
        handles.append(handle)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Extracting Activations")):
            data = data.to(device)
            _ = model(data)

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Concatenate and save activations
    if save_per_layer:
        for name in activations:
            # Concatenate all activations for the current layer
            concatenated = torch.cat(activations[name], dim=0).numpy()
            # Define the file path with .npy extension
            layer_save_path = os.path.join(save_path, f"{name.replace('.', '_')}_activations.npy")
            # Save the NumPy array
            np.save(layer_save_path, concatenated)
            print(f"Saved activations for layer '{name}' at '{layer_save_path}'")
    else:
        # Prepare a dictionary to hold all activations as NumPy arrays
        activations_np = {}
        for name in activations:
            # Concatenate all activations for the current layer and convert to NumPy
            activations_np[name] = torch.cat(activations[name], dim=0).numpy()
        
        # Define the file path for the combined activations
        combined_save_path = os.path.join(save_path, "activations.npz")
        # Save all activations into a single .npz file
        np.savez(combined_save_path, **activations_np)
        print(f"Saved all activations at '{combined_save_path}'")

# Example Usage
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets

    # Define your model
    model = torchvision.models.resnet18(pretrained=True)

    # Define your dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # Define save path
    save_path = './activations'

    # Extract and save activations from all layers
    extract_activations(
        model=model,
        dataloader=dataloader,
        save_path=save_path,
        layers=None,  # None to extract from all layers
        device='cuda' if torch.cuda.is_available() else 'cpu',
        save_per_layer=True  # Set to False to save all activations in a single file
    )