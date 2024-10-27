import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm


def get_all_layers(model, prefix=""):
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
    components = layer_name.split(".")
    module = model
    for comp in components:
        module = getattr(module, comp)
    return module


def extract_activations(
    model, dataloader, experiment_name, layers=None, device="cuda", use_cache=True
):
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
        use_cache (bool, optional): Whether to use cached activations if available.

    Returns:
        dict: Dictionary mapping layer names to activations.
    """
    if use_cache:
        save_path = os.path.join("./activations", experiment_name + ".npz")
        if os.path.exists(save_path):
            activations_np = np.load(save_path)
            activations = {}
            for key in activations_np:
                activations[key] = activations_np[key]
            print(f"Loaded activations from '{save_path}'")
            return activations
    model.eval()
    model.to(device)

    save_path = os.path.join("./activations")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, experiment_name + ".npz")

    activations = {}

    if layers is None:
        layers = get_all_layers(model)
    elif isinstance(layers, list):
        layers_dict = {}
        for name in layers:
            try:
                layer = get_layer_by_name(model, name)
                layers_dict[name] = layer
            except AttributeError:
                raise ValueError(f"Layer '{name}' not found in the model.")
        layers = layers_dict
    elif isinstance(layers, dict):
        pass
    else:
        raise ValueError(
            "layers must be None, a list of layer names, or a dict of {name: module}"
        )

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

    labels_np = np.array([]).reshape(-1, 1)
    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc="Extracting Activations")
        ):
            data = batch[0]
            rest = np.array(batch[1:]).reshape(-1, len(batch[1:]))
            labels_np = np.concatenate((labels_np, rest), axis=0)
            data = data.to(device)
            _ = model(data)

    for handle in handles:
        handle.remove()
    
    activations_np = {}
    activations_np["labels"] = labels_np
    for name, acts in activations.items():
        np_acts = torch.cat(acts).cpu().numpy()
        if "resnet" in experiment_name and "relu" in name.lower() and np_acts.shape[0] == len(labels_np)//2:
            activations_np[name + "_pre"] = np_acts[:len(labels_np)//2]
            activations_np[name + "_post"] = np_acts[len(labels_np)//2:]
        else:
            activations_np[name] = np_acts
    np.savez(save_path, **activations_np)
    print(f"Saved all activations at '{save_path}'")
    return activations_np
