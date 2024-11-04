import numpy as np
import torch


def compute_mass_mean_probe(vecs: np.ndarray, targets: np.ndarray):


    num_targets = (targets == 1).sum()
    num_notargets = (targets == 0).sum()
    weights = (targets == 1) * 1 / num_targets + (targets == 0) * 1 / num_notargets
    weights = weights / weights.max()

    X = vecs

    # Compute the mean activation over the target samples
    mean_activation_over_artifact_samples = X[targets == 1].mean(0)
    mean_activation_over_nonartifact_samples = X[targets == 0].mean(0)

    # Compute the mass mean probe
    mass_mean_probe = mean_activation_over_artifact_samples - mean_activation_over_nonartifact_samples
    mass_mean_probe = torch.tensor(mass_mean_probe, dtype=torch.float32)
    print(f"mass_mean_probe shape: {mass_mean_probe.shape}")
    return mass_mean_probe
