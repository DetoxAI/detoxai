import numpy as np
import torch


def compute_mass_mean_probe(
    vecs: np.ndarray, targets: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the mass mean probe from the activations of a model.

    Args:
      vecs(np.ndarray): Activations of the model, shape (samples, features).
      targets(np.ndarray): Target labels for the samples, shape (samples,).
      vecs: np.ndarray:
      targets: np.ndarray:

    Returns:
      tuple: A tuple containing
      - mass_mean_probe (torch.Tensor): The mass mean probe.
      - mean_activation_over_nonartifact_samples (torch.Tensor): The mean activation over non-artifact samples.
      - mean_activation_over_artifact_samples (torch.Tensor): The mean activation over artifact samples

    """
    X = vecs

    mean_activation_over_artifact_samples = X[targets == 1].mean(0)
    mean_activation_over_nonartifact_samples = X[targets == 0].mean(0)

    mass_mean_probe = (
        mean_activation_over_artifact_samples - mean_activation_over_nonartifact_samples
    )
    mass_mean_probe = torch.tensor(mass_mean_probe, dtype=torch.float32)
    mean_activation_over_nonartifact_samples = torch.tensor(
        mean_activation_over_nonartifact_samples, dtype=torch.float32
    )
    mean_activation_over_artifact_samples = torch.tensor(
        mean_activation_over_artifact_samples, dtype=torch.float32
    )

    return (
        mass_mean_probe,
        mean_activation_over_nonartifact_samples,
        mean_activation_over_artifact_samples,
    )
