import numpy as np
import torch


def cosine_similarities_batch(a, cav, eps=1e-7):
    """

    Args:
      a:
      cav:
      eps:  (Default value = 1e-7)

    Returns:

    """
    return (a * cav).sum(1) / ((eps + np.linalg.norm(a, axis=1)) * np.linalg.norm(cav))


def euclidean_dist(x, y, dim=-1):
    """

    Args:
      x:
      y:
      dim:  (Default value = -1)

    Returns:

    """
    return torch.sqrt(torch.sum((x - y) ** 2, dim=dim))


def cosine_dist(x, y, dim=-1):
    """

    Args:
      x:
      y:
      dim:  (Default value = -1)

    Returns:

    """
    return 1 - torch.nn.functional.cosine_similarity(x, y, dim)


def largest_vals(x, y, dim=-1):
    """

    Args:
      x:
      y:
      dim:  (Default value = -1)

    Returns:

    """
    return -y.sum(dim)
