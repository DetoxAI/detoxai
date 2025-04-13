import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...metrics.bias_metrics import calculate_bias_metric_torch
from ...metrics.metrics import balanced_accuracy_torch
from ...utils.dataloader import DetoxaiDataLoader
from .posthoc_base import PosthocBase

logger = logging.getLogger(__name__)


class ROCModelWrapper(nn.Module):
    """ """

    def __init__(self, base_model: nn.Module, theta: float, L_values: Dict[int, int]):
        super().__init__()
        self.base_model = base_model
        self.theta = theta
        self.L_values = L_values

    def _is_in_critical_region(self, probs: torch.Tensor) -> torch.Tensor:
        """

        Args:
          probs: torch.Tensor:

        Returns:

        """
        max_probs, _ = torch.max(probs, dim=1)
        return max_probs <= self.theta

    def forward(self, input, sensitive_features):
        """

        Args:
          input:
          sensitive_features:

        Returns:

        """
        # Get base model predictions
        output = self.base_model(input)
        if isinstance(output, tuple):
            output = output[0]

        # Apply ROC correction
        probs = F.softmax(output, dim=1)
        critical_mask = self._is_in_critical_region(probs)
        predictions = torch.argmax(probs, dim=1)

        # Modify predictions based on protected attributes
        for prot_value, L in self.L_values.items():
            mask = (sensitive_features == prot_value) & critical_mask
            predictions[mask] = L

        return predictions


class RejectOptionClassification(PosthocBase):
    """Implements Reject Option Classification (ROC) for fairness optimization.

    This class implements a post-hoc fairness optimization method that modifies model predictions
    based on a confidence threshold (theta). Predictions with confidence below theta are flipped
    to optimize for both accuracy and fairness.

    Args:

    Returns:

    """

    def __init__(
        self,
        model: nn.Module,
        experiment_name: str,
        device: str,
        dataloader: DetoxaiDataLoader,
        theta_range: Tuple[float, float] = (0.55, 0.95),
        theta_steps: int = 20,
        metric: str = "EO_GAP",
        objective_function: Optional[Callable[[float, float], float]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, experiment_name, device)

        self.dataloader = dataloader
        self.theta_range = theta_range
        self.theta_steps = theta_steps
        self.hooks: List[Any] = []

        assert (
            theta_range[0] < theta_range[1]
            and theta_range[0] >= 0.5
            and theta_range[1] <= 1.0
        )

        self.metric = metric
        self.objective_function = objective_function
        if self.objective_function is None:
            self.objective_function = lambda fairness, accuracy: fairness * accuracy

        self.best_config = {
            "theta": None,
            "L_values": {0: None, 1: None},  # L values for each protected attribute
        }

    def _evaluate_parameters(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        sensitive_features: torch.Tensor,
        theta: float,
        L_values: Dict[int, int],
    ) -> float:
        """Evaluates a specific parameter configuration.

        Args:
          preds: torch.Tensor:
          targets: torch.Tensor:
          sensitive_features: torch.Tensor:
          theta: float:
          L_values: Dict[int:
          int]:

        Returns:

        """
        # Validate theta
        if not (0.5 <= theta <= 1.0):
            raise AssertionError(f"Theta must be between 0.5 and 1.0, got {theta}")

        # Validate L_values contains all protected attribute values
        unique_protected = torch.unique(sensitive_features).tolist()
        for protected_value in unique_protected:
            if protected_value not in L_values:
                raise KeyError(
                    f"L_values missing value for protected attribute {protected_value}"
                )

        modified_preds = self._modified_prediction(
            theta, preds, sensitive_features, L_values
        )

        fairness_score = calculate_bias_metric_torch(
            self.metric, modified_preds, targets, sensitive_features
        )
        accuracy_score = balanced_accuracy_torch(modified_preds, targets)

        # Convert tensors to floats and handle NaN
        fairness_score = float(fairness_score.item())
        accuracy_score = float(accuracy_score.item())

        if np.isnan(fairness_score) or np.isnan(accuracy_score):
            return 0.0

        return float(self.objective_function(fairness_score, accuracy_score))

    def _optimize_parameters(self) -> Tuple[float, Dict[int, int]]:
        """Optimizes both theta and L values for each protected attribute value."""
        thetas = np.linspace(self.theta_range[0], self.theta_range[1], self.theta_steps)
        best_score = float("-inf")

        preds, targets, sensitive_features = self._get_model_predictions(
            self.dataloader
        )

        # Validate shapes
        assert preds.shape[1] == 2, (
            f"Expected binary classification, got {preds.shape[1]} classes"
        )
        assert targets.dim() == 1, f"Expected 1D targets, got {targets.dim()}D"
        assert sensitive_features.dim() == 1, (
            f"Expected 1D protected features, got {sensitive_features.dim()}D"
        )

        # Grid search over theta and L values
        for theta in thetas:
            for L_protected_0 in [0, 1]:
                for L_protected_1 in [0, 1]:
                    L_values = {0: L_protected_0, 1: L_protected_1}
                    score = self._evaluate_parameters(
                        preds, targets, sensitive_features, theta, L_values
                    )

                    if score > best_score:
                        best_score = score
                        self.best_config["theta"] = theta
                        self.best_config["L_values"] = L_values

        return self.best_config["theta"], self.best_config["L_values"]

    def _is_in_critical_region(self, theta: float, probs: torch.Tensor) -> torch.Tensor:
        """Determines which predictions fall in the critical region (confidence ≤ theta).

        Args:
          theta: Confidence threshold
          probs: Prediction probabilities (batch_size, 2)
          theta: float:
          probs: torch.Tensor:

        Returns:
          torch.Tensor: Boolean mask (batch_size,) indicating critical region predictions

        """
        assert probs.shape[1] == 2, (
            f"Expected binary classification, got {probs.shape[1]} classes"
        )
        max_probs, _ = torch.max(probs, dim=1)
        return max_probs <= theta

    def _modified_prediction(
        self,
        theta: float,
        probs: torch.Tensor,
        sensitive_features: torch.Tensor,
        L_values: Dict[int, int],
    ) -> torch.Tensor:
        """Modifies predictions based on critical region and protected attributes.

        Args:
          theta: Confidence threshold
          probs: Prediction probabilities (batch_size, 2)
          sensitive_features: Protected attributes (batch_size,)
          L_values: Dictionary mapping protected attribute values to labels
          theta: float:
          probs: torch.Tensor:
          sensitive_features: torch.Tensor:
          L_values: Dict[int:
          int]:

        Returns:
          torch.Tensor: Modified predictions (batch_size,)

        """
        assert probs.shape[1] == 2, "Expected binary classification"
        assert sensitive_features.dim() == 1, "Expected 1D protected features"

        critical_mask = self._is_in_critical_region(theta, probs)
        predictions = torch.argmax(probs, dim=1)

        # Apply different L values based on protected attribute
        for prot_value, L in L_values.items():
            mask = (sensitive_features == prot_value) & critical_mask
            predictions[mask] = L

        return predictions

    def apply_model_correction(self, **kwargs) -> nn.Module:
        """Returns a wrapped model that applies ROC correction during inference.

        Args:
          **kwargs:

        Returns:

        """
        theta, L_values = self._optimize_parameters()
        return ROCModelWrapper(self.model, theta, L_values)
