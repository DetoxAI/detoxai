import logging
from typing import Any, Callable, List, Optional, Tuple

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...metrics.bias_metrics import calculate_bias_metric_torch
from ...metrics.metrics import balanced_accuracy_torch
from ...utils.dataloader import DetoxaiDataLoader
from .posthoc_base import PosthocBase

logger = logging.getLogger(__name__)


class NaiveThresholdOptimizer(PosthocBase):
    """Optimizes classification threshold using forward hooks."""

    def __init__(
        self,
        model: nn.Module | L.LightningModule,
        experiment_name: str,
        device: str,
        dataloader: DetoxaiDataLoader,
        outputs_are_logits: bool = True,  # Add this parameter
        **kwargs: Any,
    ) -> None:
        super().__init__(model, experiment_name, device)

        self.dataloader = dataloader
        self.hooks: List[Any] = []
        self.best_threshold: float = 0.5
        self.outputs_are_logits = outputs_are_logits

    def _get_probabilities(self, outputs: torch.Tensor) -> torch.Tensor:
        """Convert model outputs to probabilities.

        Args:
          outputs: torch.Tensor:

        Returns:

        """
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        if self.outputs_are_logits:
            probs = F.softmax(outputs.to(self.device), dim=1)
        else:
            probs = outputs.to(self.device)

        return probs

    def __get_postive_probabilities(self, outputs: torch.Tensor) -> torch.Tensor:
        probs = self._get_probabilities(outputs)

        return probs[:, 1]  # Return probabilities for positive class

    def _threshold_hook(self, threshold: float) -> Callable:
        """Creates forward hook for threshold modification.

        Args:
          threshold: float:

        Returns:

        """

        def hook(module: nn.Module, input: Any, output: torch.Tensor) -> torch.Tensor:
            """

            Args:
              module: nn.Module:
              input: Any:
              output: torch.Tensor:

            Returns:

            """
            probs = self._get_probabilities(output)
            pos_probs = probs[:, 1]

            scaling_factor = 10.0
            pos_class = torch.sigmoid(scaling_factor * (pos_probs - threshold))

            preds = torch.zeros_like(probs, device=self.device)
            preds[:, 0] = 1 - pos_class
            preds[:, 1] = pos_class

            return preds

        return hook

    def _evaluate_threshold(
        self,
        threshold: float,
        probs: torch.Tensor,
        targets: torch.Tensor,
        sensitive_features: torch.Tensor,
        objective_function: Optional[Callable[[float, float], float]] = None,
        bias_metric: str = "EO_GAP",
    ) -> float:
        """

        Args:
          threshold: float:
          probs: torch.Tensor:
          targets: torch.Tensor:
          sensitive_features: torch.Tensor:
          objective_function: Optional[Callable[[float:
          float]:
          float]]:  (Default value = None)
          bias_metric: str:  (Default value = "EO_GAP")

        Returns:

        """
        # Ensure correct shapes for binary classification
        predictions = (probs > threshold).float()
        predictions = predictions.view(-1)  # Flatten to 1D
        targets = targets.view(-1)  # Flatten to 1D

        # Move tensors to correct device
        predictions = predictions.to(self.device)
        targets = targets.to(self.device)
        sensitive_features = sensitive_features.to(self.device)

        # Calculate metrics
        accuracy_score = balanced_accuracy_torch(predictions, targets)
        fairness_score = calculate_bias_metric_torch(
            bias_metric, predictions, targets, sensitive_features
        )

        if torch.isnan(fairness_score) or torch.isnan(accuracy_score):
            return 0.0

        return objective_function(
            float(fairness_score.item()), float(accuracy_score.item())
        )

    def _optimize_threshold(
        self,
        threshold_range: Tuple[float, float],
        threshold_steps: int,
        objective_function: Optional[Callable[[float, float], float]],
        metric: str,
    ) -> float:
        """Finds optimal threshold via grid search.

        Args:
          threshold_range: Tuple[float:
          float]:
          threshold_steps: int:
          objective_function: Optional[Callable[[float:
          float]]:
          metric: str:

        Returns:

        """
        thresholds = np.linspace(
            threshold_range[0], threshold_range[1], threshold_steps
        )

        best_score = float("-inf")
        best_threshold = 0.5

        # Get base predictions and move to device
        preds, targets, sensitive_features = self._get_model_predictions(
            self.dataloader
        )
        preds = preds.to(self.device)  #
        probs = self.__get_postive_probabilities(preds)
        scores = []
        # Grid search with fairness consideration
        for threshold in thresholds:
            score = self._evaluate_threshold(
                threshold,
                probs,
                targets,
                sensitive_features,
                objective_function,
                metric,
            )
            scores.append(score)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        logger.debug("Grid search results:")
        for threshold, score in zip(thresholds, scores):
            logger.debug(f"Threshold: {threshold:.3f} -> Score: {score:.3f}")
        logger.debug("Best result:")
        logger.debug(f"Threshold: {best_threshold:.3f} -> Score: {best_score:.3f}")
        probs = probs.to(self.device)
        targets = targets.to(self.device)
        sensitive_features = sensitive_features.to(self.device)

        # get balanced accuracy for best threshold
        balanced_acc = balanced_accuracy_torch(
            (probs > best_threshold).float(), targets
        )
        metric_value = calculate_bias_metric_torch(
            metric, (probs > best_threshold).float(), targets, sensitive_features
        )

        logger.info(
            f"Best threshold: {best_threshold}, Balanced Accuracy: {balanced_acc}, {metric}: {metric_value}, Objective: {best_score}"
        )
        return best_threshold

    def apply_model_correction(
        self,
        last_layer_name: str,
        threshold_range: Tuple[float, float] = (0.05, 0.95),
        objective_function: Optional[Callable[[float, float], float]] = None,
        threshold_steps: int = 100,
        metric: str = "EO_GAP",
        **kwargs: Any,
    ) -> None:
        """Applies threshold modification hook to model.

        Args:
          last_layer_name: str:
          threshold_range: Tuple[float:
          float]:  (Default value = (0.05)
          0.95):
          objective_function: Optional[Callable[[float:
          float]]:  (Default value = None)
          threshold_steps: int:  (Default value = 100)
          metric: str:  (Default value = "EO_GAP")
          **kwargs: Any:

        Returns:

        """

        if objective_function is None:
            objective_function = lambda fairness, accuracy: -fairness  # noqa
            logger.info(
                "No objective function provided. Using default fairness maximization."
            )
        else:
            try:
                logger.info(f"Using custom objective function: {objective_function}")
                objective_function = eval(objective_function)
            except:  # noqa
                raise ValueError("Objective function must be a valid lambda function.")

        threshold = self._optimize_threshold(
            threshold_range, threshold_steps, objective_function, metric
        )

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name == last_layer_name:
                hook = module.register_forward_hook(self._threshold_hook(threshold))
                logger.debug(f"Hook registered on layer: {name}")
                self.hooks.append(hook)

        if hasattr(self, "lightning_model"):
            self.lightning_model.model = self.model
