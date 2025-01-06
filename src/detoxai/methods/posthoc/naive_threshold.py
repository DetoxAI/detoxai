import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List, Any, Callable
import logging

from .posthoc_base import PosthocBase
from ...utils.dataloader import DetoxaiDataLoader
from ...metrics.metrics import balanced_accuracy_torch
from ...metrics.bias_metrics import calculate_bias_metric_torch

logger = logging.getLogger(__name__)

class NaiveThresholdOptimizer(PosthocBase):
    """
    Optimizes classification threshold using forward hooks.
    
    Attributes:
        threshold_range: Range for threshold optimization
        threshold_steps: Number of steps for grid search
        hooks: List of model hooks
        best_threshold: Best threshold found during optimization
    """
    
    def __init__(
        self,
        model: nn.Module,
        experiment_name: str,
        device: str,
        dataloader: DetoxaiDataLoader,
        threshold_range: Tuple[float, float] = (0.1, 0.9),
        threshold_steps: int = 20,
        metric: str = "EO_GAP",
        outputs_are_logits: bool = True,  # Add this parameter
        objective_function: Optional[Callable[[float, float], float]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, experiment_name, device)
        
        self.dataloader = dataloader
        self.threshold_range = threshold_range
        self.threshold_steps = threshold_steps
        self.hooks: List[Any] = []
        self.best_threshold: float = 0.5
        self.metric = metric
        self.outputs_are_logits = outputs_are_logits
        
        self.objective_function = objective_function
        if self.objective_function is None:
            self.objective_function = lambda fairness, accuracy: fairness * accuracy

    def _get_probabilities(self, outputs: torch.Tensor) -> torch.Tensor:
        """Convert model outputs to probabilities."""
        if isinstance(outputs, tuple):
            outputs = outputs[0]
            
        if self.outputs_are_logits:
            probs = F.softmax(outputs.to(self.device), dim=1)
        else:
            probs = outputs.to(self.device)
            
        return probs[:, 1]  # Return probabilities for positive class
    
    def _threshold_hook(self, threshold: float) -> Callable:
        """Creates forward hook for threshold modification."""
        def hook(module: nn.Module, input: Any, output: torch.Tensor) -> torch.Tensor:
            probs = self._get_probabilities(output)
            predictions = torch.where(
                probs > threshold,
                torch.ones_like(probs, device=self.device),  # Add device
                torch.zeros_like(probs, device=self.device)  # Add device
            )
            return predictions

        return hook
    
    def _evaluate_threshold(
        self,
        threshold: float,
        probs: torch.Tensor,
        targets: torch.Tensor,
        sensitive_features: torch.Tensor
    ) -> float:
        predictions = (probs > threshold).float().to(self.device)  # Add .to(self.device)
        targets = targets.to(self.device)  # Add device handling
        sensitive_features = sensitive_features.to(self.device)  # Add device handling
        
        accuracy_score = balanced_accuracy_torch(predictions, targets)
        fairness_score = calculate_bias_metric_torch(
            self.metric, predictions, targets, sensitive_features
        )
        
        if torch.isnan(fairness_score) or torch.isnan(accuracy_score):
            return 0.0
            
        return self.objective_function(
            float(fairness_score.item()),
            float(accuracy_score.item())
        )
    
    def _optimize_threshold(self) -> float:
        """Finds optimal threshold via grid search."""
        thresholds = np.linspace(
            self.threshold_range[0], 
            self.threshold_range[1], 
            self.threshold_steps
        )
        
        best_score = float('-inf')
        best_threshold = 0.5
        
        # Get base predictions and move to device
        preds, targets, sensitive_features = self._get_model_predictions(self.dataloader)
        preds = preds.to(self.device)  #
        probs = self._get_probabilities(preds)
        
        # Grid search with fairness consideration
        for threshold in thresholds:
            score = self._evaluate_threshold(threshold, probs, targets, sensitive_features)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
                
        logger.info(f"Best threshold: {best_threshold:.3f} with score: {best_score:.3f}")
        self.best_threshold = best_threshold
        return best_threshold
    
    def apply_model_correction(self, last_layer_name: str) -> None:
        """Applies threshold modification hook to model."""
        threshold = self._optimize_threshold()
        logger.info(f"Applying threshold correction with value: {threshold:.3f}")
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name == last_layer_name:
                hook = module.register_forward_hook(self._threshold_hook(threshold))
                self.hooks.append(hook)
