import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List, Any, Callable
import logging

from .posthoc_base import PosthocBase 
from ...utils.dataloader import WrappedDataLoader
from ...metrics.fairness_metrics import FairnessMetrics
from ...metrics.metrics import balanced_accuracy_torch

logger = logging.getLogger(__name__)

class RejectOptionClassification(PosthocBase):
    """
    Implements Reject Option Classification (ROC) for fairness optimization.
    
    ROC modifies model predictions in critical regions where prediction confidence
    is below a threshold (theta), replacing predictions with protected attributes.
    
    Attributes:
        dataloader (WrappedDataLoader): Dataloader containing the dataset
        theta_range (Tuple[float, float]): Range for threshold optimization
        theta_steps (int): Number of steps for threshold grid search
        hooks (List): Stores model forward hooks
        metrics (FairnessMetrics): Fairness metrics calculator
    """
    
    def __init__(
        self,
        model: nn.Module,
        experiment_name: str,
        device: str,
        dataloader: WrappedDataLoader,
        theta_range: Tuple[float, float] = (0.55, 0.95),
        theta_steps: int = 20,
        metrics_spec: Optional[Dict[str, Dict[str, List[str]]]] = None,
        objective_function: Optional[Callable[[float, float], float]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(model, experiment_name, device)
        
        self.dataloader = dataloader
        self.theta_range = theta_range
        self.theta_steps = theta_steps
        self.hooks: List[Any] = []
        
        assert metrics_spec is not None and len(metrics_spec) == 1
        assert theta_range[0] < theta_range[1] and theta_range[0] >= 0.5 and theta_range[1] <= 1.0

        self.metrics = FairnessMetrics(num_groups=2, metrics_spec=metrics_spec)
        if self.objective_function is None:
            self.objective_function = lambda fairness, accuracy: fairness * accuracy

    def _optimize_theta(self) -> float:
        """
        Optimizes the threshold parameter theta for best fairness-accuracy trade-off.
        
        Returns:
            float: Optimal theta value maximizing combined fairness and accuracy score
        """
        thetas = np.linspace(self.theta_range[0], self.theta_range[1], self.theta_steps)
        best_score = float('-inf')
        best_theta = thetas[0]

        # TODO: Verify if this is the correct way to get the target labels
        targets: torch.Tensor = self.dataloader.collator.get_target_labels()
        preds: torch.Tensor = self._get_model_predictions(self.dataloader)
        sensitive_features: torch.Tensor = self.dataloader.collator.get_group_labels()

        for theta in thetas:
            modified_preds = self._modified_prediction(preds, sensitive_features)
            fairness_score = next(iter(self.metrics(modified_preds, targets, sensitive_features).values()))
            accuracy_score = balanced_accuracy_torch(modified_preds, targets)
            combined_score = self.objective_function(fairness_score, accuracy_score)
            
            if combined_score > best_score:
                best_score = combined_score
                best_theta = theta
                
        return best_theta

    def _is_in_critical_region(self, theta: float, probs: torch.Tensor) -> torch.Tensor:
        """
        Determines which predictions fall in the critical region (confidence ≤ theta).
        
        Args:
            theta: Confidence threshold
            probs: Prediction probabilities
            
        Returns:
            torch.Tensor: Boolean mask indicating critical region predictions
        """
        max_probs, _ = torch.max(probs, dim=1)
        return max_probs <= theta

    def _modified_prediction(
        self, 
        theta: float,
        probs: torch.Tensor,
        protected_attrs: torch.Tensor
    ) -> torch.Tensor:
        """
        Modifies predictions based on the ROC algorithm.
        
        Args:
            theta: Confidence threshold
            probs: Model prediction probabilities
            protected_attrs: Protected attribute values
            
        Returns:
            torch.Tensor: Modified predictions
        """
        critical_mask = self._is_in_critical_region(theta, probs)
        predictions = torch.argmax(probs, dim=1)
        
        predictions[critical_mask] = protected_attrs[critical_mask] # TODO: discuss
        
        return predictions
        
    def apply_model_correction(self) -> None:
        theta = self._optimize_theta()
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
                
            protected_attrs = self.dataloader.collator.get_group_labels(
                self.dataloader._current_batch
            ).to(output.device)
            
            probs = F.softmax(output, dim=1)
            predictions = self._modified_prediction(theta, probs, protected_attrs)
            
            return predictions
            
        final_layer = list(self.model.modules())[-1]
        hook = final_layer.register_forward_hook(forward_hook)
        self.hooks.append(hook)