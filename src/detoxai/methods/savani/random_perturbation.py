import logging
import sys
from copy import deepcopy

import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...metrics.bias_metrics import BiasMetrics

# Project imports
from .savani_base import SavaniBase

logger = logging.getLogger(__name__)


class SavaniRP(SavaniBase):
    """ """

    def __init__(
        self,
        model: nn.Module | L.LightningModule,
        experiment_name: str,
        device: str,
        seed: int = 123,
        **kwargs,
    ) -> None:
        super().__init__(model, experiment_name, device, seed)

    def apply_model_correction(
        self,
        dataloader: DataLoader,
        last_layer_name: str,
        epsilon: float = 0.1,
        T_iters: int = 15,
        bias_metric: BiasMetrics | str = BiasMetrics.EO_GAP,
        optimizer_maxiter: int = 100,
        tau_init: float = 0.5,
        outputs_are_logits: bool = True,
        options: dict = {},
        eval_batch_size: int = 128,
        n_eval_batches: int = 3,
        soft_thresh_temperature: float = 10.0,
        **kwargs,
    ) -> None:
        """Apply random weights perturbation to the model, then select threshold 'tau' that maximizes phi

        To change perturbation parameters, you can pass the mean and std of the Gaussian noise
        options = {'mean': 1.0, 'std': 0.1}

        Args:
          dataloader: DataLoader:
          last_layer_name: str:
          epsilon: float:  (Default value = 0.1)
          T_iters: int:  (Default value = 15)
          bias_metric: BiasMetrics | str:  (Default value = BiasMetrics.EO_GAP)
          optimizer_maxiter: int:  (Default value = 100)
          tau_init: float:  (Default value = 0.5)
          outputs_are_logits: bool:  (Default value = True)
          options: dict:  (Default value = {})
          eval_batch_size: int:  (Default value = 128)
          n_eval_batches: int:  (Default value = 3)
          soft_thresh_temperature: float:  (Default value = 10.0)
          **kwargs:

        Returns:

        """
        assert T_iters > 0, "T_iters must be a positive integer"
        assert self.check_layer_name_exists(last_layer_name), (
            f"Layer name {last_layer_name} not found in the model"
        )

        self.last_layer_name = last_layer_name
        self.epsilon = epsilon
        self.bias_metric = bias_metric
        self.outputs_are_logits = outputs_are_logits
        self.n_eval_batches = n_eval_batches

        self.initialize_dataloader(dataloader, eval_batch_size)

        best_model = deepcopy(self.model)

        best_tau, best_phi = self.optimize_tau(tau_init, optimizer_maxiter)

        with tqdm(
            desc=f"Random Perturbation iterations (phi: {best_phi:.3f}, tau: {best_tau:.3f})",
            total=T_iters,
            file=sys.stdout,
        ) as pbar:
            # Randomly perturb the model weights
            for i in range(T_iters):
                self._perturb_weights(self.model, **options)

                tau, phi = self.optimize_tau(tau_init, optimizer_maxiter)

                if phi > best_phi:
                    best_tau = tau
                    best_phi = phi
                    best_model = deepcopy(self.model)

                pbar.set_description(
                    f"Random Perturbation iterations (phi: {best_phi:.3f}, tau: {best_tau:.3f})"
                )
                pbar.update(1)

        self.model = best_model
        self.best_tau = best_tau

        if hasattr(self, "lightning_model"):
            self.lightning_model.model = best_model

        # Add a hook with the best transformation
        self.apply_hook(best_tau, soft_thresh_temperature)

    def _perturb_weights(
        self, module: nn.Module, mean: float = 1.0, std: float = 0.1, **kwargs
    ) -> None:
        """Add Gaussian noise to the weights of the module by multiplying the weights with a number ~ N(mean, std)

        Args:
          module: nn.Module:
          mean: float:  (Default value = 1.0)
          std: float:  (Default value = 0.1)
          **kwargs:

        Returns:

        """
        with torch.no_grad():
            for param in module.parameters():
                param.data = param.data * torch.normal(
                    mean, std, param.data.shape, device=self.device
                )
