import sys
import torch
import lightning as L
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy
from scipy import optimize
from scipy.optimize import OptimizeResult
from tqdm import tqdm
import logging

# Project imports
from .savani_base import SavaniBase
from ...metrics.bias_metrics import BiasMetrics

logger = logging.getLogger(__name__)


class SavaniRP(SavaniBase):
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
        T_iters: int = 10,
        bias_metric: BiasMetrics | str = BiasMetrics.DP_GAP,
        data_to_use: float | int = 128,
        optimizer_maxiter: int = 50,
        tau_init: float = 0.5,
        options: dict = {},
        **kwargs,
    ) -> None:
        """
        Apply random weights perturbation to the model, then select threshold 'tau' that maximizes phi

        In options you can specify that your model already outputs probabilities, in which case the model will not apply the softmax function
        options = {'outputs_are_logits': False}

        To change perturbation parameters, you can pass the mean and std of the Gaussian noise
        options = {'mean': 1.0, 'std': 0.1}
        """
        assert 0 <= data_to_use <= 1 or isinstance(
            data_to_use, int
        ), "frac_of_batches_to_use must be in [0, 1] or an integer"
        assert T_iters > 0, "T_iters must be a positive integer"
        assert self.check_layer_name_exists(
            last_layer_name
        ), f"Layer name {last_layer_name} not found in the model"

        self.last_layer_name = last_layer_name
        self.epsilon = epsilon
        self.options = options
        self.bias_metric = bias_metric

        best_tau = tau_init
        best_model = deepcopy(self.model)
        best_phi = -1
        best_bias = -1

        # Unpack multiple batches of the dataloader
        self.X_torch, self.Y_true_torch, self.ProtAttr_torch = self.unpack_batches(
            dataloader, data_to_use
        )

        self.Y_true_np = self.Y_true_torch.detach().cpu().numpy()
        self.ProtAttr_np = self.ProtAttr_torch.detach().cpu().numpy()

        with tqdm(
            desc=f"Random Perturbation iterations (phi: {best_phi}, tau: {best_tau})",
            total=T_iters,
            file=sys.stdout,
        ) as pbar:
            # Randomly perturb the model weights
            for i in range(T_iters):
                self._perturb_weights(self.model, **options)

                # Optimize the threshold tau
                res: OptimizeResult = optimize.minimize_scalar(
                    self.objective_thresh("np", True),
                    bounds=(0, 1),
                    method="bounded",
                    options={"maxiter": optimizer_maxiter},
                )

                if res.success:
                    tau = res.x
                    phi = -res.fun
                    bias = self.phi_np(tau)[1]

                    logger.debug(f"tau: {tau:.3f}, phi: {phi:.3f}, bias: {bias:.3f}")

                    if phi > best_phi:
                        best_tau = tau
                        best_model = deepcopy(self.model)
                        best_phi = phi
                        best_bias = bias

                else:
                    logger.warning(f"Optimization failed: {res.message}")

                pbar.set_description(
                    f"Random Perturbation iterations (phi: {best_phi:.3f}, tau: {best_tau:.3f}, bias: {best_bias:.3f})"
                )
                pbar.update(1)

        self.model = best_model
        self.best_tau = best_tau

        if hasattr(self, "lightning_model"):
            self.lightning_model.model = best_model

        # Add a hook with the best transformation
        self.apply_hook(best_tau)

    def _perturb_weights(
        self, module: nn.Module, mean: float = 1.0, std: float = 0.1, **kwargs
    ) -> None:
        """
        Add Gaussian noise to the weights of the module by multiplying the weights with a number ~ N(mean, std)
        """
        with torch.no_grad():
            for param in module.parameters():
                param.data = param.data * torch.normal(
                    mean, std, param.data.shape, device=self.device
                )
