import sys
import torch
import lightning as L
import logging
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from copy import deepcopy

from scipy import optimize
from scipy.optimize import OptimizeResult
from tqdm import tqdm
from skopt import gbrt_minimize, gp_minimize, dummy_minimize, forest_minimize
from skopt.space import Real

# Project imports
from .savani_base import SavaniBase
from ...metrics.bias_metrics import BiasMetrics

logger = logging.getLogger(__name__)


def flatten_with_map(arr, indices):
    return arr[indices].flatten()


def unflatten_with_map(original, flat_arr, indices):
    original[indices] = flat_arr.reshape(original[indices].shape)
    return original


class SavaniLWO(SavaniBase):
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
        bias_metric: BiasMetrics | str = BiasMetrics.DP_GAP,
        data_to_use: float | int = 128,
        n_layers_to_optimize: int | str = "all",
        optimizer_maxiter: int = 10,
        thresh_optimizer_maxiter: int = 100,
        beta: float = 2.2,
        neuron_frac: float = 0.1,
        tau_init: float = 0.5,
        options: dict = {},
        **kwargs,
    ) -> None:
        """
        Do layer-wise optimization to find the best weights for each layer and the best threshold tau

        In options you can specify that your model already outputs probabilities, in which case the model will not apply the softmax function
        options = {'outputs_are_logits': False}

        """
        assert 0 <= data_to_use <= 1 or isinstance(
            data_to_use, int
        ), "frac_of_batches_to_use must be in [0, 1] or an integer"
        assert self.check_layer_name_exists(
            last_layer_name
        ), f"Layer name {last_layer_name} not found in the model"

        self.last_layer_name = last_layer_name
        self.tau_init = tau_init
        self.epsilon = epsilon
        self.bias_metric = bias_metric
        self.options = options

        best_tau = tau_init
        best_model = deepcopy(self.model)
        best_phi = -1
        best_bias = 1

        # Unpack multiple batches of the dataloader
        self.X_torch, self.Y_true_torch, self.ProtAttr_torch = self.unpack_batches(
            dataloader, data_to_use
        )

        total_layers = len(list(self.model.parameters()))
        if n_layers_to_optimize == "all":
            n_layers_to_optimize = total_layers
        assert (
            n_layers_to_optimize <= total_layers
        ), "n_layers_to_optimize must be less than the total number of layers"

        with tqdm(
            desc=f"LWO layer -1 (global phi: {best_phi:.3f}, tau: {best_tau:.3f}, bias: {best_bias:.3f})",
            total=n_layers_to_optimize,
            file=sys.stdout,
        ) as pbar:
            for i, parameters in enumerate(self.model.parameters()):
                # We're optimizing the last n_layers_to_optimize layers
                if i < total_layers - n_layers_to_optimize:
                    logger.debug(f"Skipping layer {i}")
                    continue

                self.parameters_np = parameters.detach().cpu().numpy()
                std = self.parameters_np.std()
                n = max(int(neuron_frac * len(self.parameters_np)), 1)
                # Cap the number of neurons to optimize, this is useful for large models
                if "max_neurons_to_optimize" in options:
                    n = min(n, options["max_neurons_to_optimize"])
                logger.debug(
                    f"Optimizing layer {i} with {n} out of {len(self.parameters_np)} neurons"
                )
                self.idx = np.random.choice(len(self.parameters_np), n, replace=False)

                flat_parameters = flatten_with_map(self.parameters_np, self.idx)

                space = [
                    Real(
                        x - beta * std,
                        x + beta * std,
                    )
                    for x in flat_parameters
                ]

                res = forest_minimize(
                    self.objective_LWO(parameters, tau_init),
                    space,
                    n_calls=optimizer_maxiter,
                )

                if -res.fun > best_phi:
                    best_params = res.x

                    # Update the weights
                    with torch.no_grad():
                        parameters.data = torch.tensor(
                            unflatten_with_map(
                                self.parameters_np, np.array(best_params), self.idx
                            ),
                            device=self.device,
                        )

                    # Optimize the threshold tau
                    res: OptimizeResult = optimize.minimize_scalar(
                        self.objective_thresh("torch", True),
                        bounds=(0, 1),
                        method="bounded",
                        options={"maxiter": thresh_optimizer_maxiter},
                    )

                    if res.success:
                        tau = res.x
                        _phi = -res.fun
                        bias = self.phi_torch(tau)[1].detach().cpu().numpy()
                        logger.debug(
                            f"tau: {tau:.3f}, phi: {_phi:.3f}, bias: {bias:.3f}"
                        )

                        if _phi > best_phi:
                            best_tau = tau
                            best_model = deepcopy(self.model)
                            best_phi = _phi
                            best_bias = bias

                    else:
                        logger.info(f"Optimization failed: {res.message}")

                pbar.update(1)
                pbar.set_description(
                    f"LWO layer {i} (global phi: {best_phi:.3f}, tau: {best_tau:.3f}, bias: {best_bias:.3f})"
                )

        self.model = best_model
        self.best_tau = best_tau

        if hasattr(self, "lightning_model"):
            self.lightning_model.model = best_model

        # Add a hook with the best transformation
        self.apply_hook(best_tau)

    def objective_LWO(self, parameters, tau):
        def objective(new_parameters) -> float:
            nonlocal tau
            ps = unflatten_with_map(
                self.parameters_np, np.array(new_parameters), self.idx
            )

            # Update the weights
            with torch.no_grad():
                parameters.data = torch.tensor(ps, device=self.device)

            phi, _ = self.phi_torch(tau)

            return -phi.detach().cpu().numpy()

        return objective
