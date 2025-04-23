import logging
import sys
from copy import deepcopy

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from skopt import forest_minimize  # noqa

# from sklearn.ensemble import RandomForestRegressor
from skopt.learning.forest import RandomForestRegressor
from skopt.space import Real
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...metrics.bias_metrics import BiasMetrics

# Project imports
from .savani_base import SavaniBase

logger = logging.getLogger(__name__)


class SavaniLWO(SavaniBase):
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
        bias_metric: BiasMetrics | str = BiasMetrics.EO_GAP,
        n_layers_to_optimize: int | str = "all",
        thresh_optimizer_maxiter: int = 100,
        beta: float = 2.2,
        params_to_opt: int | float = 0.5,
        never_more_than: int = 50_000,
        tau_init: float = 0.5,
        outputs_are_logits: bool = True,
        n_eval_batches: int = 3,
        eval_batch_size: int = 128,
        skopt_verbose: bool = False,
        skopt_njobs: int = 4,
        skopt_npoints: int = 1000,
        skopt_maxiter: int = 10,
        soft_thresh_temperature: float = 10.0,
        **kwargs,
    ) -> None:
        """Do layer-wise optimization to find the best weights for each layer and the best threshold tau

        Args:
          dataloader: DataLoader:
          last_layer_name: str:
          epsilon: float:  (Default value = 0.1)
          bias_metric: BiasMetrics | str:  (Default value = BiasMetrics.EO_GAP)
          n_layers_to_optimize: int | str:  (Default value = "all")
          thresh_optimizer_maxiter: int:  (Default value = 100)
          beta: float:  (Default value = 2.2)
          params_to_opt: int | float:  (Default value = 0.5)
          never_more_than: int:  (Default value = 50_000)
          tau_init: float:  (Default value = 0.5)
          outputs_are_logits: bool:  (Default value = True)
          n_eval_batches: int:  (Default value = 3)
          eval_batch_size: int:  (Default value = 128)
          skopt_verbose: bool:  (Default value = False)
          skopt_njobs: int:  (Default value = 4)
          skopt_npoints: int:  (Default value = 1000)
          skopt_maxiter: int:  (Default value = 10)
          soft_thresh_temperature: float:  (Default value = 10.0)
          **kwargs:

        Returns:

        """
        assert self.check_layer_name_exists(last_layer_name), (
            f"Layer name {last_layer_name} not found in the model"
        )

        self.last_layer_name = last_layer_name
        self.tau_init = tau_init
        self.epsilon = epsilon
        self.bias_metric = bias_metric
        self.outputs_are_logits = outputs_are_logits
        self.n_eval_batches = n_eval_batches

        self.initialize_dataloader(dataloader, eval_batch_size)

        best_model = deepcopy(self.model)

        best_phi, best_tau = self.optimize_tau(tau_init, thresh_optimizer_maxiter)

        total_layers = len(list(self.model.parameters()))
        if n_layers_to_optimize == "all":
            n_layers_to_optimize = total_layers
        assert n_layers_to_optimize <= total_layers, (
            "n_layers_to_optimize must be less than the total number of layers"
        )

        with tqdm(
            desc=f"LWO layer -1 (global phi: {best_phi:.3f}, tau: {best_tau:.3f})",
            total=n_layers_to_optimize,
            file=sys.stdout,
        ) as pbar:
            for i, (name, o_params) in enumerate(self.model.named_parameters()):
                # We're optimizing the last n_layers_to_optimize layers
                # -3 to avoid the last layer (2 outputs) weights and bias, then to avoid second to last layer's bias, we dont want to optimize bias as it doesn't make sense
                if i < total_layers - n_layers_to_optimize - 1 or i >= total_layers - 1:
                    continue

                logger.debug(f"Optimizing {name} layer ({i})")

                total_params_cnt = o_params.numel()

                if isinstance(params_to_opt, float):
                    n = max(int(params_to_opt * total_params_cnt), 1)
                else:
                    n = params_to_opt

                    if n > total_params_cnt:
                        n = total_params_cnt
                        logger.info(
                            f"Even though you asked for {params_to_opt} of the parameters, we're capping it to {total_params_cnt}"
                        )

                # Cap the number of neurons to optimize, this is useful for large models
                # Otherwise skopt will literally kill your machine
                if n > never_more_than:
                    n = never_more_than
                    logger.info(
                        f"Even though you asked for {params_to_opt} of the parameters, we're capping it to {never_more_than}"
                    )

                # Cap the number of neurons to optimize, this is useful for large models
                logger.debug(f"Optimizing lay. {i} w. {n}/{total_params_cnt} params")

                sel_params, indices = self.flatten_select(o_params, n, total_params_cnt)

                logging.debug(f"Flattened parameters cnt: {sel_params.numel()}")

                std = o_params.std().detach().cpu().numpy()
                space = [
                    Real(
                        x - beta * std,
                        x + beta * std,
                    )
                    for x in sel_params.detach().cpu().numpy()
                ]

                logger.debug(f"Optimizing {len(space)} parameters")

                regressor = RandomForestRegressor(
                    n_estimators=50,
                    n_jobs=skopt_njobs,
                    max_depth=10,
                    verbose=skopt_verbose,
                    min_samples_leaf=2,
                    random_state=self.seed,
                    min_impurity_decrease=1e-4,
                )

                res = forest_minimize(
                    self.objective_LWO(o_params, best_tau, indices),
                    dimensions=space,
                    base_estimator=regressor,
                    n_calls=skopt_maxiter,
                    n_jobs=skopt_njobs,
                    random_state=self.seed,
                    n_points=skopt_npoints,
                    verbose=skopt_verbose,
                )

                if -res.fun > best_phi:
                    best_p: list = res.x
                    best_p_t = torch.tensor(best_p, device=self.device)

                    # Update the weights
                    with torch.no_grad():
                        o_params.data = self.unflatten(o_params, best_p_t, indices)

                    tau, phi = self.optimize_tau(tau_init, thresh_optimizer_maxiter)

                    if phi > best_phi:
                        best_phi = phi
                        best_tau = tau
                        best_model = deepcopy(self.model)
                        logger.debug(f"New best phi: {best_phi}, best tau: {best_tau}")

                pbar.update(1)
                pbar.set_description(
                    f"LWO layer {i} (global phi: {best_phi:.3f}, tau: {best_tau:.3f})"
                )

        self.model = best_model
        self.best_tau = best_tau

        if hasattr(self, "lightning_model"):
            self.lightning_model.model = best_model

        # Add a hook with the best transformation
        self.apply_hook(best_tau, soft_thresh_temperature)

    def objective_LWO(
        self, o_params: torch.Tensor, tau: float, indices: list
    ) -> callable:
        """Objective function for the layer-wise optimization

        Args:
          o_params: The original parameters (torch.Tensor)
          tau: The threshold value (float)
          indices: The indices of the selected neurons (list)
          o_params: torch.Tensor:
          tau: float:
          indices: list:

        Returns:
          : The objective function

        """

        if not isinstance(tau, torch.Tensor):
            tau = torch.tensor(tau, device=self.device)

        def objective(new_params: list) -> float:
            """

            Args:
              new_params: list:

            Returns:

            """
            nonlocal tau, o_params, indices

            # Update the weights
            with torch.no_grad():
                np_trch = torch.tensor(new_params, device=self.device)
                o_params.data = self.unflatten(o_params, np_trch, indices)

            phi, _ = self.phi_torch(tau)

            return -phi.detach().cpu().numpy()

        return objective

    def flatten_select(
        self, params: torch.Tensor, select_cnt: float | int, total_params: int
    ) -> tuple[torch.Tensor, list]:
        """Take an n-dimensional array,

        Args:

        Args:
          select_cnt: The number of neurons to select
          total_params: The total number of parameters
          params: torch.Tensor:
          select_cnt: float | int:
          total_params: int:

        Returns:
          A 1-dimensional array of selected neurons
          A 1-dimensional array of indices of the selected neurons

        """

        if isinstance(select_cnt, float):
            select_cnt = int(select_cnt * total_params)
        assert select_cnt <= total_params, (
            "select_cnt must be less than the total number of parameters"
        )

        indices = np.random.choice(total_params, select_cnt, replace=False)
        indices = list(indices)

        return params.flatten()[indices], indices

    def unflatten(
        self, o_params: torch.Tensor, f_params: torch.Tensor, indices: list
    ) -> torch.Tensor:
        """Unflatten the parameters

        Args:
          o_params: The original parameters
          f_params: The flattened parameters
          indices: The indices of the selected neurons
          o_params: torch.Tensor:
          f_params: torch.Tensor:
          indices: list:

        Returns:
          : The unflattened parameters

        """
        o_shape = o_params.shape
        o_params = o_params.flatten()
        o_params[indices] = f_params
        return o_params.reshape(o_shape)
