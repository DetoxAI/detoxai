import torch
import lightning as L
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy
from scipy import optimize
from scipy.optimize import OptimizeResult
from tqdm import tqdm

# Project imports
from ..model_correction import ModelCorrectionMethod
from .utils import BiasMetrics, phi_np


class RandomPerturbation(ModelCorrectionMethod):
    def __init__(
        self, model: nn.Module | L.LightningModule, experiment_name: str, device: str
    ) -> None:
        super().__init__(model, experiment_name, device)
        if isinstance(model, L.LightningModule):
            self.lightning_model = model

    def apply_model_correction(
        self,
        dataloader: DataLoader,
        last_layer_name: str,
        epsilon: float = 0.05,
        T_iters: int = 10,
        bias_metric: BiasMetrics | str = BiasMetrics.TPR_GAP,
        frac_of_batches_to_use: float = 1.0,
        optimizer_maxiter: int = 500,
    ) -> None:
        """
        Apply random weights perturbation to the model, then select threshold 'tau' that maximizes phi
        """
        assert (
            0 <= frac_of_batches_to_use <= 1
        ), "frac_of_batches_to_use must be in [0, 1]"
        assert T_iters > 0, "T_iters must be a positive integer"
        assert self._check_layer_name_exists(
            last_layer_name
        ), f"Layer name {last_layer_name} not found in the model"

        self.last_layer_name = last_layer_name

        best_tau = None
        best_model = deepcopy(self.model)
        best_phi = -1

        # Unpack multiple batches of the dataloader
        X, Y_true, ProtAttr = self._unpack_batches(dataloader, frac_of_batches_to_use)

        with tqdm(
            desc=f"Random Perturbation iterations (phi: {best_phi}, tau: {best_tau})",
            total=T_iters,
        ) as pbar:
            # Randomly perturb the model weights
            for i in range(T_iters):
                self._perturb_weights(self.model)

                # Compute phi
                y_raw_preds = self.model(X)[:, 0]  # Assuming binary classification task

                def objective(tau):
                    y_preds = (y_raw_preds > tau).detach().cpu().numpy()
                    return -phi_np(Y_true, y_preds, ProtAttr, epsilon, bias_metric)

                result: OptimizeResult = optimize.minimize_scalar(
                    objective,
                    bounds=(0, 1),
                    method="bounded",
                    options={"maxiter": optimizer_maxiter},
                )

                if -result["x"] > best_phi:
                    best_tau = result["x"]
                    best_model = deepcopy(self.model)
                    best_phi = -result["fun"]

                pbar.set_description(
                    f"Random Perturbation iterations (phi: {best_phi:.3f}, tau: {best_tau:.3f})"
                )
                pbar.update(1)

        self.model = best_model
        self.best_tau = best_tau

        # Add a hook with the best transformation
        self.apply_hook(best_tau)

    def _perturb_weights(
        self, module: nn.Module, mean: float = 1.0, std: float = 0.1
    ) -> None:
        """
        Add Gaussian noise to the weights of the module by multiplying the weights with a number ~ N(mean, std)
        """
        with torch.no_grad():
            for param in module.parameters():
                param.data = param.data * torch.normal(
                    mean, std, param.data.shape, device=self.device
                )

    def _unpack_batches(
        self, dataloader: DataLoader, frac: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X, Y_true, ProtAttr = [], [], []
        all_batches = len(dataloader)
        n_batches = int(all_batches * frac)
        for i, batch in enumerate(dataloader):
            X.append(batch[0])
            Y_true.append(batch[1])
            ProtAttr.append(batch[2])
            if i == n_batches:
                break
        X = torch.cat(X)
        Y_true = torch.cat(Y_true)
        ProtAttr = torch.cat(ProtAttr)
        return X, Y_true, ProtAttr

    def _check_layer_name_exists(self, layer_name: str) -> bool:
        for name, _ in self.model.named_modules():
            if name == layer_name:
                return True
        return False

    def get_corrected_model(self) -> L.LightningModule | nn.Module:
        if hasattr(self, "lightning_model"):
            return self.lightning_model
        else:
            return self.model

    def apply_hook(self, tau: float) -> None:
        def hook(module, input, output):
            output = (output > tau).int()
            return output

        hook_fn = hook

        # Register the hook on the model
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name == self.last_layer_name:
                handle = module.register_forward_hook(hook_fn)
                hooks.append(handle)

        self.hooks = hooks
