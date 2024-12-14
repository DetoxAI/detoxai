import numpy as np
import sys
import torch
import lightning as L
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy
from scipy import optimize
from scipy.optimize import OptimizeResult
from tqdm import tqdm
from torch.nn.functional import softmax, sigmoid

# Project imports
from ..model_correction import ModelCorrectionMethod
from .utils import BiasMetrics, phi_np


class SavaniRP(ModelCorrectionMethod):
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
        bias_metric: BiasMetrics | str = BiasMetrics.EO_GAP,
        frac_of_batches_to_use: float = 1.0,
        optimizer_maxiter: int = 10,
        options: dict = {},
    ) -> None:
        """
        Apply random weights perturbation to the model, then select threshold 'tau' that maximizes phi

        In options you can specify that your model already outputs probabilities, in which case the model will not apply the softmax function
        options = {'outputs_are_logits': False}

        To change perturbation parameters, you can pass the mean and std of the Gaussian noise
        options = {'mean': 1.0, 'std': 0.1}
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
        Y_true_np = Y_true.detach().cpu().numpy()
        ProtAttr_np = ProtAttr.detach().cpu().numpy()

        with tqdm(
            desc=f"Random Perturbation iterations (phi: {best_phi}, tau: {best_tau})",
            total=T_iters,
            file=sys.stdout,
        ) as pbar:
            # Randomly perturb the model weights
            for i in range(T_iters):
                self._perturb_weights(self.model, **options)

                # Compute phi
                with torch.no_grad():
                    # Assuming binary classification and logits
                    y_raw_preds = self.model(X)
                    if options.get("outputs_are_logits", True):
                        y_probs = softmax(y_raw_preds, dim=1)
                    else:
                        y_probs = y_raw_preds
                    y_preds = y_probs[:, 1]  # Probability of class 1
                    y_preds_np = y_preds.detach().cpu().numpy()

                def objective(tau):
                    return -phi_np(
                        Y_true_np, y_preds_np > tau, ProtAttr_np, epsilon, bias_metric
                    )[0]

                # for _tau in np.linspace(0, 1, optimizer_maxiter):
                #     _phi, bias = objective(_tau)

                #     print(f"tau: {_tau:.3f}, phi: {_phi:.3f}, bias: {bias:.3f}")

                #     if _phi > best_phi:
                #         best_tau = _tau
                #         best_model = deepcopy(self.model)
                #         best_phi = _phi
                #         best_bias = bias

                # Optimize the threshold tau
                res: OptimizeResult = optimize.minimize_scalar(
                    objective,
                    bounds=(0, 1),
                    method="bounded",
                    options={"maxiter": optimizer_maxiter},
                )

                if res.success:
                    tau = res.x
                    phi = -res.fun
                    bias = phi_np(
                        Y_true_np, y_preds_np > tau, ProtAttr_np, epsilon, bias_metric
                    )[1]
                    print(f"tau: {tau:.3f}, phi: {phi:.3f}, bias: {bias:.3f}")

                    if phi > best_phi:
                        best_tau = tau
                        best_model = deepcopy(self.model)
                        best_phi = phi
                        best_bias = bias

                else:
                    print(f"Optimization failed: {res.message}")

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

    def _sigmoid_np(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def get_corrected_model(self) -> L.LightningModule | nn.Module:
        if hasattr(self, "lightning_model"):
            return self.lightning_model
        else:
            return self.model

    def apply_hook(self, tau: float) -> None:
        def hook(module, input, output):
            # output = (output > tau).int() # doesn't allow gradients to flow
            # Assuming binary classification
            output[:, 1] = sigmoid((output[:, 1] - tau) * 10)  # soft thresholding
            output[:, 0] = 1 - output[:, 1]
            print(f"Hook applied, threshold: {tau}")
            return output

        hook_fn = hook

        # Register the hook on the model
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name == self.last_layer_name:
                handle = module.register_forward_hook(hook_fn)
                print(f"Hook registered on layer: {name}")
                hooks.append(handle)

        self.hooks = hooks
