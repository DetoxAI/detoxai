import numpy as np

import logging
import torch
import lightning as L
import torch.nn as nn
from torch.utils.data import DataLoader

from torch.nn.functional import softmax, sigmoid

from abc import ABC, abstractmethod

# Project imports
from ..model_correction import ModelCorrectionMethod
from ...utils.dataloader import DetoxaiDataLoader
from .utils import phi_torch

logger = logging.getLogger(__name__)


class SavaniBase(ModelCorrectionMethod, ABC):
    def __init__(
        self,
        model: nn.Module | L.LightningModule,
        experiment_name: str,
        device: str,
        seed: int = 123,
    ) -> None:
        super().__init__(model, experiment_name, device)

        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    @abstractmethod
    def apply_model_correction(self) -> None:
        raise NotImplementedError

    def optimize_tau(
        self, tau_init: float, thresh_optimizer_maxiter: int
    ) -> tuple[float, float]:
        objective_fn = self.objective_thresh("torch", True, "max")

        best_phi = 1e-6
        tau = tau_init

        for _tau in torch.linspace(0, 1, thresh_optimizer_maxiter):
            phi = objective_fn(_tau)

            if phi > best_phi:
                best_phi = phi
                tau = _tau

        return tau, best_phi

    def objective_thresh(
        self, backend: str, cache_preds: bool = True, direction: str = "min"
    ) -> callable:
        if cache_preds:
            y_probs, y_true, prot_attr = self.get_pred_true_prot(self.internal_dl)
            y_preds = y_probs[:, 1]

        if direction == "min":
            d_mul = -1
        elif direction == "max":
            d_mul = 1
        else:
            raise ValueError(f"Direction {direction} not supported")

        if backend == "torch":

            def objective(tau):
                phi, _ = self.phi_torch(tau, (y_preds, y_true, prot_attr))
                return phi.detach().cpu().numpy() * d_mul
        elif backend == "np":
            raise NotImplementedError("Numpy backend not implemented")
        else:
            raise ValueError(f"Backend {backend} not supported")

        return objective

    def phi_torch(
        self, tau: torch.Tensor, cached: tuple | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the phi metric for a given threshold tau
        """
        if cached is None:
            y_probs, y_true, prot_attr = self.get_pred_true_prot(self.internal_dl)
            y_preds = y_probs[:, 1]
        else:
            y_preds, y_true, prot_attr = cached

            return phi_torch(
                y_true,
                y_preds > tau.to(self.device),
                prot_attr,
                self.epsilon,
                self.bias_metric,
            )

    def apply_hook(self, tau: float) -> None:
        def hook(module, input, output):
            # output = (output > tau).int() # doesn't allow gradients to flow
            # Assuming binary classification

            if self.outputs_are_logits:
                probs = softmax(output, dim=1)
                output[:, 1] = sigmoid((probs[:, 1] - tau) * 10)  # soft thresholding
                output[:, 0] = 1 - output[:, 1]
            else:
                output[:, 1] = sigmoid((output[:, 1] - tau) * 10)  # soft thresholding
                output[:, 0] = 1 - output[:, 1]

            # logger.debug(f"Savani hook fired in layer: {module}")

            return output

        hook_fn = hook

        # Register the hook on the model
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name == self.last_layer_name:
                handle = module.register_forward_hook(hook_fn)
                logger.debug(f"Hook registered on layer: {name}")
                hooks.append(handle)

        self.hooks = hooks

    def get_pred_true_prot(
        self, dataloader: DataLoader
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Y_preds, Y_true, ProtAttr = [], [], []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if self.max_batches_eval is not None and i >= self.max_batches_eval:
                    break
                x, y_true, prot = batch
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                prot = prot.to(self.device)

                y_logit = self.model(x)

                if self.outputs_are_logits:
                    y_probs = softmax(y_logit, dim=1)
                else:
                    y_probs = y_logit

                Y_preds.append(y_probs)
                Y_true.append(y_true)
                ProtAttr.append(prot)

            Y_preds = torch.cat(Y_preds).to(self.device)
            Y_true = torch.cat(Y_true).to(self.device)
            ProtAttr = torch.cat(ProtAttr).to(self.device)

        return Y_preds, Y_true, ProtAttr

    def check_layer_name_exists(self, layer_name: str) -> bool:
        for name, _ in self.model.named_modules():
            if name == layer_name:
                return True
        return False

    # def unpack_batches(
    #     self, dataloader: DataLoader, frac: float | int
    # ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """
    #     frac can be either an integer or a float
    #     If frac is an integer, it will return that many samples
    #     If frac is a float, it will return that fraction of the batches available in the dataloader
    #     """
    #     X, Y_true, ProtAttr = [], [], []
    #     all_batches = len(dataloader)

    #     if isinstance(frac, int):
    #         n_batches = frac
    #     else:
    #         n_batches = max(int(all_batches * frac), 1)

    #     n = 0

    #     for i, batch in enumerate(dataloader):
    #         X.append(batch[0])
    #         Y_true.append(batch[1])
    #         ProtAttr.append(batch[2])

    #         n += len(batch[0])

    #         if n >= n_batches:
    #             break

    #         if i == n_batches and isinstance(frac, float):
    #             break

    #     X = torch.cat(X).to(self.device)
    #     Y_true = torch.cat(Y_true).to(self.device)
    #     ProtAttr = torch.cat(ProtAttr).to(self.device)

    #     # Shave off the extra samples
    #     if isinstance(frac, int):
    #         X = X[:frac]
    #         Y_true = Y_true[:frac]
    #         ProtAttr = ProtAttr[:frac]

    #     return X, Y_true, ProtAttr

    def sample_minibatch(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = torch.randperm(self.X_torch.shape[0])[:batch_size]
        return self.X_torch[idx], self.Y_true_torch[idx], self.ProtAttr_torch[idx]

    def sample_batch(
        self, idx: int = -1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx == -1:
            idx = np.random.randint(0, len(self.internal_dl))
        else:
            idx = idx % len(self.internal_dl)

        x, y, p = self.internal_dl.get_nth_batch2(idx)
        return x.to(self.device), y.to(self.device), p.to(self.device)
