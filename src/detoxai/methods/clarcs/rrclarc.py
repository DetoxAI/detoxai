import logging
import types
from copy import deepcopy
from enum import Enum
from typing import Callable

import lightning as L
import torch

from .clarc import CLARC

_logger = logging.getLogger(__name__)


# Enum masking patterns
class RRMaskingPattern(Enum):
    """ """

    MAX_LOGIT = "max_logit"
    TARGET_LOGIT = "target_logit"
    ALL_LOGITS = "all_logits"
    ALL_LOGITS_RANDOM = "all_logits_random"
    LOGPROBS = "logprobs"


# Enum RR loss types
class RRLossType(Enum):
    """ """

    L2 = "l2"
    L1 = "l1"
    COSINE = "cosine"


class RRCLARC(CLARC):
    """ """

    def __init__(
        self,
        model: L.LightningModule,
        experiment_name: str,
        device: str,
        rr_config: dict = {},
        **kwargs,
    ) -> None:
        super().__init__(model, experiment_name, device)

        self.lambda_rr = rr_config.get("lambda_rr", 1.0)
        self.rr_loss_type = rr_config.get("rr_loss_type", RRLossType.L2)
        self.masking = rr_config.get("masking_pattern", RRMaskingPattern.MAX_LOGIT)
        self.target_class = rr_config.get("target_class", None)

    def apply_model_correction(
        self,
        cav_layers: list[str],
        dataloader: torch.utils.data.DataLoader,
        logger: object | bool = False,
        fine_tune_epochs: int = 1,
        ft_lr: float = 1e-3,
        **kwargs,
    ) -> None:
        """

        Args:
          cav_layers: list[str]:
          dataloader: torch.utils.data.DataLoader:
          logger: object | bool:  (Default value = False)
          fine_tune_epochs: int:  (Default value = 1)
          ft_lr: float:  (Default value = 1e-3)
          **kwargs:

        Returns:

        """
        assert len(cav_layers) == 1, "RR-CLARC only supports one CAV layer"
        self.cav_layer = cav_layers[0]

        # Register rr_clarc_hook and build list of module names in order
        cav_layer_found = False
        module_names_in_order = []
        for name, module in self.model.named_modules():
            if name:  # Skip empty string (root module)
                module_names_in_order.append(name)
            if name == self.cav_layer:
                hook_fn = self.rr_clarc_hook()
                handle = module.register_forward_hook(hook_fn)
                self.hooks.append(handle)
                _logger.debug(f"Added RR-CLARC hook to layer: {name}")
                cav_layer_found = True

        if not cav_layer_found:
            raise ValueError(f"CAV layer '{self.cav_layer}' not found in model")

        # Get the index of cav_layer and the set of modules to freeze (all before cav_layer)
        cav_layer_idx = module_names_in_order.index(self.cav_layer)
        modules_to_freeze = set(module_names_in_order[:cav_layer_idx])

        # Store original requires_grad state and freeze layers before cav_layer
        self.original_requires_grad = {}
        for name, param in self.model.named_parameters():
            self.original_requires_grad[name] = param.requires_grad

            # Check if this parameter belongs to a module that should be frozen
            # For parameter 'layer1.0.conv.weight', we check if 'layer1', 'layer1.0',
            # or 'layer1.0.conv' is in modules_to_freeze
            param_parts = name.rsplit(".", 1)[0] if "." in name else name
            should_freeze = False
            # Check all possible parent module paths
            parts = param_parts.split(".")
            for i in range(1, len(parts) + 1):
                module_path = ".".join(parts[:i])
                if module_path in modules_to_freeze:
                    should_freeze = True
                    break

            if should_freeze:
                param.requires_grad = False
                _logger.debug(f"Frozen parameter: {name}")

        # Override training_step in lightning model by modified_training_step
        clone_original_training_step = deepcopy(self.lightning_model.training_step)
        self.lightning_model.training_step = types.MethodType(
            self.modified_training_step(), self.lightning_model
        )

        def configure_optimizers(self):
            """ """
            # Only optimize parameters that require gradients (from cav_layer onwards)
            params_to_optimize = [p for p in self.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(params_to_optimize, lr=ft_lr)
            return optimizer

        self.lightning_model.configure_optimizers = types.MethodType(
            configure_optimizers, self.lightning_model
        )

        # Make sure model is in training mode
        self.lightning_model.train()

        trainer = L.Trainer(
            max_epochs=fine_tune_epochs,
            logger=logger,
            log_every_n_steps=1,
            enable_model_summary=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
            devices=self.devices_indices,
        )

        trainer.fit(self.lightning_model, dataloader)

        # Go back to eval mode
        self.lightning_model.eval()

        # Restore original requires_grad state
        for name, param in self.model.named_parameters():
            if name in self.original_requires_grad:
                param.requires_grad = self.original_requires_grad[name]

        # Remove hooks
        self.remove_hooks()

        # Restore original training_step
        self.lightning_model.training_step = clone_original_training_step

    def rr_clarc_hook(self) -> Callable:
        """ """

        def hook(m, i, output):
            """

            Args:
              m:
              i:
              output:

            Returns:

            """
            self.intermediate_a = output
            return output

        return hook

    def masked_criterion(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """

        Args:
          y_hat: torch.Tensor:
          y: torch.Tensor:

        Returns:

        """
        match self.masking:
            case RRMaskingPattern.MAX_LOGIT:
                return y_hat.max(1)[0]
            case RRMaskingPattern.TARGET_LOGIT:
                target_class = self.target_class
                return y_hat[range(len(y)), target_class]
            case RRMaskingPattern.ALL_LOGITS:
                return (y_hat).sum(1)
            case RRMaskingPattern.ALL_LOGITS_RANDOM:
                return (y_hat * torch.sign(0.5 - torch.rand_like(y_hat))).sum(1)
            case RRMaskingPattern.LOGPROBS:
                return (y_hat.softmax(1) + 1e-5).log().mean(1)
            case _:
                raise ValueError(f"Invalid masking pattern: {self.masking}")

    def rr_loss(self, gradient: torch.Tensor) -> torch.Tensor:
        """

        Args:
          gradient: torch.Tensor:

        Returns:

        """
        cav = self.cav[self.cav_layer]

        # TODO: Figure out what it was
        # if "mean" in self.rr_loss_type and gradient.dim() != 2:
        #   gradient = gradient.mean((2, 3), keepdim=True).expand_as(gradient)

        # TODO: This too
        # g_flat = gradient.permute(1, 0, 2, 3).flatten(start_dim=1).permute(1, 0)
        g_flat = gradient.flatten(start_dim=1)

        match self.rr_loss_type:
            case RRLossType.COSINE:
                return torch.nn.functional.cosine_similarity(g_flat, cav).abs().mean(0)
            case RRLossType.L2:
                return ((g_flat * cav).sum(1) ** 2).mean(0)
            case RRLossType.L1:
                return (g_flat * cav).sum(1).abs().mean(0)
            case _:
                raise NotImplementedError

    def modified_training_step(self) -> Callable:
        """ """

        def training_step(lightning_obj, batch, batch_idx):
            """

            Args:
              lightning_obj:
              batch:
              batch_idx:

            Returns:

            """
            with torch.enable_grad():
                x = batch[0]
                y = batch[1]

                y_hat = lightning_obj.model(x)  # logits

                rr_y_hat = self.masked_criterion(y_hat, y)

                rr_grad = torch.autograd.grad(
                    rr_y_hat,
                    self.intermediate_a,
                    create_graph=True,
                    retain_graph=True,
                    grad_outputs=torch.ones_like(rr_y_hat),
                )[0]

                rr_loss = self.rr_loss(rr_grad)

                loss = lightning_obj.criterion(y_hat, y) + self.lambda_rr * rr_loss

            lightning_obj.log("train_loss", loss)
            return {"loss": loss, "preds": y_hat}

        return training_step
