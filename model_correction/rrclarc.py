from enum import Enum
from typing import Callable
import lightning as L
import torch
from copy import deepcopy

from .base_model_correction import ModelCorrectionMethod


# Enum masking patterns
class MaskingPattern(Enum):
    MAX_LOGIT = "max_logit"
    TARGET_LOGIT = "target_logit"
    ALL_LOGITS = "all_logits"
    ALL_LOGITS_RANDOM = "all_logits_random"
    LOGPROBS = "logprobs"


# Enum RR loss types
class RRLossType(Enum):
    L2 = "l2"
    L1 = "l1"
    COSINE = "cosine"


class RRCLARC(ModelCorrectionMethod):
    def __init__(
        self,
        model,
        dataloader,
        experiment_name,
        layers,
        device,
        rr_config: dict,
    ):
        super().__init__(model, dataloader, experiment_name, layers, device)

        self.lambda_rr = rr_config.get("lambda_rr", 1.0)
        self.rr_loss_type = rr_config.get("rr_loss_type", RRLossType.L2)
        self.masking = rr_config.get("masking_pattern", MaskingPattern.MAX_LOGIT)

    def apply_model_correction(
        self,
        cav_layer: str,
        lightning_model: L.LightningModule,
        dataloader_train: torch.utils.data.DataLoader,
        logger: object,
        fine_tune_epochs: int = 1,
        alpha: float = 1.0,
    ) -> None:
        # Register rr_clarc_hook
        for name, module in self.model.named_modules():
            if name == cav_layer:
                hook_fn = self.rr_clarc_hook()
                handle = module.register_forward_hook(hook_fn)
                self.hooks.append(handle)
                print(f"DEBUG: Added RR-CLARC hook to layer: {name}")

        # Override training_step in lightning model by modified_training_step
        clone_original_training_step = deepcopy(lightning_model.training_step)
        lightning_model.training_step = self.modified_training_step()

        # Make sure model is in training mode
        self.model.train()

        trainer = L.Trainer(
            max_epochs=fine_tune_epochs, logger=logger, log_every_n_steps=1
        )
        trainer.fit(lightning_model, dataloader_train)

        # Go back to eval mode
        self.model.eval()

        # Remove hooks
        self.remove_hooks()

        # Restore original training_step
        lightning_model.training_step = clone_original_training_step

    def rr_clarc_hook(self) -> Callable:
        def hook(m, i, output):
            self.intermediate_a = output.clone()
            return output

        return hook

    def masked_criterion(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        match self.masking:
            case MaskingPattern.MAX_LOGIT:
                return y_hat.max(1)[0]
            case MaskingPattern.TARGET_LOGIT:
                target_class = self.config.get("target_class", y)
                return y_hat[range(len(y)), target_class]
            case MaskingPattern.ALL_LOGITS:
                return (y_hat).sum(1)
            case MaskingPattern.ALL_LOGITS_RANDOM:
                return (y_hat * torch.sign(0.5 - torch.rand_like(y_hat))).sum(1)
            case MaskingPattern.LOGPROBS:
                return (y_hat.softmax(1) + 1e-5).log().mean(1)
            case _:
                raise NotImplementedError

    def rr_loss(self, gradient: torch.Tensor) -> torch.Tensor:
        cav = self.cav.to(gradient)

        # TODO: Figure out what it is
        if "mean" in self.aggregation and gradient.dim() != 2:
            gradient = gradient.mean((2, 3), keepdim=True).expand_as(gradient)

        # TODO: This too
        g_flat = gradient.permute(1, 0, 2, 3).flatten(start_dim=1).permute(1, 0)

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
        def training_step(lightning_obj, batch, batch_idx):
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
