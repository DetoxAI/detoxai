import logging

import lightning as L
import torch
import torch.nn as nn
from torch import autograd
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...metrics.bias_metrics import (
    BiasMetrics,
)

# Project imports
from .savani_base import SavaniBase

logger = logging.getLogger(__name__)


class ZhangM(SavaniBase):
    """Brian Hu Zhang, Blake Lemoine, Margaret Mitchell - "Mitigating unwanted biases with adversarial learning"""

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
        iterations: int = 5,
        critic_iterations: int = 5,
        model_iterations: int = 2,
        train_batch_size: int = 128,
        thresh_optimizer_maxiter: int = 100,
        tau_init: float = 0.5,
        # alpha: float = 5.0,
        critic_lr: float = 2e-4,
        model_lr: float = 1e-4,
        critic_linear: list[int] = [256, 256, 256],
        outputs_are_logits: bool = True,
        n_eval_batches: int = 3,
        soft_thresh_temperature: float = 10.0,
        **kwargs,
    ) -> None:
        """backward
        Do layer-wise optimization to find the best weights for each layer and the best threshold tau

        In options you can specify that your model already outputs probabilities, in which case the model will not apply the softmax function
        options = {'outputs_are_logits': False}

        Args:
          dataloader: DataLoader:
          last_layer_name: str:
          epsilon: float:  (Default value = 0.1)
          bias_metric: BiasMetrics | str:  (Default value = BiasMetrics.EO_GAP)
          iterations: int:  (Default value = 5)
          critic_iterations: int:  (Default value = 5)
          model_iterations: int:  (Default value = 2)
          train_batch_size: int:  (Default value = 128)
          thresh_optimizer_maxiter: int:  (Default value = 100)
          tau_init: float:  (Default value = 0.5)
          # alpha: float:  (Default value = 5.0)
          critic_lr: float:  (Default value = 2e-4)
          model_lr: float:  (Default value = 1e-4)
          critic_linear: list[int]:  (Default value = [256)
          256:
          256]:
          outputs_are_logits: bool:  (Default value = True)
          n_eval_batches: int:  (Default value = 3)
          soft_thresh_temperature: float:  (Default value = 10.0)
          **kwargs:

        Returns:

        """
        assert self.check_layer_name_exists(last_layer_name), (
            f"Layer name {last_layer_name} not found in the model"
        )

        assert outputs_are_logits, "Only logits are supported at the moment"

        self.last_layer_name = last_layer_name
        self.tau_init = tau_init
        self.epsilon = epsilon
        self.bias_metric = bias_metric
        self.outputs_are_logits = outputs_are_logits
        self.n_eval_batches = n_eval_batches

        self.initialize_dataloader(dataloader, train_batch_size)

        if bias_metric.value == BiasMetrics.DP_GAP.value:
            # 2 because wepass only the predictions as input
            self.critic = self.get_critic(2, critic_linear)
        elif bias_metric.value == BiasMetrics.EO_GAP.value:
            # 4 because we pass the predictions and the true labels as input
            self.critic = self.get_critic(3, critic_linear)
        else:
            raise ValueError(f"Not supported: {bias_metric.value}")

        critic_criterion = nn.CrossEntropyLoss()
        critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        model_optimizer = torch.optim.Adam(self.model.parameters(), lr=model_lr)
        model_loss = nn.CrossEntropyLoss()

        for i in tqdm(range(iterations + 1), desc="Zhang: Adversarial Fine Tuning"):
            t = i  # t is the iteration number, starting from 1
            alpha = t**0.5  # as in the paper alpha = sqrt(t)
            # in the paper it is lr = 1/t, but we want to be able to set a base lr,
            # so we multiply the base lr by 1/t
            if t > 0:
                new_model_lr = model_lr * (1 / t)
                for g in model_optimizer.param_groups:
                    g["lr"] = new_model_lr

            logger.debug(f"Minibatch no. {i}")

            for param in self.critic.parameters():
                param.requires_grad = True
            for param in self.model.parameters():
                param.requires_grad = False

            self.model.eval()
            self.critic.train()
            # Train the critic
            for j in range(critic_iterations):
                x, y_true, prot_attr = self.sample_batch()

                with torch.no_grad():
                    y_logits = self.model(x)

                if bias_metric.value == BiasMetrics.DP_GAP.value:
                    c_pred = self.critic(y_logits)
                elif bias_metric.value == BiasMetrics.EO_GAP.value:
                    combined = torch.cat([y_logits, y_true.unsqueeze(1)], dim=1)
                    c_pred = self.critic(combined)
                else:
                    raise ValueError(f"Not supported: {bias_metric.value}")

                c_loss = critic_criterion(c_pred, prot_attr.long())

                c_loss.backward()
                critic_optimizer.step()
                critic_optimizer.zero_grad()
                model_optimizer.zero_grad()

                logger.debug(f"[{j}] Critic loss: {c_loss.item()}")

            for param in self.critic.parameters():
                param.requires_grad = False
            for param in self.model.parameters():
                param.requires_grad = True
            self.model.train()
            self.critic.eval()

            if i > 0:  # Skip the first iteration
                # Train the model
                for j in range(model_iterations):
                    x, y_true, prot_attr = self.sample_batch()

                    y_logits = self.model(x)

                    if bias_metric.value == BiasMetrics.DP_GAP.value:
                        c_pred = self.critic(y_logits).squeeze()
                    elif bias_metric.value == BiasMetrics.EO_GAP.value:
                        combined = torch.cat([y_logits, y_true.unsqueeze(1)], dim=1)
                        c_pred = self.critic(combined).squeeze()
                    else:
                        raise ValueError(f"Not supported: {bias_metric.value}")

                    c_loss = critic_criterion(c_pred, prot_attr.long())

                    m_loss = model_loss(y_logits, y_true.long())

                    for name, param in self.model.named_parameters():
                        try:
                            m_grad = autograd.grad(m_loss, param, retain_graph=True)[0]
                            c_grad = autograd.grad(c_loss, param, retain_graph=True)[0]
                        except RuntimeError as e:
                            logger.warning(
                                RuntimeError(
                                    f"[{i},{j}] Grad error in layer {name}: {e}"
                                )
                            )
                            continue
                        shape = c_grad.shape
                        m_grad = m_grad.flatten()
                        c_grad = c_grad.flatten()

                        m_grad_proj = (m_grad.T @ c_grad) * c_grad
                        grad = m_grad - m_grad_proj - alpha * c_grad
                        grad = grad.reshape(shape)
                        param.backward(grad)

                    model_optimizer.step()
                    model_optimizer.zero_grad()
                    critic_optimizer.zero_grad()

                    logger.debug(f"[{j}] Model loss: {m_loss.item()}")

        tau, phi = self.optimize_tau(tau_init, thresh_optimizer_maxiter)
        logger.info(f"Best tau: {tau}, Best phi: {phi}")

        if hasattr(self, "lightning_model"):
            self.lightning_model.model = self.model

        # Add a hook with the best transformation
        self.apply_hook(tau, soft_thresh_temperature)

    def get_critic(
        self,
        input_dim: int,
        critic_linear: list[int],
    ) -> nn.Module:
        """

        Args:
          input_dim: int:
          critic_linear: list[int]:

        Returns:

        """
        critic_layers = [
            nn.Linear(input_dim, critic_linear[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
        ]

        for i in range(1, len(critic_linear)):
            critic_layers += [
                nn.Linear(critic_linear[i - 1], critic_linear[i]),
                nn.ReLU(),
                nn.Dropout(0.2),
            ]

        critic_layers.append(nn.Linear(critic_linear[-1], 2))

        return nn.Sequential(*critic_layers).to(self.device)
