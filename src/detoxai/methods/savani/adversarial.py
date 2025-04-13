import logging

import lightning as L
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...metrics.bias_metrics import (
    BiasMetrics,
    calculate_bias_metric_torch,
)

# Project imports
from .savani_base import SavaniBase

logger = logging.getLogger(__name__)


class SavaniAFT(SavaniBase):
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
        iterations: int = 10,
        critic_iterations: int = 5,
        model_iterations: int = 5,
        train_batch_size: int = 128,
        thresh_optimizer_maxiter: int = 100,
        tau_init: float = 0.5,
        lam: float = 1.0,
        delta: float = 0.01,
        critic_lr: float = 1e-4,
        model_lr: float = 1e-4,
        critic_filters: list[int] = [8, 16, 32],
        critic_linear: list[int] = [32],
        outputs_are_logits: bool = True,
        n_eval_batches: int = 3,
        soft_thresh_temperature: float = 10.0,
        **kwargs,
    ) -> None:
        """backward
        Do layer-wise optimization to find the best weights for each layer and the best threshold tau

        Args:
          dataloader: DataLoader:
          last_layer_name: str:
          epsilon: float:  (Default value = 0.1)
          bias_metric: BiasMetrics | str:  (Default value = BiasMetrics.EO_GAP)
          iterations: int:  (Default value = 10)
          critic_iterations: int:  (Default value = 5)
          model_iterations: int:  (Default value = 5)
          train_batch_size: int:  (Default value = 128)
          thresh_optimizer_maxiter: int:  (Default value = 100)
          tau_init: float:  (Default value = 0.5)
          lam: float:  (Default value = 1.0)
          delta: float:  (Default value = 0.01)
          critic_lr: float:  (Default value = 1e-4)
          model_lr: float:  (Default value = 1e-4)
          critic_filters: list[int]:  (Default value = [8)
          16:
          32]:
          critic_linear: list[int]:  (Default value = [32])
          outputs_are_logits: bool:  (Default value = True)
          n_eval_batches: int:  (Default value = 3)
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
        self.lam = lam
        self.delta = delta
        self.n_eval_batches = n_eval_batches

        self.initialize_dataloader(dataloader, train_batch_size)
        self.__sample_example, _, _ = self.sample_batch()

        channels = self.__sample_example.shape[1]

        self.critic = self.get_critic(
            channels, critic_filters, critic_linear, train_batch_size
        )

        critic_criterion = nn.MSELoss()
        critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        model_optimizer = torch.optim.Adam(self.model.parameters(), lr=model_lr)
        self.model_loss = nn.CrossEntropyLoss()

        for i in tqdm(range(iterations), desc="Savani: Adversarial Fine Tuning"):
            logger.debug(f"Minibatch no. {i}")

            # Train the critic
            for j in range(critic_iterations):
                self.model.eval()
                self.critic.train()

                x, y_true, prot_attr = self.sample_batch()

                with torch.no_grad():
                    # Assuming binary classification and logits
                    y_logits = self.model(x)

                    if self.outputs_are_logits:
                        y_pred = softmax(y_logits, dim=1)
                    else:  # probabilties
                        y_pred = y_logits

                    y_pred = torch.argmax(y_pred, dim=1)

                bias = calculate_bias_metric_torch(
                    self.bias_metric, y_pred, y_true, prot_attr
                )

                c_loss = critic_criterion(self.critic(x)[0], bias)
                critic_optimizer.zero_grad()
                c_loss.backward()
                critic_optimizer.step()

                logger.debug(f"[{j}] Critic loss: {c_loss.item()}")

            # Train the model
            for j in range(model_iterations):
                self.model.train()
                self.critic.eval()

                x, y_true, prot_attr = self.sample_batch()

                y_logits = self.model(x)

                m_loss = self.fair_loss(y_logits, y_true, x)

                model_optimizer.zero_grad()
                m_loss.backward()
                model_optimizer.step()

                logger.debug(f"[{j}] Model loss: {m_loss.item()}")

        tau, phi = self.optimize_tau(tau_init, thresh_optimizer_maxiter)
        logger.info(f"Best tau: {tau}, Best phi: {phi}")

        if hasattr(self, "lightning_model"):
            self.lightning_model.model = self.model

        # Add a hook with the best transformation
        self.apply_hook(tau, soft_thresh_temperature)

    def fair_loss(self, y_logits, y_true, input):
        """

        Args:
          y_logits:
          y_true:
          input:

        Returns:

        """
        fair = torch.max(
            torch.tensor(1, dtype=torch.float32, device=self.device),
            self.lam * (self.critic(input).squeeze() - self.epsilon + self.delta) + 1,
        )
        return self.model_loss(y_logits, y_true) * fair

    def get_critic(
        self,
        channels: int,
        critic_filters: list[int],
        critic_linear: list[int],
        batch_size: int,
    ) -> nn.Module:
        """

        Args:
          channels: int:
          critic_filters: list[int]:
          critic_linear: list[int]:
          batch_size: int:

        Returns:

        """
        encoder_layers = [
            nn.Conv2d(channels, critic_filters[0], 3, padding="same"),
            nn.ReLU(),
        ]

        for i in range(1, len(critic_filters)):
            encoder_layers += [
                nn.Conv2d(critic_filters[i - 1], critic_filters[i], 3, padding="same"),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]

        # Add adaptive pooling layer

        encoder_layers.append(nn.AdaptiveAvgPool2d(3))

        encoder_layers.append(nn.Flatten(start_dim=0))

        encoder = nn.Sequential(*encoder_layers).to(self.device)

        with torch.no_grad():
            size_after = encoder(self.__sample_example[:batch_size]).shape[0]

        critic_layers = [encoder, nn.Linear(size_after, critic_linear[0]), nn.ReLU()]

        for i in range(1, len(critic_linear)):
            critic_layers += [
                nn.Linear(critic_linear[i - 1], critic_linear[i]),
                nn.ReLU(),
            ]

        critic_layers.append(nn.Linear(critic_linear[-1], 1))

        return nn.Sequential(*critic_layers).to(self.device)
