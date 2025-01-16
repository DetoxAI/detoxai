import torch
import lightning as L
import logging
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.functional import softmax


from scipy import optimize
from scipy.optimize import OptimizeResult
from tqdm import tqdm

# Project imports
from .savani_base import SavaniBase
from ...metrics.bias_metrics import (
    BiasMetrics,
    calculate_bias_metric_torch,
)

logger = logging.getLogger(__name__)


class SavaniAFT(SavaniBase):
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
        iterations: int = 10,
        critic_iterations: int = 5,
        model_iterations: int = 5,
        train_batch_size: int = 16,
        thresh_optimizer_maxiter: int = 100,
        tau_init: float = 0.5,
        lam: float = 1.0,
        delta: float = 0.01,
        critic_lr: float = 1e-4,
        model_lr: float = 1e-4,
        critic_filters: list[int] = [8, 16, 32],
        critic_linear: list[int] = [32],
        options: dict = {},
        **kwargs,
    ) -> None:
        """backward
        Do layer-wise optimization to find the best weights for each layer and the best threshold tau

        In options you can specify that your model already outputs probabilities, in which case the model will not apply the softmax function
        options = {'outputs_are_logits': False}

        """
        assert 0 <= data_to_use <= 1 or isinstance(data_to_use, int), (
            "frac_of_batches_to_use must be in [0, 1] or an integer"
        )
        assert self.check_layer_name_exists(last_layer_name), (
            f"Layer name {last_layer_name} not found in the model"
        )

        self.last_layer_name = last_layer_name
        self.tau_init = tau_init
        self.epsilon = epsilon
        self.bias_metric = bias_metric
        self.options = options
        self.lam = lam
        self.delta = delta

        # Unpack multiple batches of the dataloader
        self.X_torch, self.Y_true_torch, self.ProtAttr_torch = self.unpack_batches(
            dataloader, data_to_use
        )

        channels = self.X_torch.shape[1]

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

                x, y_true, prot_attr = self.sample_minibatch(train_batch_size)

                with torch.no_grad():
                    # Assuming binary classification and logits
                    y_pred = self.model(x)
                    if self.options.get("outputs_are_logits", True):
                        y_pred = softmax(y_pred, dim=1)
                    y_pred = y_pred[:, 1]

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

                x, y_true, prot_attr = self.sample_minibatch(train_batch_size)

                y_pred = self.model(x)
                m_loss = self.fair_loss(y_pred, y_true, x)

                model_optimizer.zero_grad()
                m_loss.backward()
                model_optimizer.step()

                logger.debug(f"[{j}] Model loss: {m_loss.item()}")

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
            logger.debug(f"tau: {tau:.3f}, phi: {_phi:.3f}, bias: {bias:.3f}")
        else:
            tau = tau_init
            logger.debug(f"Optimization failed: {res.message}")

        if hasattr(self, "lightning_model"):
            self.lightning_model.model = self.model

        # Add a hook with the best transformation
        self.apply_hook(tau)

    def fair_loss(self, y_pred, y_true, input):
        fair = torch.max(
            torch.tensor(1, dtype=torch.float32, device=self.device),
            self.lam * (self.critic(input).squeeze() - self.epsilon + self.delta) + 1,
        )
        return self.model_loss(y_pred, y_true) * fair

    def get_critic(
        self,
        channels: int,
        critic_filters: list[int],
        critic_linear: list[int],
        batch_size: int,
    ) -> nn.Module:
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
        encoder_layers.append(nn.Flatten(start_dim=0))

        encoder = nn.Sequential(*encoder_layers).to(self.device)

        with torch.no_grad():
            size_after = encoder(self.X_torch[:batch_size]).shape[0]

        critic_layers = [encoder, nn.Linear(size_after, critic_linear[0]), nn.ReLU()]

        for i in range(1, len(critic_linear)):
            critic_layers += [
                nn.Linear(critic_linear[i - 1], critic_linear[i]),
                nn.ReLU(),
            ]

        critic_layers.append(nn.Linear(critic_linear[-1], 1))

        return nn.Sequential(*critic_layers).to(self.device)
