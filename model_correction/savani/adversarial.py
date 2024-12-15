import torch
import lightning as L
import logging
import torch.nn as nn
from torch.utils.data import DataLoader


from scipy import optimize
from scipy.optimize import OptimizeResult
from tqdm import tqdm

# Project imports
from .savani_base import SavaniBase
from .utils import (
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
    ) -> None:
        super().__init__(model, experiment_name, device, seed)
        if isinstance(model, L.LightningModule):
            self.lightning_model = model

    def apply_model_correction(
        self,
        dataloader: DataLoader,
        last_layer_name: str,
        epsilon: float = 0.05,
        bias_metric: BiasMetrics | str = BiasMetrics.EO_GAP,
        frac_of_batches_to_use: float = 1.0,
        iterations: int = 10,
        critic_iterations: int = 5,
        model_iterations: int = 5,
        train_batch_size: int = 16,
        thresh_optimizer_maxiter: int = 100,
        tau_init: float = 0.5,
        lam: float = 1.0,
        delta: float = 0.01,
        options: dict = {},
    ) -> None:
        """backward
        Do layer-wise optimization to find the best weights for each layer and the best threshold tau

        In options you can specify that your model already outputs probabilities, in which case the model will not apply the softmax function
        options = {'outputs_are_logits': False}

        """
        assert (
            0 <= frac_of_batches_to_use <= 1
        ), "frac_of_batches_to_use must be in [0, 1]"
        assert self.check_layer_name_exists(
            last_layer_name
        ), f"Layer name {last_layer_name} not found in the model"

        self.last_layer_name = last_layer_name
        self.tau_init = tau_init
        self.epsilon = epsilon
        self.bias_metric = bias_metric
        self.options = options
        self.lam = lam
        self.delta = delta

        # Unpack multiple batches of the dataloader
        self.X_torch, self.Y_true_torch, self.ProtAttr_torch = self.unpack_batches(
            dataloader, frac_of_batches_to_use
        )

        channels = self.X_torch.shape[1]

        encoder = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(start_dim=0),  # Yes, also flatten over the batch dimension
        ).to(self.device)

        with torch.no_grad():
            size_after = encoder(self.X_torch[:train_batch_size]).shape[0]

        self.critic = nn.Sequential(
            encoder,
            nn.Linear(size_after, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ).to(self.device)

        critic_criterion = nn.MSELoss()
        critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

        model_optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.model_loss = nn.CrossEntropyLoss()

        for i in tqdm(range(iterations), desc="Adversarial Fine Tuning"):
            logger.debug(f"Minibatch no. {i}")

            # Train the critic
            for j in range(critic_iterations):
                self.model.eval()
                self.critic.train()

                x, y_true, prot_attr = self.sample_minibatch(train_batch_size)

                with torch.no_grad():
                    y_pred = self.model(x)

                bias = calculate_bias_metric_torch(self.bias_metric, y_pred, prot_attr)

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

    def sample_minibatch(self, batch_size: int) -> tuple:
        idx = torch.randperm(self.X_torch.shape[0])[:batch_size]
        return self.X_torch[idx], self.Y_true_torch[idx], self.ProtAttr_torch[idx]
