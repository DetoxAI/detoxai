import torch
import lightning as L
import logging
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import autograd


from scipy import optimize
from scipy.optimize import OptimizeResult
from tqdm import tqdm

# Project imports
from .savani_base import SavaniBase
from ...metrics.bias_metrics import (
    BiasMetrics,
)

logger = logging.getLogger(__name__)


class ZhangM(SavaniBase):
    """Brian Hu Zhang, Blake Lemoine, Margaret Mitchell - "Mitigating unwanted biases with adversarial learning" """

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
        critic_iterations: int = 15,
        model_iterations: int = 15,
        train_batch_size: int = 16,
        thresh_optimizer_maxiter: int = 100,
        tau_init: float = 0.5,
        alpha: float = 5.0,
        critic_lr: float = 1e-4,
        model_lr: float = 1e-4,
        critic_linear: list[int] = [64, 32, 16],
        options: dict = {},
        **kwargs,
    ) -> None:
        """backward
        Do layer-wise optimization to find the best weights for each layer and the best threshold tau

        In options you can specify that your model already outputs probabilities, in which case the model will not apply the softmax function
        options = {'outputs_are_logits': False}

        """
        assert 0 <= data_to_use <= 1 or isinstance(
            data_to_use, int
        ), "frac_of_batches_to_use must be in [0, 1] or an integer"
        assert self.check_layer_name_exists(
            last_layer_name
        ), f"Layer name {last_layer_name} not found in the model"

        self.last_layer_name = last_layer_name
        self.tau_init = tau_init
        self.epsilon = epsilon
        self.bias_metric = bias_metric
        self.options = options

        # Unpack multiple batches of the dataloader
        self.X_torch, self.Y_true_torch, self.ProtAttr_torch = self.unpack_batches(
            dataloader, data_to_use
        )
        self.ProtAttr_torch = self.ProtAttr_torch.to(dtype=torch.float32)
        self.Y_true_torch = self.ProtAttr_torch.to(dtype=torch.float32)

        self.critic = self.get_critic(2, critic_linear)  # Binary classification

        critic_criterion = nn.BCELoss()
        critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        model_optimizer = torch.optim.Adam(self.model.parameters(), lr=model_lr)
        model_loss = nn.BCELoss()

        for i in tqdm(range(iterations), desc="Zhang: Adversarial Fine Tuning"):
            logger.debug(f"Minibatch no. {i}")

            for param in self.critic.parameters():
                param.requires_grad = True
            for param in self.model.parameters():
                param.requires_grad = False

            self.model.eval()
            self.critic.train()
            # Train the critic
            for j in range(critic_iterations):
                x, y_true, prot_attr = self.sample_minibatch(train_batch_size)

                y_pred = self.model(x)
                c_pred = self.critic(y_pred)[:, 0]
                if bias_metric.value == BiasMetrics.DP_GAP.value:
                    c_loss = critic_criterion(c_pred, prot_attr)
                elif bias_metric.value == BiasMetrics.EO_GAP.value:
                    c_loss = critic_criterion(c_pred, y_true)
                else:
                    raise ValueError(f"Not supported: {bias_metric.value}")
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

            # Train the model
            for j in range(model_iterations):
                x, y_true, prot_attr = self.sample_minibatch(train_batch_size)

                y_pred = self.model(x)

                c_pred = self.critic(y_pred)[:, 0]

                if bias_metric.value == BiasMetrics.DP_GAP.value:
                    c_loss = critic_criterion(c_pred, prot_attr)
                elif bias_metric.value == BiasMetrics.EO_GAP.value:
                    c_loss = critic_criterion(c_pred, y_true)
                else:
                    raise ValueError(f"Not supported: {bias_metric.value}")

                if self.options.get("outputs_are_logits", True):
                    y_pred = torch.softmax(y_pred, dim=1)[:, 0]

                m_loss = model_loss(y_pred, y_true)

                for name, param in self.model.named_parameters():
                    try:
                        m_grad = autograd.grad(m_loss, param, retain_graph=True)[0]
                        c_grad = autograd.grad(c_loss, param, retain_graph=True)[0]
                    except RuntimeError as e:
                        logger.warning(
                            RuntimeError(f"[{i},{j}] Grad error in layer {name}: {e}")
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

    def get_critic(
        self,
        input_dim: int,
        critic_linear: list[int],
    ) -> nn.Module:
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
        critic_layers.append(nn.Softmax(dim=1))

        return nn.Sequential(*critic_layers).to(self.device)
