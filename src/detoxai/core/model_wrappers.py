from typing import Optional

import lightning as L
import torch
import torch.optim
from torchmetrics import MetricCollection


class BaseLightningWrapper(L.LightningModule):
    """ """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: Optional[torch.nn.Module] = torch.nn.CrossEntropyLoss(),
        performance_metrics: Optional[MetricCollection] = None,
        learning_rate: Optional[float] = 1e-3,
        optimizer: Optional[torch.optim.Optimizer] = torch.optim.Adam,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.train_performance_metrics = (
            performance_metrics.clone(prefix="train_") if performance_metrics else None
        )
        self.test_performance_metrics = (
            performance_metrics.clone(prefix="test_") if performance_metrics else None
        )
        self.valid_performance_metrics = (
            performance_metrics.clone(prefix="valid_") if performance_metrics else None
        )
        self.learning_rate = learning_rate
        self.optimizer = optimizer

    def training_step(self, batch, batch_idx):
        """

        Args:
          batch:
          batch_idx:

        Returns:

        """
        super().training_step(batch, batch_idx)
        inputs = batch[0]
        labels = batch[1]
        preds = self.model(inputs)  # softmax is included in the model
        loss = self.criterion(preds, labels)
        self.log("train_loss", loss)
        return {"loss": loss, "preds": preds}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """

        Args:
          outputs:
          batch:
          batch_idx:

        Returns:

        """
        super().on_train_batch_end(outputs, batch, batch_idx)
        preds = outputs["preds"]
        labels = batch[1]
        if self.train_performance_metrics:
            batch_performance_metrics = self.train_performance_metrics(preds, labels)
            self.log_dict(batch_performance_metrics)

    def on_train_epoch_end(self):
        """ """
        if self.train_performance_metrics:
            epoch_performance_metrics = self.train_performance_metrics.compute()
            self.log_dict(epoch_performance_metrics)
            self.train_performance_metrics.reset()

    def test_step(self, batch, batch_idx):
        """

        Args:
          batch:
          batch_idx:

        Returns:

        """
        super().test_step(batch, batch_idx)
        inputs = batch[0]
        labels = batch[1]
        preds = self.model(inputs)
        loss = self.criterion(preds, labels)
        return {"loss": loss, "preds": preds}

    def on_test_batch_end(self, outputs, batch, batch_idx):
        """

        Args:
          outputs:
          batch:
          batch_idx:

        Returns:

        """
        super().on_test_batch_end(outputs, batch, batch_idx)
        preds = outputs["preds"]
        labels = batch[1]
        if self.test_performance_metrics:
            batch_performance_metrics = self.test_performance_metrics(preds, labels)
            self.log_dict(batch_performance_metrics)

    def on_test_epoch_end(self):
        """ """
        if self.test_performance_metrics:
            epoch_performance_metrics = self.test_performance_metrics.compute()
            self.log_dict(epoch_performance_metrics)
            self.test_performance_metrics.reset()

    def configure_optimizers(self):
        """ """
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, x):
        """

        Args:
          x:

        Returns:

        """
        return self.model(x)

    def predict_step(self, batch):
        """

        Args:
          batch:

        Returns:

        """
        inputs = batch[0]
        preds = self.model(inputs)
        return preds


class FairnessLightningWrapper(BaseLightningWrapper):
    """ """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: Optional[torch.nn.Module] = torch.nn.CrossEntropyLoss(),
        performance_metrics: Optional[MetricCollection] = None,
        fairness_metrics: Optional[MetricCollection] = None,
        learning_rate: Optional[float] = 1e-3,
        optimizer: Optional[torch.optim.Optimizer] = torch.optim.Adam,
    ):
        super().__init__(
            model, criterion, performance_metrics, learning_rate, optimizer
        )
        self.save_hyperparameters()
        self.train_fairness_metrics = (
            fairness_metrics.clone(prefix="train_") if fairness_metrics else None
        )
        self.test_fairness_metrics = (
            fairness_metrics.clone(prefix="test_") if fairness_metrics else None
        )
        self.valid_fairness_metrics = (
            fairness_metrics.clone(prefix="valid_") if fairness_metrics else None
        )

    def training_step(self, batch, batch_idx):
        """

        Args:
          batch:
          batch_idx:

        Returns:

        """
        out = super().training_step(batch, batch_idx)
        return out

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """

        Args:
          outputs:
          batch:
          batch_idx:

        Returns:

        """
        super().on_train_batch_end(outputs, batch, batch_idx)
        preds = outputs["preds"]
        pred_for_1 = preds[:, 1]
        labels = batch[1]
        protected_attributes = batch[2]
        if self.train_fairness_metrics:
            batch_fairness_metrics = self.train_fairness_metrics(
                pred_for_1, labels, protected_attributes
            )
            self.log_dict(batch_fairness_metrics)

    def on_train_epoch_end(self):
        """ """
        super().on_train_epoch_end()
        if self.train_fairness_metrics:
            epoch_fairness_metrics = self.train_fairness_metrics.compute()
            self.log_dict(epoch_fairness_metrics)
            self.train_fairness_metrics.reset()

    def test_step(self, batch, batch_idx):
        """

        Args:
          batch:
          batch_idx:

        Returns:

        """
        out = super().test_step(batch, batch_idx)
        return out

    def on_test_batch_end(self, outputs, batch, batch_idx):
        """

        Args:
          outputs:
          batch:
          batch_idx:

        Returns:

        """
        super().on_test_batch_end(outputs, batch, batch_idx)
        preds = outputs["preds"]
        pred_for_1 = preds[:, 1]
        labels = batch[1]
        protected_attributes = batch[2]
        if self.test_fairness_metrics:
            batch_fairness_metrics = self.test_fairness_metrics(
                pred_for_1, labels, protected_attributes
            )
            self.log_dict(batch_fairness_metrics)

    def on_test_epoch_end(self):
        """ """
        super().on_test_epoch_end()
        if self.test_fairness_metrics:
            epoch_fairness_metrics = self.test_fairness_metrics.compute()
            self.log_dict(epoch_fairness_metrics)
            self.test_fairness_metrics.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """

        Args:
          batch:
          batch_idx:
          dataloader_idx:  (Default value = None)

        Returns:

        """
        inputs = batch[0]
        preds = self.model(inputs)
        return preds
