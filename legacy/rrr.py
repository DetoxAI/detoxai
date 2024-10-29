import torch

from model_correction.correction_methods.base_correction_method import LitClassifier, Freeze


class RRR_CE(LitClassifier):
    """
    Classifier with Right for the Right Reasons (RRR) loss.
    The loss is based on the cosine similarity between the gradient of the CE loss and the artifact mask.
    """

    def __init__(self, model, config, **kwargs):
        super().__init__(model, config, kwargs=kwargs)
        self.lamb = self.config["lamb"]  # 100
        self.layer_name = self.config['layer_name']

    def training_step(self, batch, batch_idx):
        x, y, z = batch
        loss = self.default_step(x, y, z, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, z = batch
        self.default_step(x, y, z, stage="valid")

    def test_step(self, batch, batch_idx):
        x, y, z = batch
        self.default_step(x, y, z, stage="test")

    def criterion_fn(self, y_hat, y):
        return self.loss(y_hat, y)

    def default_step(self, x, y, z, stage):

        with torch.enable_grad():
            x.requires_grad = True
            y_hat = self(x)
            num_samples = (z.sum((1, 2)) > 0).sum()
            yc_hat = self.criterion_fn(y_hat, y)
            if num_samples > 0:
                grad = torch.autograd.grad(outputs=yc_hat, inputs=x, create_graph=True, retain_graph=True)[0]
                aux_loss = torch.nn.functional.cosine_similarity(
                    grad.sum(1).abs().flatten(start_dim=1), z.flatten(start_dim=1).to(x)).mean(0)
            else:
                aux_loss = 0.0

        loss = self.loss(y_hat, y) + self.lamb * aux_loss
        self.log_dict(
            {f"{stage}_loss": loss,
             f"{stage}_acc": self.get_accuracy(y_hat, y),
             f"{stage}_auc": self.get_auc(y_hat, y),
             f"{stage}_f1": self.get_f1(y_hat, y),
             f"{stage}_auxloss": aux_loss},
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def configure_callbacks(self):
        return [
            Freeze()
            ]
    

class RRR_ExpLogSum(RRR_CE):
    """
    Classifier with Right for the Right Reasons (RRR) loss.
    The loss is based on the cosine similarity between the gradient of the predicted logit and the artifact mask.
    """

    def __init__(self, model, config, **kwargs):
        super().__init__(model, config, kwargs=kwargs)

    def criterion_fn(self, y_hat, y):
        return (y_hat.softmax(1) + 1e-5).log().mean(1).sum()
