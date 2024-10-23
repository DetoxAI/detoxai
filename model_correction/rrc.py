import torch

from model_correction.base_correction_method import Freeze
from model_correction.clarc import Clarc


class RRClarc(Clarc):
    """
    Classifier with Right Reasons loss for latent concept unlearning.
    """

    def __init__(self, model, config, **kwargs):
        super().__init__(model, config, **kwargs)
        self.lamb = self.config["lamb"]
        self.aggregation = self.config.get("compute", "l1")
        self.gradient_target = self.config.get("criterion", "max")
        self.intermediate = torch.tensor(0.0)
        self.layer_name = config["layer_name"]

    def clarc_hook(self, m, i, o):
        self.intermediate = o
        return o.clone()

    def criterion_fn(self, y_hat, y):
        # Construct target vector yc_hat, w.r.t. which the gradient will be computed.
        # Based on config (gradient_target), this can be class-specific (target_class) or class-inspecific (all_logits/all_logits_random)
        if self.gradient_target == 'max_logit':
            return y_hat.max(1)[0]
        elif self.gradient_target == 'target_logit':
            target_class = self.config.get("target_class", y)
            return y_hat[range(len(y)), target_class]
        elif self.gradient_target == 'all_logits':
            return (y_hat).sum(1)
        elif self.gradient_target == 'all_logits_random':
            return (y_hat * torch.sign(0.5 - torch.rand_like(y_hat))).sum(1)
        elif self.gradient_target == 'logprobs':
            return (y_hat.softmax(1) + 1e-5).log().mean(1)
        else:
            raise NotImplementedError

    def loss_compute(self, gradient):
        # Compute right-reason loss based on defined similarity metric (cosine/L1/L2) between CAV and latent gradient (w.r.t. target)
        
        cav = self.cav.to(gradient)
        if "mean" in self.aggregation and gradient.dim() != 2:
            gradient = gradient.mean((2, 3), keepdim=True).expand_as(gradient)

        g_flat = gradient.permute(1, 0, 2, 3).flatten(start_dim=1).permute(1, 0)

        if "cosine" in self.aggregation:
            return torch.nn.functional.cosine_similarity(g_flat, cav).abs().mean(0)  
        elif "l2" in self.aggregation:
            return ((g_flat * cav).sum(1) ** 2).mean(0) 
        elif "l1" in self.aggregation:
            return (g_flat * cav).sum(1).abs().mean(0)
        else:
            raise NotImplementedError

    def default_step(self, x, y, stage):
        with torch.enable_grad():
            x.requires_grad = True
            y_hat = self(x)

            # build target vector yc_hat, w.r.t. which the gradient is computed (all/random/class-specific)
            yc_hat = self.criterion_fn(y_hat, y)

            # compute gradient of latent activations w.r.t. yc_hat
            grad = torch.autograd.grad(outputs=yc_hat,
                                       inputs=self.intermediate,
                                       create_graph=True,
                                       retain_graph=True,
                                       grad_outputs=torch.ones_like(yc_hat))[0]
            
            # compute RR loss as similarity between gradient and CAV
            rr_loss = self.loss_compute(grad)

        # Compute total loss as weighted sum between classification loss and RR loss
        loss = self.loss(y_hat, y) + self.lamb * rr_loss
        self.log_dict(
            {f"{stage}_loss": loss,
             f"{stage}_acc": self.get_accuracy(y_hat, y),
             f"{stage}_auc": self.get_auc(y_hat, y),
             f"{stage}_f1": self.get_f1(y_hat, y),
             f"{stage}_auxloss": rr_loss},
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def configure_callbacks(self):
        return [
            Freeze()
        ]
