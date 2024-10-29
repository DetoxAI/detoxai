import os

import numpy as np
import torch
from zennit.core import stabilize

from model_correction.base_correction_method import LitClassifier, Freeze
from tcav.cav import compute_cav


class Clarc(LitClassifier):
    def __init__(self, model, config, **kwargs):
        super().__init__(model, config, **kwargs)
        self.std = None
        self.layer_name = config["layer_name"]
        self.dataset_name = config["dataset_name"]
        self.model_name = config["model_name"]
        self.cav_scope = config.get("cav_scope", "attacked_class")

        assert "artifact_sample_ids" in kwargs.keys(), "artifact_sample_ids have to be passed to ClArC correction methods"
        assert "sample_ids" in kwargs.keys(), "all sample_ids have to be passed to ClArC correction methods"
        assert "n_classes" in kwargs.keys(), "n_classes has to be passed to ClArC correction methods"
        assert "mode" in kwargs.keys(), "mode has to be passed to ClArC correction methods"

        self.artifact_sample_ids = kwargs["artifact_sample_ids"]
        self.sample_ids = kwargs["sample_ids"]
        self.n_classes = kwargs["n_classes"]

        self.direction_mode = config.get("direction_mode", "signal")
        self.mode = kwargs['mode']

        print(f"Using {len(self.artifact_sample_ids)} artifact samples.")

        artifact_type = config.get('artifact_type', None)
        artifact_extension = f"_{artifact_type}-{config['p_artifact']}" if artifact_type else ""
        self.path = f"results/global_relevances_and_activations/{self.dataset_name}{artifact_extension}/{self.model_name}"

        cav, mean_length, mean_length_targets = self.compute_cav(self.mode, norm=False)
        print(f"Using {len(self.artifact_sample_ids)} artifact samples.")
        self.cav = cav
        self.mean_length = mean_length
        self.mean_length_targets = mean_length_targets
        print(f"Using {len(self.artifact_sample_ids)} artifact samples.")
        hooks = []
        for n, m in self.model.named_modules():
            if n == self.layer_name:
                print("Registered forward hook.")
                hooks.append(m.register_forward_hook(self.clarc_hook))
        self.hooks = hooks
        print(f"Using {len(self.artifact_sample_ids)} artifact samples.")

    def compute_cav(self, mode, norm=False):
        vecs = []
        sample_ids = []

        path = self.path
        classes = range(self.n_classes)
        if self.cav_scope == "attacked_class":
            if "celeba" in self.dataset_name:
                classes = [1]
            if "imagenet" in self.dataset_name:
                classes = [0]
            if "isic" in self.dataset_name:
                classes = [0]
            if "bone" in self.dataset_name:
                classes = [2]
        for class_id in classes:
            data = torch.load(f"{path}/{self.layer_name}_class_{class_id}_all.pth")
            if data['samples']:
                sample_ids += data['samples']
                vecs.append(torch.stack(data[mode], 0))

        vecs = torch.cat(vecs, 0)

        sample_ids = np.array(sample_ids)

        # Only keep samples that are in self.sample_ids (usually training set)
        all_sample_ids = np.array(self.sample_ids)
        filter_sample = np.array([id in all_sample_ids for id in sample_ids])
        vecs = vecs[filter_sample]
        sample_ids = sample_ids[filter_sample]

        target_ids = np.array(
            [np.argwhere(sample_ids == id_)[0][0] for id_ in self.artifact_sample_ids if
             np.argwhere(sample_ids == id_)])
        targets = np.array([1 * (j in target_ids) for j, x in enumerate(sample_ids)])
        X = vecs.detach().cpu().numpy()

        cav = compute_cav(
            X, targets, cav_type=self.direction_mode
        )

        mean_length = (vecs[targets == 0] * cav).sum(1).mean(0)
        mean_length_targets = (vecs[targets == 1] * cav).sum(1).mean(0)
        print(f"Computed CAV. {mean_length:.1f} vs {mean_length_targets:.1f}")

        expected_sim = -torch.nn.functional.cosine_similarity(
            vecs[targets == 0] - vecs[targets == 1].mean(0, keepdim=True), cav).mean()
        print(f"Expected Clean CAV similarity: {expected_sim:.2f}")

        expected_sim = -torch.nn.functional.cosine_similarity(
            vecs[targets == 0].mean(0, keepdim=True) - vecs[targets == 1], cav).mean()
        print(f"Expected Target CAV similarity: {expected_sim:.2f}")

        return cav, mean_length, mean_length_targets

    def clarc_hook(self, m, i, o):
        pass

    def configure_callbacks(self):
        return [Freeze()]


class PClarc(Clarc):
    def __init__(self, model, config, **kwargs):
        super().__init__(model, config, **kwargs)

        self.path = f"results/global_relevances_and_activations/{os.path.basename(config['config_file'])[:-5]}"
        if os.path.exists(self.path):
            print("Re-computing CAV.")
            cav, mean_length, mean_length_targets = self.compute_cav(self.mode)
            self.cav = cav
            self.mean_length = mean_length
            self.mean_length_targets = mean_length_targets
        else:
            if self.hooks:
                for hook in self.hooks:
                    print("Removed hook. No hook should be active for training.")
                    hook.remove()
                self.hooks = []

    def clarc_hook(self, m, i, o):
        outs = o + 0
        cav = self.cav.to(outs)
        length = (outs.flatten(start_dim=2).max(2).values * cav).sum(1)
        v = self.cav.to(outs)
        beta = (cav * v).sum(1)
        mag = (self.mean_length - length).to(outs) / stabilize(beta)
        print(mag.shape, v.shape)
        addition = (mag[:, None, None, None] * v[..., None, None])
        acts = outs + addition
        return acts


class AClarc(Clarc):
    def __init__(self, model, config, **kwargs):
        super().__init__(model, config, **kwargs)
        self.lamb = self.config["lamb"]  # 10

    def clarc_hook(self, m, i, o):
        outs = o + 0
        cav = self.cav.to(outs)
        length = (outs.flatten(start_dim=2).max(2).values * cav).sum(1)
        v = self.cav.to(outs)
        beta = (cav * v).sum(1)
        mag = (self.mean_length_targets - length).to(outs) / stabilize(beta)
        addition = (mag[:, None, None, None] * v[..., None, None]).requires_grad_()
        acts = outs + addition
        return acts
