import copy
import gc
import logging
import os
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision.transforms as T
import tqdm
import wandb
import yaml
from torch.utils.data import DataLoader

from datasets import get_dataset, get_dataset_kwargs
from tcav.prepare_for_evaluation import prepare_model_for_evaluation
from model_correction.correction_methods.clarc import Clarc
from models import get_fn_model_loader
from utils.artificial_artifact import get_artifact_kwargs

torch.random.manual_seed(0)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file', default="config_files/YOUR_CONFIG.yaml")
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    print(args.config_file)
    with open(args.config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            config["wandb_id"] = os.path.basename(args.config_file)[:-5]
        except yaml.YAMLError as exc:
            print(exc)
            config = {}

    if config.get('wandb_api_key', None):
        os.environ["WANDB_API_KEY"] = config['wandb_api_key']
        wandb.init(id=config['wandb_id'], project=config['wandb_project_name'], resume=True)

    config['config_file'] = args.config_file

    measure_quality_cav(config)


def measure_quality_cav(config):
    """ Computes TCAV scores
    Args:
        config (dict): config for model correction run
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_name = config['dataset_name']
    if "celeba" not in config['dataset_name'] and "attacked" not in config['dataset_name']:
        dataset_name += "_attacked"

    model_name = config['model_name']

    data_paths = config['data_paths']
    artifact_name = config["artifact"]
    if "celeba" in dataset_name:
        artifact_name = "collar"
    artifacts_file = config.get('artifacts_file', None)
    artifact_kwargs = get_artifact_kwargs(config)
    dataset_specific_kwargs = get_dataset_kwargs(config)
    config["device"] = device

    dataset_name = f"{dataset_name}_hm" if not "_hm"  in dataset_name else dataset_name

    dataset = get_dataset(dataset_name)(data_paths=data_paths,
                                        normalize_data=True,
                                        p_artifact=config.get('p_artifact', 1.0),
                                        artifact_type=config.get('artifact_type', None),
                                        attacked_classes=config.get('attacked_classes', []),
                                        artifact_ids_file=artifacts_file,
                                        artifact=artifact_name,
                                        **artifact_kwargs, **dataset_specific_kwargs)
    
    n_classes = len(dataset.class_names)

    ckpt_path = config['ckpt_path_corrected']
    model = get_fn_model_loader(model_name=model_name)(n_class=n_classes, ckpt_path=ckpt_path).to(device)

    # Construct correction kwargs
    kwargs_correction = {}
    artifact_idxs_train = [i for i in dataset.idxs_train if i in dataset.sample_ids_by_artifact[config['artifact']]]
    mode = "cavs_max"
    kwargs_correction['n_classes'] = len(dataset.class_names)
    kwargs_correction['artifact_sample_ids'] = artifact_idxs_train
    kwargs_correction['sample_ids'] = dataset.idxs_train
    kwargs_correction['mode'] = mode

    model_cav = Clarc(copy.deepcopy(model), config, **kwargs_correction)
    model = prepare_model_for_evaluation(model, dataset, device, config)
    cav = model_cav.cav
    sets = {
        'train': dataset.idxs_train,
        'val': dataset.idxs_val,
        'test': dataset.idxs_test,
    }
    torch.cuda.empty_cache();
    gc.collect()
    model.eval()

    gaussian = T.GaussianBlur(kernel_size=41, sigma=8.0)

    if "imagenet" in dataset_name:
        all_classes = list(dataset.label_map.keys())
        if config.get("subset_correction", False):
            sets['test'] = sets['test'][::10]
            sets['val'] = sets['val'][::10]
    elif "bone" in dataset_name:
        all_classes = range(len(dataset.class_names))
    else:
        all_classes = dataset.class_names

    dataset = get_dataset(dataset_name)(data_paths=data_paths,
                                        normalize_data=True,
                                        p_artifact=1.0,
                                        artifact_type=config.get('artifact_type', None),
                                        attacked_classes=all_classes,
                                        artifact_ids_file=artifacts_file,
                                        artifact=artifact_name,
                                        **artifact_kwargs, **dataset_specific_kwargs)

    results = {}
    for split in [
        # 'train',
        'test',
        'val'
    ]:
        split_set = sets[split]

        artifact_ids_split = [i for i in dataset.sample_ids_by_artifact[artifact_name] if i in split_set]
        dataset_artifact_only = dataset.get_subset_by_idxs(artifact_ids_split)

        dl_art = DataLoader(dataset_artifact_only, batch_size=1, shuffle=False)

        def get_activation(module, input_, output_):
            global activations
            activations = output_
            return output_.clone()

        layer = config["layer_name"]
        for n, m in model.named_modules():
            if n.endswith(layer):
                m.register_forward_hook(get_activation)

        if "celeba" in dataset_name:
            attacked_class = 1
        elif "imagenet" in dataset_name:
            attacked_class = 0
        elif "isic" in dataset_name:
            attacked_class = 0
        elif "bone" in dataset_name:
            attacked_class = 2
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not supported")

        TCAV_sens_list = []
        TCAV_pos = 0
        TCAV_neg = 0
        TCAV_pos_masked = 0
        TCAV_neg_masked = 0
        TCAV_pos_mean = 0
        TCAV_neg_mean = 0
        for sample in tqdm.tqdm(dl_art):
            x_att, _, x_mask = sample
            
            # Compute latent representation
            with torch.enable_grad():
                x_att.requires_grad = True
                x_att = x_att.to(device)
                y_hat = model(x_att)
                yc_hat = y_hat[:, attacked_class]

                grad = torch.autograd.grad(outputs=yc_hat,
                                           inputs=activations,
                                           retain_graph=True,
                                           grad_outputs=torch.ones_like(yc_hat))[0]

                grad = grad.detach().cpu()
                model.zero_grad()

                acts = activations

                _, num_channels, h, w = acts.shape

                resizer = T.Compose([gaussian, T.Resize((h, w), interpolation=T.functional.InterpolationMode.BILINEAR)])
                latent_mask = torch.stack([resizer(x_mask.float())] * num_channels, dim=1)
                latent_mask = latent_mask > .2

                TCAV_pos += ((grad * cav[..., None, None]).sum(1).flatten() > 0).sum().item()
                TCAV_neg += ((grad * cav[..., None, None]).sum(1).flatten() < 0).sum().item()

                TCAV_pos_masked += ((grad * latent_mask * cav[..., None, None]).sum(1).flatten() > 0).sum().item()
                TCAV_neg_masked += ((grad * latent_mask * cav[..., None, None]).sum(1).flatten() < 0).sum().item()

                TCAV_pos_mean += ((grad * cav[..., None, None]).sum(1).mean((1, 2)).flatten() > 0).sum().item()
                TCAV_neg_mean += ((grad * cav[..., None, None]).sum(1).mean((1, 2)).flatten() < 0).sum().item()

                TCAV_sensitivity = (grad * cav[..., None, None]).sum(1).abs().flatten().numpy()
                TCAV_sens_list.append(TCAV_sensitivity)

        TCAV_sens_list = np.concatenate(TCAV_sens_list)

        tcav_quotient = TCAV_pos / (TCAV_neg + TCAV_pos)
        tcav_quotient_masked = TCAV_pos_masked / (TCAV_neg_masked + TCAV_pos_masked + 1e-10)
        mean_tcav_quotient = TCAV_pos_mean / (TCAV_neg_mean + TCAV_pos_mean)
        mean_tcav_sensitivity = TCAV_sens_list.mean()
        mean_tcav_sensitivity_sem = np.std(TCAV_sens_list) / np.sqrt(len(TCAV_sens_list))


        quotient_sderr = np.sqrt(tcav_quotient * (1 - tcav_quotient) / (TCAV_neg + TCAV_pos))
        mean_quotient_sderr = np.sqrt(mean_tcav_quotient * (1 - mean_tcav_quotient) / (TCAV_neg_mean + TCAV_pos_mean))

        print(f"TCAV quotient: {tcav_quotient} +- {quotient_sderr}")
        print(f"Mean TCAV quotient: {mean_tcav_quotient} +- {mean_quotient_sderr}")
        print(f"TCAV quotient masked: {tcav_quotient_masked}")

        results[f"{split}_mean_tcav_quotient"] = mean_tcav_quotient
        results[f"{split}_mean_quotient_sderr"] = mean_quotient_sderr

        results[f"{split}_tcav_quotient"] = tcav_quotient
        results[f"{split}_tcav_quotient_masked"] = tcav_quotient_masked
        results[f"{split}_quotient_sderr"] = quotient_sderr

        results[f"{split}_mean_tcav_sensitivity"] = mean_tcav_sensitivity
        results[f"{split}_mean_tcav_sensitivity_sem"] = mean_tcav_sensitivity_sem

        if config.get('wandb_api_key', None):
            wandb.log(results)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main()
