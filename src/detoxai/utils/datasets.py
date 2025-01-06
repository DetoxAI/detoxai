import random
from typing import Optional, Tuple, Union, Dict, List
from pathlib import Path

import os
import PIL
import PIL.Image
import numpy as np
import pandas as pd
import torch

# from torch.utils.data import Dataset
import yaml
from torchvision.datasets.folder import VisionDataset

DETOXAI_DATASET_PATH = os.environ.get("DETOXAI_DATASET_PATH", Path.home() / ".detoxai")

# NOTE: transforms and the combination of transform and target_transform are mutually exclusive

CELEBA_DATASET_CONFIG = {
    "name": "celeba",
    "variant": "default",  # or None
    "target": "Male",  # target attribute that should be predicted
    "splits": {"train": 0.6, "test": 0.2, "unlearn": 0.2},
}

CELEBA_VARIANT_CONFIG = {
    "dataset": "celeba",
    "variant": "default",
    "fraction": 1.0,
    "splits": {
        "train": {
            "fraction": 0.3,
            "balancing": [
                {
                    "attribute_combination": [
                        {"attribute": "Male", "label": 0},
                        {"attribute": "Smiling", "label": 1},
                    ],
                    "percentage": 0.95,
                }
            ],
        },
        "test": {
            "fraction": 0.5,
            "balancing": [
                {
                    "attribute_combination": [
                        {"attribute": "Male", "label": 1},
                        {"attribute": "Smiling", "label": 1},
                    ],
                    "percentage": 0.5,
                }
            ],
        },
        "unlearn": {
            "fraction": 0.2,
            "balancing": [
                {
                    "attribute_combination": [
                        {"attribute": "Male", "label": 1},
                        {"attribute": "Smiling", "label": 1},
                    ],
                    "percentage": 0.5,
                }
            ],
        },
    },
}


def make_detoxai_datasets_variant(variant_config):
    variant_path = (
        Path(DETOXAI_DATASET_PATH)
        / variant_config["dataset"]
        / "variants"
        / variant_config["variant"]
        / "splits"
    )
    os.makedirs(variant_path, exist_ok=True)

    labels = pd.read_csv(
        Path(DETOXAI_DATASET_PATH) / variant_config["dataset"] / "labels.csv"
    )
    labels_fraction = labels.iloc[: int(variant_config["fraction"] * len(labels))]

    assert (
        variant_config["fraction"] <= 1.0
    ), "Fraction should be less than or equal to 1.0"
    assert (
        sum(
            [
                split_config["fraction"]
                for split_name, split_config in variant_config["splits"].items()
            ]
        )
        <= 1.0
    ), "Fractions should add up to less than or equal to 1.0"

    for split_name, split_config in variant_config["splits"].items():
        split_path = variant_path / f"{split_name}.txt"
        split_indices_stop_index = int(split_config["fraction"] * len(labels_fraction))
        df_split = labels_fraction.iloc[:split_indices_stop_index]

        final_split_indices = []
        for balancing_config in split_config["balancing"]:
            attribute_combination = balancing_config["attribute_combination"]
            percentage = balancing_config["percentage"]
            all_indices = df_split.index.to_numpy()

            # filter the split_indices based on the attribute_combination
            filtered_df = df_split
            for attribute in attribute_combination:
                filtered_df = filtered_df[
                    filtered_df[attribute["attribute"]] == attribute["label"]
                ]
            attribute_combination_indices = filtered_df.index.to_numpy()
            rest_indices = np.setdiff1d(all_indices, attribute_combination_indices)

            final_num_samples = len(attribute_combination_indices) / percentage
            if final_num_samples > len(all_indices):
                raise ValueError("final_num_samples is greater than len(all_indices)")

            final_rest_num_samples = final_num_samples - len(
                attribute_combination_indices
            )
            np.random.shuffle(rest_indices)
            rest_indices = rest_indices[: int(final_rest_num_samples)]

            final_split_indices.extend(attribute_combination_indices)
            final_split_indices.extend(rest_indices)

        final_split_df = df_split.loc[final_split_indices]
        np.savetxt(split_path, final_split_df.index.to_numpy(), fmt="%d", delimiter=",")

    with open(str(variant_path / "variant_config.yaml"), "w") as f:
        yaml.dump(variant_config, f)

    return variant_path


def get_detoxai_datasets(
    config: dict,
    transform: Optional[
        callable
    ] = None,  # takes in a PIL image and returns a transformed version
    transforms: Optional[
        callable
    ] = None,  # takes in an image and a label and returns the transformed versions of both
    target_transform: Optional[
        callable
    ] = None,  # A function/transform that takes in the target and transforms it.
    download: bool = False,
    seed: Optional[int] = None,
    device: str = None,
    saved_variant: Optional[str] = None,
) -> Dict[str, "DetoxaiDataset"]:
    detoxai_dataset_path = Path(DETOXAI_DATASET_PATH)

    if saved_variant is not None:
        variant_path = (
            Path(DETOXAI_DATASET_PATH) / config["name"] / "variants" / saved_variant
        )
        split_files = list(variant_path.glob("splits/*.txt"))
        split_indices = {}
        for split_file in split_files:
            split_name = split_file.stem
            split_indices[split_name] = np.loadtxt(split_file, dtype=int, delimiter=",")
    else:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # generate indices for all the splits randomly

        labels = pd.read_csv(detoxai_dataset_path / config["name"] / "labels.csv")
        all_indices = np.arange(len(labels))
        np.random.shuffle(all_indices)

        split_indices = {}
        start = 0
        for split_name, frac in config["splits"].items():
            end = start + int(frac * len(all_indices))
            split_indices[split_name] = all_indices[start:end]
            start = end

    datasets = {}
    for split, indices in split_indices.items():
        datasets[split] = DetoxaiDataset(
            config,
            detoxai_dataset_path,
            indices,
            transform=transform,
            transforms=transforms,
            target_transform=target_transform,
            download=download,
            seed=seed,
            device=device,
        )

    return datasets


class DetoxaiDataset(VisionDataset):
    def __init__(
        self,
        config: dict,
        root: Union[str, Path],
        split_indices: np.ndarray,
        transform: Optional[
            callable
        ] = None,  # takes in a PIL image and returns a transformed version
        transforms: Optional[
            callable
        ] = None,  # takes in an image and a label and returns the transformed versions of both
        target_transform: Optional[
            callable
        ] = None,  # A function/transform that takes in the target and transforms it.
        download: bool = False,
        seed: Optional[int] = None,
        device: str = None,
    ) -> None:
        super().__init__(
            root,
            transform=transform,
            transforms=transforms,
            target_transform=target_transform,
        )
        self.config = config
        self.root = Path(root)
        self.device = device

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it"
            )

        self.labels = self._read_labels_from_file()
        self.labels_mapping = self._read_labels_mapping_from_file()
        # self._target_labels_translation = self.get_target_label_translation()
        self.split_indices = split_indices

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def _read_labels_from_file(self) -> pd.DataFrame:
        df = pd.read_csv(self.root / self.config["name"] / "labels.csv")
        return df

    def _read_labels_mapping_from_file(self) -> pd.DataFrame:
        labels_mapping_from_yaml = yaml.safe_load(
            (self.root / self.config["name"] / "labels_mapping.yaml").open()
        )
        return labels_mapping_from_yaml

    def download(self):
        pass

    def _check_integrity(self) -> bool:
        return (self.root / self.config["name"]).exists()

    def __len__(self) -> int:
        return len(self.split_indices)

    def __getitem__(self, idx: int) -> Tuple[PIL.Image.Image, int, dict]:
        img = self._load_image(self.split_indices[idx])
        label = self._load_label(self.split_indices[idx])
        fairness_attributes = self._load_fairness_attributes(self.split_indices[idx])

        if self.transforms is not None:
            img, label = self.transforms(img, label)
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                label = self.target_transform(label)

        return img, label, fairness_attributes

    def _load_image(self, idx: int) -> PIL.Image.Image:
        img_path = (
            self.root / self.config["name"] / "data" / self.labels.iloc[idx]["image_id"]
        )
        img = PIL.Image.open(img_path)
        return img

    def _load_label(self, idx: int):
        label = self.labels.iloc[idx][self.config["target"]]
        return label

    def _load_fairness_attributes(self, idx: int) -> dict:
        fairness_attributes = {}
        for key, value in self.labels_mapping.items():
            # fairness_attributes[key] = value[self.labels.iloc[idx][key]]
            fairness_attributes[key] = self.labels.iloc[idx][key]
        return fairness_attributes

    def get_class_names(self) -> List[str]:
        return [
            f"{self.config['target']}_{item.replace(' ', '_')}"
            for key, item in self.labels_mapping[self.config["target"]].items()
        ]

    # def get_target_label_translation(self) -> dict:
    #     return {i: name for i, name in enumerate(self.get_class_names())}

    def get_num_classes(self) -> int:
        return len(self.labels_mapping[self.config["target"]])

    def get_collate_fn(self, protected_attribute: str, protected_attribute_value: str):
        def collate_fn(
            batch: List[Tuple[torch.Tensor, str, Dict[str, Union[str, int]]]],
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            images = torch.stack([item[0] for item in batch])
            labels = torch.tensor([item[1] for item in batch])
            protected_attributes = torch.tensor(
                [
                    int(item[2].get(protected_attribute) == protected_attribute_value)
                    for item in batch
                ]
            )
            return images, labels, protected_attributes

        return collate_fn
