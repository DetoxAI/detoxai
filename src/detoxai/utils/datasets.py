import logging
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import PIL
import PIL.Image
import torch

# from torch.utils.data import Dataset
import yaml
from torchvision.datasets.folder import VisionDataset

logger = logging.getLogger(__name__)

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
                    "percentage": 0.1,
                },
                {
                    "attribute_combination": [
                        {"attribute": "Male", "label": 1},
                        {"attribute": "Smiling", "label": 0},
                    ],
                    "percentage": 0.9,
                },
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


def calculate_max_samples(df: pd.DataFrame, config: dict) -> int:
    """Calculate the maximum number of total samples possible given the constraints
    to avoid duplicates and maintain percentages.

    Args:
      df: pd.DataFrame:
      config: dict:

    Returns:

    """
    max_possible_samples = []

    for balance_rule in config["balancing"]:
        mask = pd.Series([True] * len(df), index=df.index)
        for condition in balance_rule["attribute_combination"]:
            mask &= df[condition["attribute"]] == condition["label"]

        available_samples = mask.sum()

        # Calculate maximum total samples possible for this rule
        # Example: if we need 30% and have 100 samples, max total is 100/0.3 = 333
        if balance_rule["percentage"] > 0:
            max_total = int(available_samples / balance_rule["percentage"])
            max_possible_samples.append(max_total)

    return min(max_possible_samples) if max_possible_samples else len(df)


def balance_dataset(df: pd.DataFrame, config: dict) -> Tuple[np.ndarray, int]:
    """

    Args:
      df: pd.DataFrame:
      config: dict:

    Returns:

    """
    total_samples = calculate_max_samples(df, config)
    selected_indices: Set[int] = set()
    remaining_indices = set(df.index.tolist())
    total_matching_indices = set()

    for balance_rule in config["balancing"]:
        n_samples = int(balance_rule["percentage"] * total_samples)

        mask = pd.Series([True] * len(df), index=df.index)
        for condition in balance_rule["attribute_combination"]:
            attr = condition["attribute"]
            label = condition["label"]
            mask &= df[attr] == label

        matching_indices = df[mask].index.tolist()

        available_indices = list(set(matching_indices) - selected_indices)

        if len(available_indices) < n_samples:
            logger.warning(
                f"Warning: Reducing total samples. Not enough samples for combination "
                f"{balance_rule['attribute_combination']}. "
                f"Requested {n_samples}, but only {len(available_indices)} available."
            )
            total_samples = int(len(available_indices) / balance_rule["percentage"])
            n_samples = int(balance_rule["percentage"] * total_samples)

        if available_indices:
            selected = np.random.choice(
                available_indices, size=n_samples, replace=False
            )
            selected_indices.update(selected)
            remaining_indices -= set(selected)

        total_matching_indices.update(matching_indices)

    total_percentage = sum(rule["percentage"] for rule in config["balancing"])

    if total_percentage < 1 and remaining_indices:
        remaining_samples = int((1 - total_percentage) * total_samples)
        indices_that_do_not_adhere_to_any_rule = (
            remaining_indices - total_matching_indices
        )
        remaining_to_sample = min(
            remaining_samples, len(indices_that_do_not_adhere_to_any_rule)
        )
        remaining_selected = np.random.choice(
            list(indices_that_do_not_adhere_to_any_rule),
            size=remaining_to_sample,
            replace=False,
        )
        selected_indices.update(remaining_selected)

    return np.array(list(selected_indices)), total_samples


def make_detoxai_datasets_variant(variant_config):
    """

    Args:
      variant_config:

    Returns:

    """
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

    assert variant_config["fraction"] <= 1.0, (
        "Fraction should be less than or equal to 1.0"
    )
    assert (
        sum(
            [
                split_config["fraction"]
                for split_name, split_config in variant_config["splits"].items()
            ]
        )
        <= 1.0
    ), "Fractions should add up to less than or equal to 1.0"

    split_index_offset = 0
    for split_name, split_config in variant_config["splits"].items():
        split_path = variant_path / f"{split_name}.txt"
        split_num_samples = int(split_config["fraction"] * len(labels_fraction))
        df_split = labels_fraction.iloc[
            split_index_offset : split_index_offset + split_num_samples
        ]
        split_index_offset += split_num_samples

        final_split_indices, total_samples = balance_dataset(df_split, split_config)

        final_split_df = df_split.loc[final_split_indices]
        np.savetxt(split_path, final_split_df.index.to_numpy(), fmt="%d", delimiter=",")

    with open(str(variant_path / "variant_config.yaml"), "w") as f:
        yaml.dump(variant_config, f)

    return variant_path


def get_detoxai_datasets(
    config: dict,
    transform: Optional[
        Callable
    ] = None,  # takes in a PIL image and returns a transformed version
    transforms: Optional[
        Callable
    ] = None,  # takes in an image and a label and returns the transformed versions of both
    target_transform: Optional[
        Callable
    ] = None,  # A function/transform that takes in the target and transforms it.
    download: bool = False,
    seed: Optional[int] = None,
    device: Union[str, None] = None,
    saved_variant: Optional[str] = None,
) -> Dict[str, "DetoxaiDataset"]:
    """

    Args:
      config: dict:
      transform: Optional[Callable]:  (Default value = None)
      # takes in a PIL image and returns a transformed versiontransforms: Optional[Callable]:  (Default value = None)
      # takes in an image and a label and returns the transformed versions of bothtarget_transform: Optional[Callable]:  (Default value = None)
      # A function/transform that takes in the target and transforms it.download: bool:  (Default value = False)
      seed: Optional[int]:  (Default value = None)
      device: Union[str:
      None]:  (Default value = None)
      saved_variant: Optional[str]:  (Default value = None)

    Returns:

    """
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
    """ """

    def __init__(
        self,
        config: dict,
        root: Union[str, Path],
        split_indices: np.ndarray,
        transform: Optional[
            Callable
        ] = None,  # takes in a PIL image and returns a transformed version
        transforms: Optional[
            Callable
        ] = None,  # takes in an image and a label and returns the transformed versions of both
        target_transform: Optional[
            Callable
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
        """ """
        df = pd.read_csv(self.root / self.config["name"] / "labels.csv")
        return df

    def _read_labels_mapping_from_file(self) -> pd.DataFrame:
        """ """
        labels_mapping_from_yaml = yaml.safe_load(
            (self.root / self.config["name"] / "labels_mapping.yaml").open()
        )
        return labels_mapping_from_yaml

    def download(self):
        """ """
        pass

    def _check_integrity(self) -> bool:
        """ """
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
        """

        Args:
          idx: int:

        Returns:

        """
        img_path = (
            self.root / self.config["name"] / "data" / self.labels.iloc[idx]["image_id"]
        )
        img = PIL.Image.open(img_path)
        return img

    def _load_label(self, idx: int):
        """

        Args:
          idx: int:

        Returns:

        """
        label = self.labels.iloc[idx][self.config["target"]]
        return label

    def _load_fairness_attributes(self, idx: int) -> dict:
        """

        Args:
          idx: int:

        Returns:

        """
        fairness_attributes = {}
        for key, value in self.labels_mapping.items():
            # fairness_attributes[key] = value[self.labels.iloc[idx][key]]
            fairness_attributes[key] = self.labels.iloc[idx][key]
        return fairness_attributes

    def get_class_names(self) -> List[str]:
        """ """
        return [
            f"{self.config['target']}_{str(item).replace(' ', '_')}"
            for key, item in self.labels_mapping[self.config["target"]].items()
        ]

    # def get_target_label_translation(self) -> dict:
    #     return {i: name for i, name in enumerate(self.get_class_names())}

    def get_num_classes(self) -> int:
        """ """
        return len(self.labels_mapping[self.config["target"]])

    def get_collate_fn(self, protected_attribute: str, protected_attribute_value: str):
        """

        Args:
          protected_attribute: str:
          protected_attribute_value: str:

        Returns:

        """

        def collate_fn(
            batch: List[Tuple[torch.Tensor, str, Dict[str, Union[str, int]]]],
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """

            Args:
              batch: List[Tuple[torch.Tensor:
              str:
              Dict[str:
              Union[str:
              int]]]]:

            Returns:

            """
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
