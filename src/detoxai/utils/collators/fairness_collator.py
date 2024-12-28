from typing import Dict, List, Tuple, Union

import numpy as np
import torch

from .base_collator import BaseCollator


class FairnessCollator(BaseCollator):
    def __init__(
        self,
        class_names: List[str],
        protected_attribute: str,
        protected_attribute_value: Union[str, int],
        device: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(class_names, device, *args, **kwargs)
        self.protected_attribute = protected_attribute
        self.protected_attribute_value = protected_attribute_value

    def __repr__(self) -> str:
        return (
            f"FairnessCollator(\n"
            f"  class_names: {self.class_names},\n"
            f"  protected_attribute: {self.protected_attribute},\n"
            f"  protected_attribute_value: {self.protected_attribute_value}\n)"
        )

    @classmethod
    def from_config(cls, config: Dict) -> "FairnessCollator":
        class_names = config["class_names"]
        protected_attribute = config["protected_attribute"]
        protected_attribute_value = config["protected_attribute_value"]
        return cls(
            class_names=class_names,
            protected_attribute=protected_attribute,
            protected_attribute_value=protected_attribute_value,
        )

    def get_collate_fn(self):
        def collate_fn(
            batch: List[Tuple[torch.Tensor, str, Dict[str, Union[str, int]]]],
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            images = torch.stack([item[0] for item in batch])
            labels = torch.tensor([self.label_translation[item[1]] for item in batch])
            protected_attributes = torch.tensor(
                [
                    int(
                        item[2].get(self.protected_attribute)
                        == self.protected_attribute_value
                    )
                    for item in batch
                ]
            )

            if self.device is not None:
                images = images.to(self.device)
                labels = labels.to(self.device)
                protected_attributes = protected_attributes.to(self.device)

            return images, labels, protected_attributes

        return collate_fn

    def get_memmap_collate_fn(self):
        def collate_fn(
            batch: List[Tuple[np.ndarray, int, int]],
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            images = torch.stack(
                [torch.tensor(item[0].reshape(3, 28, 28)) for item in batch]
                # [torch.tensor(item[0].reshape(3, 218, 218)) for item in batch] #celeba
            )
            labels = torch.tensor(
                [torch.tensor(item[1], dtype=torch.long) for item in batch]
            )
            protected_attributes = torch.tensor(
                [torch.tensor(item[2], dtype=torch.long) for item in batch]
            )

            if self.device is not None:
                images = images.to(self.device)
                labels = labels.to(self.device)
                protected_attributes = protected_attributes.to(self.device)

            return images, labels, protected_attributes

        return collate_fn

    def translate_back_item(
        self, item: Tuple[torch.Tensor, int, int]
    ) -> Tuple[torch.Tensor, str, Union[str, None]]:
        image, label, protected_attribute = item
        return (
            image,
            self.translate_back_label(label),
            self.protected_attribute_value if protected_attribute == 1 else None,
        )

    def translate_back_batch(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> List[Tuple[torch.Tensor, str, Union[str, None]]]:
        images, labels, protected_attributes = batch
        translated_labels = [
            self.translate_back_label(label.item()) for label in labels
        ]
        return [
            (
                image,
                label,
                self.protected_attribute_value if protected_attribute == 1 else None,
            )
            for image, label, protected_attribute in zip(
                images, translated_labels, protected_attributes
            )
        ]

    def get_group_labels(self) -> List[int]:
        return [0, 1]

    def get_group_label_names(self) -> List[str]:
        return [
            "Not_"
            + str(self.protected_attribute)
            + "_"
            + str(self.protected_attribute_value),
            str(self.protected_attribute) + "_" + str(self.protected_attribute_value),
        ]
