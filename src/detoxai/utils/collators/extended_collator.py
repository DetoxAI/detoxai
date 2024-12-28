from typing import Dict, List, Tuple

import torch

from .base_collator import BaseCollator


class ExtendedCollator(BaseCollator):
    def __init__(self, class_names: List[str], *args, **kwargs):
        super().__init__(class_names, *args, **kwargs)

    def __repr__(self) -> str:
        return (
            f"ExtendedCollator(\n"
            f"  class_names: {self.class_names},\n"
            f"  Label Translation: {self.label_translation}\n)"
        )

    @classmethod
    def from_config(cls, config: Dict) -> "ExtendedCollator":
        class_names = config["class_names"]
        return cls(class_names=class_names)

    def get_collate_fn(self):
        def collate_fn(
            batch: List[Tuple[torch.Tensor, str, Dict[str, Dict]]],
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            images = torch.stack([item[0] for item in batch])
            labels = torch.tensor([self.label_translation[item[1]] for item in batch])
            extra_dict = torch.tensor([item[2] for item in batch])
            return images, labels, extra_dict

        return collate_fn

    def translate_back_item(
        self, item: Tuple[torch.Tensor, int, Dict[str, Dict]]
    ) -> Tuple[torch.Tensor, str, Dict[str, Dict]]:
        image, label, extra = item
        return (
            image,
            self.translate_back_label(label),
            extra,
        )

    def translate_back_batch(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> List[Tuple[torch.Tensor, str, Dict[str, Dict]]]:
        images, labels, extra = batch
        translated_labels = [
            self.translate_back_label(label.item()) for label in labels
        ]
        return [
            (image, label, extra)
            for image, label, extra in zip(images, translated_labels, extra)
        ]
