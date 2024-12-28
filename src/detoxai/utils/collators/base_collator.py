from typing import Dict, List, Tuple

import numpy as np
import torch

# from ..datasets import MemMappedDataset


class BaseCollator:
    def __init__(self, class_names: List[str], device: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_names = class_names
        self.label_translation = {name: i for i, name in enumerate(class_names)}
        self.device = device

    def __repr__(self) -> str:
        return (
            f"BaseCollator(\n"
            f"  class_names: {self.class_names},\n"
            f"  Label Translation: {self.label_translation}\n)"
        )

    @classmethod
    def from_config(cls, config: Dict) -> "BaseCollator":
        class_names = config["class_names"]
        return cls(class_names=class_names)

    def infer_best_collate_fn(self, dataset: torch.utils.data.Dataset):
        # if isinstance(dataset, MemMappedDataset):
        #     return self.get_memmap_collate_fn()
        return self.get_collate_fn()

    def get_collate_fn(self):
        return self.get_base_collate_fn()

    def get_base_collate_fn(self):
        def collate_fn(
            batch: List[Tuple[torch.Tensor, str]],
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            images = torch.stack([item[0] for item in batch])
            labels = torch.tensor([self.label_translation[item[1]] for item in batch])

            if self.device is not None:
                images = images.to(self.device)
                labels = labels.to(self.device)

            return images, labels

        return collate_fn

    def get_memmap_collate_fn(self):
        def collate_fn(
            batch: List[Tuple[np.ndarray, int]],
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            images = torch.stack(
                [torch.tensor(item[0].reshape(3, 28, 28)) for item in batch]
            )
            labels = torch.tensor(
                [torch.tensor(item[1], dtype=torch.long) for item in batch]
            )

            if self.device is not None:
                images = images.to(self.device)
                labels = labels.to(self.device)

            return images, labels

        return collate_fn

    def translate_back_label(self, label: int) -> str:
        return self.class_names[label]

    def translate_back_item(
        self, item: Tuple[torch.Tensor, int]
    ) -> Tuple[torch.Tensor, str]:
        image, label = item
        return image, self.translate_back_label(label)

    def translate_back_batch(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> List[Tuple[torch.Tensor, str]]:
        images, labels = batch
        output = []
        for image, label in zip(images, labels):
            output.append((image, self.translate_back_label(label.item())))
        return output

    def get_num_classes(self) -> int:
        return len(self.class_names)

    def get_class_names(self, to_strings=True) -> List[str]:
        if to_strings:
            return [str(name) for name in self.class_names]
        return [name for name in self.class_names]
