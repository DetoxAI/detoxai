from typing import Dict, List

from .base_collator import BaseCollator


class BinaryCollator(BaseCollator):
    def __init__(self, class_names: List[str], target_class: str, *args, **kwargs):
        super().__init__(class_names, *args, **kwargs)
        self.target_class = target_class
        assert target_class in class_names  # target class must be in class names
        self.label_translation = {
            name: (1 if name == target_class else 0) for name in class_names
        }

    def __repr__(self) -> str:
        return (
            f"BinaryCollator(\n"
            f"  class_names: {self.class_names},\n"
            f"  target_class: {self.target_class},\n"
            f"  Label Translation: {self.label_translation}\n)"
        )

    @classmethod
    def from_config(cls, config: Dict) -> "BinaryCollator":
        class_names = config["class_names"]
        target_class = config["target_class"]
        return cls(class_names=class_names, target_class=target_class)

    def get_num_classes(self) -> int:
        return 2

    def get_class_names(self) -> List[str]:
        return ["Not_" + str(self.target_class), str(self.target_class)]

    def translate_back_label(self, label: int) -> str:
        return self.target_class if label == 1 else f"Not_{self.target_class}"
