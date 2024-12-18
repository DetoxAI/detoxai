from typing import Dict, List

from .base_collator import BaseCollator


class MultiClassCollator(BaseCollator):
    def __init__(self, class_names: List[str], *args, **kwargs):
        super().__init__(class_names, *args, **kwargs)

    def __repr__(self) -> str:
        return (
            f"MultiClassCollator(\n"
            f"  class_names: {self.class_names},\n"
            f"  Label Translation: {self.label_translation}\n)"
        )

    @classmethod
    def from_config(cls, config: Dict) -> "MultiClassCollator":
        class_names = config["class_names"]
        return cls(class_names=class_names)
