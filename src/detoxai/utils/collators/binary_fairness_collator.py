from typing import Dict, List, Union

from .binary_collator import BinaryCollator
from .fairness_collator import FairnessCollator


class BinaryClassFairnessCollator(BinaryCollator, FairnessCollator):
    def __init__(
        self,
        class_names: List[str],
        protected_attribute: str,
        protected_attribute_value: Union[str, int],
        target_class: str,
        *args,
        **kwargs,
    ):
        super().__init__(
            class_names=class_names,
            target_class=target_class,
            protected_attribute=protected_attribute,
            protected_attribute_value=protected_attribute_value,
            *args,
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"BinaryClassFairnessCollator(\n"
            f"  class_names: {self.class_names},\n"
            f"  target_class: {self.target_class},\n"
            f"  protected_attribute: {self.protected_attribute},\n"
            f"  protected_attribute_value: {self.protected_attribute_value}\n)"
        )

    @classmethod
    def from_config(cls, config: Dict) -> "BinaryClassFairnessCollator":
        class_names = config["class_names"]
        target_class = config["target_class"]
        protected_attribute = config["protected_attribute"]
        protected_attribute_value = config["protected_attribute_value"]
        return cls(
            class_names=class_names,
            target_class=target_class,
            protected_attribute=protected_attribute,
            protected_attribute_value=protected_attribute_value,
        )
