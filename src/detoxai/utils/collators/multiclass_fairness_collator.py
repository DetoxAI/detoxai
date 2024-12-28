from typing import Dict, List, Union

from .fairness_collator import FairnessCollator
from .multiclass_collator import MultiClassCollator


class MultiClassFairnessCollator(MultiClassCollator, FairnessCollator):
    def __init__(
        self,
        class_names: List[str],
        protected_attribute: str,
        protected_attribute_value: Union[str, int],
        *args,
        **kwargs,
    ):
        super().__init__(
            class_names=class_names,
            protected_attribute=protected_attribute,
            protected_attribute_value=protected_attribute_value,
            *args,
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"MultiClassFairnessCollator(\n"
            f"  class_names: {self.class_names},\n"
            f"  protected_attribute: {self.protected_attribute},\n"
            f"  protected_attribute_value: {self.protected_attribute_value}\n)"
        )

    @classmethod
    def from_config(cls, config: Dict) -> "MultiClassFairnessCollator":
        class_names = config["class_names"]
        protected_attribute = config["protected_attribute"]
        protected_attribute_value = config["protected_attribute_value"]
        return cls(
            class_names=class_names,
            protected_attribute=protected_attribute,
            protected_attribute_value=protected_attribute_value,
        )
