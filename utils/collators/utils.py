from typing import Dict, List

from .base_collator import BaseCollator
from .binary_collator import BinaryCollator
from .binary_fairness_collator import BinaryClassFairnessCollator
from .multiclass_collator import MultiClassCollator
from .multiclass_fairness_collator import MultiClassFairnessCollator


# create factory method that will infer the correct collator based on the config
def infer_and_create_collator_from_config(config: Dict) -> BaseCollator:
    if "protected_attribute" in config:
        if "target_class" in config:
            return BinaryClassFairnessCollator.from_config(config)
        return MultiClassFairnessCollator.from_config(config)
    elif "target_class" in config:
        return BinaryCollator.from_config(config)
    return MultiClassCollator.from_config(config)


# create factory method that will infer the correct collator based on the config
def get_collators_from_config(
    config: Dict, fairness=True, base=True
) -> List[BaseCollator]:
    collators = {}
    if base:
        if "target_class" in config:
            collators["base"] = BinaryCollator.from_config(config)
        else:
            collators["base"] = MultiClassCollator.from_config(config)
    if fairness:
        if "protected_attribute" in config:
            if "target_class" in config:
                collators["fairness"] = BinaryClassFairnessCollator.from_config(config)
            else:
                collators["fairness"] = MultiClassFairnessCollator.from_config(config)
    return collators
