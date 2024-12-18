import torch.nn as nn
from copy import deepcopy
from torch.utils.data import DataLoader

# Project imports
from .methods import (
    SavaniRP,
    SavaniLWO,
    SavaniAFT,
    ZhangM,
    RRCLARC,
    PCLARC,
    ACLARC,
    LEACE,
)
from .model_wrappers import FairnessLightningWrapper
from .results_class import CorrectionResult
from .metrics.fairness_metrics import AllMetrics
from .evaluation import evaluate_model
from .mcda_helpers import filter_pareto_front, select_best_method
from .interface_helpers import construct_metrics_config, load_supported_tags

SUPPORTED_METHODS = [
    "SAVANIRP",
    "SAVANILWO",
    "SAVANIAFT",
    "ZHANGM",
    "RRCLARC",
    "PCLARC",
    "ACLARC",
    "LEACE",
]


def debias(
    model: nn.Module,
    harmful_concept: str,
    methods: list[str] | str = "all",
    metrics: list[str] | str = "all",
    methods_config: dict = {},
    return_type: str = "pareto-front",
) -> CorrectionResult | list[CorrectionResult]:
    """
    Run a suite of correction methods on the model and return the results

    Args:
        methods: List of correction methods to run
        model: Model to run the correction methods on
        methods_config: Configuration for each correction method
        return_type (optional): Type of results to return. Options are 'pareto-front', 'all', 'best'
            "pareto-front": Return the results CorrectionResult objects only for results on the pareto front
            "all": Return the results for all correction methods
            "best": Return the results for the best correction method, chosen with ideal point method from pareto front


    ***
    `TEMPLATE FOR METHODS CONFIG`

    methods_config_template = {
        "global": {
            "last_layer_name": "fc",
            "epsilon": 0.05,
            "bias_metric": "equal_opportunity",
        },
        "method_specific": {
            r"SavaniLWO": {
                "iterations": 10,
            }
        },
    }

    """
    # Parse methods
    if methods == "all":
        methods = SUPPORTED_METHODS
    else:
        # Ensure all methods passed are supported
        for method in methods:
            if method.upper() not in SUPPORTED_METHODS:
                raise ValueError(f"Method {method} not supported")

    # ------------------------------------------------
    # DATASET HANDLING IS TODO HERE
    # Load supported tags ie. protected attributes
    supported_tags = load_supported_tags()
    if harmful_concept not in supported_tags["attributes"]:
        raise ValueError(
            f"Attribute {harmful_concept} not found in supported attributes"
        )
    else:
        prot_attr_arity = len(supported_tags["mapping"][harmful_concept])
        class_labels = NotImplementedError  # TODO: Take it from somewhere

    pass

    # Create an AllMetrics object
    metrics_calculator = AllMetrics(
        construct_metrics_config(metrics),
        class_labels=class_labels,  # TODO: what is this?
        num_groups=prot_attr_arity,
    )

    # Wrap model
    model = FairnessLightningWrapper(
        model,
        performance_metrics=metrics_calculator.get_performance_metrics(),
        fairness_metrics=metrics_calculator.get_fairness_metrics(),
    )

    results = []
    for method in methods:
        method_kwargs = methods_config[method]
        method_kwargs["model"] = model
        result = run_correction(method, method_kwargs, deepcopy(metrics_calculator))
        results.append(result)

    if return_type == "pareto-front":
        return filter_pareto_front(results)
    elif return_type == "all":
        return results
    elif return_type == "best":
        return select_best_method(results)
    else:
        raise ValueError(f"Invalid return type {return_type}")


def run_correction(
    method: str,
    dataloader: DataLoader,
    method_kwargs: dict,
    metrics_calculator: AllMetrics,
) -> CorrectionResult:
    """
    Run the specified correction method

    Args:
        method: Correction method to run
        kwargs: Arguments for the correction method
    """
    match method.upper():
        case "SAVANIRP":
            corrector = SavaniRP(**method_kwargs)
        case "SAVANILWO":
            corrector = SavaniLWO(**method_kwargs)
        case "SAVANIAFT":
            corrector = SavaniAFT(**method_kwargs)
        case "ZHANGM":
            corrector = ZhangM(**method_kwargs)
        case "RRCLARC":
            corrector = RRCLARC(**method_kwargs)
        case "PCLARC":
            corrector = PCLARC(**method_kwargs)
        case "ACLARC":
            corrector = ACLARC(**method_kwargs)
        case "LEACE":
            corrector = LEACE(**method_kwargs)
        case _:
            raise ValueError(f"Correction method {method} not found")

    try:
        corrector.apply_model_correction(**method_kwargs)
    except Exception as e:
        print(f"Error running correction method {method}: {e}")
        return None

    # TODO: evaluate the model after correction
    model = corrector.get_corrected_model()
    metrics = evaluate_model(model, dataloader, metrics_calculator)

    return CorrectionResult(method=method, model=model, metrics=metrics)
