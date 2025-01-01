import torch.nn as nn
from copy import deepcopy
import logging
import traceback

# Project imports
from ..methods import (
    SavaniRP,
    SavaniLWO,
    SavaniAFT,
    ZhangM,
    RRCLARC,
    PCLARC,
    ACLARC,
    LEACE,
    RejectOptionClassification,
)
from .model_wrappers import FairnessLightningWrapper
from .results_class import CorrectionResult
from ..utils.dataloader import WrappedDataLoader
from ..metrics.fairness_metrics import AllMetrics
from .evaluation import evaluate_model
from .mcda_helpers import filter_pareto_front, select_best_method
from .interface_helpers import construct_metrics_config

logger = logging.getLogger(__name__)

SUPPORTED_METHODS = [
    "SAVANIRP",
    "SAVANILWO",
    "SAVANIAFT",
    "ZHANGM",
    "RRCLARC",
    "PCLARC",
    "ACLARC",
    "LEACE",
    "RejectOptionClassification",
]


DEFAULT_METHODS_CONFIG = {
    "global": {
        "last_layer_name": "last",
        "experiment_name": "default",
        "device": "cpu",
        "dataloader": None,
    },
    "PCLARC": {
        "cav_type": "signal",
        "cav_layers": "penultimate",
        "use_cache": True,
    },
    "ACLARC": {
        "cav_type": "signal",
        "cav_layers": "penultimate",
        "use_cache": True,
    },
    "RRCLARC": {
        "cav_type": "signal",
        "cav_layers": "penultimate",
        "use_cache": True,
    },
    "LEACE": {
        "intervention_layers": "penultimate",
        "use_cache": True,
    },
    "SAVANIRP": {
        "frac_of_batches_to_use": 0.15,
    },
    "SAVANILWO": {
        "frac_of_batches_to_use": 0.15,
        "n_layers_to_optimize": 4,
    },
    "SAVANIAFT": {
        "frac_of_batches_to_use": 0.15,
    },
    "ZHANGM": {
        "frac_of_batches_to_use": 0.15,
    },
    "RejectOptionClassification": {
        "theta_range": (0.55, 0.95),
        "theta_steps": 20,
        "metrics_spec": {
            "EqualizedOdds": {"reduce": ["difference"]},
        },
        "objective_function": lambda fairness, accuracy: fairness * accuracy,
    },
}


def debias(
    model: nn.Module,
    dataloader: WrappedDataLoader,  # bez concept labeli
    # harmful_concept: str,
    methods: list[str] | str = "all",
    metrics: list[str] | str = "all",
    methods_config: dict = DEFAULT_METHODS_CONFIG,
    pareto_metrics: list[str] = ["balanced_accuracy", "equalized_odds"],
    return_type: str = "pareto-front",
    device: str = "cpu",
) -> CorrectionResult | list[CorrectionResult]:
    """
    Run a suite of correction methods on the model and return the results

    Args:
        `model`: Model to run the correction methods on
        `dataloader`: WrappedDataLoader object with the dataset
        `harmful_concept`: Concept to debias -- this is the protected attribute # NOT SUPPORTED YET
        `methods`: List of correction methods to run
        `metrics`: List of metrics to include in the configuration
        `methods_config`: Configuration for each correction method
        `pareto_metrics`: List of metrics to use for the pareto front and selection of best method
        `return_type` (optional): Type of results to return. Options are 'pareto-front', 'all', 'best'
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

        # Capitalize all methods
        methods = [method.upper() for method in methods]

    methods_config["global"]["device"] = device

    # # ------------------------------------------------
    # # DATASET HANDLING IS TODO HERE
    # # Load supported tags ie. protected attributes
    # supported_tags = load_supported_tags()
    # if harmful_concept not in supported_tags["attributes"]:
    #     raise ValueError(
    #         f"Attribute {harmful_concept} not found in supported attributes"
    #     )
    # else:
    #     prot_attr_arity = len(supported_tags["mapping"][harmful_concept])
    #     class_labels = NotImplementedError  # TODO: Take it from somewhere

    # pass

    class_labels = dataloader.collator.get_class_names()
    prot_attr_arity = len(dataloader.collator.get_group_label_names())

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
        logger.info("=" * 50 + f" Running method {method} " + "=" * 50)
        method_kwargs = methods_config[method] | methods_config["global"]
        method_kwargs["model"] = deepcopy(model)
        method_kwargs["dataloader"] = dataloader
        result = run_correction(method, method_kwargs, pareto_metrics)
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
    method_kwargs: dict,
    pareto_metrics: list[str] | None = None,
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
        case "RejectOptionClassification":
            corrector = RejectOptionClassification(**method_kwargs) 
        case _:
            raise ValueError(f"Correction method {method} not found")

    # Parse intervention layers
    if "intervention_layers" in method_kwargs:
        method_kwargs["intervention_layers"] = infer_layers(
            corrector, method_kwargs["intervention_layers"]
        )

    # Parse cav layers
    if "cav_layers" in method_kwargs:
        method_kwargs["cav_layers"] = infer_layers(
            corrector, method_kwargs["cav_layers"]
        )

    # Parse last layer name
    if "last_layer_name" in method_kwargs:
        method_kwargs["last_layer_name"] = infer_layers(
            corrector, method_kwargs["last_layer_name"]
        )[0]

    # Precompute CAVs if required
    if corrector.requires_acts:
        if "intervention_layers" not in method_kwargs:
            lays = method_kwargs["cav_layers"]
        else:
            lays = method_kwargs["intervention_layers"]
        corrector.extract_activations(method_kwargs["dataloader"], lays)

        logger.debug(f"Computing CAVs on layers: {lays}")

        if corrector.requires_cav:
            corrector.compute_cavs(method_kwargs["cav_type"], lays)

    logger.debug(f"Running correction method {method}")

    # Here we finally run the correction method
    try:
        corrector.apply_model_correction(**method_kwargs)
    except Exception as e:
        logger.error(f"Error running correction method {method}: {e}")
        logger.error(traceback.format_exc())
        return None

    logger.debug(f"Correction method {method} applied")

    method_kwargs["model"] = corrector.get_lightning_model()
    metrics = evaluate_model(
        method_kwargs["model"],
        method_kwargs["dataloader"],
        pareto_metrics,
        device=method_kwargs["device"],
    )

    return CorrectionResult(
        method=method, model=method_kwargs["model"], metrics=metrics
    )


def infer_layers(corrector, layers: list[str] | str) -> list[str]:
    """
    Infer the layers to use for the correction method

    Args:
        corrector: Correction method object
        layers: Layer specification

    There are wildcards available:
    - "last": Use the last layer
    - "penultimate": Use the penultimate layer

    Otherwise, a list of *actual* layer names can be passed
    """
    if layers == "last":
        last_layer = list(corrector.model.named_modules())[-1][0]
        return [last_layer]
    elif layers == "penultimate":
        penultimate_layer = list(corrector.model.named_modules())[-2][0]
        return [penultimate_layer]
    elif isinstance(layers, list):
        # Verify that all layers exist in the model
        all_layers = set([layer[0] for layer in corrector.model.named_modules()])
        if set(layers).issubset(all_layers):
            return layers
        else:
            not_found = set(layers) - all_layers
            raise ValueError(f"Layers {not_found} not found in the model")
    else:
        raise ValueError(f"Invalid layer specification {layers}")
