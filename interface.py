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
from ..src.models.LightningWrapper import BaseLightningWrapper


class CorrectionResult:
    def __init__(self):
        self.method: str
        self.model: BaseLightningWrapper
        self.metrics: dict

    def __str__(self):
        return f"Results for: {self.method}"

    def __repr__(self):
        return self.__str__()

    def get_all_metrics(self) -> dict:
        return self.metrics

    def get_metric(self, metric: str) -> float:
        return self.metrics[metric]

    def get_model(self) -> BaseLightningWrapper:
        return self.model

    def get_method(self) -> str:
        return self.method


def run_correction(method: str, method_kwargs: dict) -> CorrectionResult:
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
    # model = corrector.get_model()
    # metrics = evaluate_model(model)

    return CorrectionResult(
        method=method, model=corrector.get_corrected_model(), metrics={}
    )


def evaluate_model(model: BaseLightningWrapper) -> dict:
    """
    Evaluate the model on various metrics


    ***
    `TEMPLATE FOR METRICS DICT`
    ***

    metrics_dict_template = {
        "pareto": {
            "balanced_accuracy": 0.0,
            "equal_opportunity": 0.0,
        },
        "all": {
            "balanced_accuracy": 0.0,
            "equal_opportunity": 0.0,
            "equalized_odds": 0.0,
            "demographic_parity": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        },
    }

    Args:
        model: Model to evaluate
    """
    raise NotImplementedError


def filter_pareto_front(results: list[CorrectionResult]) -> list[CorrectionResult]:
    """
    Filter the results to only include those on the pareto front

    Args:
        results: List of CorrectionResult objects to filter
    """
    raise NotImplementedError


def select_best_method(results: list[CorrectionResult]) -> CorrectionResult:
    """
    Select the best correction method from the results using the ideal point method

    Args:
        results: List of CorrectionResult objects to choose from
    """
    pf = filter_pareto_front(results)
    raise NotImplementedError


def run_suite(
    methods: list[str],
    model: BaseLightningWrapper,
    methods_config: dict,
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

    results = []
    for method in methods:
        method_kwargs = methods_config[method]
        method_kwargs["model"] = model
        result = run_correction(method, method_kwargs)
        results.append(result)

    if return_type == "pareto-front":
        return filter_pareto_front(results)
    elif return_type == "all":
        return results
    elif return_type == "best":
        return select_best_method(results)
    else:
        raise ValueError(f"Invalid return type {return_type}")
