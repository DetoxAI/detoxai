from .results_class import CorrectionResult


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
