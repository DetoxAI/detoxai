import logging

import numpy as np

from .results_class import CorrectionResult

logger = logging.getLogger(__name__)

# IF YOU ADD A NEW METRIC, MAKE SURE TO ADD IT TO MINIMIZE IF IT IS A COST TYPE METRIC
MINIMIZE = ["EOO", "DP", "EO"]


# Faster than is_pareto_efficient_simple, but less readable.
def is_pareto_efficient(costs: np.ndarray, return_mask: bool = True) -> np.ndarray:
    """Find the pareto-efficient points

    Args:
      costs: An (n_points, n_costs) array
      return_mask: True to return a mask
      costs: np.ndarray:
      return_mask: bool:  (Default value = True)

    Returns:
      An array of indices of pareto-efficient points.
      If return_mask is True, this will be an (n_points, ) boolean array
      Otherwise it will be a (n_efficient_points, ) integer array of indices.

      Credit: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python

    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def filter_pareto_front(
    results: dict[str, CorrectionResult],
) -> dict[str, CorrectionResult]:
    """
    Filter the results to only include those on the pareto front

    Args:
      results: List of CorrectionResult objects to filter
      results: list[CorrectionResult]:

    Returns:

    """

    metrics = list(results.values())[0].get_all_metrics()["pareto"].keys()
    data = []
    for method, result in results.items():
        d = []
        for met in metrics:
            if met in MINIMIZE:
                d.append(result.get_metric(met))
            else:
                d.append(-result.get_metric(met))
        data.append(d)

    data = np.array(data)
    mask = is_pareto_efficient(data)

    logger.info(f"Pareto front: {list(zip(results, mask))}")

    return {method: result for (method, result), m in zip(results.items(), mask) if m}


def select_best_method(results: dict[str, CorrectionResult]) -> CorrectionResult:
    """
    Select the best correction method from the results using the ideal point method

    Args:
      results: List of CorrectionResult objects to choose from
      results: list[CorrectionResult]:

    Returns:

    """
    pf = filter_pareto_front(results)

    if len(pf) == 0:
        mess = "No methods on the pareto front, defaulting to ideal point method on all results"
        logger.warning(mess)
        pf = results

    metrics = list(results.values())[0].get_all_metrics()["pareto"].keys()

    # Get the ideal point
    ideal_point = [0] * len(metrics)
    for result in pf.values():
        for i, met in enumerate(metrics):
            v = result.get_metric(met)
            if met in MINIMIZE:
                if met in MINIMIZE:
                    ideal_point[i] = min(ideal_point[i], v)
                else:
                    ideal_point[i] = max(ideal_point[i], v)

    # Get the best method as L1 distance from the ideal point
    best_method = None
    best_score = None

    for result in results.values():
        score = 0
        for i, met in enumerate(metrics):
            v = result.get_metric(met)
            score += abs(v - ideal_point[i])

        if best_score is None or score < best_score:
            best_score = score
            best_method = result

    return best_method
