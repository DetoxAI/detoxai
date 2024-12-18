import logging
from EasyMCDM.models.Pareto import Pareto
from .results_class import CorrectionResult

logger = logging.getLogger(__name__)

# IF YOU ADD A NEW METRIC, MAKE SURE TO ADD IT TO MINIMIZE IF IT IS A COST TYPE METRIC
MINIMIZE = ["EOO", "DP", "EO"]


def filter_pareto_front(results: list[CorrectionResult]) -> list[CorrectionResult]:
    """
    Filter the results to only include those on the pareto front

    Args:
        results: List of CorrectionResult objects to filter
    """

    metrics = results[0].get_all_metrics()["pareto"].keys()
    data = {}
    for result in results:
        data[result] = [
            result.get_all_metrics()["pareto"][metric] for metric in metrics
        ]

    prefs = ["min" if metric in MINIMIZE else "max" for metric in metrics]
    indices = list(range(len(prefs)))

    p = Pareto(data)
    res = p.solve(indexes=indices, prefs=prefs)

    mask = []

    for _, d in res.items():
        w_dom = len(d["Weakly-dominated-by"])
        dom = len(d["Dominated-by"])

        if w_dom == 0 and dom == 0:
            mask.append(True)
        else:
            mask.append(False)

    return [result for result, m in zip(results, mask) if m]


def select_best_method(results: list[CorrectionResult]) -> CorrectionResult:
    """
    Select the best correction method from the results using the ideal point method

    Args:
        results: List of CorrectionResult objects to choose from
    """
    pf = filter_pareto_front(results)

    if len(pf) == 0:
        mess = "No methods on the pareto front, defaulting to ideal point method on all results"
        logger.warning(mess)
        pf = results

    metrics = results[0].get_all_metrics()["pareto"].keys()

    # Get the ideal point
    ideal_point = [0] * len(metrics)
    for result in pf:
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

    for result in results:
        score = 0
        for i, met in enumerate(metrics):
            v = result.get_metric(met)
            score += abs(v - ideal_point[i])

        if best_score is None or score < best_score:
            best_score = score
            best_method = result

    return best_method
