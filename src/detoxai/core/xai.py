import numpy as np
from abc import ABC, abstractmethod
import scipy.stats as stats
import torch.nn as nn
import torch
from tqdm import tqdm

from ..utils.dataloader import DetoxaiDataLoader
from ..visualization import LRPHandler, ConditionOn


class XAIMetricsCalculator:
    def __init__(self, dataloader: DetoxaiDataLoader, lrphandler: LRPHandler) -> None:
        self.dataloader = dataloader
        self.lrphandler = lrphandler

    def _symmetrize(
        self, sailmaps: np.ndarray, neutral_point: float = 0.5
    ) -> np.ndarray:
        return np.abs(sailmaps - neutral_point)

    def calculate_metrics(
        self,
        model: nn.Module,
        rect_pos: tuple[int, int],
        rect_size: tuple[int, int],
        vanilla_model: nn.Module = None,
        sailmap_metrics: list[str] = [
            "RRF",
            "HRF",
            "MRR",
            "DET",
            "ADR",
            "DIF",
            "RDDT",
        ],
        batches: int = 2,
        condition_on: str = ConditionOn.PROPER_LABEL.value,
        verbose: bool = False,
        # source_range: tuple[float, float] = (0, 1),
        neutral_point: float = 0.5,
        abs_on_neutral: bool = True,
    ) -> dict[str, float]:
        """
        Calculate the metrics for the given model and sailmaps

        Parameters:
            - `model` (nn.Module): The model to use for LRP
            - `rect_pos` (tuple[int, int]): The position of the rectangle
            - `rect_size` (tuple[int, int]): The size of the rectangle
            - `vanilla_model` (nn.Module): The vanilla model to use for comparison
            - `sailmap_metrics` (list[str]): The list of metrics to calculate
            - `batches` (int): The number of batches to calculate
            - `condition_on` (str): The condition to calculate the metrics on
            - `verbose` (bool): Whether to show progress bar or not

        Returns:
            - `dict[str, float]`: The calculated metrics where the key is the metric name and
                the value is the calculated metric

        """
        metrics_calcs: list["SailRectMetric"] = []

        for metric in sailmap_metrics:
            if metric == "RRF":
                metrics_calcs.append(RRF())
            elif metric == "HRF":
                metrics_calcs.append(HRF())
            elif metric == "MRR":
                metrics_calcs.append(MRR())
            elif metric == "DET":
                metrics_calcs.append(DET())
            elif metric == "ADR":
                if vanilla_model is None:
                    raise ValueError("ADR requires a vanilla model for comparison")
                metrics_calcs.append(ADR())
            elif metric == "DIF":
                if vanilla_model is None:
                    raise ValueError("DIF requires a vanilla model for comparison")
                metrics_calcs.append(DIF())
            elif metric == "RDDT":
                if vanilla_model is None:
                    raise ValueError("RDDT requires a vanilla model for comparison")
                metrics_calcs.append(RDDT())
            else:
                raise ValueError(f"Metric {metric} is not supported")

        for i in tqdm(range(batches), disable=not verbose, desc="Calculating metrics"):
            lrpres = self.lrphandler.calculate(model, self.dataloader, batch_num=i)

            if vanilla_model is not None:
                vanilla_lrpres = self.lrphandler.calculate(
                    vanilla_model, self.dataloader, batch_num=i
                )

            _, labels, _ = self.dataloader.get_nth_batch(i)  # noqa

            conditioned = []
            for i, label in enumerate(labels):
                # Assuming binary classification
                label = (
                    label
                    if condition_on == ConditionOn.PROPER_LABEL.value
                    else 1 - label
                )
                conditioned.append(lrpres[label, i])

            sailmaps: torch.Tensor = torch.stack(conditioned).to(dtype=float)
            sailmaps = sailmaps.cpu().detach().numpy()
            if abs_on_neutral:
                sailmaps = self._symmetrize(sailmaps, neutral_point)

            if vanilla_model is not None:
                vanilla_sailmaps = torch.stack(
                    [vanilla_lrpres[label, i] for i, label in enumerate(labels)]
                ).to(dtype=float)
                vanilla_sailmaps = vanilla_sailmaps.cpu().detach().numpy()

                if abs_on_neutral:
                    vanilla_sailmaps = self._symmetrize(vanilla_sailmaps, neutral_point)

            for metric in metrics_calcs:
                if isinstance(metric, (ADR, DIF, RDDT)):
                    metric.aggregate(sailmaps, rect_pos, rect_size, vanilla_sailmaps)
                else:
                    metric.aggregate(sailmaps, rect_pos, rect_size)

        ret = {}
        for metric in metrics_calcs:
            ret[str(metric)] = metric.reduce()

        return ret


class SailRectMetric(ABC):
    """ """

    def __init__(self) -> None:
        self.sailmaps = None
        self.metvals: np.ndarray = []

    def _sailmaps_rect(
        self,
        sailmaps: np.ndarray,
        rect_pos: tuple[int, int],
        rect_size: tuple[int, int],
    ) -> np.ndarray:
        assert isinstance(sailmaps, np.ndarray), "Sailmaps should be a numpy array"

        return sailmaps[
            :,
            rect_pos[0] : rect_pos[0] + rect_size[0],
            rect_pos[1] : rect_pos[1] + rect_size[1],
        ]

    def calculate_batch(
        self,
        sailmaps: np.ndarray,
        rect_pos: tuple[int, int],
        rect_size: tuple[int, int],
        ret_format: tuple[str] = ("mean", "std"),
    ) -> dict[str, float]:
        """
        Calculate the metric for a single batch of sailmaps
        """
        c = self._core(sailmaps, rect_pos, rect_size)
        return self.structure_output(c, ret_format)

    def reduce(self, ret_format: tuple[str] = ("mean", "std")) -> dict[str, float]:
        """
        Calculate the metric for already aggregated sailmaps
        """
        return self.structure_output(self.metvals, ret_format)

    def aggregate(
        self,
        sailmaps: np.ndarray,
        rect_pos: tuple[int, int],
        rect_size: tuple[int, int],
        vanilla_sailmaps: np.ndarray = None,
    ):
        """
        Aggregate sailmaps for later calculation
        """
        if vanilla_sailmaps is not None:
            c = self._core(sailmaps, rect_pos, rect_size, vanilla_sailmaps)
        else:
            c = self._core(sailmaps, rect_pos, rect_size)

        assert isinstance(c, np.ndarray), "Output should be a numpy array"
        self.metvals.extend(c)

    @abstractmethod
    def _core(
        self,
        sailmaps: np.ndarray,
        rect_pos: tuple[int, int],
        rect_size: tuple[int, int],
    ) -> np.ndarray:
        pass

    def structure_output(
        self, per_sample: np.ndarray[float], ret_format: tuple[str] = ("mean", "std")
    ) -> dict[str, float]:
        ret = {}
        if "mean" in ret_format:
            ret["mean"] = np.mean(per_sample)

        if "std" in ret_format:
            ret["std"] = np.std(per_sample)

        if "min" in ret_format:
            ret["min"] = np.min(per_sample)

        if "max" in ret_format:
            ret["max"] = np.max(per_sample)

        if "median" in ret_format:
            ret["median"] = np.median(per_sample)

        return ret

    def __str__(self) -> str:
        if hasattr(self, "name"):
            return self.name

        return self.__class__.__name__

    def __repr__(self) -> str:
        return self.__str__()


class RRF(SailRectMetric):
    """
    Rectangle Relevance Fraction
    \begin{equation}
    \mathbf{RRF} = \frac{\displaystyle \sum_{(i,j) \in R} p_{ij}}{\displaystyle \sum_{i = 1}^N \sum_{j = 1}^M p_{ij}}
    \end{equation}

    Here, $\mathbf{RRF}$ measures the fraction of total relevance that falls within ROI.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = "RRF"

    def _core(
        self,
        sailmaps: np.ndarray,
        rect_pos: tuple[int, int],
        rect_size: tuple[int, int],
    ) -> np.ndarray:
        sm_rect = self._sailmaps_rect(sailmaps, rect_pos, rect_size)

        r_sum = sm_rect.reshape(len(sm_rect), -1).sum(axis=1)
        s_sum = sailmaps.reshape(len(sm_rect), -1).sum(axis=1)

        return r_sum / s_sum  # safe bc s_sum > r_sum and never 0


class HRF(SailRectMetric):
    """
    \subsection{High-Relevance Fraction (HRF)}
    \begin{equation}
    \mathbf{HRF} = \displaystyle \frac{1}{\vert R \vert} \sum_{(i,j) \in R} \mathbbm{1}_{\{p_{ij} > \epsilon\}}
    \end{equation}

    $\mathbf{HRF}$ quantifies the proportion of pixels inside the ROI whose relevance exceeds a predefined threshold $\epsilon$, indicating how many pixels are highly important for prediction.
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon

        self.name = "HRF"

    def _core(
        self,
        sailmaps: np.ndarray,
        rect_pos: tuple[int, int],
        rect_size: tuple[int, int],
    ) -> np.ndarray:
        sm_rect = self._sailmaps_rect(sailmaps, rect_pos, rect_size)

        print(
            f"Describe qunaitles etc summary: {np.quantile(sm_rect, [0.25, 0.5, 0.75])}"
        )
        print(f"epsilon: {self.epsilon}")

        high_relevance = np.sum(sm_rect > self.epsilon, axis=1)
        return high_relevance / sm_rect.size


class MRR(SailRectMetric):
    """
        \subsection{Mean Relevance Ratio (MRR)}

    \begin{equation}
        \mathbf{MRR} = \frac{\displaystyle \frac{1}{\vert R \vert} \sum_{(i,j) \in R} p_{ij}}{\displaystyle \frac{1}{N M - \vert R \vert} \sum_{(i,j) \notin R} p_{ij}},
    \end{equation}
    $\mathbf{MRR}$ quantifies the ratio of the mean pixel value inside the ROI to the mean pixel value outside it. $\mathbf{MRR} = 1$ indicates that the mean values are equal, while $\mathbf{MRR} > 1$ says the mean pixel within the ROI has a higher intensity.

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = "MRR"

    def _core(
        self,
        sailmaps: np.ndarray,
        rect_pos: tuple[int, int],
        rect_size: tuple[int, int],
    ) -> np.ndarray:
        sm_rect = self._sailmaps_rect(sailmaps, rect_pos, rect_size)
        sm_outside = sailmaps.copy()
        sm_outside[
            :,
            rect_pos[0] : rect_pos[0] + rect_size[0],
            rect_pos[1] : rect_pos[1] + rect_size[1],
        ] = 0

        sm_outside_sum = sm_outside.reshape(len(sm_outside), -1).sum(axis=1)
        total_pixels = sm_outside[0].size
        rect_pixels = sm_rect[0].size

        sm_outside_mean = sm_outside_sum / (total_pixels - rect_pixels)
        sm_rect_mean = sm_rect.reshape(len(sm_rect), -1).sum(axis=1) / rect_pixels

        return sm_rect_mean / sm_outside_mean  #


class DET(SailRectMetric):
    """ 
    
    \subsection{Distribution Equivalence Testing (DET)}

    The goal of the statistical test is to determine whether the pixels \textit{inside} the rectangle have higher intensity than those \textit{outside} the rectangle. Since the number of pixels and their intensity distributions inside and outside the ROI can vary, a non-parametric, unpaired statistical Mann-Whitney-Wilcoxon test is used. This permutation test assesses whether the intensity values from one group (inside) tend to be higher than those from the other (outside).

    The null hypothesis $H_0$ for the test is that the intensity distributions inside and outside the rectangle are equal:
    \begin{equation}
    \begin{split}
        H_0: F_{\text{inside}}(x) &= F_{\text{outside}}(x) \\
        H_1: F_{\text{inside}}(x) &> F_{\text{outside}}(x)
    \end{split}
    \end{equation}

    To perform the test, all pixel intensities are ranked, and the sum of ranks for each group (inside and outside the ROI) is computed. The test then evaluates the probability that the intensity values inside the rectangle are statistically higher than those outside. The final outcome of the DET is a binary decision: \textbf{TRUE} indicates that the null hypothesis is rejected (i.e., there is statistically significant evidence that the pixels inside the rectangle have higher intensity), while \textbf{FALSE} signifies that we fail to reject the null hypothesis, meaning that the evidence is inconclusive regarding a higher intensity inside the rectangle.

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = "DET"

    def _core(
        self,
        sailmaps: np.ndarray,
        rect_pos: tuple[int, int],
        rect_size: tuple[int, int],
    ) -> np.ndarray:
        sm_rect = self._sailmaps_rect(sailmaps, rect_pos, rect_size)
        sm_outside = sailmaps.copy()
        sm_outside[
            :,
            rect_pos[0] : rect_pos[0] + rect_size[0],
            rect_pos[1] : rect_pos[1] + rect_size[1],
        ] = 0

        scores = np.zeros(sm_rect.shape[0])
        # Per image
        for i in range(sm_rect.shape[0]):
            _, p = stats.mannwhitneyu(
                sm_rect[i].flatten(), sm_outside[i].flatten(), alternative="greater"
            )

            if p < 0.01:
                scores[i] = 1

        return scores


class ADR(SailRectMetric):
    """
    Average Difference in Region (ADR)
    
    ADR measures the mean pixel-wise difference between vanilla and debiased saliency maps 
    within the region of interest (ROI). A positive value indicates that vanilla saliency 
    values are generally higher than debiased ones in the region.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = "ADR"

    def _core(
        self,
        sailmaps: np.ndarray,
        rect_pos: tuple[int, int],
        rect_size: tuple[int, int],
        vanilla_sailmaps: np.ndarray = None,
    ) -> np.ndarray:
        sm_rect = self._sailmaps_rect(sailmaps, rect_pos, rect_size)
        vanilla_sm_rect = self._sailmaps_rect(vanilla_sailmaps, rect_pos, rect_size)
        
        # Calculate mean difference per image
        diff = vanilla_sm_rect - sm_rect
        return diff.reshape(len(diff), -1).mean(axis=1)


class DIF(SailRectMetric):
    """
    Decreased Intensity Fraction (DIF)
    
    DIF measures the ratio of pixels showing decreased intensity in the debiased model 
    compared to the vanilla model. It represents the fraction of pixels inside a rectangle 
    that significantly flipped their saliency value.
    """

    def __init__(self, threshold: float = 0.1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = "DIF"
        self.threshold = threshold

    def _core(
        self,
        sailmaps: np.ndarray,
        rect_pos: tuple[int, int],
        rect_size: tuple[int, int],
        vanilla_sailmaps: np.ndarray = None,
    ) -> np.ndarray:
        sm_rect = self._sailmaps_rect(sailmaps, rect_pos, rect_size)
        vanilla_sm_rect = self._sailmaps_rect(vanilla_sailmaps, rect_pos, rect_size)
        
        # Calculate fraction of pixels where debiased < vanilla
        diff = vanilla_sm_rect - sm_rect
        decreased = (diff > self.threshold).reshape(len(diff), -1)
        return decreased.sum(axis=1) / decreased.shape[1]


class RDDT(SailRectMetric):
    """
    Rectangle Difference Distribution Testing (RDDT)
    
    Performs a Wilcoxon signed rank test to determine if pixels from the vanilla model
    have significantly higher intensity than those from the debiased model within the ROI.
    Returns 1 if the test rejects the null hypothesis (indicating vanilla has higher intensity),
    0 otherwise.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = "RDDT"

    def _core(
        self,
        sailmaps: np.ndarray,
        rect_pos: tuple[int, int],
        rect_size: tuple[int, int],
        vanilla_sailmaps: np.ndarray = None,
    ) -> np.ndarray:
        sm_rect = self._sailmaps_rect(sailmaps, rect_pos, rect_size)
        vanilla_sm_rect = self._sailmaps_rect(vanilla_sailmaps, rect_pos, rect_size)
        
        scores = np.zeros(sm_rect.shape[0])
        # Per image
        for i in range(sm_rect.shape[0]):
            _, p = stats.wilcoxon(
                vanilla_sm_rect[i].flatten(),
                sm_rect[i].flatten(),
                alternative="greater"
            )
            if p < 0.01:
                scores[i] = 1
                
        return scores


