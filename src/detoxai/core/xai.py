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

    def calculate_metrics(
        self,
        model: nn.Module,
        rect_pos: tuple[int, int],
        rect_size: tuple[int, int],
        vanilla_model: nn.Module = None,
        sailmap_metrics: list[str] = ["RRF", "HRF", "MRR", "DET", "RMSDR", "DDT", "DCR"],
        batches: int = 2,
        condition_on: str = ConditionOn.PROPER_LABEL.value,
        verbose: bool = False,
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
            elif metric == "RMSDR":
                if vanilla_model is None:
                    raise ValueError("RMSDR requires a vanilla model for comparison")
                metrics_calcs.append(RMSDR())
            elif metric == "DDT":
                if vanilla_model is None:
                    raise ValueError("DDT requires a vanilla model for comparison")
                metrics_calcs.append(DDT())
            elif metric == "DCR":
                if vanilla_model is None:
                    raise ValueError("DCR requires a vanilla model for comparison")
                metrics_calcs.append(DCR())
            else:
                raise ValueError(f"Metric {metric} is not supported")

        for i in tqdm(range(batches), disable=not verbose, desc="Calculating metrics"):
            lrpres = self.lrphandler.calculate(model, self.dataloader, batch_num=i)

            if vanilla_model is not None:
                vanilla_lrpres = self.lrphandler.calculate(vanilla_model, self.dataloader, batch_num=i)

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

            if vanilla_model is not None:
                vanilla_sailmaps = torch.stack(
                    [vanilla_lrpres[label, i] for i, label in enumerate(labels)]
                ).to(dtype=float)
                vanilla_sailmaps = vanilla_sailmaps.cpu().detach().numpy()

            for metric in metrics_calcs:
                if isinstance(metric, (RMSDR, DDT, DCR)):
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
    ):
        """
        Aggregate sailmaps for later calculation
        """
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
        epsilon: float = 0.5,
        source_range: tuple[float, float] = (0, 1),
        symmetrical: bool = False,
        neutral_point: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.source_range = source_range
        self.symmetrical = symmetrical
        self.neutral_point = neutral_point

        self.name = "HRF"

    def _core(
        self,
        sailmaps: np.ndarray,
        rect_pos: tuple[int, int],
        rect_size: tuple[int, int],
    ) -> np.ndarray:
        if self.symmetrical:
            sm_rect = (
                self._sailmaps_rect(sailmaps, rect_pos, rect_size) - self.neutral_point
            )
            sm_rect = np.abs(sm_rect).reshape(len(sm_rect), -1)

            # Rescale to [0, 1]
            sm_rect = (sm_rect - sm_rect.min(axis=1)) / (
                sm_rect.max(axis=1) - sm_rect.min(axis=1)
            )
        else:
            sm_rect = self._sailmaps_rect(sailmaps, rect_pos, rect_size)

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


class RMSDR(SailRectMetric):
    """
    Root Mean Square Difference Ratio (RMSDR)
    
    Compares the root mean square differences between debiased and vanilla saliency maps
    inside vs outside the ROI.
    """
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = "RMSDR"

    def _core(
        self,
        sailmaps: np.ndarray,  # debiased maps
        rect_pos: tuple[int, int],
        rect_size: tuple[int, int],
        vanilla_maps: np.ndarray = None,  # vanilla maps
    ) -> np.ndarray:
        if vanilla_maps is None:
            raise ValueError("RMSDR requires both debiased and vanilla saliency maps")
            
        # Get rectangles for both maps
        d_rect = self._sailmaps_rect(sailmaps, rect_pos, rect_size)
        v_rect = self._sailmaps_rect(vanilla_maps, rect_pos, rect_size)
        
        # Calculate differences inside rectangle
        diff_rect = (d_rect - v_rect) ** 2
        rms_inside = np.sqrt(diff_rect.reshape(len(diff_rect), -1).mean(axis=1))
        
        # Calculate differences outside rectangle
        d_outside = sailmaps.copy()
        v_outside = vanilla_maps.copy()
        
        d_outside[:, rect_pos[0]:rect_pos[0]+rect_size[0], 
                    rect_pos[1]:rect_pos[1]+rect_size[1]] = 0
        v_outside[:, rect_pos[0]:rect_pos[0]+rect_size[0], 
                    rect_pos[1]:rect_pos[1]+rect_size[1]] = 0
        
        diff_outside = (d_outside - v_outside) ** 2
        rms_outside = np.sqrt(diff_outside.reshape(len(diff_outside), -1).mean(axis=1))
        
        return rms_inside / rms_outside


class DDT(SailRectMetric):
    """
    Difference Distribution Testing (DDT)
    
    Tests whether the distribution of differences between debiased and vanilla maps
    is different inside vs outside the ROI using Mann-Whitney-Wilcoxon test.
    """
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = "DDT"

    def _core(
        self,
        sailmaps: np.ndarray,  # debiased maps
        rect_pos: tuple[int, int],
        rect_size: tuple[int, int],
        vanilla_maps: np.ndarray = None,  # vanilla maps
    ) -> np.ndarray:
        if vanilla_maps is None:
            raise ValueError("DDT requires both debiased and vanilla saliency maps")
            
        # Calculate differences
        diff_maps = sailmaps - vanilla_maps
        
        # Get differences inside and outside rectangle
        diff_rect = self._sailmaps_rect(diff_maps, rect_pos, rect_size)
        
        diff_outside = diff_maps.copy()
        diff_outside[:, rect_pos[0]:rect_pos[0]+rect_size[0], 
                       rect_pos[1]:rect_pos[1]+rect_size[1]] = 0
        
        scores = np.zeros(diff_rect.shape[0])
        # Per image
        for i in range(diff_rect.shape[0]):
            _, p = stats.mannwhitneyu(
                diff_rect[i].flatten(), 
                diff_outside[i].flatten(), 
                alternative="two-sided"
            )
            if p < 0.01:
                scores[i] = 1
                
        return scores


class DCR(SailRectMetric):
    """
    Direction Change Ratio (DCR)
    
    Measures the ratio of pixels showing decreased intensity in the debiased model
    compared to the vanilla model, inside vs outside the ROI.
    """
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = "DCR"

    def _core(
        self,
        sailmaps: np.ndarray,  # debiased maps
        rect_pos: tuple[int, int],
        rect_size: tuple[int, int],
        vanilla_maps: np.ndarray = None,  # vanilla maps
    ) -> np.ndarray:
        if vanilla_maps is None:
            raise ValueError("DCR requires both debiased and vanilla saliency maps")
            
        # Get rectangles for both maps
        d_rect = self._sailmaps_rect(sailmaps, rect_pos, rect_size)
        v_rect = self._sailmaps_rect(vanilla_maps, rect_pos, rect_size)
        
        # Calculate proportion of decreased pixels inside rectangle
        decreased_inside = (d_rect < v_rect).reshape(len(d_rect), -1).mean(axis=1)
        
        # Calculate proportion of decreased pixels outside rectangle
        d_outside = sailmaps.copy()
        v_outside = vanilla_maps.copy()
        
        d_outside[:, rect_pos[0]:rect_pos[0]+rect_size[0], 
                    rect_pos[1]:rect_pos[1]+rect_size[1]] = 0
        v_outside[:, rect_pos[0]:rect_pos[0]+rect_size[0], 
                    rect_pos[1]:rect_pos[1]+rect_size[1]] = 0
        
        decreased_outside = (d_outside < v_outside).reshape(len(d_outside), -1).mean(axis=1)
        
        return decreased_inside / decreased_outside
