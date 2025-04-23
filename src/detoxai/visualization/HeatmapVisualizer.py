import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..utils.dataloader import DetoxaiDataLoader
from .enums import ConditionOn
from .ImageVisualizer import ImageVisualizer
from .LRPHandler import LRPHandler
from .utils import get_nth_batch


class HeatmapVisualizer(ImageVisualizer):
    """ """

    def __init__(
        self,
        data_loader: DetoxaiDataLoader | DataLoader,
        model: nn.Module,
        lrp_object: LRPHandler = None,
        plot_config: dict = {},
        draw_rectangles: bool = False,
        rectangle_config: dict = {},
    ) -> None:
        self.data_loader = data_loader
        self.model = model
        
        if not isinstance(data_loader, DetoxaiDataLoader):
            # Check if the user passed an LRPHandler object with n_classes != None
            if lrp_object is None or lrp_object.n_classes is None:
                raise ValueError(
                    "If you pass a DataLoader that is not a subclass of `DetoxaiDataLoader`, you must pass an LRPHandler with `n_classes` set."
                )
            

        if lrp_object is None:
            lrp_object = LRPHandler()  # Default LRPHandler

        self.lrp_object = lrp_object

        self.init_rectangle_painter(draw_rectangles, rectangle_config)
        self.set_up_plots_configuration(plot_config)

    def visualize_batch(
        self,
        batch_num: int,
        condition_on: ConditionOn = ConditionOn.PROPER_LABEL,
        show_cbar: bool = True,
        max_images: int | None = 36,
        return_fig: bool = False,
    ) -> None:
        """

        Args:
          batch_num: int:
          condition_on: ConditionOn:  (Default value = ConditionOn.PROPER_LABEL)
          show_cbar: bool:  (Default value = True)
          max_images: int | None:  (Default value = 36)
          return_fig: bool:  (Default value = False)

        Returns:

        """
        images = self._get_heatmaps(batch_num, condition_on, max_images)

        if max_images is None:
            max_images = images.shape[0]

        images_to_show = min(images.shape[0], max_images)
        rows = int(images_to_show**0.5)
        cols = int(images_to_show**0.5)

        fig, ax = self.get_canvas(
            rows=rows,
            cols=cols,
            shape=(
                int(rows) * self.plot_shape_multiplier,
                int(cols) * self.plot_shape_multiplier,
            ),
        )

        for i, img in enumerate(images[:max_images]):
            im = ax[i // cols, i % cols].imshow(img, cmap="seismic", vmin=0, vmax=1)

            self.maybe_paint_rectangle(ax[i // cols, i % cols])

        if show_cbar:
            # Show colorbar at the bottom
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.89, 0.15, 0.05, 0.7])

            cbar = fig.colorbar(im, cax=cbar_ax)

            # Modify ticks in the colorbar
            cbar.set_ticks([0, 0.5, 1])
            cbar.set_ticklabels(["-1", "0", "1"])

            # Make cbar slimmer in width
            cbar_ax.set_aspect(25)

        if return_fig:
            return fig, ax

    def visualize_agg(
        self,
        batch_num: int,
        condition_on: ConditionOn = ConditionOn.PROPER_LABEL,
    ) -> None:
        """

        Args:
          batch_num: int:
          condition_on: ConditionOn:  (Default value = ConditionOn.PROPER_LABEL)

        Returns:

        """
        _, labels, prot_attr = get_nth_batch(self.data_loader, batch_num)  # noqa

        images = self._get_heatmaps(batch_num, condition_on, None)

        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels)
        if isinstance(prot_attr, np.ndarray):
            prot_attr = torch.tensor(prot_attr)
        if isinstance(images, np.ndarray):
            images = torch.tensor(images)

        ulab = labels.unique()
        uprot = prot_attr.unique()

        fig, ax = self.get_canvas(
            rows=len(ulab), cols=len(uprot), shape=(len(ulab) * 3, len(uprot) * 3)
        )

        for row, label in enumerate(ulab):
            for col, prot_a in enumerate(uprot):
                mask = (labels == label) & (prot_attr == prot_a)

                img = images[mask].mean(dim=0).cpu().detach().numpy()
                ax[row, col].imshow(img, cmap="seismic", vmin=0, vmax=1)

                self.maybe_paint_rectangle(ax[row, col])

    def _get_heatmaps(
        self, batch_num: int, condition_on: ConditionOn, max_images: int | None
    ) -> np.ndarray:
        """

        Args:
          batch_num: int:
          condition_on: ConditionOn:
          max_images: int | None:

        Returns:

        """
        images, labels, prot_attr = get_nth_batch(self.data_loader, batch_num)  # noqa

        if max_images is None:
            max_images = images.shape[0]

        heatmaps = self.lrp_object.calculate(
            self.model, self.data_loader, batch_num, max_images
        )

        conditioned = []
        for i, label in enumerate(labels[:max_images]):
            # Assuming binary classification
            label = label if condition_on == ConditionOn.PROPER_LABEL else 1 - label
            conditioned.append(heatmaps[label, i])

        images: torch.Tensor = torch.stack(conditioned).to(dtype=float)
        images = images.cpu().detach().numpy()

        return images
