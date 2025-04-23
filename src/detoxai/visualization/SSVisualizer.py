import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from ..utils.dataloader import DetoxaiDataLoader
from ..utils.datasets import DetoxaiDataset
from .DataVisualizer import DataVisualizer
from .HeatmapVisualizer import ConditionOn, HeatmapVisualizer
from .ImageVisualizer import ImageVisualizer
from .LRPHandler import LRPHandler
from .utils import get_nth_batch


class SSVisualizer(ImageVisualizer):
    """SS - Side by Side visualizer for Heatmaps and Data"""

    def __init__(
        self,
        data_loader: DetoxaiDataLoader | DataLoader,
        model: nn.Module,
        lrp_object: LRPHandler = None,
        plot_config: dict = {},
        draw_rectangles: bool = False,
        rectangle_config: dict = {},
    ) -> None:
        """
        Args:
          data_loader: DetoxaiDataLoader | DataLoader:
          model: nn.Module:
          lrp_object: LRPHandler = None:
          plot_config: dict = {}:
          draw_rectangles: bool = False:
          rectangle_config: dict = {}:

        Returns:
            None

        If you pass a DataLoader that is not a subclass of `DetoxaiDataLoader`, you must pass an LRPHandler with `n_classes` set!
        """

        self.data_loader = data_loader
        self.model = model
        self.set_up_plots_configuration(plot_config)
        self.init_rectangle_painter(draw_rectangles, rectangle_config)

        if not isinstance(data_loader, DetoxaiDataLoader):
            # Check if the user passed an LRPHandler object with n_classes != None
            if lrp_object is None or lrp_object.n_classes is None:
                raise ValueError(
                    "If you pass a DataLoader that is not a subclass of `DetoxaiDataLoader`, you must pass an LRPHandler with `n_classes` set."
                )

        if not isinstance(data_loader.dataset, DetoxaiDataset):
            # Check if the user passed an LRPHandler object with n_classes != None
            if lrp_object is None or lrp_object.n_classes is None:
                raise ValueError(
                    "If you pass a DataLoader that has a dataset that is not a subclass of `DetoxaiDataset`, you must pass an LRPHandler with `n_classes` set."
                )

        self.lrp_vis = HeatmapVisualizer(
            data_loader,
            model,
            lrp_object,
            plot_config,
            draw_rectangles,
            rectangle_config,
        )
        self.data_vis = DataVisualizer(
            data_loader, plot_config, draw_rectangles, rectangle_config
        )

        self.model_device = next(model.parameters()).device

    def visualize_batch(
        self,
        batch_num: int,
        lrp_condition_on: ConditionOn = ConditionOn.PROPER_LABEL,
        lrp_show_cbar: bool = True,
        max_images: int | None = 36,
        show_labels: bool = True,
    ) -> None:
        """

        Args:
          batch_num: int:
          lrp_condition_on: ConditionOn:  (Default value = ConditionOn.PROPER_LABEL)
          lrp_show_cbar: bool:  (Default value = True)
          max_images: int | None:  (Default value = 36)
          show_labels: bool:  (Default value = True)

        Returns:

        """
        self.lrp_vis.visualize_batch(
            batch_num, lrp_condition_on, lrp_show_cbar, max_images
        )

        if show_labels:
            data = get_nth_batch(self.data_loader, batch_num)[0].to(self.model_device)
            preds = self.model(data).argmax(dim=1)
        else:
            preds = None

        self.data_vis.visualize_batch(
            batch_num, max_images, batch_preds=preds, show_labels=show_labels
        )

        self.__build_plot()

    def visualize_agg(self, batch_num: int) -> None:
        """

        Args:
          batch_num: int:

        Returns:

        """
        self.lrp_vis.visualize_agg(batch_num)
        self.data_vis.visualize_agg(batch_num)

        self.__build_plot()

    def __build_plot(self) -> None:
        f1 = self.lrp_vis.figure
        f2 = self.data_vis.figure

        def figure_to_array(fig):
            """Convert a Matplotlib figure to a numpy array.

            Args:
              fig:

            Returns:

            """
            fig.canvas.draw()  # Draw the figure
            # Get the image as an RGBA buffer and convert it to a numpy array
            img = np.asarray(fig.canvas.buffer_rgba())
            return img[..., :3]  # Convert RGBA to RGB if needed

        # Extract image data from the figures
        img1 = figure_to_array(f1)
        img2 = figure_to_array(f2)

        # Plot them side by side
        fig, axs = self.get_canvas(cols=2, shape=(10, 6))
        axs[0].imshow(img2)
        axs[1].imshow(img1)

        # Close plots in the sub-visualizers
        self.lrp_vis.close_plot()
        self.data_vis.close_plot()
