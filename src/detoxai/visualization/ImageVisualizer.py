from abc import ABC, abstractmethod

import matplotlib.patches as patches

# Project imports
from .Visualizer import Visualizer


class ImageVisualizer(Visualizer, ABC):
    """ """

    @abstractmethod
    def visualize_batch(self, batch_num: int) -> None:
        """

        Args:
          batch_num: int:

        Returns:

        """
        pass

    @abstractmethod
    def visualize_agg(self, batch_num: int) -> None:
        """

        Args:
          batch_num: int:

        Returns:

        """
        pass

    def init_rectangle_painter(self, draw_rectangles: bool, rectangle_config: dict):
        """Initialize the rectangle painter.

        Args:
          draw_rectangles(bool): Whether to draw rectangles on the image.
          rectangle_config(dict): The configuration for the rectangles.
        Should contain the following keys:
        rect (tuple): The rectangle coordinates in the form (x, y, width, height). Mandatory.
        color (str): The color of the rectangle. Default is "black".
        linewidth (int): The width of the rectangle's border. Default is 2.
          draw_rectangles: bool:
          rectangle_config: dict:

        Returns:

        """
        self.rectangle_painter_init = True
        self.draw_rectangles = draw_rectangles
        self.rectangle_config = rectangle_config

    def maybe_paint_rectangle(self, ax) -> None:
        """Maybe paint a rectangle on the image. The rectangle is painted only if the
        `draw_rectangles` flag is set to True. Before calling this method, the
        `init_rectangle_painter` method should be called.

        Args:
          ax:

        Returns:

        """
        assert hasattr(self, "rectangle_painter_init"), (
            "Rectangle painter not initialized"
        )

        if self.draw_rectangles:
            x, y, width, height = self.rectangle_config["rect"]  # This one is mandatory
            color = self.rectangle_config.get("color", "black")
            linewidth = self.rectangle_config.get("linewidth", 2)

            ax.add_patch(
                patches.Rectangle(
                    (x, y),
                    width,
                    height,
                    fill=False,
                    edgecolor=color,
                    linewidth=linewidth,
                )
            )
