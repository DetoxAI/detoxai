from abc import ABC, abstractmethod

# Project imports
from .Visualizer import Visualizer


class ImageVisualizer(Visualizer, ABC):
    @abstractmethod
    def visualize_batch(self, batch_num: int) -> None:
        pass

    @abstractmethod
    def visualize_agg(self, batch_num: int) -> None:
        pass
