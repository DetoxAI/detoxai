from torch.utils.data import DataLoader

from .datasets import DetoxaiDataset


class DetoxaiDataLoader(DataLoader):
    def __init__(self, dataset: DetoxaiDataset, **kwargs):
        super().__init__(dataset, **kwargs)

    def get_class_names(self):
        assert isinstance(
            self.dataset, DetoxaiDataset
        ), "Dataset must be an instance of DetoxaiDataset, as we rely on its internal structure"
        return self.dataset.get_class_names()

    def get_nth_batch(self, n: int) -> tuple:
        for i, batch in enumerate(self):
            if i == n:
                return batch
        return None
