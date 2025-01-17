import logging
from torch.utils.data import DataLoader
import itertools

from .datasets import DetoxaiDataset

logger = logging.getLogger(__name__)


class DetoxaiDataLoader(DataLoader):
    def __init__(self, dataset: DetoxaiDataset, **kwargs):
        super().__init__(dataset, **kwargs)

    def get_class_names(self):
        assert isinstance(self.dataset, DetoxaiDataset), (
            "Dataset must be an instance of DetoxaiDataset, as we rely on its internal structure"
        )
        return self.dataset.get_class_names()

    def get_nth_batch(self, n: int) -> tuple:
        for i, batch in enumerate(self):
            if i == n:
                return batch
        return None

    def get_nth_batch2(self, n: int) -> tuple:
        if n < 0 or n >= len(self):
            return None

        # Create a new iterator
        dataiter = iter(self)
        # Use itertools.islice to get to the desired batch directly
        batch = next(itertools.islice(dataiter, n, n + 1), None)
        return batch


def copy_data_loader(
    dataloader: DataLoader, batch_size: int | None = None
) -> DetoxaiDataLoader:
    """
    Copy the dataloader
    """
    if batch_size is None:
        batch_size = dataloader.batch_size

    params = {
        "batch_size": batch_size,
        "num_workers": dataloader.num_workers,
        "collate_fn": dataloader.collate_fn,
    }

    logger.debug(f"Copying dataloader with params: {params}")

    return DetoxaiDataLoader(
        dataset=dataloader.dataset,
        batch_size=batch_size,
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
    )
