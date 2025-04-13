import itertools
import logging

from torch.utils.data import DataLoader, Dataset

from .datasets import DetoxaiDataset

logger = logging.getLogger(__name__)


class DetoxaiDataLoader(DataLoader):
    """ """

    def __init__(self, dataset: DetoxaiDataset, **kwargs):
        super().__init__(dataset, **kwargs)

    def get_class_names(self):
        """ """
        assert isinstance(self.dataset, DetoxaiDataset), (
            "Dataset must be an instance of DetoxaiDataset, as we rely on its internal structure"
        )
        return self.dataset.get_class_names()

    def get_nth_batch(self, n: int) -> tuple:
        """

        Args:
          n: int:

        Returns:

        """
        for i, batch in enumerate(self):
            if i == n:
                return batch
        return None

    def get_nth_batch2(self, n: int) -> tuple:
        """

        Args:
          n: int:

        Returns:

        """
        if n < 0 or n >= len(self):
            return None

        # Create a new iterator
        dataiter = iter(self)
        # Use itertools.islice to get to the desired batch directly
        batch = next(itertools.islice(dataiter, n, n + 1), None)
        return batch

    def get_num_classes(self) -> int:
        assert isinstance(self.dataset, DetoxaiDataset), (
            "Dataset must be an instance of DetoxaiDataset, as we rely on its internal structure"
        )
        return self.dataset.get_num_classes()


class WrappedDataLoader(DetoxaiDataLoader):
    def __init__(self, dataset: Dataset, num_of_classes: int, **kwargs):
        super().__init__(dataset, **kwargs)
        self.num_of_classes = num_of_classes

    def get_num_classes(self) -> int:
        return self.num_of_classes

    def get_class_names(self) -> list[str]:
        """Name them A B C .."""
        return [chr(i) for i in range(65, 65 + self.num_of_classes)]


def copy_data_loader(
    dataloader: DataLoader,
    batch_size: int | None = None,
    shuffle: bool = False,
    drop_last: bool = False,
) -> DetoxaiDataLoader | WrappedDataLoader:
    """Copy the dataloader

    Args:
      dataloader: DataLoader:
      batch_size: int | None:  (Default value = None)
      shuffle: bool:  (Default value = False)
      drop_last: bool:  (Default value = False)

    Returns:

    """
    if batch_size is None:
        batch_size = dataloader.batch_size

    if isinstance(dataloader, DetoxaiDataLoader):
        return DetoxaiDataLoader(
            dataset=dataloader.dataset,
            batch_size=batch_size,
            num_workers=dataloader.num_workers,
            collate_fn=dataloader.collate_fn,
            shuffle=shuffle,
            drop_last=drop_last,
        )
    elif isinstance(dataloader, WrappedDataLoader):
        return WrappedDataLoader(
            dataset=dataloader.dataset,
            num_of_classes=dataloader.num_of_classes,
            batch_size=batch_size,
            num_workers=dataloader.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
        )
    else:
        raise ValueError(
            f"Unsupported DataLoader type: {type(dataloader)}. "
            "Please use DetoxaiDataLoader or WrappedDataLoader."
        )
