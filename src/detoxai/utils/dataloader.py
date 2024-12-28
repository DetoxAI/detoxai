from torch.utils.data import DataLoader

from .collators import BaseCollator


class WrappedDataLoader(DataLoader):
    def __init__(self, dataset, collator: BaseCollator, **kwargs):
        self.collator = collator
        collate_fn = self.collator.infer_best_collate_fn(dataset)
        # filter out collator from kwargs, our collator has precedence
        kwargs = {k: v for k, v in kwargs.items() if k != "collate_fn"}
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)

    def get_nth_batch(self, n: int) -> tuple:
        for i, batch in enumerate(self):
            if i == n:
                return batch
        return None
