from torch.utils.data import DataLoader
from torch import Tensor

def get_nth_batch(dataloader: DataLoader, n: int) -> tuple:
    """
    Get the nth batch from a DataLoader.
    
    Args:
        dataloader (DataLoader): The DataLoader from which to get the batch.
        n (int): The index of the batch to retrieve.
        
    Returns:
        tuple: The nth batch from the DataLoader.
    """
    
    # If dataloader has get_nth_batch method, use it
    if hasattr(dataloader, "get_nth_batch"):
        return dataloader.get_nth_batch(n)
    # Otherwise, iterate through the dataloader to get the nth batch
    for i, batch in enumerate(dataloader):
        if i == n:
            return batch
        
    raise IndexError(f"Batch number {n} exceeds the number of batches in the dataloader.")