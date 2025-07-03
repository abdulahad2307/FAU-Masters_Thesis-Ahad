import torch
from torch.utils.data import DataLoader

class DocFormerDataLoader:
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=None,
        prefetch_factor=2,
        persistent_workers=True,
    ):
        """
        Args:
            dataset: Dataset
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data at every epoch
            num_workers: Number of CPU workers for parallel loading
            pin_memory: Whether to use pinned memory (recommended for GPU training)
            drop_last: Drop last incomplete batch
            collate_fn: Custom collate function for batching
            prefetch_factor: Number of batches prefetched by each worker
            persistent_workers: Keep workers alive between epochs for speed
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers

        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    def get_loader(self):
        """Returns the DataLoader."""
        return self.dataloader