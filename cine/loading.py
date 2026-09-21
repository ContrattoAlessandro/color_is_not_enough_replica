"""Persistent data workers and reproducible per-epoch sample order."""
import torch
from torch.utils.data import DataLoader, Sampler
from .data import collate


class EpochSampler(Sampler):
    def __init__(self, dataset, seed=42):
        self.size, self.seed, self.epoch = len(dataset), seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter((i, self.epoch) for i in torch.randperm(self.size, generator=generator).tolist())

    def __len__(self):
        return self.size


def loader(dataset, batch_size, workers=0, shuffle=False, seed=42, prefetch_factor=2, persistent_workers=True):
    kwargs = dict(dataset=dataset, batch_size=batch_size, num_workers=workers, collate_fn=collate, pin_memory=True, generator=torch.Generator().manual_seed(seed))
    if shuffle:
        kwargs['sampler'] = EpochSampler(dataset, seed)
    if workers:
        kwargs.update(prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)
    return DataLoader(**kwargs)


def shutdown_loader(batches):
    # Explicitly release persistent workers when a loader is no longer needed.
    iterator = getattr(batches, '_iterator', None)
    if iterator is not None:
        iterator._shutdown_workers()
        batches._iterator = None
