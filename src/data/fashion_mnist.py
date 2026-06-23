"""FashionMNIST data module satisfying the engine's data contract.

Exposes ``train_dataset`` / ``val_dataset`` / ``batch_size`` / ``num_workers``. The
engine builds the ``DataLoader``s (injecting the seeded generator + ``seed_worker``),
so this class only describes *what* the data is and *where* it lives.

The data location is config-driven (``data_dir``) — never hard-coded — so the same
config works against a Drive mount on Colab or scratch on a cluster.
"""

from __future__ import annotations

from torchvision import transforms
from torchvision.datasets import FashionMNIST

# Dataset-level mean/std (documented constant, part of the fixed preprocessing).
_MEAN, _STD = (0.2860,), (0.3530,)


class FashionMNISTData:
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 128,
        num_workers: int = 2,
        download: bool = True,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)]
        )
        self.train_dataset = FashionMNIST(
            data_dir, train=True, download=download, transform=transform
        )
        self.val_dataset = FashionMNIST(
            data_dir, train=False, download=download, transform=transform
        )
