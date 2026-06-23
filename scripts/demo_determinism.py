#!/usr/bin/env python
"""The "break it, then fix it" determinism demo.

Run on the `example` branch::

    python scripts/demo_determinism.py

It trains the same tiny ViT on FashionMNIST twice, two ways:

1. **Uncontrolled** — no seeding. The two runs DIVERGE: different init, different
   shuffles, different numbers. This is what most student repos do by default.
2. **Controlled** — `seed_everything` before each run (incl. the DataLoader generator
   and worker_init_fn). The two runs now produce IDENTICAL numbers on this machine.

The honest footnote (printed at the end): identical here means *same machine, same
device*. Across CPU vs GPU or different GPU models the numbers will still differ —
which is why you report mean +/- std over seeds, not a single magic number.

Kept tiny (a few hundred training samples, one pass) so it finishes in seconds on a
laptop CPU.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset

from src.data.fashion_mnist import FashionMNISTData
from src.metrics import Accuracy
from src.models.vit import ViT
from src.reproducibility import seed_everything, seed_worker

N_TRAIN = 512
N_VAL = 512
BATCH_SIZE = 64


def _train_once(seed: int | None) -> float:
    """Train for one pass and return validation accuracy.

    If ``seed`` is None, nothing is seeded (the 'broken' case). Otherwise every RNG —
    including the DataLoader's — is seeded (the 'fixed' case).
    """
    generator = None
    worker_init = None
    if seed is not None:
        generator = seed_everything(seed, deterministic=True)
        worker_init = seed_worker

    data = FashionMNISTData(num_workers=0)
    train_ds = Subset(data.train_dataset, range(N_TRAIN))
    val_ds = Subset(data.val_dataset, range(N_VAL))

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        worker_init_fn=worker_init,
        generator=generator,
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    device = torch.device("cpu")  # pin to CPU so the demo is identical everywhere
    model = ViT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()
    accuracy = Accuracy()

    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0.0
    n = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            correct += accuracy(model(inputs), targets) * inputs.size(0)
            n += inputs.size(0)
    return correct / n


def _report(title: str, a: float, b: float) -> None:
    same = abs(a - b) < 1e-9
    print(f"\n{title}")
    print(f"  run 1 val acc: {a:.6f}")
    print(f"  run 2 val acc: {b:.6f}")
    print(f"  identical?     {'YES' if same else 'NO'}  (|diff| = {abs(a - b):.2e})")


def main() -> None:
    print("Training the same model twice, each way (CPU, ~1 pass over 512 samples)...")

    _report(
        "[1/2] Uncontrolled (no seeding; expect divergence):",
        _train_once(seed=None),
        _train_once(seed=None),
    )
    _report(
        "[2/2] Controlled (everything seeded - expect identical):",
        _train_once(seed=1234),
        _train_once(seed=1234),
    )

    print(
        "\nNote: 'identical' means same machine + same device (CPU here). Across "
        "CPU vs GPU or different GPUs, results still differ — so report mean +/- std "
        "over seeds (scripts/run_seeds.py), not a single number."
    )


if __name__ == "__main__":
    main()
