"""CPU-only tests for the reproducibility utilities.

These need no task, model, or dataset — they verify the framework's core promise:
the same seed yields the same numbers, including inside DataLoader workers, and a
captured RNG state can be restored to continue the *same* random trajectory.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.reproducibility import (
    capture_rng_state,
    restore_rng_state,
    seed_everything,
    seed_worker,
)


def _draw_all():
    """One draw from each of the four RNGs seed_everything controls."""
    return (
        random.random(),
        float(np.random.rand()),
        torch.rand(1).item(),
    )


def test_seed_everything_is_repeatable():
    seed_everything(123, deterministic=False)
    first = _draw_all()
    seed_everything(123, deterministic=False)
    second = _draw_all()
    assert first == second


def test_different_seeds_differ():
    seed_everything(1, deterministic=False)
    a = _draw_all()
    seed_everything(2, deterministic=False)
    b = _draw_all()
    assert a != b


def test_seed_everything_returns_seeded_generator():
    g1 = seed_everything(7, deterministic=False)
    perm1 = torch.randperm(10, generator=g1)
    g2 = seed_everything(7, deterministic=False)
    perm2 = torch.randperm(10, generator=g2)
    assert torch.equal(perm1, perm2)


def test_capture_restore_continues_same_trajectory():
    """capture -> advance -> restore -> next draws match the captured point."""
    seed_everything(99, deterministic=False)
    _draw_all()  # advance a bit so the captured state isn't the seeded start
    state = capture_rng_state()
    expected = _draw_all()  # what comes next from the captured point

    _draw_all()  # advance the live RNGs further so a no-op restore would differ
    restore_rng_state(state)
    assert _draw_all() == expected


def test_restore_handles_cpu_only_checkpoint():
    """A checkpoint captured on CPU (torch_cuda == []) restores without raising."""
    seed_everything(3, deterministic=False)
    state = capture_rng_state()
    assert state["torch_cuda"] == [] or torch.cuda.is_available()
    restore_rng_state(state)  # must not raise on a CPU-only box


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA to move RNG tensors off CPU"
)
def test_restore_after_cuda_map_location(tmp_path):
    """Regression: ``load_checkpoint`` uses ``map_location=device``; on CUDA that moves
    the RNG ByteTensors to the GPU, and the setters reject non-CPU tensors with
    ``RNG state must be a torch.ByteTensor``. restore must coerce them back to CPU.
    """
    generator = seed_everything(5, deterministic=False)
    _draw_all()
    state = capture_rng_state(generator)
    expected = _draw_all()

    path = tmp_path / "rng.pth"
    torch.save({"rng": state}, path)
    loaded = torch.load(path, map_location="cuda", weights_only=False)["rng"]
    assert loaded["torch"].is_cuda  # the precondition that used to raise

    _draw_all()  # advance the live RNGs so a no-op restore would differ
    restore_rng_state(loaded, generator)
    assert _draw_all() == expected


class _NumpyDataset(Dataset):
    """A dataset whose items depend on numpy's RNG — the worker-reseed trap."""

    def __len__(self) -> int:
        return 8

    def __getitem__(self, idx: int):
        # If workers are not reseeded deterministically, this varies run-to-run.
        return torch.tensor(np.random.rand(), dtype=torch.float32)


def _collect_with_workers(seed: int) -> list[float]:
    generator = seed_everything(seed, deterministic=False)
    loader = DataLoader(
        _NumpyDataset(),
        batch_size=2,
        shuffle=True,
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return [v.item() for batch in loader for v in batch]


def test_dataloader_workers_are_reproducible():
    assert _collect_with_workers(42) == _collect_with_workers(42)


def test_dataloader_workers_differ_across_seeds():
    assert _collect_with_workers(1) != _collect_with_workers(2)
