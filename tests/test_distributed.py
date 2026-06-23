"""Tests for the device-agnostic helpers in the single-process (non-DDP) path."""

from __future__ import annotations

import torch

from src.distributed import (
    get_device,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    reduce_mean,
)


def test_get_device_explicit():
    assert get_device("cpu") == torch.device("cpu")


def test_get_device_auto_never_raises():
    device = get_device("auto")
    assert isinstance(device, torch.device)
    assert device.type in {"cpu", "cuda", "mps"}


def test_single_process_defaults():
    assert is_distributed() is False
    assert is_main_process() is True
    assert get_rank() == 0
    assert get_world_size() == 1


def test_reduce_mean_is_identity_single_process():
    t = torch.tensor([2.0, 4.0])
    assert torch.equal(reduce_mean(t), t)
