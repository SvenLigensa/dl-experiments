"""Tests for atomic, resumable checkpointing.

These verify the contracts the cluster path relies on: a save/load roundtrip
restores weights *and* the RNG trajectory, writes are atomic (no truncated files
on a mid-write crash), and the retention policy keeps the right files — never
touching ``best.pth``.
"""

from __future__ import annotations

import glob
import os
import time

import pytest
import torch
from omegaconf import OmegaConf

from src.checkpointing import (
    find_latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)
from tests.helpers import TinyModel


def _cfg():
    return OmegaConf.create({"trainer": {"seed": 0}})


def _save(path, *, model=None, optimizer=None, epoch=0, generator=None, best=None):
    model = model or TinyModel()
    optimizer = optimizer or torch.optim.SGD(model.parameters(), lr=0.1)
    save_checkpoint(
        path,
        model,
        optimizer,
        None,
        epoch,
        _cfg(),
        generator=generator,
        best_metric=best,
    )
    return model, optimizer


def _touch_after(path):
    """Bump mtime so file ordering by mtime is unambiguous on fast filesystems."""
    time.sleep(0.01)
    os.utime(path, None)


def test_save_load_roundtrip(run_dir):
    device = torch.device("cpu")
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # Take a step so optimizer/model state is non-trivial.
    out = model(torch.randn(4, 8)).sum()
    out.backward()
    optimizer.step()

    path = os.path.join(run_dir, "ckpt_epoch0.pth")
    save_checkpoint(path, model, optimizer, None, 3, _cfg(), best_metric=0.5)

    fresh = TinyModel()
    fresh_opt = torch.optim.SGD(fresh.parameters(), lr=0.1)
    ckpt = load_checkpoint(path, fresh, fresh_opt, None, device)

    for a, b in zip(
        model.state_dict().values(), fresh.state_dict().values(), strict=True
    ):
        assert torch.equal(a, b)
    assert ckpt["epoch"] == 3
    assert ckpt["best_metric"] == 0.5


def test_load_restores_generator_state(run_dir):
    """The DataLoader generator continues the same shuffle order after a load."""
    gen = torch.Generator().manual_seed(7)
    gen.manual_seed(7)
    torch.randperm(5, generator=gen)  # advance it off the seeded start
    path = os.path.join(run_dir, "ckpt_epoch0.pth")
    _save(path, generator=gen)
    expected = torch.randperm(5, generator=gen)  # next draw from the saved point

    restored = torch.Generator()
    model = TinyModel()
    load_checkpoint(path, model, None, None, torch.device("cpu"), generator=restored)
    assert torch.equal(torch.randperm(5, generator=restored), expected)


def test_save_is_atomic_no_tmp_left(run_dir):
    path = os.path.join(run_dir, "ckpt_epoch0.pth")
    _save(path)
    assert os.path.exists(path)
    assert not glob.glob(os.path.join(run_dir, "*.tmp"))
    # And the file is fully loadable (not truncated).
    torch.load(path, map_location="cpu", weights_only=False)


def test_failed_save_never_corrupts_existing(run_dir, monkeypatch):
    """If torch.save dies mid-write, the original target is left intact."""
    path = os.path.join(run_dir, "ckpt_epoch0.pth")
    _save(path, best=1.0)  # a good existing checkpoint

    def boom(*_a, **_k):
        raise RuntimeError("killed mid-write")

    monkeypatch.setattr("src.checkpointing.torch.save", boom)
    with pytest.raises(RuntimeError):
        _save(path, best=2.0)

    # Original survives, unchanged; no half-written temp left behind.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["best_metric"] == 1.0
    assert not glob.glob(os.path.join(run_dir, "*.tmp"))


def test_find_latest_checkpoint(run_dir):
    assert find_latest_checkpoint(run_dir) is None
    for epoch in range(3):
        p = os.path.join(run_dir, f"ckpt_epoch{epoch}.pth")
        _save(p, epoch=epoch)
        _touch_after(p)
    # best.pth must be ignored even if it's the newest file.
    best = os.path.join(run_dir, "best.pth")
    _save(best)
    _touch_after(best)

    assert find_latest_checkpoint(run_dir) == os.path.join(run_dir, "ckpt_epoch2.pth")


def test_prune_keeps_last_n_and_spares_best(run_dir):
    for epoch in range(5):
        p = os.path.join(run_dir, f"ckpt_epoch{epoch}.pth")
        _save(p, epoch=epoch)
        _touch_after(p)
    best = os.path.join(run_dir, "best.pth")
    _save(best)

    prune_checkpoints(run_dir, keep_last=2)

    remaining = sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(run_dir, "*.pth"))
    )
    assert remaining == ["best.pth", "ckpt_epoch3.pth", "ckpt_epoch4.pth"]


def test_prune_noop_when_keep_last_non_positive(run_dir):
    for epoch in range(3):
        _save(os.path.join(run_dir, f"ckpt_epoch{epoch}.pth"), epoch=epoch)
    prune_checkpoints(run_dir, keep_last=0)
    assert len(glob.glob(os.path.join(run_dir, "ckpt_epoch*.pth"))) == 3
