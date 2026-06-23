"""End-to-end tests that drive the *real* Trainer on a tiny synthetic task.

These are the headline tests: they prove the guarantees the whole template exists
to make — same seed produces identical weights, and an interrupted run that
auto-resumes from a checkpoint continues *bit-identically* to an uninterrupted run.
Marked ``slow`` (they run several epochs) so ``pytest -m "not slow"`` stays fast.
"""

from __future__ import annotations

import glob
import os

import pytest
import torch

from tests.helpers import build_trainer, make_cfg

pytestmark = pytest.mark.slow


def _weights(trainer) -> list[torch.Tensor]:
    return [p.detach().clone() for p in trainer._unwrapped.parameters()]


def _assert_same_weights(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert torch.equal(x, y)


def _epoch_ckpts(run_dir):
    return sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(run_dir, "ckpt_epoch*.pth"))
    )


def test_smoke_train_runs_and_checkpoints(run_dir):
    trainer = build_trainer(make_cfg(run_dir, epochs=2), run_dir)
    metrics = trainer.train()

    assert "val/loss" in metrics and "train/loss" in metrics
    assert os.path.exists(os.path.join(run_dir, "ckpt_epoch1.pth"))
    assert os.path.exists(os.path.join(run_dir, "run_metadata.json"))


def test_same_seed_gives_identical_weights(tmp_path):
    dir_a, dir_b = str(tmp_path / "a"), str(tmp_path / "b")
    os.makedirs(dir_a)
    os.makedirs(dir_b)

    a = build_trainer(make_cfg(dir_a, epochs=3, seed=0), dir_a)
    a.train()
    b = build_trainer(make_cfg(dir_b, epochs=3, seed=0), dir_b)
    b.train()

    _assert_same_weights(_weights(a), _weights(b))


def test_different_seed_gives_different_weights(tmp_path):
    dir_a, dir_b = str(tmp_path / "a"), str(tmp_path / "b")
    os.makedirs(dir_a)
    os.makedirs(dir_b)

    a = build_trainer(make_cfg(dir_a, epochs=2, seed=0), dir_a)
    a.train()
    b = build_trainer(make_cfg(dir_b, epochs=2, seed=1), dir_b)
    b.train()

    same = all(torch.equal(x, y) for x, y in zip(_weights(a), _weights(b), strict=True))
    assert not same


def test_reproducible_resume_matches_uninterrupted(tmp_path):
    """The crown jewel: resume from a checkpoint == an uninterrupted run, bit-for-bit."""
    ref_dir = str(tmp_path / "ref")
    res_dir = str(tmp_path / "res")
    os.makedirs(ref_dir)
    os.makedirs(res_dir)

    # Reference: a clean 4-epoch run.
    ref = build_trainer(make_cfg(ref_dir, epochs=4, seed=0), ref_dir)
    ref.train()
    reference_weights = _weights(ref)

    # Interrupted: train 2 epochs, then resume to 4 with a brand-new Trainer.
    first = build_trainer(make_cfg(res_dir, epochs=2, seed=0), res_dir)
    first.train()
    assert os.path.exists(os.path.join(res_dir, "ckpt_epoch1.pth"))

    resumed = build_trainer(make_cfg(res_dir, epochs=4, seed=0), res_dir)
    assert resumed.start_epoch == 2  # auto-resumed from the last completed epoch
    resumed.train()

    _assert_same_weights(reference_weights, _weights(resumed))


def test_pruning_during_run_keeps_last_n(run_dir):
    trainer = build_trainer(make_cfg(run_dir, epochs=5, ckpt_keep_last=2), run_dir)
    trainer.train()
    assert _epoch_ckpts(run_dir) == ["ckpt_epoch3.pth", "ckpt_epoch4.pth"]


def test_best_checkpoint_tracks_min_val_loss(run_dir):
    class RecordingTracker:
        def __init__(self):
            self.val_losses = []

        def log(self, metrics, step=None):
            if "val/loss" in metrics:
                self.val_losses.append(metrics["val/loss"])

        def finish(self):
            pass

    tracker = RecordingTracker()
    trainer = build_trainer(make_cfg(run_dir, epochs=4), run_dir, tracker=tracker)
    trainer.train()

    best_path = os.path.join(run_dir, "best.pth")
    assert os.path.exists(best_path)
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    assert ckpt["best_metric"] == pytest.approx(min(tracker.val_losses))


def test_preemption_writes_no_checkpoint(run_dir):
    trainer = build_trainer(make_cfg(run_dir, epochs=2), run_dir)
    trainer._stop_requested = True  # simulate SIGTERM before any epoch completes
    trainer.train()

    assert trainer.interrupted is True
    assert _epoch_ckpts(run_dir) == []
    assert not os.path.exists(os.path.join(run_dir, "best.pth"))
