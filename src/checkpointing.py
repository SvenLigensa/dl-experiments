"""Device-agnostic checkpoint save / load / auto-resume — built for preemption.

Checkpoints store enough state to *exactly* continue a run: model + optimizer +
scheduler weights, the epoch, the resolved config, **and the RNG state** of every
generator. That last part is what makes a resume *reproducible* rather than merely
possible: without it, the numbers after a resume drift from an uninterrupted run
(see ``capture_rng_state`` in ``src/reproducibility.py``).

Two more robustness properties matter on a real cluster:

* **Atomic writes.** ``torch.save`` straight to the final path means a kill signal
  mid-write leaves a *truncated, unloadable* ``.pth`` — and you lose everything. We
  write to ``<path>.tmp`` and then ``os.replace`` it into place, which is atomic on
  POSIX and Windows: a reader only ever sees either the old file or the complete new one.
* **Retention.** ``prune_checkpoints`` keeps only the most recent N epoch checkpoints
  so a long run doesn't fill the disk; ``best.pth`` is tracked separately by the
  engine and never pruned.

Auto-resume (``find_latest_checkpoint`` + ``load_checkpoint``) lets a job that was
pre-empted on a cluster pick up from the latest checkpoint with no manual fuss.
"""

from __future__ import annotations

import glob
import os
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from .reproducibility import capture_rng_state, restore_rng_state


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    epoch: int,
    cfg: DictConfig,
    generator: torch.Generator | None = None,
    best_metric: float | None = None,
) -> None:
    """Atomically write a resumable checkpoint to ``path``.

    ``epoch`` is the index of the last *completed* epoch; resume starts at ``epoch+1``.
    ``generator`` is the DataLoader's seeded generator, captured so shuffling continues
    identically after a resume.
    """
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "rng": capture_rng_state(generator),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Atomic: write fully to a temp file, then rename over the target in one syscall.
    tmp = f"{path}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Load a checkpoint onto ``device``, restoring weights + RNG state in place.

    Returns the raw checkpoint dict so the caller can read ``epoch`` / ``best_metric``.
    """
    # weights_only=False: the payload includes RNG state (numpy/python objects), not
    # just tensors. Only load checkpoints you trust.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if checkpoint.get("rng") is not None:
        restore_rng_state(checkpoint["rng"], generator)
    return checkpoint


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Return the most recently modified epoch checkpoint in ``output_dir``, if any.

    Only ``ckpt_epoch*.pth`` files are considered — ``best.pth`` is for model selection,
    not for continuing the training sequence.
    """
    checkpoints = glob.glob(os.path.join(output_dir, "ckpt_epoch*.pth"))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)


def prune_checkpoints(output_dir: str, keep_last: int) -> None:
    """Delete all but the ``keep_last`` most recent ``ckpt_epoch*.pth`` files.

    ``keep_last <= 0`` keeps everything. ``best.pth`` is never matched, so it survives.
    """
    if keep_last <= 0:
        return
    checkpoints = sorted(
        glob.glob(os.path.join(output_dir, "ckpt_epoch*.pth")),
        key=os.path.getmtime,
    )
    for stale in checkpoints[:-keep_last]:
        os.remove(stale)
