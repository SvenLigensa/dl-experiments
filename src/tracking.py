"""Experiment tracking and provenance.

A result is only reproducible if you can point to the *exact code + config + data*
that produced it. This module captures that provenance and ships it to two places:

1. ``<run_dir>/run_metadata.json`` — always written, no account or network needed.
2. Weights & Biases — when enabled. W&B has an **offline mode**
   (``mode="offline"``, sync later with ``wandb sync``) which matters on air-gapped
   cluster nodes that cannot open an outbound connection.

The single most important habit this enforces: every run records the **git commit
hash** (and whether the working tree was dirty). If the tree is dirty, the commit
hash alone does not reproduce the run — so we surface that loudly.

Alternatives (same idea, different tool):
    * MLflow  — ``mlflow.start_run()`` logs to a local ``./mlruns`` directory by
      default, no account required.
    * TensorBoard — ``torch.utils.tensorboard.SummaryWriter`` for minimal, local logging.
The provenance pattern below (commit hash + resolved config) applies to all three.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
from datetime import UTC, datetime
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf


def _run(*args: str) -> str | None:
    try:
        out = subprocess.run(
            list(args),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git(*args: str) -> str | None:
    return _run("git", *args)


def environment_provenance() -> dict[str, Any]:
    """Capture the runtime environment a run actually executed in.

    The lockfile pins Python *packages*; this records what a lockfile cannot — the
    interpreter build, the OS, and the GPU/driver/CUDA stack — because results are not
    bitwise-portable across them (see ``docs/reproducibility.md`` layer 2). Best-effort:
    every field degrades to ``None`` rather than failing a run.
    """
    import torch

    cuda = torch.cuda.is_available()
    env: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": _module_version("numpy"),
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "gpus": (
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if cuda
            else []
        ),
        # The driver lives below the CUDA toolkit and is not pinned by anything above.
        "nvidia_driver": (
            _run(
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            )
            if cuda
            else None
        ),
    }
    return env


def _module_version(name: str) -> str | None:
    try:
        return __import__(name).__version__
    except Exception:
        return None


def git_provenance() -> dict[str, Any]:
    """Capture the git commit hash, branch, and dirty-tree flag of the repo."""
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "git_commit": commit,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # dirty == there are uncommitted changes, so the commit hash is not enough
        # to reproduce this run.
        "git_dirty": bool(status) if status is not None else None,
    }


def write_run_metadata(run_dir: str, cfg: DictConfig) -> dict[str, Any]:
    """Write provenance + resolved config to ``<run_dir>/run_metadata.json``."""
    metadata = {
        **git_provenance(),
        "environment": environment_provenance(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    return metadata


class Tracker:
    """Thin wrapper over W&B that always records local provenance.

    Works with ``mode`` in {"online", "offline", "disabled"}. Even when disabled,
    ``run_metadata.json`` is written, so provenance is never lost.
    """

    def __init__(self, cfg: DictConfig, run_dir: str, enabled: bool = True) -> None:
        self.enabled = enabled
        self.metadata = write_run_metadata(run_dir, cfg)
        self._wandb = None

        if not enabled:
            return

        import wandb

        self._wandb = wandb
        wandb.init(
            entity=cfg.tracking.entity,
            project=cfg.tracking.project,
            name=cfg.tracking.run_name,
            mode=cfg.tracking.mode,  # online | offline | disabled
            dir=run_dir,
            config=cast("dict[str, Any]", OmegaConf.to_container(cfg, resolve=True)),
        )
        # Make the git provenance prominent in the run's config.
        wandb.config.update(
            {k: v for k, v in self.metadata.items() if k != "config"},
            allow_val_change=True,
        )

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
