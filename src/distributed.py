"""Device-agnostic helpers with an optional DistributedDataParallel (DDP) path.

The default, portable path is single-device: the same script runs on a laptop CPU,
a Colab T4, or a cluster A100 without changes, because everything is routed through
:func:`get_device` instead of hard-coding ``.cuda()``.

The optional DDP path activates automatically when the standard torchrun
environment variables (``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``) are present, e.g.::

    torchrun --nproc_per_node=4 -m src.train +experiment=...

When they are absent (the common case) every helper degrades to a sensible
single-process default, so the rest of the code never needs an ``if ddp:`` branch.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    """True if launched under torchrun with more than one process."""
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def is_main_process() -> bool:
    """True on rank 0 (or in single-process runs). Guard logging/checkpoints with this."""
    return not is_distributed() or dist.get_rank() == 0


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def get_device(prefer: str = "auto") -> torch.device:
    """Resolve the device to use.

    Args:
        prefer: ``"auto"`` (CUDA if available, else MPS, else CPU), or an explicit
            string like ``"cpu"`` / ``"cuda"`` / ``"cuda:1"`` / ``"mps"``.
    """
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device(f"cuda:{local_rank}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_distributed(device: torch.device) -> None:
    """Initialise the process group if torchrun env vars are present. No-op otherwise."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return
    backend = "nccl" if device.type == "cuda" else "gloo"
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(backend=backend, init_method="env://")
    dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    """Average a scalar tensor across processes (identity in single-process runs)."""
    if not is_distributed():
        return value
    value = value.clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value /= dist.get_world_size()
    return value
