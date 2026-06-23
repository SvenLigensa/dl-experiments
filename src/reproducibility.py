"""Controlled randomness for reproducible experiments.

This is the heart of the template. Reproducibility starts with controlling every
source of randomness in the stack, but it does *not* end with the illusion of a
single "true number".

There are four independent random number generators a typical PyTorch program
touches, and all four must be seeded:

1. Python's ``random`` module
2. NumPy's global RNG
3. PyTorch on the CPU (``torch.manual_seed``)
4. PyTorch on every CUDA device (``torch.cuda.manual_seed_all``)

Two traps catch almost everyone:

* **DataLoader workers reseed themselves.** Each worker process forks with its own
  RNG state, so ``numpy``/``random`` calls inside a ``Dataset`` (augmentations,
  shuffling) are *not* governed by the seed you set in the main process. The fix is
  a ``worker_init_fn`` (:func:`seed_worker`) plus a seeded ``generator`` passed to
  the ``DataLoader``. See https://pytorch.org/docs/stable/notes/randomness.html
* **Algorithm choice is nondeterministic by default.** cuDNN benchmarks kernels and
  some ops use atomic adds. :func:`seed_everything` with ``deterministic=True`` opts
  into deterministic algorithms where they exist.

The honest caveat, worth teaching explicitly
---------------------------------------------
Bitwise-identical results are often **unattainable** across different GPUs, driver
versions, or CUDA versions, and some operations have *no* deterministic kernel at
all (``torch.use_deterministic_algorithms(True)`` will raise on those). CPU vs GPU
results differ too. The goal is therefore *controlled* randomness, not a single
magic number — which is exactly why you report **mean +/- std over several seeds**
(see ``scripts/run_seeds.py``) rather than the best of N runs.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> torch.Generator:
    """Seed every RNG and (optionally) request deterministic algorithms.

    Args:
        seed: The base seed applied to ``random``, ``numpy`` and ``torch``.
        deterministic: If ``True``, force deterministic algorithms and disable
            cuDNN autotuning. This can be slower and will raise a ``RuntimeError``
            if an op without a deterministic implementation is used — that error is
            informative, not a bug. If ``False``, only the seeds are set (fast, but
            runs are not exactly repeatable).

    Returns:
        A seeded :class:`torch.Generator` to hand to a ``DataLoader`` via its
        ``generator=`` argument, so shuffling order is reproducible.
    """
    # PYTHONHASHSEED governs hash randomization (e.g. set iteration order). It must
    # be set before the interpreter starts to take full effect, but setting it here
    # still helps subprocesses spawned later (e.g. DataLoader workers).
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Opt into deterministic kernels. warn_only=False means we *want* to know
        # (via an exception) when an op cannot be made deterministic.
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Required for deterministic CUDA matmuls (cuBLAS). No effect on CPU.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.backends.cudnn.benchmark = True

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def capture_rng_state(generator: torch.Generator | None = None) -> dict[str, Any]:
    """Snapshot every RNG so a resumed run can continue the *same* random trajectory.

    Seeding (:func:`seed_everything`) only fixes the *start*. To resume a run without
    changing its numbers, you must also restore where each RNG had advanced *to* at the
    checkpoint boundary — otherwise "resumable" is not "reproducible-resumable". This
    captures the same four RNGs that :func:`seed_everything` sets, plus the DataLoader's
    ``generator`` (which drives shuffling and per-worker seeding).
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "generator": generator.get_state() if generator is not None else None,
    }


def restore_rng_state(
    state: dict[str, Any], generator: torch.Generator | None = None
) -> None:
    """Restore RNGs captured by :func:`capture_rng_state`. Best-effort and safe.

    The CUDA state is only restored when CUDA is available *and* the device count
    matches what was saved, so a GPU checkpoint resumed on CPU (or on a differently
    sized box) degrades gracefully instead of raising.

    RNG states are CPU ``ByteTensor``s, but a checkpoint loaded with
    ``map_location="cuda"`` moves *every* tensor in it to the GPU — including these.
    The RNG setters reject non-CPU tensors (``RNG state must be a torch.ByteTensor``),
    so we force each one back to CPU before restoring. ``.cpu()`` is a no-op when the
    tensor is already on CPU.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    cuda_state = state.get("torch_cuda") or []
    if torch.cuda.is_available() and len(cuda_state) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([s.cpu() for s in cuda_state])
    if generator is not None and state.get("generator") is not None:
        generator.set_state(state["generator"].cpu())


def seed_worker(worker_id: int) -> None:  # noqa: ARG001 - signature fixed by PyTorch
    """``worker_init_fn`` that reseeds NumPy and ``random`` inside DataLoader workers.

    PyTorch already gives each worker a distinct ``torch`` seed derived from the
    main generator, exposed via :func:`torch.initial_seed`. We reuse it to seed the
    *other* two RNGs so that augmentations relying on ``numpy``/``random`` are
    reproducible across runs.

    Pass this together with a seeded generator::

        DataLoader(ds, worker_init_fn=seed_worker, generator=seed_everything(seed))
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
