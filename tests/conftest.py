"""Shared fixtures.

The autouse ``reset_determinism`` fixture is the load-bearing one: ``seed_everything``
flips *process-global* state (``torch.use_deterministic_algorithms``, cuDNN flags,
and a couple of env vars). Without a reset, a test that requests determinism would
silently change the behaviour of every test that runs after it. Resetting keeps the
suite order-independent.
"""

from __future__ import annotations

import os

import pytest
import torch


@pytest.fixture(autouse=True)
def reset_determinism():
    yield
    # Undo the global flips made by seed_everything(deterministic=True).
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    for var in ("CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED"):
        os.environ.pop(var, None)


@pytest.fixture
def run_dir(tmp_path) -> str:
    """A clean per-test output directory for checkpoints and run metadata."""
    d = tmp_path / "run"
    d.mkdir()
    return str(d)
