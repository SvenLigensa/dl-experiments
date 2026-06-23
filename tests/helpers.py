"""A tiny, self-contained task used to drive the *real* engine in tests.

Everything here is deliberately minimal and CPU-only: a 2-layer MLP, an in-memory
random-tensor classification dataset, and a config builder that mirrors
``configs/`` closely enough to exercise ``src/engine.py`` end-to-end. No
FashionMNIST, no disk I/O, no network — so the headline guarantees (determinism,
reproducible resume) can be checked in well under a second per test.
"""

from __future__ import annotations

import logging

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import TensorDataset

from src.distributed import get_device
from src.engine import Trainer
from src.reproducibility import seed_everything
from src.tracking import Tracker

IN_FEATURES = 8
NUM_CLASSES = 4


class TinyModel(nn.Module):
    """Linear(8->16) -> ReLU -> Linear(16->4). Cheap and deterministic on CPU."""

    def __init__(self, in_features: int = IN_FEATURES, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_dataset(n: int, seed: int) -> TensorDataset:
    """A fixed (inputs, targets) classification dataset, identical every call.

    Built from a local generator so it never depends on — or perturbs — the global
    RNG that ``seed_everything`` controls.
    """
    g = torch.Generator().manual_seed(seed)
    inputs = torch.randn(n, IN_FEATURES, generator=g)
    targets = torch.randint(0, NUM_CLASSES, (n,), generator=g)
    return TensorDataset(inputs, targets)


class TinyDataModule:
    """Satisfies the engine's data contract (see ``src/engine.py`` docstring)."""

    def __init__(self, n_train: int = 32, n_val: int = 16, batch_size: int = 8):
        self.train_dataset = _make_dataset(n_train, seed=1234)
        self.val_dataset = _make_dataset(n_val, seed=5678)
        self.batch_size = batch_size
        # 0 workers: deterministic and free of subprocess flakiness. The dedicated
        # worker-reproducibility test in test_reproducibility.py covers num_workers>0.
        self.num_workers = 0


def make_cfg(
    run_dir: str,
    epochs: int = 2,
    seed: int = 0,
    deterministic: bool = True,
    **trainer_overrides: object,
) -> DictConfig:
    """Build a minimal but valid config wiring the tiny task into the engine.

    Mirrors ``configs/trainer/default.yaml`` for the ``trainer.*`` block. SGD with
    no momentum keeps optimizer state trivial and bit-reproducible. The scheduler is
    omitted (``cfg.get("scheduler") is None``) so the engine skips it.
    """
    trainer = {
        "seed": seed,
        "deterministic": deterministic,
        "epochs": epochs,
        "device": "cpu",
        "amp": False,
        "clip_grad": 0.0,
        "auto_resume": True,
        "print_freq": 1000,
        "ckpt_keep_last": 3,
        "ckpt_monitor": "val/loss",
        "ckpt_monitor_mode": "min",
    }
    trainer.update(trainer_overrides)
    cfg = {
        "trainer": trainer,
        "model": {"_target_": "tests.helpers.TinyModel"},
        "data": {
            "_target_": "tests.helpers.TinyDataModule",
            "batch_size": 8,
        },
        "loss": {"_target_": "torch.nn.CrossEntropyLoss"},
        "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.1},
        "tracking": {
            "enabled": False,
            "entity": None,
            "project": "test",
            "run_name": f"tiny_seed{seed}",
            "mode": "disabled",
        },
    }
    return OmegaConf.create(cfg)


def quiet_logger() -> logging.Logger:
    logger = logging.getLogger("test")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def build_trainer(cfg: DictConfig, run_dir: str, tracker=None) -> Trainer:
    """Wire up a real ``Trainer`` on CPU, exactly as ``src/train.py`` does.

    Pass a custom ``tracker`` to capture logged metrics; otherwise a disabled
    ``Tracker`` is used (which writes run metadata but never imports wandb).
    """
    device = get_device("cpu")
    generator = seed_everything(
        cfg.trainer.seed, deterministic=cfg.trainer.deterministic
    )
    if tracker is None:
        tracker = Tracker(cfg, run_dir, enabled=False)
    return Trainer(cfg, device, generator, run_dir, tracker, quiet_logger())
