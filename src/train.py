"""Hydra entrypoint.

Run with::

    python -m src.train +experiment=<your_experiment>

Hydra gives every run its own timestamped output directory (configured in
``configs/config.yaml``), so "send me your run dir" works identically on a laptop,
in Colab, and on a cluster. Command-line overrides compose on top of the config,
e.g. ``trainer.epochs=5 trainer.device=cpu optimizer.lr=1e-3``.

The core is task-independent: ``cfg.model`` / ``cfg.data`` / ``cfg.loss`` default to
mandatory-missing (``???``) values, so running with no task selected fails fast with
a clear message pointing you at an experiment config.
"""

from __future__ import annotations

import json
import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from . import distributed as dist_utils
from .engine import Trainer
from .logging_utils import get_logger
from .reproducibility import seed_everything
from .tracking import Tracker


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_dir = HydraConfig.get().runtime.output_dir

    # 1. Controlled randomness — before anything touches an RNG.
    generator = seed_everything(
        cfg.trainer.seed, deterministic=cfg.trainer.deterministic
    )

    # 2. Device + optional distributed setup.
    device = dist_utils.get_device(cfg.trainer.device)
    dist_utils.setup_distributed(device)
    is_main = dist_utils.is_main_process()

    logger = get_logger(output_dir=run_dir, is_main=is_main)
    if is_main:
        logger.info(f"Run directory: {run_dir}")
        logger.info(f"Device: {device}  |  world size: {dist_utils.get_world_size()}")
        logger.info("Resolved config:\n" + OmegaConf.to_yaml(cfg, resolve=True))

    # 3. Tracking + provenance (writes run_metadata.json with the git commit hash).
    tracker = Tracker(cfg, run_dir, enabled=is_main and cfg.tracking.enabled)
    if is_main and tracker.metadata.get("git_dirty"):
        logger.warning(
            "Working tree is DIRTY: the recorded commit hash alone will not "
            "reproduce this run. Commit your changes for a reproducible result."
        )

    try:
        trainer = Trainer(cfg, device, generator, run_dir, tracker, logger)
        final_metrics = trainer.train()
        if is_main and trainer.interrupted:
            # Pre-empted: a checkpoint was written, but these are not final results.
            # Don't write final_metrics.json — re-run (auto_resume) to finish first.
            logger.warning(
                "Run was interrupted before completing; final_metrics.json not "
                "written. Re-launch the same command to resume from the last "
                "completed-epoch checkpoint."
            )
        elif is_main:
            # Used by scripts/run_seeds.py to aggregate mean +/- std across seeds.
            with open(os.path.join(run_dir, "final_metrics.json"), "w") as f:
                json.dump(final_metrics, f, indent=2)
    finally:
        if is_main:
            tracker.finish()
        dist_utils.cleanup_distributed()


if __name__ == "__main__":
    main()
