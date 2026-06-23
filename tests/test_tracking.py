"""Tests for provenance capture and the W&B-optional Tracker."""

from __future__ import annotations

import json
import os

from omegaconf import OmegaConf

from src.tracking import (
    Tracker,
    environment_provenance,
    git_provenance,
    write_run_metadata,
)


def test_git_provenance_keys():
    prov = git_provenance()
    assert set(prov) == {"git_commit", "git_branch", "git_dirty"}


def test_environment_provenance_keys():
    env = environment_provenance()
    for key in (
        "timestamp_utc",
        "hostname",
        "python_version",
        "platform",
        "torch_version",
        "cuda_available",
        "gpus",
    ):
        assert key in env
    # On a CPU-only box the GPU fields degrade rather than crash.
    if not env["cuda_available"]:
        assert env["gpus"] == []
        assert env["nvidia_driver"] is None


def test_write_run_metadata(tmp_path):
    cfg = OmegaConf.create({"trainer": {"seed": 42}, "foo": "bar"})
    metadata = write_run_metadata(str(tmp_path), cfg)
    path = os.path.join(tmp_path, "run_metadata.json")
    assert os.path.exists(path)
    with open(path) as f:
        on_disk = json.load(f)
    assert on_disk["config"]["trainer"]["seed"] == 42
    assert "git_commit" in metadata


def test_tracker_disabled_writes_metadata_without_wandb(tmp_path):
    cfg = OmegaConf.create({"trainer": {"seed": 1}, "tracking": {"mode": "disabled"}})
    tracker = Tracker(cfg, str(tmp_path), enabled=False)

    assert os.path.exists(os.path.join(tmp_path, "run_metadata.json"))
    assert tracker._wandb is None
    # log/finish are safe no-ops when disabled.
    tracker.log({"loss": 1.0}, step=0)
    tracker.finish()
