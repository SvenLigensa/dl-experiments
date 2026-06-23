#!/usr/bin/env python
"""Profile forward/backward time and memory for the configured model.

Wraps `torch.profiler` around a few training steps of whatever task you select with
`+experiment=...`. Works on **CPU and CUDA**, so it runs in the laptop demo too.

This is the tool behind the "report your compute budget" habit (Dodge et al., *Show
Your Work*): it tells you where time and memory actually go, and emits a Chrome trace
you can attach to a writeup.

    uv run python scripts/profile_model.py +experiment=fashion_mnist_vit trainer.device=cpu

Tune the step counts on the CLI: `+profile.warmup=10 +profile.active=30`.
"""

from __future__ import annotations

import os

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader

from src.distributed import get_device
from src.reproducibility import seed_everything, seed_worker


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Optional knobs; default if the `profile` group wasn't supplied.
    n_warmup, n_active = 5, 10
    if "profile" in cfg:
        n_warmup = int(cfg.profile.get("warmup", n_warmup))
        n_active = int(cfg.profile.get("active", n_active))

    generator = seed_everything(cfg.trainer.seed, deterministic=False)
    device = get_device(cfg.trainer.device)
    print(f"Profiling on {device} | warmup={n_warmup} active={n_active} steps")

    datamodule = hydra.utils.instantiate(cfg.data)
    loader = DataLoader(
        datamodule.train_dataset,
        batch_size=datamodule.batch_size,
        shuffle=True,
        num_workers=datamodule.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    model = hydra.utils.instantiate(cfg.model).to(device)
    criterion = hydra.utils.instantiate(cfg.loss)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    model.train()

    data_iter = iter(loader)

    def next_batch():
        nonlocal data_iter
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        inputs, targets = batch
        return inputs.to(device, non_blocking=True), targets.to(
            device, non_blocking=True
        )

    def step():
        inputs, targets = next_batch()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()

    for _ in range(n_warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities, record_shapes=True, profile_memory=True
    ) as prof:
        for _ in range(n_active):
            step()
        if device.type == "cuda":
            torch.cuda.synchronize()

    sort_key = (
        "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    )
    print(prof.key_averages().table(sort_by=sort_key, row_limit=20))

    run_dir = HydraConfig.get().runtime.output_dir
    trace_path = os.path.join(run_dir, "trace.json")
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace written to {trace_path}")
    print("Open it at chrome://tracing or https://ui.perfetto.dev")

    # Memory-over-time plot from the trace (best-effort: needs the viz extra).
    from src.memory_viz import _format_peaks, plot_memory_timeline

    png_path = os.path.join(run_dir, "memory_timeline.png")
    try:
        peaks = plot_memory_timeline(
            trace_path, png_path, title=f"Memory — {cfg.model._target_} on {device}"
        )
        print(f"Memory timeline written to {png_path}")
        print(_format_peaks(peaks))
    except ImportError as err:
        print(f"(skipping memory plot: {err})")


if __name__ == "__main__":
    main()
