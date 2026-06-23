#!/usr/bin/env python
"""Find the training batch size with the best throughput for the current GPU.

Sweeps powers of two, measuring step time, throughput and peak memory, and stops at
the first out-of-memory error. Picks the batch size with the highest throughput.

**CUDA-only by nature** (it watches `torch.cuda` memory and catches CUDA OOM). On CPU
or MPS it exits with a message, because batch-size tuning is a GPU-memory concern.

    uv run python scripts/find_batch_size.py +experiment=fashion_mnist_vit

The model and optimizer are rebuilt fresh for each batch size so earlier runs don't
leak memory into later measurements.
"""

from __future__ import annotations

import gc
import time

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.distributed import get_device
from src.reproducibility import seed_everything, seed_worker

WARMUP_IMAGES = 256
MEASURE_IMAGES = 1024
MAX_BATCH_SIZE = 4096


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = get_device(cfg.trainer.device)
    if device.type != "cuda":
        print(
            f"find_batch_size needs a CUDA GPU; got device '{device}'. "
            "Skipping (batch-size tuning is a GPU-memory concern)."
        )
        return

    generator = seed_everything(cfg.trainer.seed, deterministic=False)
    datamodule = hydra.utils.instantiate(cfg.data)
    criterion = hydra.utils.instantiate(cfg.loss)

    results: list[dict] = []
    batch_sizes = [2**i for i in range(13) if 2**i <= MAX_BATCH_SIZE]

    print(f"Sweeping batch sizes on {device}")
    for bs in batch_sizes:
        n_warmup = max(1, WARMUP_IMAGES // bs)
        n_measure = max(1, MEASURE_IMAGES // bs)
        loader = DataLoader(
            datamodule.train_dataset,
            batch_size=bs,
            shuffle=True,
            num_workers=datamodule.num_workers,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=generator,
        )
        if len(loader) < n_warmup + n_measure:
            print(f"  not enough data for bs={bs}; stopping.")
            break

        model = hydra.utils.instantiate(cfg.model).to(device)
        model.train()
        optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()

        try:
            data_iter = iter(loader)
            times: list[float] = []
            for i in range(n_warmup + n_measure):
                start = time.time()
                inputs, targets = next(data_iter)
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(inputs), targets)
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                if i >= n_warmup:
                    times.append(time.time() - start)

            avg = sum(times) / len(times)
            peak_mib = torch.cuda.max_memory_reserved() / 1024**2
            throughput = bs / avg
            results.append({"bs": bs, "throughput": throughput, "oom": False})
            print(
                f"  bs={bs:>5}: {avg * 1000:8.1f} ms/step | "
                f"{peak_mib:8.0f} MiB peak | {throughput:8.1f} img/s"
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
            if (
                isinstance(err, torch.cuda.OutOfMemoryError)
                or "out of memory" in str(err).lower()
            ):
                print(f"  bs={bs:>5}: OOM")
                results.append({"bs": bs, "oom": True})
                torch.cuda.empty_cache()
                gc.collect()
                break
            raise
        finally:
            del loader

    valid = [r for r in results if not r["oom"]]
    if valid:
        best = max(valid, key=lambda r: r["throughput"])
        print(
            f"\nRecommended batch size: {best['bs']} "
            f"({best['throughput']:.1f} img/s). Set it via data.batch_size={best['bs']}."
        )
    else:
        print("\nNo batch size completed without OOM.")


if __name__ == "__main__":
    main()
