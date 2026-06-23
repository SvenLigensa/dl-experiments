"""The training engine: a generic, device-agnostic train/validate loop.

The engine knows *nothing* about any specific task. The model, datasets, loss,
optimizer, scheduler and metrics are all built from config via
``hydra.utils.instantiate`` and handed in. Adding a new task therefore requires
**zero changes to this file** — you only add config + your own classes (see
``docs/extending.md`` and the ``example`` branch).

Data contract
-------------
``cfg.data`` must instantiate a lightweight "data module" object exposing:

* ``train_dataset`` : a ``torch.utils.data.Dataset``
* ``val_dataset``   : a ``torch.utils.data.Dataset``
* ``batch_size``    : int
* ``num_workers``   : int

The engine builds the ``DataLoader``s itself so it can inject the seeded
``generator`` and ``worker_init_fn`` centrally — this is what makes shuffling and
augmentation reproducible.

Batches are expected to be ``(inputs, targets)`` pairs. Metrics (optional) are
callables ``metric(outputs, targets) -> float``.
"""

from __future__ import annotations

import logging
import os
import signal
from collections.abc import Callable
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from . import distributed as dist_utils
from .checkpointing import (
    find_latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)
from .reproducibility import seed_worker
from .tracking import Tracker


class Trainer:
    def __init__(
        self,
        cfg: DictConfig,
        device: torch.device,
        generator: torch.Generator,
        run_dir: str,
        tracker: Tracker,
        logger: logging.Logger,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.generator = generator
        self.run_dir = run_dir
        self.tracker = tracker
        self.logger = logger
        self.start_epoch = 0
        # Best monitored metric so far (for best.pth); restored on resume.
        self.best_metric: float | None = None
        # Set by a SIGTERM/SIGUSR1 handler when the scheduler pre-empts us; the loop
        # checks it, flushes a checkpoint, and exits cleanly so the job can requeue.
        self._stop_requested = False
        self.interrupted = False

        # --- Build everything from config (task-agnostic) ---
        datamodule = hydra.utils.instantiate(cfg.data)
        self.train_loader = self._make_loader(
            datamodule.train_dataset, datamodule, shuffle=True
        )
        self.val_loader = self._make_loader(
            datamodule.val_dataset, datamodule, shuffle=False
        )

        model = hydra.utils.instantiate(cfg.model).to(device)
        self.model = self._maybe_wrap_ddp(model)

        self.criterion = hydra.utils.instantiate(cfg.loss)
        # _partial_ optimizers/schedulers are completed here with the live objects.
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters()
        )
        self.scheduler = (
            hydra.utils.instantiate(cfg.scheduler, optimizer=self.optimizer)
            if cfg.get("scheduler") is not None
            else None
        )
        self.metrics: dict[str, Callable] = (
            hydra.utils.instantiate(cfg.metrics)
            if cfg.get("metrics") is not None
            else {}
        )

        self.use_amp = bool(cfg.trainer.amp) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self._maybe_auto_resume()
        self._install_preemption_handlers()

    # ------------------------------------------------------------------ setup
    def _make_loader(
        self, dataset: Dataset, datamodule: Any, shuffle: bool
    ) -> DataLoader:
        sampler = None
        if dist_utils.is_distributed():
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False
        return DataLoader(
            dataset,
            batch_size=datamodule.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=datamodule.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=shuffle,
            # The two lines that make data loading reproducible:
            worker_init_fn=seed_worker,
            generator=self.generator,
        )

    def _maybe_wrap_ddp(self, model: torch.nn.Module) -> torch.nn.Module:
        if not dist_utils.is_distributed():
            return model
        device_ids = [self.device.index] if self.device.type == "cuda" else None
        return torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids)

    def _maybe_auto_resume(self) -> None:
        if not self.cfg.trainer.auto_resume:
            return
        ckpt = find_latest_checkpoint(self.run_dir)
        if ckpt is None:
            return
        checkpoint = load_checkpoint(
            ckpt,
            self._unwrapped,
            self.optimizer,
            self.scheduler,
            self.device,
            self.generator,  # restore the DataLoader RNG so shuffling continues
        )
        self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
        self.best_metric = checkpoint.get("best_metric", self.best_metric)
        self.logger.info(
            f"Auto-resumed from {ckpt} (starting at epoch {self.start_epoch}, "
            f"best={self.best_metric})"
        )

    def _install_preemption_handlers(self) -> None:
        """Turn SIGTERM/SIGUSR1 into a graceful stop request.

        SLURM sends SIGTERM (or SIGUSR1 with ``--signal=B:USR1@<seconds>``) before
        killing a pre-empted job. The handler only sets a flag; the training loop stops
        at the next step *without* writing a mid-epoch checkpoint and exits cleanly, so
        the job can requeue (``#SBATCH --requeue``) and auto-resume restarts from the
        last completed epoch.
        """

        def _request_stop(signum: int, _frame: Any) -> None:
            self._stop_requested = True
            self.logger.warning(
                f"Received signal {signum}; stopping after the current step "
                "(no mid-epoch checkpoint; resume re-runs this epoch)."
            )

        signal.signal(signal.SIGTERM, _request_stop)
        if hasattr(signal, "SIGUSR1"):  # not available on Windows
            signal.signal(signal.SIGUSR1, _request_stop)

    @property
    def _unwrapped(self) -> torch.nn.Module:
        # Under DDP the real model is at .module; cast because nn.Module.__getattr__
        # is typed as returning Tensor | Module.
        if dist_utils.is_distributed():
            return cast(torch.nn.Module, self.model.module)
        return self.model

    # ------------------------------------------------------------------ loops
    def train(self) -> dict[str, float]:
        final_metrics: dict[str, float] = {}
        for epoch in range(self.start_epoch, self.cfg.trainer.epochs):
            if isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)
            train_loss = self._train_one_epoch(epoch)

            if self._stop_requested:
                # Pre-empted mid-epoch. We deliberately write *no* checkpoint here:
                # checkpoints are only ever taken at epoch boundaries (with full RNG
                # state), so resume restarts cleanly from the last *completed* epoch and
                # re-runs this one — a bitwise-reproducible resume. The partial epoch is
                # discarded (finishing it mid-way would need step-level state; see the
                # checkpointing note in docs/reproducibility.md). Exit so we can requeue.
                self.interrupted = True
                self.logger.warning(
                    f"Stopped during epoch {epoch} after preemption; no mid-epoch "
                    "checkpoint written. Resume restarts from the last completed epoch."
                )
                break

            final_metrics = self._validate(epoch)
            final_metrics["train/loss"] = train_loss
            if dist_utils.is_main_process():
                self.tracker.log({"epoch": epoch, **final_metrics}, step=epoch)
            self._save_checkpoint(epoch, final_metrics)
        return final_metrics

    def _save_checkpoint(self, epoch: int, metrics: dict[str, float]) -> None:
        """Save the epoch checkpoint (+ best.pth) and prune old ones. Main only."""
        if dist_utils.is_main_process():
            self._maybe_save_best(epoch, metrics)  # may bump self.best_metric
            save_checkpoint(
                os.path.join(self.run_dir, f"ckpt_epoch{epoch}.pth"),
                self._unwrapped,
                self.optimizer,
                self.scheduler,
                epoch,
                self.cfg,
                generator=self.generator,
                best_metric=self.best_metric,
            )
            prune_checkpoints(self.run_dir, int(self.cfg.trainer.ckpt_keep_last))
        dist_utils.barrier()

    def _maybe_save_best(self, epoch: int, metrics: dict[str, float]) -> None:
        monitor = self.cfg.trainer.ckpt_monitor
        if monitor not in metrics:
            return
        value = metrics[monitor]
        mode = self.cfg.trainer.ckpt_monitor_mode
        improved = self.best_metric is None or (
            value < self.best_metric if mode == "min" else value > self.best_metric
        )
        if improved:
            self.best_metric = value
            save_checkpoint(
                os.path.join(self.run_dir, "best.pth"),
                self._unwrapped,
                self.optimizer,
                self.scheduler,
                epoch,
                self.cfg,
                generator=self.generator,
                best_metric=self.best_metric,
            )
            self.logger.info(f"New best {monitor}={value:.4f} — saved best.pth")

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        running = torch.zeros(1, device=self.device)
        n = 0
        for step, (inputs, targets) in enumerate(self.train_loader):
            if self._stop_requested:
                self.logger.warning(
                    f"Preemption: interrupting epoch {epoch} at "
                    f"step {step}/{len(self.train_loader)}."
                )
                break
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            if self.cfg.trainer.clip_grad:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.trainer.clip_grad
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running += loss.detach() * inputs.size(0)
            n += inputs.size(0)

            if step % self.cfg.trainer.print_freq == 0:
                self.logger.info(
                    f"[epoch {epoch} | step {step}/{len(self.train_loader)}] "
                    f"loss {loss.item():.4f}"
                )

        if self.scheduler is not None:
            self.scheduler.step()
        avg = dist_utils.reduce_mean(running / max(n, 1)).item()
        return avg

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        running = torch.zeros(1, device=self.device)
        metric_sums = {name: 0.0 for name in self.metrics}
        n = 0
        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            running += loss.detach() * inputs.size(0)
            for name, fn in self.metrics.items():
                metric_sums[name] += float(fn(outputs, targets)) * inputs.size(0)
            n += inputs.size(0)

        n = max(n, 1)
        results = {"val/loss": dist_utils.reduce_mean(running / n).item()}
        for name, total in metric_sums.items():
            value = dist_utils.reduce_mean(torch.tensor(total / n, device=self.device))
            results[f"val/{name}"] = value.item()
        self.logger.info(
            f"[epoch {epoch}] " + " ".join(f"{k}={v:.4f}" for k, v in results.items())
        )
        return results
