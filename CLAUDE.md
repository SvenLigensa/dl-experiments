# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A **task-independent** scaffold for reproducible deep-learning experiments. `main` ships
**no model or dataset on purpose** — `cfg.model` / `cfg.data` / `cfg.loss` default to
`???` (mandatory-missing), so `python -m src.train` is *supposed* to fail fast with a
self-documenting error. A complete runnable example (ViT on FashionMNIST) lives on the
**`example` branch**, not `main`. The engine is generic; you add a task by writing config,
never by editing engine code.

## Branch policy (IMPORTANT — keep the branches in sync)

There are two long-lived branches with a strict relationship:

- **`main`** — the framework only. No task; running fails fast by design.
- **`example`** — **everything `main` has, plus** the few task files needed to run the
  worked example (ViT on FashionMNIST): `src/models/`, `src/data/`, `src/metrics.py`,
  the `configs/{model,data,loss,metrics,experiment}` entries, `scripts/demo_determinism.py`,
  and `docs/student_guide.md`.

**Rule: every framework feature on `main` must also be on `example`.** The framework
files are meant to be **byte-identical on both branches**:

- `src/` except the task modules `src/models/`, `src/data/`, `src/metrics.py`
- `configs/{config,trainer,optimizer,scheduler}` (the task groups `model`/`data`/`loss`/
  `metrics`/`experiment` exist only on `example`)
- `scripts/`, `docs/reproducibility.md`, `docs/extending.md`, `tests/`
- `pyproject.toml` — **identical except** `example` adds the `torchvision` dependency
  (the lone deliberate divergence; `uv.lock` follows from it)

### Workflow — `example` forks from `main` and stays ahead of it (ALWAYS follow this)

`example` must always have `main`'s tip as an ancestor, holding **only** the task-delta
commit on top. Keep it that way with one rule and one routine:

- **Never commit framework changes on `example`.** Do all framework work on `main`. The
  task files (`src/models/`, `src/data/`, `src/metrics.py`, `scripts/demo_determinism.py`,
  the `configs/{model,data,loss,metrics,experiment}` entries, `docs/student_guide.md`,
  and the `torchvision`/lockfile bump) are the *only* things that live on `example`.
- **After every change to `main`, rebase `example` onto it** (in the same unit of work):

  ```bash
  git checkout main        # ... commit framework change ...
  git checkout example && git rebase main
  git push --force-with-lease origin example   # example was rewritten
  ```

  Conflicts only ever surface in `pyproject.toml` / `uv.lock`; resolve and `--continue`.

This makes the "byte-identical framework" policy and the fork structure the same thing.

### Verify parity (true framework diff — task paths excluded)

```bash
# 1. example descends from main's tip
git merge-base --is-ancestor main example && echo "ok: example is ahead of main"

# 2. framework files are byte-identical (NOTE: must exclude the task paths, or the
#    plain `git diff main..example -- src/ ...` will list the task additions as "drift")
git diff --name-only main..example -- \
  src/ scripts/ tests/ \
  configs/config.yaml configs/trainer configs/optimizer configs/scheduler \
  docs/reproducibility.md docs/extending.md \
  ':(exclude)src/models' ':(exclude)src/data' ':(exclude)src/metrics.py' \
  ':(exclude)scripts/demo_determinism.py'
# expected output: empty
```

Both checks are enforced automatically by the version-controlled `hooks/pre-push` whenever
a push includes the `example` ref. Because `core.hooksPath` is a local git setting, enable
it **once per clone**:

```bash
git config core.hooksPath hooks
```

Bypass only in emergencies with `git push --no-verify`.

The branches are currently at full framework parity (reproducible-resume, `best.pth` +
`ckpt_monitor` selection, `prune_checkpoints` retention, atomic checkpoint writes,
SIGTERM/SIGUSR1 preemption handling, and the `tests/` suite are all present on both).

## Commands

```bash
uv sync                       # install the exact pinned env (the lockfile is the contract)
uv sync --extra viz           # + matplotlib, needed by profile_model.py / visualize_memory.py

# Run a training (needs an experiment that fills model/data/loss — see example branch):
uv run python -m src.train +experiment=<exp> trainer.epochs=5 trainer.device=cpu
# CLI overrides compose on the config: optimizer.lr=1e-3 trainer.amp=true etc.

# Tests
uv run pytest                 # full suite
uv run pytest -m "not slow"   # skip end-to-end multi-epoch training tests
uv run pytest tests/test_engine.py::test_name   # a single test

# Lint / format / type-check (all Astral; same as pre-commit)
uv run ruff check --fix .
uv run ruff format .
uv run ty check                # ty is beta; resolves types from .venv, so `uv sync` first
pre-commit run --all-files
```

`pre-commit` order matters: `ruff-check --fix` runs **before** `ruff-format` (a fix may
need reformatting).

## Architecture

Config flows down; the engine builds task pieces from config via `hydra.utils.instantiate`
and runs one device-agnostic loop.

- **`src/train.py`** — Hydra entrypoint. Fixed order: `seed_everything` (before any RNG is
  touched) → resolve device + optional DDP → logger → `Tracker` (provenance) → `Trainer`.
  Writes `final_metrics.json` only on clean completion (not on preemption).
- **`src/engine.py`** — `Trainer`. Knows nothing about any task. Instantiates model, data,
  loss, optimizer, scheduler, metrics from `cfg.*`. **Builds the `DataLoader`s itself** so it
  can inject the seeded `generator` + `worker_init_fn` centrally — that injection is what
  makes shuffling/augmentation reproducible, so don't move loader construction elsewhere.
- **`src/reproducibility.py`** — the centerpiece. Seeds all four RNGs (python/numpy/torch
  CPU/torch CUDA), `seed_worker` reseeds numpy+random inside DataLoader workers, and
  `capture_rng_state`/`restore_rng_state` make resume *bitwise-reproducible*, not just
  resumable. `seed_everything(deterministic=True)` flips **process-global** state.
- **`src/distributed.py`** — single-device is the default portable path; everything routes
  through `get_device()` (`cuda`→`mps`→`cpu`). DDP activates **automatically** when torchrun
  env vars (`RANK`/`WORLD_SIZE`/`LOCAL_RANK`) are present, so there are no `if ddp:` branches
  in the rest of the code. Guard side-effects with `is_main_process()`.
- **`src/checkpointing.py`** — atomic save, auto-resume, prune. Checkpoints are taken **only
  at epoch boundaries** (with full RNG state). `best.pth` is tracked via
  `trainer.ckpt_monitor` / `ckpt_monitor_mode`.
- **`src/tracking.py`** — W&B, **defaulting to offline** (works air-gapped; `wandb sync`
  later). Writes `run_metadata.json` with the git commit hash; warns loudly if the working
  tree is dirty (a dirty tree means the commit hash alone won't reproduce the run).

### Data contract

`cfg.data` must instantiate a "data module" object exposing `train_dataset`, `val_dataset`,
`batch_size`, `num_workers`. Batches are `(inputs, targets)`. Metrics are callables
`metric(outputs, targets) -> float`. Optimizers/schedulers are `_partial_` in config and
completed in the engine with the live `params=`/`optimizer=`.

### Config layout (Hydra groups under `configs/`)

`config.yaml` composes one option from each of `trainer`, `optimizer`, `scheduler`, `model`,
`data`, `loss`. Add a task by writing config whose `_target_` points at your class — **no
engine changes**. See `docs/extending.md`. Per-run output goes to `outputs/<date>/<time>/`
(`run_metadata.json`, checkpoints, `train.log`, `final_metrics.json`).

## Operator scripts (`scripts/`)

All read the same `+experiment=...` config and build via `instantiate`, so they work for any
task: `run_seeds.py` (mean ± std — report this, never a single number), `profile_model.py`
(torch.profiler, writes Chrome trace + memory plot), `visualize_memory.py`,
`find_batch_size.py` (CUDA-only, no-ops on CPU/MPS), `capture_env.sh`.

## Gotchas

- The `example` branch is where runnable training and demos live (`demo_determinism.py`,
  `student_guide.md`). On `main`, a no-task run failing fast is correct behavior.
- Because `seed_everything` mutates process-global determinism flags, the autouse
  `reset_determinism` fixture in `tests/conftest.py` undoes them after each test to keep the
  suite order-independent. Preserve that pattern when adding tests that seed.
- Results are **not** bitwise-identical across CPU/GPU or GPU models — expected, hence
  reporting mean ± std over seeds. Concepts in `docs/reproducibility.md`.
