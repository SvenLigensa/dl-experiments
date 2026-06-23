# Student Guide — Deep Learning example pipeline

You are on the **`example`** branch: the template's plug-in points (model, data, loss,
metric) are filled with a small **Vision Transformer on FashionMNIST**, so you can run
the whole pipeline end to end on a laptop CPU. This one file is your starting point.

- **The concepts** — *why* each piece exists — are in
  [`reproducibility.md`](reproducibility.md). Read it alongside this guide.
- **Adding your own model/dataset** is in [`extending.md`](extending.md).
- **The big picture** (architecture diagram, tooling choices) is in the
  [README](../README.md).
- This file is the **hands-on path**: understanding how all the pieces come together
  to carry out Deep Learning experiments in a principled manner

## What was added on top of `main`

Only config + small classes were added — **the engine (`src/engine.py`) is untouched.**
That is the whole point: a new task is *config*, not new training code.

| Piece | File(s) |
|------|------|
| Model | `src/models/vit.py` + `configs/model/vit.yaml` |
| Data  | `src/data/fashion_mnist.py` + `configs/data/fashion_mnist.yaml` |
| Loss  | `configs/loss/cross_entropy.yaml` (`torch.nn.CrossEntropyLoss`) |
| Metric | `src/metrics.py` + `configs/metrics/classification.yaml` |
| Experiment | `configs/experiment/fashion_mnist_vit.yaml` (ties it together) |

## Setup

```bash
git clone <repo-url> && cd dl-experiments
git switch example
uv sync                  # installs the EXACT versions from uv.lock — the portable contract
export DATA_DIR=./data   # config-driven location; FashionMNIST auto-downloads here
```

No `uv`? `pip install uv` first, or use `conda` (the point is a *lockfile*, not loose
pins). On Colab `DATA_DIR` is typically a Drive mount; on a cluster, a scratch path.

## The session arc — break it, then fix it

**1. Break it (layer 1).** Run the same training twice with no seeding — the numbers
diverge.

```bash
uv run python scripts/demo_determinism.py
```

You will see the *uncontrolled* pair differ and the *controlled* pair come out
bit-identical. That is the whole lesson on one screen.

**2. Fix it, on one machine.** The full pipeline is seeded by default (`trainer.seed`,
`trainer.deterministic=true`). Run this twice → identical `val/accuracy`:

```bash
uv run python -m src.train +experiment=fashion_mnist_vit trainer.epochs=1 trainer.device=cpu tracking.enabled=false
```

**3. It still differs across hardware — and that's correct.** Run once on CPU, once on
GPU:

```bash
uv run python -m src.train +experiment=fashion_mnist_vit trainer.device=cuda
```

The numbers will *not* match across CPU/GPU (or across GPU models). Bitwise determinism
across hardware is not attainable; chasing it is the wrong goal — this is layer 1's
honest caveat.

**4. Pin the environment + track provenance (layers 2 & 4).** Every run already writes
`outputs/DATE/TIME/run_metadata.json` with the git commit hash (and a warning if your
working tree was dirty), and `uv.lock` pins the exact dependency set. "Send me your run
dir" is enough to reproduce — open `run_metadata.json` to see the commit that produced
your number.

**5. The real deliverable — report variance, not a magic number (layer 1).**

```bash
uv run python scripts/run_seeds.py +experiment=fashion_mnist_vit trainer.device=cpu --seeds 0 1 2 3 4
```

Report the printed **mean ± std**, not the best of the five runs.

> **Caveats to flag up front**
> - Compute nodes (Colab/cluster) often can't reach the internet — keep W&B in its
>   default `offline` mode (or `tracking.enabled=false`); don't make students log in.
> - `trainer.deterministic=true` can raise on ops without a deterministic kernel — that
>   error is *informative*, not a setup failure.

## Syncing experiment tracking (W&B)

W&B defaults to **offline** mode: runs write to a local `wandb/` folder with no account
or network. Upload them later with:

```bash
uv run wandb sync                     # all offline runs
uv run wandb sync wandb/offline-run-* # or a specific run
```

`wandb` lives inside the project's uv-managed `.venv`, so always prefix with `uv run`
(a bare `wandb` is "command not found"). On a cluster, `uv run wandb login` once and
sync from a **login node** (compute nodes usually have no outbound internet). To stream
live instead: `tracking.enabled=true tracking.mode=online tracking.entity=<your-entity>`.

## Profiling, memory & batch-size tuning

These are the framework's optional operator tools (documented generically under
[Convenience tools](../README.md#convenience-tools) in the README); here they are on the
ViT. The memory plot needs matplotlib: `uv sync --extra viz`.

```bash
# Where does time/memory go? Writes a Chrome trace (open in chrome://tracing or
# perfetto.dev) + a memory-over-time plot, memory_timeline.png, in the run dir.
uv run python scripts/profile_model.py +experiment=fashion_mnist_vit trainer.device=cpu
#   tune the step counts: +profile.warmup=10 +profile.active=30

# Re-plot memory from an existing trace (e.g. to compare runs)
uv run python scripts/visualize_memory.py outputs/DATE/TIME/trace.json

# What batch size maximises GPU throughput? Sweeps powers of two until OOM.
# CUDA-only — on a CPU/MPS laptop it prints a "needs a GPU" message and exits.
uv run python scripts/find_batch_size.py +experiment=fashion_mnist_vit
```

`memory_timeline.png` shows allocated (solid) and reserved (dashed) memory per device
over time; the sawtooth is the allocate-on-forward/backward, free-after pattern of each
step.

## Now make it yours

Swap pieces on the CLI without touching code:

```bash
uv run python -m src.train +experiment=fashion_mnist_vit optimizer.lr=1e-3 trainer.epochs=10 model.depth=6
uv run python -m src.train +experiment=fashion_mnist_vit optimizer=sgd scheduler=none
```

Then add your *own* model/dataset following [`extending.md`](extending.md). The model
here is deliberately tiny — this template is about *reproducibility*, not accuracy.
