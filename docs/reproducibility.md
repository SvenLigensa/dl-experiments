# Reproducibility, in depth

Reproducibility is not "I got lucky and the number came back." It is a property you
*engineer* across five layers:

| Layer | What it means | Where it lives here |
|------|----------------|---------------------|
| **1. Controlled randomness** | Seed all RNGs; fix the DataLoader-worker trap; opt into deterministic algorithms — and report variance, not one number. | [`src/reproducibility.py`](../src/reproducibility.py), [`scripts/run_seeds.py`](../scripts/run_seeds.py) |
| **2. Environment capture** | A *lockfile*, not a loose `requirements.txt`. A container as an optional upgrade. | `uv.lock`, [`Dockerfile`](../Dockerfile), [`scripts/capture_env.sh`](../scripts/capture_env.sh) |
| **3. Configuration management** | Config separate from code; CLI overrides; automatic per-run output dirs. | [`configs/`](../configs/) via Hydra |
| **4. Experiment tracking** | Log hyperparameters, metrics, **and the git commit hash** of every run. | [`src/tracking.py`](../src/tracking.py) |
| **5. Data & code versioning** | Git for code (commit hash logged into every run); documented, config-driven data location. | `run_metadata.json`, [`configs/data`](../configs/data) |

The [student guide](https://github.com/SvenLigensa/dl-experiments/blob/example/docs/student_guide.md)
(on the `example` branch) walks these five layers hands-on with a runnable task.

## 1. Controlled randomness & determinism

Four independent RNGs must be seeded (`src/reproducibility.py::seed_everything`):
`random`, `numpy`, `torch` (CPU), and `torch.cuda` (all devices). Two traps:

- **DataLoader workers reseed independently.** Each worker forks with its own RNG, so
  `numpy`/`random` calls inside your `Dataset` are *not* covered by the seed in the
  main process. Fix: `worker_init_fn=seed_worker` **and** a seeded `generator=` passed
  to the `DataLoader`. Both are wired centrally in `src/engine.py`.
- **Nondeterministic kernels.** cuDNN autotunes (`benchmark=True`) and some ops use
  atomic adds. `torch.use_deterministic_algorithms(True)` + `cudnn.deterministic=True`
  + `cudnn.benchmark=False` opt into deterministic behavior.

**The honest caveat (teach this explicitly):** bitwise-identical results are often
*unattainable* across different GPUs, driver versions, or CUDA versions, and some ops
have **no** deterministic kernel (the deterministic flag will raise on them). CPU and
GPU results differ too. The goal is *controlled* randomness — which is exactly why the
deliverable is **mean ± std over several seeds**, not the best of N runs
(`scripts/run_seeds.py`).

## 2. Environment capture

"Works on my machine" failures usually live here. Pin dependencies with a **lockfile**
(`uv.lock`), not a loose `requirements.txt` — an unpinned `pip install` resolves to
different versions a month apart. A container (`Dockerfile`) is an *optional upgrade*
that also captures the OS/CUDA stack; on clusters that forbid Docker, Apptainer
(Singularity) is the no-root equivalent. The teachable point: **the lockfile is the
portable contract; the container is an optimization on top of it.** `conda env export`
is the conda-world equivalent. `scripts/capture_env.sh` snapshots the *actual*
environment (lockfile + git + `nvidia-smi`) a given run executed in.

## 3. Configuration management

Config lives outside the code (`configs/`, Hydra + OmegaConf): CLI overrides, config
composition/groups, and automatic per-run output directories. This replaces the typical
`argparse` sprawl and means every run on every machine produces the same folder
structure — so "send me your run dir" actually works.

## 4. Experiment tracking

A result is reproducible only if you can point to the exact **code + config + data**.
`src/tracking.py` logs the resolved config, the **git commit hash + dirty flag**, and a
snapshot of the **runtime environment** (interpreter, OS, and the live torch / CUDA /
cuDNN / GPU-driver stack — the things a lockfile *cannot* pin) to both
`run_metadata.json` (always, offline) and W&B (offline-capable). If the working tree is
dirty, the commit hash alone won't reproduce the run — the trainer warns you.
`scripts/capture_env.sh` is the heavier complement: it also exports the full resolved
lockfile for a run's forensics.
Alternatives with the same provenance pattern: MLflow (local `./mlruns`, no account),
TensorBoard (minimal, local).

## 5. Data & code versioning

Git versions the code (commit hash logged per run). For data, document fixed
train/val/test splits and preprocessing, keep the data location config-driven, and be
aware of DVC / dataset hashing for versioning larger artifacts.

## Reproducible resume (checkpoints & preemption)

A checkpoint that restores only weights is *resumable*; one that also restores the RNG
state is **reproducibly** resumable — the numbers after a resume match an uninterrupted
run. `src/checkpointing.py` saves the four RNGs + the DataLoader generator at every
epoch boundary, and writes atomically (temp file + `os.replace`) so a kill mid-save
cannot corrupt the file. On a cluster, `SIGTERM`/`SIGUSR1` is caught to stop cleanly and
let the job requeue (`#SBATCH --requeue`, `--signal=B:USR1@<seconds>`); no mid-epoch
checkpoint is written, so resume simply re-runs the interrupted epoch from the last
completed one. Retention (`ckpt_keep_last`) and the `best.pth` metric (`ckpt_monitor`,
`ckpt_monitor_mode`) are configured in `configs/trainer/`.
