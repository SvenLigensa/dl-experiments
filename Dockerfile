# OPTIONAL: a container is the "cluster-grade upgrade" on top of the uv lockfile.
# The lockfile is the portable contract that works everywhere (laptop, Colab, login
# node) without root. The container additionally pins the OS / CUDA stack — use it
# where you have Docker (or convert to Apptainer/Singularity on clusters that forbid
# Docker; see README).
#
#   docker build -t reproducible-dl .
#   docker run --rm -it --gpus all reproducible-dl python -m src.train +experiment=...
#
# For a GPU build, swap the base image for an nvidia/cuda runtime that matches your
# pinned torch wheel.
FROM python:3.11-slim

# Install uv (fast, reproducible installs from the lockfile).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first (cached layer) from the lockfile only.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Then the project itself.
COPY . .
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "-m", "src.train", "--help"]
