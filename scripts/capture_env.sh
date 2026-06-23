#!/usr/bin/env bash
# Capture the full environment of a run into a directory, for "works on my machine"
# forensics. The uv lockfile is the portable *contract*; this snapshot records the
# *actual* environment a specific run executed in.
#
# Usage: scripts/capture_env.sh [output_dir]   (default: ./env_capture)
set -euo pipefail

OUT="${1:-./env_capture}"
mkdir -p "$OUT"

echo "Capturing environment into $OUT"

# Exact, resolved dependency set (the portable contract).
uv export --frozen >"$OUT/requirements.lock.txt" 2>/dev/null \
  || echo "uv export failed (is uv installed?)" >"$OUT/requirements.lock.txt"

# Git provenance: exact code that ran.
{
  echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "dirty: $([ -n "$(git status --porcelain 2>/dev/null)" ] && echo yes || echo no)"
} >"$OUT/git.txt"

# Platform / interpreter / hardware.
{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "uname: $(uname -a)"
  echo "python: $(python --version 2>&1)"
} >"$OUT/platform.txt"

# CUDA / GPU stack (captured here because the lockfile cannot pin drivers).
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi >"$OUT/nvidia-smi.txt" 2>&1
else
  echo "no nvidia-smi (CPU-only or no NVIDIA driver)" >"$OUT/nvidia-smi.txt"
fi

echo "Done. See $OUT/"
