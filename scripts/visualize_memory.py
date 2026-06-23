#!/usr/bin/env python
"""Plot the memory-over-time timeline from a profiler trace.

Companion to `scripts/profile_model.py` (which already auto-generates this plot for
its own run). Use this standalone to re-plot, or to compare a trace produced
elsewhere.

    uv run python scripts/visualize_memory.py outputs/DATE/TIME/trace.json
    uv run python scripts/visualize_memory.py trace.json --out memory.png

Requires the viz extra: `uv sync --extra viz`.
"""

from __future__ import annotations

import argparse
import os

from src.memory_viz import _format_peaks, plot_memory_timeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", help="Path to a profiler trace.json")
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path (default: memory_timeline.png next to the trace).",
    )
    args = parser.parse_args()

    out = args.out or os.path.join(
        os.path.dirname(args.trace) or ".", "memory_timeline.png"
    )
    peaks = plot_memory_timeline(args.trace, out)
    print(f"Memory timeline written to {out}")
    print(_format_peaks(peaks))


if __name__ == "__main__":
    main()
