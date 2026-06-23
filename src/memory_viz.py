"""Plot memory-over-time from a PyTorch profiler trace.

`scripts/profile_model.py` runs the profiler with `profile_memory=True` and writes a
Chrome trace (`trace.json`). That trace contains `[memory]` instant events carrying a
timestamp plus "Total Allocated" / "Total Reserved" bytes per device. This module
turns those into an allocated/reserved-vs-time plot — the same idea as the reference
`visualize_profiles.py`, trimmed to one dependency (matplotlib, an optional extra).

Works for whatever device the trace recorded: CUDA memory on a GPU run, CPU-allocator
memory on a laptop run.
"""

from __future__ import annotations

import json
from typing import Any


def parse_memory_trace(trace_path: str) -> dict[str, dict[str, list[float]]]:
    """Extract per-device memory timelines from a profiler trace.

    Returns ``{device_label: {"t_ms": [...], "allocated_mib": [...],
    "reserved_mib": [...]}}`` with time normalised to start at zero.
    """
    with open(trace_path) as f:
        data = json.load(f)
    events = data["traceEvents"] if isinstance(data, dict) else data

    raw: dict[str, list[tuple[float, float, float]]] = {}
    for event in events:
        if event.get("name") != "[memory]":
            continue
        args = event.get("args", {})
        if "Total Allocated" not in args:
            continue
        device_type = args.get("Device Type", 0)
        device_id = args.get("Device Id", -1)
        label = "CPU" if device_type == 0 else f"CUDA:{device_id}"
        raw.setdefault(label, []).append(
            (event["ts"], args.get("Total Allocated", 0), args.get("Total Reserved", 0))
        )

    series: dict[str, dict[str, list[float]]] = {}
    for label, records in raw.items():
        records.sort(key=lambda r: r[0])
        t0 = records[0][0]
        series[label] = {
            "t_ms": [(ts - t0) / 1000.0 for ts, _, _ in records],  # ts is microseconds
            "allocated_mib": [a / 1024**2 for _, a, _ in records],
            "reserved_mib": [r / 1024**2 for _, _, r in records],
        }
    return series


def plot_memory_timeline(
    trace_path: str, out_png: str, title: str | None = None
) -> dict[str, dict[str, float]]:
    """Plot allocated/reserved memory over time; return peak stats per device.

    Raises ``ImportError`` if matplotlib is not installed (it is an optional extra:
    ``uv sync --extra viz``) and ``ValueError`` if the trace has no memory events.
    """
    try:
        import matplotlib  # ty: ignore[unresolved-import]  # optional viz extra

        matplotlib.use("Agg")  # headless: works on clusters with no display
        import matplotlib.pyplot as plt  # ty: ignore[unresolved-import]
    except ImportError as err:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "matplotlib is required for the memory plot. Install the viz extra: "
            "`uv sync --extra viz`."
        ) from err

    series = parse_memory_trace(trace_path)
    if not series:
        raise ValueError(
            f"No memory events in {trace_path}. Profile with profile_memory=True."
        )

    fig, ax = plt.subplots(figsize=(10, 5))
    peaks: dict[str, dict[str, float]] = {}
    for label, s in series.items():
        peak_alloc = max(s["allocated_mib"])
        peak_res = max(s["reserved_mib"])
        peaks[label] = {"allocated_mib": peak_alloc, "reserved_mib": peak_res}
        line = ax.plot(
            s["t_ms"],
            s["allocated_mib"],
            label=f"{label} allocated (peak {peak_alloc:.0f} MiB)",
        )
        ax.plot(
            s["t_ms"],
            s["reserved_mib"],
            linestyle="--",
            color=line[0].get_color(),
            alpha=0.6,
            label=f"{label} reserved (peak {peak_res:.0f} MiB)",
        )

    ax.set_xlabel("time (ms)")
    ax.set_ylabel("memory (MiB)")
    ax.set_title(title or "Memory usage over time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return peaks


def _format_peaks(peaks: dict[str, dict[str, Any]]) -> str:
    return " | ".join(
        f"{label}: peak allocated {p['allocated_mib']:.0f} MiB, "
        f"reserved {p['reserved_mib']:.0f} MiB"
        for label, p in peaks.items()
    )
