"""Test the profiler-trace memory parser (no matplotlib needed)."""

from __future__ import annotations

import json

from src.memory_viz import parse_memory_trace


def _write_trace(path, events):
    with open(path, "w") as f:
        json.dump({"traceEvents": events}, f)


def test_parse_memory_trace_normalises_and_converts(tmp_path):
    mib = 1024**2
    trace = tmp_path / "trace.json"
    _write_trace(
        trace,
        [
            {"name": "other", "args": {}},
            {
                "name": "[memory]",
                "ts": 1000.0,  # microseconds
                "args": {
                    "Total Allocated": 1 * mib,
                    "Total Reserved": 2 * mib,
                    "Device Type": 0,
                    "Device Id": -1,
                },
            },
            {
                "name": "[memory]",
                "ts": 2000.0,
                "args": {
                    "Total Allocated": 3 * mib,
                    "Total Reserved": 4 * mib,
                    "Device Type": 0,
                    "Device Id": -1,
                },
            },
        ],
    )

    series = parse_memory_trace(str(trace))
    assert set(series) == {"CPU"}
    cpu = series["CPU"]
    assert cpu["t_ms"] == [0.0, 1.0]  # normalised to start, microseconds -> ms
    assert cpu["allocated_mib"] == [1.0, 3.0]
    assert cpu["reserved_mib"] == [2.0, 4.0]


def test_parse_memory_trace_empty(tmp_path):
    trace = tmp_path / "trace.json"
    _write_trace(trace, [{"name": "other", "args": {}}])
    assert parse_memory_trace(str(trace)) == {}
