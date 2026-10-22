#!/usr/bin/env python3
"""Unprivileged CUDA/NVTX tracing; publish only whitelisted, relative records."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import shutil
import sqlite3
import statistics
import subprocess
import sys

from public_data import public_text, write_public


def extract(database, repetitions):
    c = sqlite3.connect(database)
    c.row_factory = sqlite3.Row
    tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
    strings = dict(c.execute("select id,value from StringIds"))
    kernels = list(c.execute("select * from CUPTI_ACTIVITY_KIND_KERNEL order by start"))
    if not kernels:
        raise RuntimeError("trace contains no GPU kernels")
    runtime = list(c.execute("select * from CUPTI_ACTIVITY_KIND_RUNTIME"))
    launch = {r["correlationId"]: r for r in runtime}
    ranges = list(c.execute("select * from NVTX_EVENTS where end is not null")) if "NVTX_EVENTS" in tables else []
    base = min(k["start"] for k in kernels)
    end = max(k["end"] for k in kernels)
    records, stages = [], defaultdict(list)
    intervals = []
    allowed = {"Gate", "Up", "SwiGLU", "Down", "Dispatch", "Combine",
               "Router", "MoE", "Attention", "Prefill", "Decode"}
    by_thread = defaultdict(list)
    for r in ranges:
        label = r["text"] if "text" in r.keys() else None
        if not label and "textId" in r.keys():
            label = strings.get(r["textId"])
        if label in allowed or label == "Measured workload":
            by_thread[r["globalTid"]].append((r["start"], r["end"], label))
    for thread in by_thread:
        by_thread[thread].sort()
    starts = {thread: [r[0] for r in v] for thread, v in by_thread.items()}
    for k in kernels:
        name = public_text(strings[k["shortName"]])
        api = launch.get(k["correlationId"])
        matches, phases = [], []
        if api:
            thread, timestamp = api["globalTid"], api["start"]
            index = bisect_right(starts.get(thread, []), timestamp) - 1
            while index >= 0:
                begin, finish, label = by_thread[thread][index]
                if timestamp <= finish:
                    if label != "Measured workload":
                        matches.append((finish - begin, label))
                    if label in {"Prefill", "Decode"}:
                        phases.append(label)
                    if label in {"Prefill", "Decode", "Measured workload"}:
                        break
                index -= 1
        stage = min(matches)[1] if matches else name
        duration = k["end"] - k["start"]
        stages[stage].append(duration)
        intervals.append((k["start"], k["end"]))
        records.append({"kernel": name, "stage": stage, "start_ns": k["start"] - base,
                        "phase": phases[0] if phases else None,
                        "duration_ns": duration, "grid": [k["gridX"], k["gridY"], k["gridZ"]],
                        "block": [k["blockX"], k["blockY"], k["blockZ"]],
                        "registers_per_thread": k["registersPerThread"],
                        "shared_memory_bytes": k["staticSharedMemory"] + k["dynamicSharedMemory"]})
    union, left, right = 0, *intervals[0]
    for x, y in intervals[1:]:
        if x <= right:
            right = max(right, y)
        else:
            union += right - left
            left, right = x, y
    union += right - left
    copies = []
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        for r in c.execute("select * from CUPTI_ACTIVITY_KIND_MEMCPY"):
            copies.append({"bytes": r["bytes"], "duration_ns": r["end"] - r["start"]})
    return {"kernel_count": len(records), "repetitions": repetitions,
            "gpu_span_ms": (end - base) / 1e6,
            "gpu_kernel_sum_ms_per_invocation": sum(r["duration_ns"] for r in records) / 1e6 / repetitions,
            "gpu_active_fraction_of_trace_span": union / (end - base),
            "stages": {s: {"launches": len(v), "total_ms_per_invocation": sum(v) / 1e6 / repetitions,
                            "median_kernel_us": statistics.median(v) / 1e3} for s, v in stages.items()},
            "kernels": records, "memcpy_events": copies,
            "scope": "CUDA activity timestamps, not SM utilization, DRAM counters, or NoC packets"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", nargs="+", type=int, default=[1, 16, 128, 512, 2048])
    p.add_argument("--patterns", nargs="+", default=["balanced", "random", "hotspot"])
    p.add_argument("--private", type=Path, required=True, help="untracked run/ directory for raw reports")
    p.add_argument("--output", type=Path, required=True, help="new public output directory")
    a = p.parse_args()
    if "run" not in a.private.parts:
        p.error("raw reports must be stored under an ignored run/ directory")
    a.private.mkdir(parents=True, exist_ok=False)
    a.output.mkdir(parents=True, exist_ok=False)
    nsys = shutil.which("nsys")
    if not nsys:
        p.error("Nsight Systems CLI is required")
    cases = [(t, "block", pattern, serial) for t in a.tokens for pattern in a.patterns for serial in [False, True]]
    cases += [(t, "single", "balanced", False) for t in [1, 128]]
    summaries = []
    for t, mode, pattern, serial in cases:
        name = f"t{t}-{mode}-{pattern}" + ("-serial" if serial else "")
        prefix = a.private / name
        command = [nsys, "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
                   "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--force-overwrite=false",
                   "-o", str(prefix), sys.executable, str(Path(__file__).with_name("benchmark.py")),
                   "--tokens", str(t), "--mode", mode, "--pattern", pattern,
                   "--profile", "--profile-repeat", "3", "--output", str(prefix) + "-workload"]
        if serial:
            command.append("--serial")
        with prefix.with_suffix(".log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run([nsys, "export", "--type", "sqlite", "--output", str(prefix) + ".sqlite",
                            str(prefix) + ".nsys-rep"], stdout=log, stderr=subprocess.STDOUT, check=True)
        data = extract(str(prefix) + ".sqlite", 3)
        data["case"] = {"tokens": t, "mode": mode, "pattern": pattern, "serial": serial}
        write_public(a.output / (name + ".json"), data)
        summaries.append({"case": name, **data["case"], "kernel_count": data["kernel_count"],
                          "gpu_kernel_sum_ms_per_invocation": data["gpu_kernel_sum_ms_per_invocation"],
                          "gpu_active_fraction_of_trace_span": data["gpu_active_fraction_of_trace_span"],
                          "stages": data["stages"]})
        write_public(a.output / "summary.json", summaries)
        print(name, "PASS", data["kernel_count"], flush=True)
    write_public(a.output / "method.json", {
        "tool": subprocess.check_output([nsys, "--version"], text=True).strip(),
        "permissions": "ordinary user; no sudo; no CPU sampling or context-switch tracing",
        "capture": "cudaProfilerApi; CUDA and NVTX; three warmed-up invocations",
        "privacy": "raw binary and SQLite reports are private; only whitelisted relative records are exported",
        "counter_access": "NCU hardware counters unavailable; Nsight Systems activity tracing available"})


if __name__ == "__main__":
    main()
