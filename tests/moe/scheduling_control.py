#!/usr/bin/env python3
"""Paired padded/compact controls and unprivileged compact-grid traces."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from nsight_profile import extract
from public_data import write_public


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--private", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if "run" not in a.private.parts:
        p.error("private reports must stay under run/")
    a.private.mkdir(parents=True, exist_ok=False)
    (a.output / "hardware").mkdir(parents=True, exist_ok=False)
    (a.output / "nsight").mkdir()
    benchmark = Path(__file__).with_name("benchmark.py")
    rows = []
    for tokens in [128, 2048]:
        for pattern in ["balanced", "hotspot"]:
            for schedule in ["padded", "compact"]:
                name = f"t{tokens}-{pattern}-{schedule}"
                command = [sys.executable, str(benchmark), "--tokens", str(tokens),
                           "--pattern", pattern, "--mode", "block", "--schedule", schedule]
                with (a.private / (name + ".log")).open("w") as log:
                    subprocess.run(command + ["--output", str(a.private / (name + "-workload"))],
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                record = json.loads((a.private / (name + "-workload/result.json")).read_text())
                write_public(a.output / "hardware" / (name + ".json"), record)
                rows.append({"case": name, "tokens": tokens, "pattern": pattern,
                             "schedule": schedule, "median_ms": record["median_ms"],
                             "gate_launched_ctas": record["analytical_not_measured"]["gate_launched_ctas"],
                             "gate_useful_ctas": record["analytical_not_measured"]["gate_useful_ctas"]})
                print(name, "PASS", record["median_ms"], flush=True)
    write_public(a.output / "hardware-summary.json", rows)
    for tokens in [128, 2048]:
        for pattern in ["balanced", "hotspot"]:
            name = f"t{tokens}-{pattern}-compact"
            prefix = a.private / ("trace-" + name)
            command = [shutil.which("nsys"), "profile", "--trace=cuda,nvtx", "--sample=none",
                       "--cpuctxsw=none", "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
                       "-o", str(prefix), sys.executable, str(benchmark), "--tokens", str(tokens),
                       "--pattern", pattern, "--mode", "block", "--schedule", "compact", "--profile",
                       "--output", str(prefix) + "-workload"]
            with prefix.with_suffix(".log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                subprocess.run(["nsys", "export", "--type", "sqlite", "--output", str(prefix) + ".sqlite",
                                str(prefix) + ".nsys-rep"], stdout=log, stderr=subprocess.STDOUT, check=True)
            record = extract(str(prefix) + ".sqlite", 3)
            record["case"] = {"tokens": tokens, "pattern": pattern, "schedule": "compact"}
            write_public(a.output / "nsight" / (name + ".json"), record)
            print(name, "TRACE PASS", flush=True)


if __name__ == "__main__":
    main()
