#!/usr/bin/env python3
"""Run hardware controls sequentially; capture/profiling are separate runs."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 128])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    benchmark = Path(__file__).with_name("benchmark.py")
    cases = []
    for t in a.tokens:
        cases.append((t, "single", "balanced", False))
        for pattern in ["balanced", "random", "hotspot"]:
            for mode in ["dispatch", "combine", "experts", "block"]:
                cases.append((t, mode, pattern, False))
            cases.append((t, "experts", pattern, True))
    rows = []
    for tokens, mode, pattern, serial in cases:
        name = f"t{tokens}-{mode}-{pattern}" + ("-serial" if serial else "")
        out = a.output / name
        command = [sys.executable, str(benchmark), "--tokens", str(tokens), "--mode", mode,
                   "--pattern", pattern, "--output", str(out)]
        if serial:
            command.append("--serial")
        with (a.output / (name + ".log")).open("w") as log:
            r = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError(f"failed case {name}; see log")
        data = json.loads((out / "result.json").read_text())
        rows.append({"case": name, "tokens": tokens, "mode": mode,
                     "pattern": pattern, "serial": serial, "median_ms": data["median_ms"],
                     "max_mean_load": data["max_mean_load"],
                     **data["analytical_not_measured"]})
        with (a.output / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(name, data["median_ms"], flush=True)


if __name__ == "__main__":
    main()
