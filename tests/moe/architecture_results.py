#!/usr/bin/env python3
"""Publish architecture timings and phase aggregates without profiler identities."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from nsight_profile import extract
from public_data import write_public


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--traced", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    summary = []
    for t in [128, 512, 2048]:
        native = json.loads((a.native / f"t{t}/result.json").read_text())
        write_public(a.output / f"t{t}.json", native)
        trace = extract(a.traced / f"t{t}.sqlite", 1)
        phases = {"Prefill": defaultdict(float), "Decode": defaultdict(float)}
        unknown = 0
        for kernel in trace["kernels"]:
            phase = kernel["phase"]
            if phase in phases:
                stage = kernel["stage"]
                if stage in {"Prefill", "Decode"}:
                    stage = "Other"
                phases[phase][stage] += kernel["duration_ns"] / 1e6
            else:
                unknown += 1
        trace["phase_gpu_ms"] = phases
        trace["unassigned_kernel_count"] = unknown
        trace["timeline_sample"] = [k for phase in ["Prefill", "Decode"] for k in [k for k in trace["kernels"] if k["phase"] == phase][:256]]
        del trace["kernels"]
        trace["scope"] = "one synthetic-weight full-architecture prefill and eight cached decode forwards; GPU aggregates cover the full trace"
        trace["timeline_sample_scope"] = "first 256 kernels per phase only"
        write_public(a.output / f"t{t}-nsight.json", trace)
        summary.append({"tokens": t, "prefill_ms": native["prefill_ms"],
                        "prefill_tokens_per_second": native["prefill_tokens_per_second"],
                        "decode_tokens_per_second": native["decode_tokens_per_second"],
                        "kernel_count": trace["kernel_count"], "phase_gpu_ms": phases,
                        "unassigned_kernel_count": unknown,
                        "weights": "synthetic random BF16; not a trained checkpoint"})
    write_public(a.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
