#!/usr/bin/env python3
"""Archive small, auditable results without captured weights or runtime DSOs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from public_data import sanitize_tree


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hardware", type=Path, required=True)
    p.add_argument("--sim", type=Path, action="append", default=[])
    p.add_argument("--diagnostic", type=Path, action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-commit", default="f44abbe461eff6467ccdc40bcf62b8e692d3f2e4")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "scripts/experiments"))
    from collect_baseline_metrics import parse_log
    hardware = a.output / "hardware"
    hardware.mkdir()
    shutil.copy2(a.hardware / "summary.csv", a.output / "hardware-summary.csv")
    for src in sorted(a.hardware.glob("*/result.json")):
        shutil.copy2(src, hardware / (src.parent.name + ".json"))
    rows = []
    for run in a.sim:
        target = a.output / "simulator" / run.name
        target.mkdir(parents=True)
        for src in sorted((run / "tracking/launchers").glob("*.metrics.json")):
            record = json.loads(src.read_text())
            log = src.with_name(src.name.replace(".metrics.json", ".simulation.log"))
            parsed = parse_log(log)
            record["noc_metrics"] = parsed["noc_metrics"]
            record["noc_hotspots"] = parsed["noc_hotspots"]
            record["log_sha256"] = hashlib.sha256(log.read_bytes()).hexdigest()
            (target / src.name).write_text(json.dumps(record, indent=2) + "\n")
            shutil.copy2(log, target / log.name)
            rows.append({"run": run.name, "kernel": record["kernel"],
                         "complete": record["complete"],
                         "sim_cycles": record["metrics"].get("gpu_tot_sim_cycle"),
                         "noc_metrics": record["noc_metrics"],
                         "noc_hotspots": record["noc_hotspots"]})
    (a.output / "simulator-summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    diagnostics = a.output / "diagnostics"
    diagnostics.mkdir()
    for src in a.diagnostic:
        shutil.copy2(src, diagnostics / src.name)
    files = [root / "setup_environment", root / "tools/TritonTrace/session.py",
             *sorted((root / "tests/moe").glob("*.py")), root / "tests/moe/bf16_convert.cu"]
    for name in ["instructions.cc", "ptx.l", "ptx.y", "ptx_ir.cc", "ptx_parser.cc"]:
        files.append(root / "src/cuda-sim" / name)
    files.append(root / "src/gpgpu-sim/flash/ptx_sched/ptx_scheduler.cc")
    manifest = {"base_commit": a.base_commit,
                "source_sha256": {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
                "hardware_cases": len(list(hardware.glob("*.json"))),
                "simulation_launches": len(rows),
                "simulation_complete": sum(r["complete"] for r in rows)}
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    sanitize_tree(a.output)
    # Hash the published, sanitized logs, not the private originals.
    for metadata in (a.output / "simulator").rglob("*.metrics.json"):
        record = json.loads(metadata.read_text())
        log = metadata.with_name(metadata.name.replace(".metrics.json", ".simulation.log"))
        record["log_sha256"] = hashlib.sha256(log.read_bytes()).hexdigest()
        metadata.write_bytes((json.dumps(record, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
