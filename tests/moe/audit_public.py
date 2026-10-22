#!/usr/bin/env python3
"""Check publication privacy, dates, log hashes and workload completeness."""
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
PAGES = [ROOT / "docs/experiments/moe-RTX5090.md", ROOT / "tests/moe/README.md"]
FOLDERS = [ROOT / "docs/experiments/results/moe-baseline", ROOT / "docs/experiments/results/moe-gpu",
           ROOT / "docs/experiments/figures/moe"]
RULES = {
    "private network": r"(?<![\w.])(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}(?![\w.])",
    "home directory": r"/(?:data/)?home/[^/\s\"'<>]+|[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"'<>]+",
    "calendar date": r"\b(?:19|20)\d{2}[-/]\d{2}[-/]\d{2}\b|<dc:date>",
    "GPU identity": r"GPU-[0-9a-fA-F]{8}-",
}
FORBIDDEN_KEYS = {"timestamp_utc", "hostname", "username", "globalPid", "globalTid", "compute_processes", "nvidia_smi"}


def main():
    files = PAGES + [p for folder in FOLDERS for p in folder.rglob("*") if p.is_file()]
    errors = []
    for p in files:
        if re.search(r"(?:19|20)\d{6}", p.name):
            errors.append((p.name, "dated filename"))
        if p.suffix in {".nsys-rep", ".sqlite", ".qdstrm"}:
            errors.append((p.name, "private raw profiler report"))
        if p.suffix not in {".md", ".json", ".csv", ".log", ".txt", ".svg", ".drawio"}:
            continue
        text = p.read_text(encoding="utf-8")
        for name, pattern in RULES.items():
            if re.search(pattern, text):
                errors.append((str(p.relative_to(ROOT)), name))
        if p.suffix == ".json":
            record = json.loads(text)
            def check_keys(value):
                if isinstance(value, dict):
                    for k, v in value.items():
                        if k in FORBIDDEN_KEYS:
                            errors.append((str(p.relative_to(ROOT)), "identity/calendar key"))
                        check_keys(v)
                elif isinstance(value, list):
                    for v in value:
                        check_keys(v)
            check_keys(record)
            if p.name.endswith(".metrics.json") and "log_sha256" in record:
                log = p.with_name(p.name.replace(".metrics.json", ".simulation.log"))
                if hashlib.sha256(log.read_bytes()).hexdigest() != record["log_sha256"]:
                    errors.append((str(p.relative_to(ROOT)), "published log hash mismatch"))
    base = json.loads((FOLDERS[0] / "simulator-summary.json").read_text())
    compact = json.loads((FOLDERS[1] / "compact-simulator/simulator-summary.json").read_text())
    if len(base + compact) != 19 or not all(r["complete"] for r in base + compact):
        errors.append(("simulation", "incomplete run"))
    traces = json.loads((FOLDERS[1] / "nsight/summary.json").read_text())
    if len(traces) != 32 or not all(len(r["stages"]) == 6 for r in traces if r["mode"] == "block"):
        errors.append(("nsight", "missing stage coverage"))
    release = FOLDERS[1] / "release-manifest.json"
    if release.exists():
        manifest = json.loads(release.read_text())
        for name, expected in manifest["artifact_sha256"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                errors.append((name, "release artifact hash mismatch"))
    if errors:
        # Print locations and categories only, never repeat a sensitive value.
        print(json.dumps(errors, indent=2))
        sys.exit(1)
    print(f"PASS: {len(files)} publication files; privacy/date checks; 19 complete simulations; Nsight stage coverage; public log hashes")


if __name__ == "__main__":
    main()
