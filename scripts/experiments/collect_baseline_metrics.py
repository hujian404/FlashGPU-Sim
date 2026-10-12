#!/usr/bin/env python3
"""Extract reproducible FlashGPU-Sim metrics from one completed run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


SCALAR_PATTERNS = {
    "sim_cycles": r"^gpu_tot_sim_cycle\s*=\s*(\d+)",
    "sim_instructions": r"^gpu_tot_sim_insn\s*=\s*(\d+)",
    "ipc": r"^gpu_tot_ipc\s*=\s*([0-9.eE+-]+)",
    "issued_ctas": r"^gpu_tot_issued_cta\s*=\s*(\d+)",
    "occupancy_percent": r"^gpu_tot_occupancy\s*=\s*([0-9.eE+-]+)%",
    "tma_transactions": r"^gpgpu_n_tma_tx_completed\s*=\s*(\d+)",
    "tma_read_transactions": r"^gpgpu_n_tma_read_tx_completed\s*=\s*(\d+)",
    "tma_write_transactions": r"^gpgpu_n_tma_write_tx_completed\s*=\s*(\d+)",
    "l2_accesses": r"^L2_total_cache_accesses\s*=\s*(\d+)",
    "l2_misses": r"^L2_total_cache_misses\s*=\s*(\d+)",
    "l2_miss_rate": r"^L2_total_cache_miss_rate\s*=\s*([0-9.eE+-]+)",
}

PROGRESS_RE = re.compile(
    r"^Progress: cycle=(?P<cycle>\d+).*completed(?:\s+\+\d+)?="
    r"(?P<completed>\d+)/(?P<total>\d+)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def last_match(pattern: str, text: str) -> str | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return matches[-1] if matches else None


def parse_number(value: str | None) -> int | float | None:
    if value is None:
        return None
    return float(value) if any(c in value.lower() for c in (".", "e")) else int(value)


def parse_time_file(path: Path | None) -> dict[str, int | float | None]:
    result: dict[str, int | float | None] = {
        "elapsed_seconds": None,
        "max_rss_kb": None,
    }
    if path is None:
        return result
    text = path.read_text(errors="replace")
    for key, value in re.findall(
        r"\b(elapsed_seconds|max_rss_kb)=([0-9.eE+-]+)", text
    ):
        if key in result:
            result[key] = parse_number(value)
    return result


def parse_log(path: Path) -> dict[str, object]:
    text = path.read_text(errors="replace")
    metrics = {
        name: parse_number(last_match(pattern, text))
        for name, pattern in SCALAR_PATTERNS.items()
    }

    progress = list(PROGRESS_RE.finditer(text))
    last_progress = None
    if progress:
        match = progress[-1]
        last_progress = {
            key: int(match.group(key)) for key in ("cycle", "completed", "total")
        }

    success_markers = {
        "simulator_exit": "GPGPU-Sim: *** exit detected ***" in text,
        "kernel_completed": "Kernel execution completed successfully" in text,
        "vectoradd_passed": "Test PASSED" in text,
        "validation_passed": "Validation PASSED" in text,
    }
    synchronization_warning = (
        "not been implemented yet CUresult cuCtxSynchronize" in text
    )

    return {
        "metrics": metrics,
        "last_progress": last_progress,
        "success_markers": success_markers,
        "synchronization_warning": synchronization_warning,
        "has_final_statistics": metrics["sim_cycles"] is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--time-file", type=Path)
    parser.add_argument("--return-code", required=True, type=int)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact", action="append", default=[], type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    parsed = parse_log(args.log)
    result = {
        "schema_version": 1,
        "workload": args.workload,
        "commit": args.commit,
        "return_code": args.return_code,
        "log": str(args.log),
        "log_sha256": sha256(args.log),
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "artifacts": [
            {"path": str(path), "sha256": sha256(path)} for path in args.artifact
        ],
        "resources": parse_time_file(args.time_file),
        **parsed,
    }

    output = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
