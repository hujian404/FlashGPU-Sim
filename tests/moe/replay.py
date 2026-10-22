#!/usr/bin/env python3
"""Build and replay each captured launch in isolation, preserving provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from public_data import write_public


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capture", type=Path, help="benchmark output directory containing tracking/")
    p.add_argument("--runtime", type=Path, required=True, help="directory containing versioned simulator libcudart.so.12 and libcuda.so")
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "configs/SM120_RTX5090")
    p.add_argument("--timeout", type=int, default=1800, help="seconds per kernel")
    p.add_argument("--kernel", help="replay one launcher stem only")
    p.add_argument("--output", type=Path, required=True, help="new replay directory, preserves original capture")
    a = p.parse_args()
    a.runtime, a.config = a.runtime.resolve(), a.config.resolve()
    for name in ["libcudart.so.12", "libcuda.so"]:
        if not (a.runtime / name).exists():
            p.error(f"missing simulator library {a.runtime / name}")
    import shutil
    a.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(a.capture / "tracking", a.output / "tracking",
                    ignore=shutil.ignore_patterns("*.simulation.log", "*.metrics.json", "core*"))
    launchers = (a.output / "tracking/launchers").resolve()
    files = sorted(launchers.glob("*_harness.cu"))
    if a.kernel:
        files = [x for x in files if x.name == a.kernel + "_harness.cu"]
    if not files:
        p.error("no matching capture harnesses")
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "1"))
    root = Path(__file__).resolve().parents[2]
    nvcc = shutil.which("nvcc")
    if not nvcc:
        p.error("nvcc must be on PATH")
    env.setdefault("CUDA_INSTALL_PATH", str(Path(nvcc).resolve().parents[1]))
    env["GPGPUSIM_ROOT"] = str(root)
    env["PATH"] = str(root / "bin") + ":" + env["PATH"]
    env["LD_LIBRARY_PATH"] = str(a.runtime) + ":" + env.get("LD_LIBRARY_PATH", "")
    for path in a.config.iterdir():
        if path.is_file():
            shutil.copy2(path, launchers / path.name)
    records = []
    for harness in files:
        stem = harness.name.removesuffix("_harness.cu")
        result_path = launchers / (stem + ".metrics.json")
        if result_path.exists():
            raise FileExistsError(f"replay already recorded: {result_path}")
        # The generic harness's 1e-2 absolute tolerance would accept all-zero
        # outputs for our small-amplitude data. Tighten BF16 for this experiment.
        source = harness.read_text()
        begin = source.index("} else if (is_bf16) {")
        end = source.index("} else if (is_fp32) {", begin)
        section = source[begin:end].replace("float rel_tolerance = 1e-3f;", "float rel_tolerance = 2e-2f;").replace("float abs_tolerance = 1e-2f;", "float abs_tolerance = 1e-8f;")
        harness.write_text(source[:begin] + section + source[end:])
        subprocess.run(["make", "-f", stem + "_Makefile", stem + "_kernel.fatbin"], cwd=launchers, check=True)
        subprocess.run(["nvcc", "-std=c++17", "-arch=sm_120a", "-cudart", "none", str(harness),
                        "-L" + str(a.runtime), "-lcudart", "-lcuda",
                        "-Xlinker=-rpath", "-Xlinker=" + str(a.runtime), "-o", stem], cwd=launchers, check=True)
        deps = subprocess.run(["ldd", "./" + stem], cwd=launchers, env=env, text=True, capture_output=True, check=True).stdout
        for name in ["libcudart.so.12", "libcuda.so"]:
            # Both runtime and driver APIs are exported by the same simulator
            # DSO; the linker can coalesce the two SONAME references.
            if name == "libcuda.so" and not re.search(r"libcuda\.so\s+=>", deps):
                continue
            if not re.search(re.escape(name) + r"\s+=>\s+" + re.escape(str(a.runtime)) + "/", deps):
                raise RuntimeError(f"unexpected library resolution: {deps}")
        start = time.monotonic()
        timed_out = False
        log = launchers / (stem + ".simulation.log")
        with log.open("x") as stream:
            try:
                result = subprocess.run(["./" + stem], cwd=launchers, env=env,
                                        stdout=stream, stderr=subprocess.STDOUT, timeout=a.timeout)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code, timed_out = None, True
        text = log.read_text(errors="replace")
        sys.path.insert(0, str(root / "scripts/experiments"))
        from collect_baseline_metrics import parse_log
        parsed = parse_log(log)
        metrics = {}
        for key, value in re.findall(r"^([A-Za-z][A-Za-z0-9_]+)\s*=\s*([0-9.eE+-]+)\s*$", text, re.M):
            try:
                metrics[key] = float(value) if any(c in value.lower() for c in ".e") else int(value)
            except ValueError:
                pass
        success = code == 0 and "Validation PASSED" in text and "Validation FAILED" not in text and "*** exit detected ***" in text and "gpu_tot_sim_cycle" in metrics and not parsed["synchronization_warning"]
        record = {"kernel": stem, "complete": success, "returncode": code,
                  "timeout": timed_out, "wall_seconds": time.monotonic() - start,
                  "metrics": metrics, "ldd": deps,
                  "noc_metrics": parsed["noc_metrics"], "noc_hotspots": parsed["noc_hotspots"],
                  "runtime_hashes": {n: digest(a.runtime / n) for n in ["libcudart.so.12", "libcuda.so"]},
                  "config_hashes": {x.name: digest(x) for x in a.config.iterdir() if x.is_file()},
                  "capture_hashes": {x.name: digest(x) for x in launchers.glob(stem + "_*") if x.is_file()},
                  "scope": "isolated launch; fresh simulator caches; never an end-to-end pipeline latency"}
        record["bf16_validation"] = {"rtol": 0.02, "atol": 1e-8}
        write_public(result_path, record)
        records.append(record)
        print(stem, "PASS" if success else "INCOMPLETE", metrics.get("gpu_tot_sim_cycle"), flush=True)
        if not success:
            raise RuntimeError(f"replay failed; inspect {log}")


if __name__ == "__main__":
    main()
