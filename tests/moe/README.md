# Single-chip BF16 MoE evaluation

See the [report](../../docs/experiments/moe-RTX5090.md). Commands run from the repository root in Linux. All output directories must be new. Use a clean CUDA environment for GPU measurements, and a separate simulator environment for replay.

## Workloads and hardware timing

The tested GPU environment provides PyTorch 2.9.0+cu128 and Triton 3.5.0. Set `PY` to its Python executable; no login or host information is required in commands or artifacts.

```bash
PY=python
$PY -m unittest discover -s tests/moe -p 'test_*.py'
$PY -m unittest discover -s tools/tests -p test_bf16_serialization.py
$PY tests/moe/suite.py --tokens 1 16 128 512 2048 --output tests/moe/run/hardware-new
$PY tests/moe/benchmark.py --tokens 128 --mode block --schedule compact --output tests/moe/run/compact-new
$PY tests/moe/benchmark.py --tokens 16 --mode single --capture --output tests/moe/run/single-new
```

Defaults are H=2048, I=1024, E=64, top-k=8. `single` overrides E=K=1. No token dropping or quantization is used. Gate/Up/Down use BF16 inputs/outputs and FP32 accumulation. Dispatch and Combine exchange data through global memory/L2; experts are not pinned to physical SMs.

The independent reference uses FP32 GEMMs rounded at BF16 boundaries. Hardware timing excludes reference checks, compilation, capture, and CPU route construction. `--serial` launches each active expert separately per projection. `--schedule compact` removes empty CTAs while preserving useful CTA order, and adds three map loads per CTA. These are distinct controls. Byte/flop/CTA formulas are labeled analytical, not measured DRAM traffic.

## Unprivileged Nsight Systems

```bash
$PY tests/moe/nsight_profile.py --private tests/moe/run/nsight-raw-new --output tests/moe/run/nsight-public-new
$PY tests/moe/scheduling_control.py --private tests/moe/run/scheduling-raw-new --output tests/moe/run/scheduling-public-new
```

Both use ordinary-user CUDA/NVTX activity tracing, CPU sampling and context-switch tracing disabled, and `cudaProfilerApi` capture ranges. The standard suite traces three warmed-up invocations per case. The control suite runs eight paired native hardware tests and four compact-grid traces. A private `run/` directory is required for raw `.nsys-rep`, SQLite and tool logs. Only whitelisted relative timing, launch geometry and stage records are exported. No hostnames, usernames, PIDs, UUIDs or calendar metadata are published.

NVTX labels distinguish the six MoE stages. A kernel is associated with a stage by its launch API correlation and the enclosing CPU NVTX range, not by overlapping GPU timestamps. GPU activity fraction is the union of kernel intervals divided by the first-to-last kernel span; it is not SM occupancy. CUPTI memcpy events do not measure kernel-internal DRAM traffic. Profiling changes launch costs; use the native suite as the latency baseline.

NCU counter access remains unavailable in this environment, including this timing-only probe:

```bash
ncu --profile-from-start off --metrics gpu__time_duration.sum --csv   "$PY" tests/moe/benchmark.py --mode block --profile --output tests/moe/run/ncu-new
```

The probe returns `ERR_NVGPUCTRPERM`. No sudo or driver configuration changes are used.

## Full architecture with synthetic weights

Use Transformers 4.57.1 in an isolated environment. `architecture_inference.py` constructs the complete 16-layer, 64-expert OLMoE architecture with 6,919,161,856 BF16 parameters on the GPU. It uses seeded synthetic token IDs and random weights, not a trained checkpoint. It measures prefill and eight cached decode steps, checks finite logits, and exports independent prefill/decode routing counts. This is an architecture workload, not a language-quality evaluation.

```bash
MODEL_PY=python
for T in 128 512 2048; do
  "$MODEL_PY" tests/moe/architecture_inference.py --prefill "$T" --output "tests/moe/run/architecture-native-new/t$T"
  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none     --capture-range=cudaProfilerApi --capture-range-end=stop     -o "tests/moe/run/architecture-traced-new/t$T"     "$MODEL_PY" tests/moe/architecture_inference.py --prefill "$T" --profile     --output "tests/moe/run/architecture-traced-new/t$T-workload"
  nsys export --type sqlite --output "tests/moe/run/architecture-traced-new/t$T.sqlite"     "tests/moe/run/architecture-traced-new/t$T.nsys-rep"
done
$PY tests/moe/architecture_results.py --native tests/moe/run/architecture-native-new   --traced tests/moe/run/architecture-traced-new --output tests/moe/run/architecture-public-new
```

Native timing is separate from tracing. Profiling adds one forward prefill and eight decode passes after warmup/timing. Hooks are only installed for that extra capture. Router NVTX covers the gate projection; softmax/top-k remain inside MoE. Public tracing aggregates cover all kernels, with at most the first 256 kernels per phase retained as a timeline sample.

## Trained checkpoint workflow

When a trained checkpoint is available, `model.py` loads it as BF16 without quantization:

```bash
"$MODEL_PY" tests/moe/model.py --model "$CHECKPOINT" --prefill-length 512   --decode 16 --repeat 3 --output tests/moe/run/checkpoint-new
$PY tests/moe/benchmark.py --route-file tests/moe/run/checkpoint-new/routes-layer-00.json   --mode block --capture --output tests/moe/run/real-routing-new
```

It also exports `routes-decode-layer-XX.json`, and supports `--profile` with the same Nsight capture flags. Top-k scores preserve `norm_topk_prob=false`. Route replay retains expert choices/scores but uses synthetic activations and weights. The trained-checkpoint path has an interface smoke test, not a full checkpoint performance result.

## Simulator replay

```bash
export CUDA_INSTALL_PATH="$CUDA_HOME"
export PATH="$CUDA_INSTALL_PATH/bin:$PATH"
source setup_environment
make -j8
RUNTIME="$PWD/tests/moe/run/runtime-new"
mkdir -p "$RUNTIME"
cp "lib/$GPGPUSIM_CONFIG/libcudart.so" "$RUNTIME/libcudart.so.12"
patchelf --set-soname libcudart.so.12 "$RUNTIME/libcudart.so.12"
ln -s libcudart.so.12 "$RUNTIME/libcudart.so"
ln -s libcudart.so.12 "$RUNTIME/libcuda.so"
python tests/moe/replay.py tests/moe/run/single-new --runtime "$RUNTIME"   --output tests/moe/run/replay-new --timeout 1800
```

Capture runs in a clean shell; replay uses the simulator runtime. Each standalone launcher starts with cold caches, so do not add isolated cycles and call the sum an end-to-end latency. The replay requires output validation, final statistics, a zero exit status and simulator exit, and rejects synchronization warnings. It tightens BF16 validation to `rtol=0.02, atol=1e-8`. RN-even BF16 conversion functionality is supported; its hardware timing is not recalibrated.

`collect_results.py` publishes small sanitized records, omitting captured weights and runtime DSOs. Log hashes are recomputed after redaction. Historical source hashes describe acquisition-time file bytes; subsequent privacy/tracing changes can change current script hashes.

## Figures and publication checks

```bash
$PY tests/moe/plot_results.py
```

The plotter reads only published JSON/CSV data and generates PNG/SVG files under `docs/experiments/figures/moe`. SVG calendar metadata is disabled. The same directory contains editable `.drawio` sources and PNG exports for the routing, NoC path, and evaluation workflow. Labels use English and concise stage names.

Raw reports remain private because they may embed machine identities, account paths and timestamps. Public exports remove those fields, retain relative GPU timing and reproducible configuration, and are checked before commit. Anonymization changes a log hash; hashes always refer to the published bytes.
