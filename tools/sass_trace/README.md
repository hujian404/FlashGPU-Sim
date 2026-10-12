# SM120 SASS trace intake (experimental)

This directory is the first, read-only intake gate for an optional SASS trace
frontend. `nvbit_inspect.py` checks Accel-Sim NVBit tracer format v6, rejects traces
from the wrong architecture, and inventories native opcodes and replay markers.
It accepts raw `.trace` and `.trace.xz` files. **It does not yet perform timing
simulation or execute a trace through FlashGPU-Sim.** The default PTX frontend
is unchanged.

```bash
python3 tools/sass_trace/nvbit_inspect.py /path/to/kernel-1.trace.xz
python3 -m unittest discover -s tools/sass_trace -p 'test_*.py'
```

## RTX 5090 feasibility check (2026-10-12)

Host: RTX 5090, driver 580.82.07, CUDA 12.8.2. Accel-Sim `dev` commit
`d930ad6` and its NVBit 1.8 tracer built successfully. A native SM120
`add_one` kernel passed its result check while instrumented; its trace reported
binary version 120, format version 6, 36 records, 4 warps, and addresses for
`LDG.E`/`STG.E`. A 128x128x128 Triton TMA GEMM also passed its native result
check. Selective tracing of its fourth launch generated a 6,520-record trace
with `UTMALDG.2D`, `UTMASTG.2D`, `HMMA.16816.F32`, barriers and native control
instructions. A tracer run on the full Python script also discovered PyTorch
library kernels, showing that capture is not restricted to custom PTX kernels.

This confirms **capture feasibility**, not Blackwell trace playback or vLLM
support. The vLLM package was not installed on this host during this check.
The TMA record contains a large variable-length payload. A timing frontend
must parse and validate it without silently dropping transfer information.

## Required next integration gates

1. Define an owned, versioned trace representation for kernel metadata,
   CTA/warp instructions, addresses, registers, TMA payload and markers.
2. Add a trace-backed shader adapter that converts records to `warp_inst_t`
   while preserving the PTX frontend as the default. Explicitly classify every
   encountered SM120 opcode and reject unsupported forms.
3. Connect trace memory addresses and async completion/barrier events to the
   existing timing backend. Re-evaluate spin loops in the simulator, rather
   than treating a fixed count from hardware as universally valid.
4. Compare instruction and memory counts, cycle timing, and deterministic
   replay against RTX 5090 hardware on simple kernels, TMA GEMM, and then
   selective vLLM layers. Keep functional validation separate: a trace-only
   timing run does not produce or verify output tensors.

Accel-Sim tracer source and format:
https://github.com/accel-sim/accel-sim-framework/tree/dev/util/tracer_nvbit

## Native `.tracez` parser gate

The Accel-Sim parser has been imported with its license under
`src/trace-parser/`. The native probe traverses each CTA and warp, decompresses
all instruction records, rejects non-SM120 traces, and counts the records:

```bash
CUDA_INSTALL_PATH=/path/to/cuda make -C tools/sass_trace
build/tools/tracez_probe /path/to/traces/kernelslist.g
```

On the RTX 5090, this matched the raw-trace counts: 36 records across 4 warps
for `add_one`, and 6,520 records across 8 warps for a Triton TMA GEMM launch.
The latter trace was converted using Accel-Sim's `post-traces-processing`.
This verifies that the simulator-side parser can read the captured Blackwell
trace format. It does not yet exercise FlashGPU-Sim's timing backend.

### SM120 asynchronous payload audit

`tracez_probe <kernelslist.g> --details` reports every TMA and `SYNCS`
record's decoded dynamic fields. The normal mode counts active TMA records and
returns exit code 5 if any active record lacks expanded memory addresses or
byte count. In the RTX 5090 Triton TMA GEMM capture, 12 TMA records include
6 predicated-off records. All 6 active records carry 8,192 addresses and a
16,384-byte transfer. All 80 `SYNCS` records carry a memory address payload.
This verifies capture completeness for this *one* kernel; it does not imply
that FlashGPU-Sim can time or synchronize these instructions yet.

The upstream FlashGPU-Sim timing path expects PTX `ptx_instruction` objects in
`shader_core_ctx::issue_warp` for dynamic mbarrier and TMA handling, while the
Accel-Sim trace frontend constructs `trace_warp_inst_t` objects. A trace
frontend must replace those PTX-specific casts with an instruction-neutral
interface and translate the tracer's expanded TMA addresses into the native
TMA engine's request representation. Copying Accel-Sim's frontend directly
would bypass or misinterpret the FlashGPU-Sim asynchronous state machine.
