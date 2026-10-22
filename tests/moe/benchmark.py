#!/usr/bin/env python3
"""Synthetic OLMoE-shaped BF16 MoE block, capture and hardware measurements."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from contextlib import nullcontext

from routing import make_routes, pack_routes
from public_data import write_public


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--intermediate", type=int, default=1024)
    p.add_argument("--experts", type=int, default=64)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--pattern", choices=["balanced", "random", "hotspot"], default="balanced")
    p.add_argument("--mode", choices=["single", "dispatch", "combine", "experts", "block"], default="block")
    p.add_argument("--serial", action="store_true", help="serialize expert launches for the concurrency control")
    p.add_argument("--schedule", choices=["padded", "compact"], default="padded")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--capture", action="store_true")
    p.add_argument("--profile", action="store_true", help="open one CUDA profiler range after warmup; use ncu --profile-from-start off")
    p.add_argument("--profile-repeat", type=int, default=3)
    p.add_argument("--route-file", type=Path, help="real routing JSON exported by model.py")
    p.add_argument("--output", type=Path, required=True, help="new directory; never overwrite a run")
    args = p.parse_args()
    if min(args.tokens, args.hidden, args.intermediate, args.repeat, args.profile_repeat) <= 0:
        p.error("dimensions and repeat must be positive")
    if args.mode == "single":
        args.experts = args.topk = 1
    if args.profile and args.capture:
        p.error("profile and capture must run separately")
    if args.serial and args.schedule == "compact":
        p.error("compact and serial are separate scheduling controls")
    routes, slots, counts = make_routes(args.tokens, args.experts, args.topk, args.pattern, args.seed)
    real_routes = None
    if args.route_file:
        if args.mode == "single":
            p.error("single mode cannot use a multi-expert route file")
        real_routes = json.loads(args.route_file.read_text())
        routes = real_routes["experts"]
        args.tokens, args.topk, args.experts = len(routes), len(routes[0]), real_routes["num_experts"]
        slots, counts = pack_routes(routes, args.experts)
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import triton
    from kernels import dispatch, combine, expert_mm, compact_mm, swiglu
    tracker = None
    if args.capture:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
        from TritonTrace import Tracker
        tracker = Tracker(args.output / "tracking", save_binaries=True,
                          capture_args=True, enabled=False)
    if os.environ.get("GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN") == "1":
        raise RuntimeError("capture/hardware measurements require a clean CUDA environment")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda"
    T, H, I, E, K = args.tokens, args.hidden, args.intermediate, args.experts, args.topk
    capacity = T  # no token dropping, accommodates worst-case skew
    rand = lambda shape: torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.02
    x = rand((T, H))
    wg, wu, wd = rand((E, I, H)), rand((E, I, H)), rand((E, H, I))
    slot = torch.tensor(slots, device=device, dtype=torch.int64)
    cnt = torch.tensor(counts, device=device, dtype=torch.int32)
    tiles = [(e, row) for e, count in enumerate(counts) for row in range(triton.cdiv(count, 16))]
    tile_maps = {}
    if args.schedule == "compact":
        # Preserve useful CTA order: expert, column tile, row tile. The only
        # removed CTAs are empty ones; compact launches add three map reads.
        for n in {I, H}:
            mapping = [(e, row, col) for e, count in enumerate(counts)
                       for col in range(triton.cdiv(n, 32))
                       for row in range(triton.cdiv(count, 16))]
            tile_maps[n] = tuple(torch.tensor([x[j] for x in mapping], device=device, dtype=torch.int32) for j in range(3))
    scores = torch.full((T, K), 1 / K, device=device, dtype=torch.float32)
    if real_routes:
        scores = torch.tensor(real_routes["scores"], device=device, dtype=torch.float32)
        if scores.shape != (T, K) or not torch.isfinite(scores).all() or (scores < 0).any():
            raise ValueError("invalid route scores")
    packed = torch.zeros((E, capacity, H), device=device, dtype=torch.bfloat16)
    g = torch.zeros((E, capacity, I), device=device, dtype=torch.bfloat16)
    u, act = torch.zeros_like(g), torch.zeros_like(g)
    y, out = torch.zeros_like(packed), torch.zeros_like(x)

    def do_dispatch():
        dispatch[(T * K, triton.cdiv(H, 256))](x, slot, packed, H=H, TOPK=K, BLOCK=256)

    def mm(a, w, c, n, k):
        if args.schedule == "compact":
            compact_mm[(len(tile_maps[n][0]),)](
                a, w, c, cnt, *tile_maps[n], CAP=capacity, N=n, K=k)
        elif args.serial:
            for e, count in enumerate(counts):
                if count:
                    expert_mm[(triton.cdiv(count, 16), triton.cdiv(n, 32), 1)](
                        a[e:e+1], w[e:e+1], c[e:e+1], cnt[e:e+1], CAP=capacity, N=n, K=k)
        else:
            expert_mm[(triton.cdiv(max(counts), 16), triton.cdiv(n, 32), E)](
                a, w, c, cnt, CAP=capacity, N=n, K=k)

    def do_experts():
        with torch.cuda.nvtx.range("Gate") if args.profile else nullcontext():
            mm(packed, wg, g, I, H)
        with torch.cuda.nvtx.range("Up") if args.profile else nullcontext():
            mm(packed, wu, u, I, H)
        with torch.cuda.nvtx.range("SwiGLU") if args.profile else nullcontext():
            swiglu[(triton.cdiv(max(counts) * I, 256), E)](g, u, act, cnt, CAP=capacity, I=I, BLOCK=256)
        with torch.cuda.nvtx.range("Down") if args.profile else nullcontext():
            mm(act, wd, y, H, I)

    def do_combine():
        combine[(T, triton.cdiv(H, 256))](y, slot, scores, out, H=H, TOPK=K, BLOCK=256)

    def block():
        with torch.cuda.nvtx.range("Dispatch") if args.profile else nullcontext():
            do_dispatch()
        do_experts()
        with torch.cuda.nvtx.range("Combine") if args.profile else nullcontext():
            do_combine()

    block()  # compile and prepare valid inputs even for isolated stages
    torch.cuda.synchronize()
    # Independent FP32 references, rounded at each BF16 intermediate boundary.
    expected_packed = torch.zeros_like(packed)
    expected_packed.view(-1, H)[slot] = x.repeat_interleave(K, dim=0)
    expected_y = torch.zeros_like(y)
    for e, count in enumerate(counts):
        if count:
            a = expected_packed[e, :count].float()
            rg = (a @ wg[e].float().T).bfloat16().float()
            ru = (a @ wu[e].float().T).bfloat16().float()
            ra = (torch.nn.functional.silu(rg) * ru).bfloat16().float()
            expected_y[e, :count] = (ra @ wd[e].float().T).bfloat16()
    expected_out = (expected_y.view(-1, H)[slot].view(T, K, H).float() * scores[:, :, None]).sum(1).bfloat16()
    torch.testing.assert_close(packed, expected_packed, rtol=0, atol=0)
    torch.testing.assert_close(y, expected_y, rtol=0.03, atol=2e-5)
    torch.testing.assert_close(out, expected_out, rtol=0.03, atol=2e-5)
    fn = {"single": do_experts, "dispatch": do_dispatch, "combine": do_combine,
          "experts": do_experts, "block": block}[args.mode]
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.repeat):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(args.profile_repeat):
            with torch.cuda.nvtx.range("Measured workload"):
                fn()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    if args.capture:
        # The tracker discovers outputs by mutation; force a fresh destination
        # after warmup without altering the input of an isolated stage.
        if args.mode in ("dispatch", "block"):
            packed.zero_()
        if args.mode in ("single", "experts", "block"):
            for tensor in (g, u, act, y):
                tensor.zero_()
        if args.mode in ("combine", "block"):
            out.zero_()
        torch.cuda.synchronize()
        tracker.enable()
        fn()
        torch.cuda.synchronize()
        tracker.disable()
        tracker.save_summary()
    def command(argv):
        r = subprocess.run(argv, text=True, capture_output=True)
        return r.stdout.strip() if r.returncode == 0 else r.stderr.strip()
    config = Path(__file__).resolve().parents[2] / "configs/SM120_RTX5090/gpgpusim.config"
    record = {
        "schema_version": 1, "kind": "synthetic_moe_block", "validation": "passed",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ["benchmark.py", "kernels.py", "routing.py"]},
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "torch": torch.__version__, "triton": triton.__version__,
        "gpu": command(["nvidia-smi", "--query-gpu=name,driver_version,memory.used,utilization.gpu", "--format=csv,noheader"]),
        "expert_counts": counts, "max_mean_load": max(counts) / (T*K/E),
        "routing_source": str(args.route_file) if real_routes else "synthetic; equal top-k weights; not model router logits",
        "median_ms": statistics.median(samples), "samples_ms": samples,
        "analytical_not_measured": {
            "resident_weight_bytes": E * 3 * H * I * 2,
            "active_unique_weight_bytes": sum(c > 0 for c in counts) * 3 * H * I * 2,
            "dispatch_payload_bytes": T*K*H*2,
            "combine_input_payload_bytes": T*K*H*2,
            "expert_gemm_flops": T*K*6*H*I,
            "gate_useful_ctas": len(tiles) * triton.cdiv(I, 32),
            "gate_launched_ctas": (len(tiles) if args.serial or args.schedule == "compact" else triton.cdiv(max(counts), 16) * E) * triton.cdiv(I, 32),
        },
    }
    write_public(args.output / "result.json", record)
    print(f"Validation PASSED; {args.mode} median={record['median_ms']:.6f} ms")


if __name__ == "__main__":
    main()
