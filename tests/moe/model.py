#!/usr/bin/env python3
"""Real OLMoE BF16 prefill/decode timing and separate router inspection pass."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time
from contextlib import nullcontext
from public_data import write_public


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    p.add_argument("--revision", default="main")
    p.add_argument("--prompt", default="Explain how a mixture of experts language model routes tokens to experts.")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--decode", type=int, default=16, help="fixed number of cached forward steps")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--profile", action="store_true", help="CUDA profiler range around measured prefill/decode")
    p.add_argument("--prefill-length", type=int, help="fixed token length from a repeated prompt")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if min(a.batch, a.decode, a.repeat) < 1:
        p.error("batch, decode and repeat must be positive")
    a.output.mkdir(parents=True, exist_ok=False)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, revision=a.revision, torch_dtype=torch.bfloat16,
        attn_implementation="eager").eval().to("cuda")
    if any(x.dtype != torch.bfloat16 for x in model.parameters() if x.is_floating_point()):
        raise RuntimeError("model parameters must remain BF16")
    inputs = tokenizer([a.prompt] * a.batch, return_tensors="pt").to("cuda")
    if a.prefill_length:
        if not 1 <= a.prefill_length <= model.config.max_position_embeddings - a.decode:
            p.error("prefill plus decode must fit the model context")
        base = inputs.input_ids[0]
        ids = base.repeat((a.prefill_length + len(base) - 1) // len(base))[:a.prefill_length]
        inputs = {"input_ids": ids[None].repeat(a.batch, 1), "attention_mask": torch.ones((a.batch, a.prefill_length), device="cuda", dtype=torch.long)}
    prefill_ms, decode_ms, generated = [], [], []
    def measured(fn, label):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.cuda.nvtx.range(label) if a.profile else nullcontext():
            result = fn()
        torch.cuda.synchronize()
        return result, (time.perf_counter() - start) * 1000
    with torch.inference_mode():
        # Warm up the full path, including KV cache and decode shapes.
        for iteration in range(a.repeat + 1):
            if a.profile and iteration == 1:
                torch.cuda.cudart().cudaProfilerStart()
            result, pre = measured(lambda: model(**inputs, use_cache=True, output_router_logits=False), "Prefill")
            cache = result.past_key_values
            token = result.logits[:, -1:].argmax(-1)
            steps, ids = [], []
            for _ in range(a.decode):
                result, ms = measured(lambda: model(input_ids=token, past_key_values=cache,
                                                     use_cache=True, output_router_logits=False), "Decode")
                cache = result.past_key_values
                token = result.logits[:, -1:].argmax(-1)
                steps.append(ms)
                ids.append(token.tolist())
            if iteration:
                prefill_ms.append(pre)
                decode_ms.append(steps)
                generated = ids
        if a.profile:
            torch.cuda.cudart().cudaProfilerStop()
        # Router instrumentation is excluded from timed forward passes.
        result = model(**inputs, use_cache=True, output_router_logits=True)
        prefill_result = result
        layers = []
        for layer, logits in enumerate(result.router_logits):
            probs = torch.softmax(logits.float(), dim=-1)
            scores, experts = torch.topk(probs, model.config.num_experts_per_tok, dim=-1)
            if model.config.norm_topk_prob:
                scores = scores / scores.sum(-1, keepdim=True)
            counts = torch.bincount(experts.flatten(), minlength=model.config.num_experts)
            route = {"layer": layer, "num_experts": model.config.num_experts,
                     "experts": experts.cpu().tolist(), "scores": scores.cpu().tolist(),
                     "counts": counts.cpu().tolist()}
            (a.output / f"routes-layer-{layer:02d}.json").write_text(json.dumps(route) + "\n")
            layers.append({"layer": layer, "counts": route["counts"],
                           "max_mean_load": (counts.max() / counts.float().mean()).item()})
        decode_result = model(input_ids=prefill_result.logits[:, -1:].argmax(-1),
                              past_key_values=prefill_result.past_key_values, use_cache=True,
                              output_router_logits=True)
        for layer, logits in enumerate(decode_result.router_logits):
            probs = torch.softmax(logits.float(), dim=-1)
            scores, experts = torch.topk(probs, model.config.num_experts_per_tok, dim=-1)
            if model.config.norm_topk_prob:
                scores = scores / scores.sum(-1, keepdim=True)
            write_public(a.output / f"routes-decode-layer-{layer:02d}.json",
                         {"num_experts": model.config.num_experts, "experts": experts.cpu().tolist(), "scores": scores.cpu().tolist()})
    total_decode_ms = [sum(x) for x in decode_ms]
    record = {
        "kind": "real_model_bf16", "model": a.model,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        "torch": torch.__version__, "transformers": transformers.__version__,
        "config": model.config.to_dict(), "prefill_ms": prefill_ms,
        "decode_step_ms": decode_ms,
        "prefill_tokens_per_second": inputs["input_ids"].numel() / (statistics.median(prefill_ms) / 1000),
        "decode_tokens_per_second": a.batch * a.decode / (statistics.median(total_decode_ms) / 1000),
        "timing_scope": "synchronized forward wall time; excludes tokenization and argmax; fixed decode steps ignoring EOS",
        "routes_scope": "prefill and one cached decode token; separate untimed passes",
        "generated_token_ids_after_decode": generated, "routing": layers,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True, capture_output=True).stdout,
    }
    write_public(a.output / "result.json", record)
    print(json.dumps({k: record[k] for k in ["resolved_revision", "prefill_tokens_per_second", "decode_tokens_per_second"]}))


if __name__ == "__main__":
    main()
