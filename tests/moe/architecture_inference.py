#!/usr/bin/env python3
"""Full OLMoE architecture on GPU with synthetic BF16 weights, not a checkpoint."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import statistics
import time
from public_data import write_public


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefill", type=int, default=128)
    p.add_argument("--decode", type=int, default=8)
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if min(a.prefill, a.decode, a.repeat) < 1 or a.prefill + a.decode > 4096:
        p.error("positive lengths required within the model context")
    a.output.mkdir(parents=True, exist_ok=False)
    import torch
    import transformers
    from transformers import OlmoeConfig, OlmoeForCausalLM
    torch.manual_seed(0)
    config = OlmoeConfig(hidden_size=2048, intermediate_size=1024,
                         num_hidden_layers=16, num_attention_heads=16,
                         num_key_value_heads=16, num_experts=64,
                         num_experts_per_tok=8, vocab_size=50304,
                         max_position_embeddings=4096, norm_topk_prob=False,
                         attention_bias=False, rope_theta=10000.0,
                         tie_word_embeddings=False, pad_token_id=1,
                         eos_token_id=50279, hidden_act="silu")
    config._attn_implementation = "eager"
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("cuda"):
            model = OlmoeForCausalLM(config).eval()
    finally:
        torch.set_default_dtype(old_dtype)
    if any(p.dtype != torch.bfloat16 for p in model.parameters()):
        raise RuntimeError("all parameters must be BF16")
    ids = torch.randint(3, config.vocab_size, (1, a.prefill), device="cuda")
    prefill, decode = [], []

    def forward_timed(fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        return result, (time.perf_counter() - start) * 1000

    with torch.inference_mode():
        for rep in range(a.repeat + 1):
            result, pre = forward_timed(lambda: model(input_ids=ids, use_cache=True))
            if rep == 0 and not torch.isfinite(result.logits).all():
                raise RuntimeError("non-finite prefill output")
            cache, token = result.past_key_values, result.logits[:, -1:].argmax(-1)
            steps = []
            for _ in range(a.decode):
                result, ms = forward_timed(lambda: model(input_ids=token, past_key_values=cache, use_cache=True))
                cache, token = result.past_key_values, result.logits[:, -1:].argmax(-1)
                steps.append(ms)
            if not torch.isfinite(result.logits).all():
                raise RuntimeError("non-finite decode output")
            if rep:
                prefill.append(pre)
                decode.append(steps)
        if a.profile:
            hooks = []
            def instrument(module, label):
                def begin(m, x):
                    torch.cuda.nvtx.range_push(label)
                hooks.append(module.register_forward_pre_hook(begin))
                # Hook return values replace inputs/outputs; always return None.
                def finish(m, x, out):
                    torch.cuda.nvtx.range_pop()
                hooks.append(module.register_forward_hook(finish))
            for layer in model.model.layers:
                instrument(layer.self_attn, "Attention")
                instrument(layer.mlp, "MoE")
                instrument(layer.mlp.gate, "Router")
            torch.cuda.cudart().cudaProfilerStart()
            with torch.cuda.nvtx.range("Prefill"):
                result = model(input_ids=ids, use_cache=True)
                cache, token = result.past_key_values, result.logits[:, -1:].argmax(-1)
            for _ in range(a.decode):
                with torch.cuda.nvtx.range("Decode"):
                    result = model(input_ids=token, past_key_values=cache, use_cache=True)
                    cache, token = result.past_key_values, result.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            for hook in hooks:
                hook.remove()
        # Export routing in separate, untimed passes.
        routing = {}
        result = model(input_ids=ids, use_cache=True, output_router_logits=True)
        decoded = model(input_ids=result.logits[:, -1:].argmax(-1), past_key_values=result.past_key_values,
                        use_cache=True, output_router_logits=True)
        for phase, output in [("prefill", result), ("decode", decoded)]:
            routing[phase] = [torch.bincount(torch.topk(torch.softmax(logits.float(), -1), 8, -1).indices.flatten(),
                                           minlength=64).cpu().tolist() for logits in output.router_logits]
    record = {"kind": "full_architecture_synthetic_weights_bf16", "weights": "random initialization; not trained checkpoint",
              "input": "seeded synthetic token IDs", "seed": 0, "batch": 1,
              "prefill_tokens": a.prefill, "decode_steps": a.decode,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
              "layers": 16, "experts_per_layer": 64, "topk": 8,
              "torch": torch.__version__, "transformers": transformers.__version__,
              "attention": config._attn_implementation, "prefill_ms": prefill, "decode_step_ms": decode,
              "prefill_tokens_per_second": a.prefill * 1000 / statistics.median(prefill),
              "decode_tokens_per_second": a.decode * 1000 / statistics.median([sum(x) for x in decode]),
              "validation": "finite prefill and decode logits; no trained-model accuracy claim",
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "routing_counts": routing,
              "timing_scope": "synchronized forward wall time, excludes initialization and argmax; no profiling hooks in timed passes",
              "profile_scope": "one additional prefill plus cached decode after unprofiled timing"}
    write_public(a.output / "result.json", record)
    print(json.dumps({k: record[k] for k in ["kind", "parameter_count", "prefill_tokens_per_second", "decode_tokens_per_second"]}), flush=True)


if __name__ == "__main__":
    main()
