# Experiment result collection

`collect_baseline_metrics.py` converts one FlashGPU-Sim log into a versioned
JSON record. It records the code commit, configuration and artifact hashes,
return code, resource use, functional success markers, final simulator metrics,
the last periodic progress sample, local request/reply network utilization,
transport service-limit counters, and the reported top hotspot nodes.

Example:

```bash
python scripts/experiments/collect_baseline_metrics.py \
  --workload flashattention-b32-h32-s512-d64-causal \
  --log "$RUN_DIR/flashattention.log" \
  --time-file "$RUN_DIR/flashattention.time" \
  --return-code "$(cat "$RUN_DIR/flashattention.rc")" \
  --commit "$(cat "$RUN_DIR/commit.txt")" \
  --config configs/SM120_RTX5090/gpgpusim.config \
  --artifact tutorials/triton-flash-attention/run/tracking/launchers/flash_attention_kernel_launch1_kernel.ptx \
  --artifact tutorials/triton-flash-attention/run/tracking/launchers/flash_attention_kernel_launch1_kernel.cubin \
  --output "$RUN_DIR/flashattention.metrics.json"
```

Treat a run as complete only when all of the following hold:

- the process return code is zero;
- the workload-specific validation marker is present;
- `simulator_exit` is true;
- `synchronization_warning` is false;
- `has_final_statistics` is true.

The last progress sample is retained for diagnosis but must not be substituted
for missing final statistics.
