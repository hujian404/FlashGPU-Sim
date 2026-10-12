#!/usr/bin/env python3

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "scripts" / "experiments" / "collect_baseline_metrics.py"
SPEC = importlib.util.spec_from_file_location("collect_baseline_metrics", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CollectBaselineMetricsTest(unittest.TestCase):
    def test_parses_time_fields_on_one_or_multiple_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.time"
            path.write_text("elapsed_seconds=158.63 max_rss_kb=4299968\n")
            parsed = MODULE.parse_time_file(path)

        self.assertEqual(parsed["elapsed_seconds"], 158.63)
        self.assertEqual(parsed["max_rss_kb"], 4299968)

    def test_parses_final_metrics_and_progress(self):
        log = """\
Progress: cycle=70000 CTA total=40 launched +0=40/40 completed +1=39/40
gpu_tot_sim_cycle = 73038
gpu_tot_sim_insn = 21826800
gpu_tot_ipc = 298.8417
gpu_tot_occupancy = 8.3333%
L2_total_cache_accesses = 829600
L2_total_cache_misses = 430084
L2_total_cache_miss_rate = 0.5184
Validation PASSED for arg[2]
GPGPU-Sim: *** exit detected ***
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.log"
            path.write_text(log)
            parsed = MODULE.parse_log(path)

        self.assertEqual(parsed["metrics"]["sim_cycles"], 73038)
        self.assertEqual(parsed["metrics"]["ipc"], 298.8417)
        self.assertEqual(parsed["last_progress"]["completed"], 39)
        self.assertTrue(parsed["success_markers"]["validation_passed"])
        self.assertTrue(parsed["has_final_statistics"])

    def test_flags_unimplemented_driver_synchronization(self):
        log = """\
Progress: cycle=730000 CTA total=8192 launched +0=8192/8192 completed +33=8187/8192
WARNING: this function has not been implemented yet CUresult cuCtxSynchronize()
Kernel execution completed successfully
GPGPU-Sim: *** exit detected ***
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.log"
            path.write_text(log)
            parsed = MODULE.parse_log(path)

        self.assertTrue(parsed["synchronization_warning"])
        self.assertFalse(parsed["has_final_statistics"])
        self.assertEqual(parsed["last_progress"]["completed"], 8187)


if __name__ == "__main__":
    unittest.main()
