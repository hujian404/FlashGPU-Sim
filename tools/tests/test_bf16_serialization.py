"""CPU-only regression: BF16 input/output captures retain exact payload bits."""
import tempfile
import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from TritonTrace.session import TrackingSession, KernelLaunchInfo


class BF16SerializationTest(unittest.TestCase):
    def test_roundtrip_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = TrackingSession(Path(tmp), save_binaries=True, capture_args=True,
                                      mode="online", target_name=None)
            x = torch.tensor([[1.0, -2.0], [0.5, 3.25]], dtype=torch.bfloat16).T
            args, snapshots = session.capture_arguments((x,), "bf16", 1, snapshot_tensors=True)
            self.assertEqual(args[0].dtype, "bfloat16")
            payload = Path(args[0].data_file).read_bytes()
            restored = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).reshape(x.shape)
            self.assertTrue(torch.equal(restored, x))
            launch = KernelLaunchInfo(timestamp="", kernel_name="bf16", kernel_hash="test",
                                      grid=(1, 1, 1), block=(32, 1, 1), shared_memory=0,
                                      num_warps=1, num_ctas=1, args_info=args, launch_id=1)
            x.add_(1)
            session.capture_outputs(launch, (x,), snapshots)
            output = torch.frombuffer(bytearray(Path(args[0].output_file).read_bytes()), dtype=torch.bfloat16).reshape(x.shape)
            self.assertTrue(torch.equal(output, x))


if __name__ == "__main__":
    unittest.main()
