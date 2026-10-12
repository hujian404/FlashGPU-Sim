import lzma
import tempfile
import unittest
from pathlib import Path

from nvbit_inspect import TraceError, inspect, parse_record


HEADER = """-kernel name = add_one
-binary version = 120
-accelsim tracer version = 6
-enable lineinfo = 0
#traces format = native
"""
RECORD = "0 0 0 0 0 0 0 0 0 0 0 0010 ffffffff 1 R1 LDG.E 1 R2 4 1 0x1000 4 0 0 NoVal\n"


class InspectorTests(unittest.TestCase):
    def test_reads_compressed_sm120_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kernel.trace.xz"
            with lzma.open(path, "wt") as stream:
                stream.write(HEADER + RECORD)
            result = inspect(path)
            self.assertEqual(result["records"], 1)
            self.assertEqual(result["warps"], 1)
            self.assertEqual(result["opcodes"], {"LDG.E": 1})

    def test_rejects_wrong_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kernel.trace"
            path.write_text(HEADER.replace("version = 120", "version = 90") + RECORD)
            with self.assertRaisesRegex(TraceError, "expected SM120"):
                inspect(path)

    def test_rejects_short_register_list(self):
        with self.assertRaises(TraceError):
            parse_record("0 0 0 0 0 0 0 0 0 0 0 0000 ffffffff 2 R1")


if __name__ == "__main__":
    unittest.main()
