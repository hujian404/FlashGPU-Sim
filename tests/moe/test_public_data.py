import json
from pathlib import Path
import tempfile
import unittest
from public_data import public_text, write_public


class PublicDataTest(unittest.TestCase):
    def test_paths_accounts_and_calendar_metadata(self):
        sample = "/".join(["", "data", "home", "example", "work", "FlashGPU-Sim-moe", "tests", "sample.py"])
        text = sample + " " + ".".join(["192", "0", "2", "5"]) + " " + "2000" + "-01-01T00:00:00Z"
        result = public_text(text)
        self.assertNotIn("example", result)
        self.assertIn("${REPO}", result)
        self.assertIn("<host>", result)
        self.assertNotIn("T00:", result)

    def test_nested_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "result.json"
            write_public(p, {"timestamp_utc": "private", "nested": {"username": "private", "duration_ns": 42}})
            self.assertEqual(json.loads(p.read_text()), {"nested": {"duration_ns": 42}})

    def test_model_config_integer_keys(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            write_public(p, {"id2label": {0: "LABEL_0"}})
            self.assertEqual(json.loads(p.read_text()), {"id2label": {"0": "LABEL_0"}})


if __name__ == "__main__":
    unittest.main()
