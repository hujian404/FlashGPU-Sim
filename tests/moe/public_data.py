"""Privacy boundary for publishable results; raw profiler reports stay in run/."""
import json
import re
from pathlib import Path

PRIVATE_KEYS = {"timestamp", "timestamp_utc", "hostname", "host", "username",
                "user", "uuid", "compute_processes", "git_status", "nvidia_smi"}


def public_text(value):
    value = re.sub(r"(?:[A-Za-z]:[\\/]|/)[^\s\"'<>]*?FlashGPU-Sim-moe(?=[/\\\s\"']|$)", "${REPO}", value)
    value = re.sub(r"/(?:data/)?home/[^/\s\"'<>]+", "${HOME}", value)
    value = re.sub(r"[A-Za-z]:[\\/]Users[\\/][^\\/\s\"'<>]+", "${HOME}", value)
    value = re.sub(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])", "<host>", value)
    value = re.sub(r"GPU-[0-9a-fA-F-]+", "<gpu>", value)
    value = re.sub(r"[\w.+-]+@[\w.-]+(?:\.[\w.-]+)+", "<account>", value)
    value = re.sub(r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z| UTC)?)?\b", "", value)
    value = re.sub(r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\d{4}", "", value)
    value = re.sub(r"moe-\d{8}-v\d+", "moe-baseline", value)
    value = re.sub(r"(?<=[/_-])(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])(?=[/_\-.]|$)", "", value)
    return value


def public_value(value):
    if isinstance(value, dict):
        return {public_text(str(k)): public_value(v) for k, v in value.items()
                if str(k).lower() not in PRIVATE_KEYS}
    if isinstance(value, list):
        return [public_value(v) for v in value]
    return public_text(value) if isinstance(value, str) else value


def write_public(path, value):
    Path(path).write_bytes((json.dumps(public_value(value), indent=2) + "\n").encode("utf-8"))


def sanitize_tree(root):
    """Only use on an explicitly selected publication directory."""
    root = Path(root)
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".csv", ".md", ".log", ".txt"}:
            continue
        if path.suffix == ".json":
            write_public(path, json.loads(path.read_text(encoding="utf-8")))
        else:
            path.write_bytes(public_text(path.read_text(encoding="utf-8")).encode("utf-8"))
