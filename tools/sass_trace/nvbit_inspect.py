#!/usr/bin/env python3
"""Inspect raw Accel-Sim NVBit traces before importing them into FlashGPU-Sim.

This deliberately does not claim to simulate cycles. It checks the capture
contract and reports the native instructions that a timing frontend must map.
"""

import argparse
import collections
import json
import lzma
from pathlib import Path


class TraceError(ValueError):
    pass


def open_trace(path):
    return lzma.open(path, "rt") if str(path).endswith(".xz") else open(path)


def parse_header(lines):
    header = {}
    for line in lines:
        if not line.startswith("-"):
            continue
        if " = " in line:
            key, value = line[1:].strip().split(" = ", 1)
            header[key] = value
    return header


def parse_record(line, *, lineinfo=False, core_id=False):
    fields = line.split()
    prefix_size = 11 + (2 if core_id else 0) + (1 if lineinfo else 0)
    if len(fields) < prefix_size + 5:
        raise TraceError("short instruction record")
    prefix = fields[:11]
    try:
        cta = tuple(int(value) for value in prefix[:3])
        warp = int(prefix[3])
        cluster = tuple(int(value) for value in prefix[4:7])
        cluster_cta = tuple(int(value) for value in prefix[7:10])
        cluster_rank = int(prefix[10])
        pc = int(fields[prefix_size], 16)
        mask = int(fields[prefix_size + 1], 16)
        count = int(fields[prefix_size + 2])
        pos = prefix_size + 3
        destinations = fields[pos:pos + count]
        pos += count
        opcode = fields[pos]
        pos += 1
        count = int(fields[pos])
        pos += 1
        sources = fields[pos:pos + count]
        pos += count
    except (IndexError, ValueError) as exc:
        raise TraceError("invalid instruction fields") from exc
    if len(destinations) != int(fields[prefix_size + 2]) or len(sources) != count:
        raise TraceError("truncated register list")
    if not opcode or not 0 <= warp or mask > 0xFFFFFFFF:
        raise TraceError("invalid opcode, warp, or active mask")
    return {
        "cta": cta, "warp": warp, "cluster": cluster,
        "cluster_cta": cluster_cta, "cluster_rank": cluster_rank,
        "pc": pc, "mask": mask, "destinations": destinations,
        "opcode": opcode, "sources": sources, "payload": fields[pos:],
    }


def inspect(path, expected_arch="120"):
    with open_trace(path) as stream:
        lines = iter(stream)
        header_lines = []
        for line in lines:
            header_lines.append(line)
            if line.startswith("#traces format"):
                break
        header = parse_header(header_lines)
        if header.get("binary version") != expected_arch:
            raise TraceError(
                f"expected SM{expected_arch}, got {header.get('binary version', 'unknown')}"
            )
        if header.get("accelsim tracer version") != "6":
            raise TraceError("only Accel-Sim tracer format version 6 is accepted")
        lineinfo = header.get("enable lineinfo") == "1"
        core_id = header.get("trace core id") == "1"
        counts = collections.Counter()
        warps = set()
        ctas = set()
        memory = collections.Counter()
        markers = collections.Counter()
        records = 0
        for number, line in enumerate(lines, len(header_lines) + 1):
            if not line.strip() or line.startswith("#") or line.startswith("-"):
                continue
            try:
                record = parse_record(line, lineinfo=lineinfo, core_id=core_id)
            except TraceError as exc:
                raise TraceError(f"{path}:{number}: {exc}") from exc
            opcode = record["opcode"]
            records += 1
            if opcode in ("REPLAY_START", "REPLAY_END"):
                markers[opcode] += 1
                continue
            counts[opcode] += 1
            ctas.add(record["cta"])
            warps.add((record["cta"], record["warp"]))
            if opcode.startswith(("LDG", "STG", "ATOM", "RED", "LDGSTS")):
                memory[opcode] += 1
        if records == 0:
            raise TraceError("trace contains no instruction records")
        return {
            "source": str(path), "kernel": header.get("kernel name"),
            "binary_version": header["binary version"],
            "tracer_version": header["accelsim tracer version"],
            "records": records, "ctas": len(ctas), "warps": len(warps),
            "opcodes": dict(sorted(counts.items())),
            "memory_opcodes": dict(sorted(memory.items())),
            "replay_markers": dict(sorted(markers.items())),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="raw .trace or .trace.xz file")
    parser.add_argument("--arch", default="120", help="expected SM binary version")
    args = parser.parse_args()
    try:
        print(json.dumps(inspect(args.trace, args.arch), indent=2))
    except (OSError, TraceError) as exc:
        parser.exit(2, f"trace rejected: {exc}\n")


if __name__ == "__main__":
    main()
