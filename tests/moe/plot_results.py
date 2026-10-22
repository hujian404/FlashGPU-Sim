#!/usr/bin/env python3
"""Regenerate date-free figures from published hardware and tracing records."""
import csv
import json
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs/experiments/results"
FIGURES = ROOT / "docs/experiments/figures/moe"
COLORS = {"Gate": "#4169A1", "Up": "#269FAD", "SwiGLU": "#E4B250",
          "Down": "#8860A9", "Dispatch": "#3B9D7D", "Combine": "#D7824A"}


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / (name + ".png"), dpi=180, bbox_inches="tight", metadata={"Software": "MoE Evaluation"})
    fig.savefig(FIGURES / (name + ".svg"), bbox_inches="tight", metadata={"Date": None, "Creator": "MoE Evaluation"})
    svg = FIGURES / (name + ".svg")
    # Omit RDF metadata entirely, including calendar-like namespace URIs.
    svg.write_bytes(re.sub(r"<metadata>.*?</metadata>", "", svg.read_text(encoding="utf-8"), flags=re.S).encode("utf-8"))
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#26364A", "text.color": "#26364A",
                         "svg.fonttype": "none"})
    rows = []
    for p in [RESULTS / "moe-baseline/hardware-summary.csv", RESULTS / "moe-gpu/hardware-extended/summary.csv"]:
        with p.open() as stream:
            rows.extend(csv.DictReader(stream))
    tokens = sorted({int(r["tokens"]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    for ax, pattern in zip(axes, ["balanced", "hotspot"]):
        for serial, label, color, marker in [("False", "Grouped grid", "#4169A1", "o"), ("True", "Separate launches", "#8860A9", "s")]:
            values = [float(next(r["median_ms"] for r in rows if int(r["tokens"]) == t and r["mode"] == "experts" and r["pattern"] == pattern and r["serial"] == serial)) for t in tokens]
            ax.plot(tokens, values, marker=marker, lw=2, color=color, label=label)
        ax.set(xscale="log", xticks=tokens, xticklabels=tokens, xlabel="Tokens", ylabel="Expert stages (ms)", title=pattern.capitalize(), ylim=(0, None))
        ax.grid(axis="y", alpha=0.15)
    axes[0].legend(frameon=False)
    save(fig, "hardware-scaling")

    traces = json.loads((RESULTS / "moe-gpu/nsight/summary.json").read_text())
    chosen = [next(r for r in traces if r["tokens"] == t and r["pattern"] == p and r["mode"] == "block" and not r["serial"])
              for t in [128, 2048] for p in ["balanced", "random", "hotspot"]]
    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    x, bottom = np.arange(len(chosen)), np.zeros(len(chosen))
    for stage, color in COLORS.items():
        y = np.array([r["stages"][stage]["total_ms_per_invocation"] for r in chosen])
        ax.bar(x, y, bottom=bottom, color=color, width=0.68, label=stage)
        bottom += y
    ax.set(xticks=x, xticklabels=[str(r["tokens"]) + "\n" + r["pattern"].capitalize() for r in chosen], ylabel="GPU kernel time / invocation (ms)")
    ax.legend(frameon=False, ncol=6, loc="upper left", bbox_to_anchor=(0, 1.16), fontsize=10)
    ax.grid(axis="y", alpha=0.15)
    save(fig, "nsight-stages")

    fig, axes = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True, constrained_layout=True)
    timeline_order = ["Dispatch", "Gate", "Up", "SwiGLU", "Down", "Combine"]
    for ax, serial, label in zip(axes, [False, True], ["Grouped grid", "Separate launches"]):
        name = "t128-block-balanced" + ("-serial" if serial else "")
        d = json.loads((RESULTS / "moe-gpu/nsight" / (name + ".json")).read_text())
        first = d["kernels"][:d["kernel_count"] // d["repetitions"]]
        for i, stage in enumerate(timeline_order):
            intervals = [(k["start_ns"] / 1e6, k["duration_ns"] / 1e6) for k in first if k["stage"] == stage]
            ax.broken_barh(intervals, (i - 0.32, 0.64), facecolors=COLORS[stage])
        ax.set(yticks=range(6), yticklabels=timeline_order, title=label, ylim=(5.6, -0.6))
        ax.grid(axis="x", alpha=0.15)
    axes[-1].set_xlabel("GPU timeline (ms), first invocation")
    save(fig, "nsight-timeline")

    values, labels = [], []
    for t in [128, 2048]:
        for pattern in ["balanced", "random", "hotspot"]:
            folder = RESULTS / ("moe-baseline/hardware" if t == 128 else "moe-gpu/hardware-extended")
            d = json.loads((folder / f"t{t}-block-{pattern}.json").read_text())
            counts = np.array(d["expert_counts"])
            values.append(counts / counts.mean())
            labels.append(f"{t}  {pattern.capitalize()}")
    fig, ax = plt.subplots(figsize=(10, 3.3), constrained_layout=True)
    im = ax.imshow(values, cmap="YlOrRd", vmin=0, vmax=8, aspect="auto", interpolation="nearest")
    ax.set(yticks=range(6), yticklabels=labels, xticks=[0, 16, 32, 48, 63], xlabel="Expert ID")
    fig.colorbar(im, ax=ax, label="Tokens / mean expert load", shrink=0.9)
    save(fig, "routing-load")

    p = RESULTS / "moe-gpu/scheduling-control/hardware-summary.json"
    if p.exists():
        control = json.loads(p.read_text())
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
        for ax, t in zip(axes, [128, 2048]):
            x = np.arange(2)
            for shift, schedule, color in [(-0.18, "padded", "#8860A9"), (0.18, "compact", "#269FAD")]:
                y = [next(r["median_ms"] for r in control if r["tokens"] == t and r["pattern"] == pattern and r["schedule"] == schedule) for pattern in ["balanced", "hotspot"]]
                ax.bar(x + shift, y, width=0.35, color=color, label=schedule.capitalize())
                for xx, yy in zip(x + shift, y):
                    ax.text(xx, yy, f"{yy:.2f}", ha="center", va="bottom", fontsize=9)
            ax.set(xticks=x, xticklabels=["Balanced", "Hotspot"], title=f"{t} tokens", ylabel="MoE block (ms)", ylim=(0, None))
            ax.margins(y=0.15)
            ax.grid(axis="y", alpha=0.15)
        axes[0].legend(frameon=False)
        save(fig, "scheduling-control")

    architecture = RESULTS / "moe-gpu/architecture-inference"
    if architecture.exists():
        data = [json.loads((architecture / f"t{t}.json").read_text()) for t in [128, 512, 2048]]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
        x = np.arange(3)
        for ax, field, label, color in [(axes[0], "prefill_tokens_per_second", "Prefill tokens / s", "#4169A1"),
                                         (axes[1], "decode_tokens_per_second", "Decode tokens / s", "#269FAD")]:
            y = [d[field] for d in data]
            ax.bar(x, y, color=color, width=0.6)
            for xx, yy in zip(x, y):
                ax.text(xx, yy, f"{yy:,.1f}", ha="center", va="bottom", fontsize=10)
            ax.set(xticks=x, xticklabels=[128, 512, 2048], xlabel="Prefill tokens", ylabel=label, ylim=(0, None))
            ax.margins(y=0.15)
            ax.grid(axis="y", alpha=0.15)
        fig.suptitle("Synthetic BF16 weights", fontsize=13)
        save(fig, "architecture-throughput")

        summary = json.loads((architecture / "summary.json").read_text())
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ax, phase, scale, label in [(axes[0], "Prefill", 1, "GPU kernel time / prefill (ms)"),
                                        (axes[1], "Decode", 8, "GPU kernel time / decode token (ms)")]:
            bottom = np.zeros(3)
            for stage, color in [("Attention", "#4169A1"), ("Router", "#E4B250"), ("MoE", "#8860A9"), ("Other", "#A3ACB9")]:
                y = np.array([d["phase_gpu_ms"][phase].get(stage, 0) / scale for d in summary])
                ax.bar(x, y, bottom=bottom, color=color, width=0.6, label=stage)
                bottom += y
            ax.set(xticks=x, xticklabels=[128, 512, 2048], xlabel="Prefill tokens", ylabel=label)
            ax.grid(axis="y", alpha=0.15)
        axes[0].legend(frameon=False, ncol=4, fontsize=9, loc="upper left", bbox_to_anchor=(0, 1.14))
        fig.suptitle("Synthetic BF16 weights", fontsize=13, y=1.12)
        save(fig, "architecture-stages")

        record = data[0]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, phase in zip(axes, ["prefill", "decode"]):
            counts = np.array(record["routing_counts"][phase], dtype=float)
            normalized = counts / counts.mean(axis=1, keepdims=True)
            im = ax.imshow(normalized, cmap="YlOrRd", vmin=0, vmax=8, aspect="auto", interpolation="nearest")
            ax.set(xticks=[0, 16, 32, 48, 63], yticks=[0, 4, 8, 12, 15], xlabel="Expert ID", ylabel="Layer", title=phase.capitalize())
        fig.colorbar(im, ax=axes, label="Tokens / mean expert load", shrink=0.9)
        fig.suptitle("Synthetic BF16 weights", fontsize=13)
        save(fig, "architecture-routing")


if __name__ == "__main__":
    main()
