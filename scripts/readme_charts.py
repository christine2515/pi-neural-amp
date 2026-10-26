#!/usr/bin/env python3
"""README charts, from CSVs only (results-pi/kpi_by_block.csv, results-pi/live_summary.csv,
results-pi/sustained_per_minute_full_pipe3.csv). Writes PNGs to docs/img/.

Usage: RESULTS_DIR=results-pi python3 scripts/readme_charts.py   (needs matplotlib)
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS = os.environ.get("RESULTS_DIR") or os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "docs", "img")
DEADLINE_128 = 2666.7
BLUE, ORANGE, GREY, RED = "#2b6cb0", "#dd6b20", "#a0aec0", "#c53030"


def rows(name):
    with open(os.path.join(RESULTS, name), newline="") as f:
        return list(csv.DictReader(f))


def kpi(tag, model_suffix):
    for r in rows("kpi_by_block.csv"):
        if r["tag"] == tag and r["block"] == "128" and r["model"].endswith(model_suffix):
            return float(r["p50_us"]), float(r["p99_us"])
    raise SystemExit(f"missing kpi row {tag} {model_suffix}")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=150)
    plt.close(fig)
    print("wrote", os.path.join("docs", "img", name))


def chart_speedup():
    full = [("Baseline (stock A2 path)", "baseline", "@full", BLUE),
            ("Ring buffer mode 0", "ring0_full", "@full", BLUE),
            ("Register-blocked NEON kernel", "neon", "@full", BLUE),
            ("+ two-core pipeline", "pipe_full", "@full+pipe", BLUE),
            ("+ NEON head", "head", "@full+pipe", BLUE),
            ("Tried: fast-math auto-vectorize", "a2fm", "@full", GREY),
            ("Tried: Eigen lazyProduct", "lazy", "@full", GREY),
            ("Tried: GCC 14, pipelined (spills)", "gcc14", "@full+pipe", GREY)]
    lite = [("Baseline", "baseline", "@lite", ORANGE), ("Ring buffer mode 0", "ring0", "@lite", ORANGE),
            ("NEON lanes kernel", "neon", "@lite", ORANGE), ("+ NEON head", "head", "@lite", ORANGE),
            ("Tried: fast-math auto-vectorize", "a2fm", "@lite", GREY)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
    for ax, series, title in ((axes[0], full, "Full model (8 channels, 11,776 MAC/sample)"),
                              (axes[1], lite, "Lite model (3 channels, 1,731 MAC/sample)")):
        labels, p50s, colors = [], [], []
        for label, tag, suffix, color in series:
            p50, _ = kpi(tag, suffix)
            labels.append(label)
            p50s.append(p50)
            colors.append(color)
        y = range(len(labels))[::-1]
        ax.barh(list(y), p50s, color=colors)
        for yi, v in zip(y, p50s):
            ax.text(v + 60, yi, f"{v:,.0f} µs", va="center", fontsize=9)
        ax.set_xlim(0, max(p50s) * 1.18)
        if DEADLINE_128 < max(p50s) * 1.18:
            ax.axvline(DEADLINE_128, color=RED, ls="--", lw=1.2)
            ax.text(DEADLINE_128, -0.9, " deadline 2,667 µs", color=RED, fontsize=9, va="center")
        else:
            ax.text(0.98, 0.02, "deadline 2,667 µs is off this scale (Lite uses 32–43%)",
                    transform=ax.transAxes, ha="right", color=RED, fontsize=8)
        ax.set_yticks(list(y), labels, fontsize=9)
        ax.set_xlabel("p50 time per 128-sample block (µs), Pi 2, 3 runs")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", alpha=0.3)
    save(fig, "speedup.png")


def chart_live():
    want = [("lsw_lite", "@lite", "Lite", ORANGE), ("lsw_full_pipe", "@full+pipe", "Full, pipelined", BLUE)]
    data = rows("live_summary.csv")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    xlabels, x = [], 0
    for tag, suffix, name, color in want:
        sel = [r for r in data if r["label"].split("__")[0] == tag and r["model"].endswith(suffix)]
        sel.sort(key=lambda r: (int(r["periods"]), int(r["period"])))
        for r in sel:
            p, n = int(r["period"]), int(r["periods"])
            pipe = p if "+pipe" in r["model"] else 0
            buf_ms = (p * n + pipe) / 48.0
            ax.bar(x, float(r["proc_p99_us"]) / float(r["deadline_us"]) * 100, color=color)
            ax.text(x, float(r["proc_p99_us"]) / float(r["deadline_us"]) * 100 + 2,
                    f"{r['xruns']} xr", ha="center", fontsize=8)
            xlabels.append(f"{name}\n{p}×{n}\n{buf_ms:.2f} ms")
            x += 1
        x += 0.6
    ax.axhline(100, color=RED, ls="--", lw=1.2)
    ax.text(0, 101, "deadline (one period)", color=RED, fontsize=9)
    ticks = [i for i, _ in enumerate(xlabels)]
    # recompute tick positions with the gap
    pos, x = [], 0
    for tag, suffix, _, _ in want:
        k = sum(1 for r in data if r["label"].split("__")[0] == tag and r["model"].endswith(suffix))
        pos += [x + i for i in range(k)]
        x += k + 0.6
    ax.set_xticks(pos, xlabels, fontsize=7.5)
    ax.set_ylabel("model p99 as % of the period deadline")
    ax.set_title("Live on a Scarlett Solo (20 s per config, --rt): p99 vs deadline, underruns (xr), "
                 "in-device buffering", fontsize=10)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "live_buffers.png")


def chart_sustained():
    r = rows("sustained_per_minute_full_pipe3.csv")
    m = [int(x["minute"]) for x in r if int(x["blocks"]) > 1000]
    p50 = [float(x["p50_us"]) for x in r if int(x["blocks"]) > 1000]
    p99 = [float(x["p99_us"]) for x in r if int(x["blocks"]) > 1000]
    temp = [float(x["temp_c_max"]) for x in r if int(x["blocks"]) > 1000]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(m, p50, color=BLUE, label="p50")
    ax.plot(m, p99, color=ORANGE, label="p99")
    ax.axhline(DEADLINE_128, color=RED, ls="--", lw=1.2, label="deadline")
    ax.set_ylim(0, 3000)
    ax.set_xlabel("minute")
    ax.set_ylabel("µs per 128-sample block")
    ax2 = ax.twinx()
    ax2.plot(m, temp, color=GREY, lw=1, label="SoC °C")
    ax2.set_ylim(20, 60)
    ax2.set_ylabel("SoC temperature (°C)")
    ax.set_title("Full model, pipelined, --rt, paced: 30 minutes (675,000 blocks)", fontsize=10)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "sustained.png")


if __name__ == "__main__":
    chart_speedup()
    chart_live()
    chart_sustained()
