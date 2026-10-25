#!/usr/bin/env python3
"""Turn results/*.csv into PNG charts in results/charts/. Reads CSVs only (run kpis.py first).

Usage: python3 scripts/plot.py [--blocks results/blocks/baseline/<model>__64__r1.csv]
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# RESULTS_DIR=results-pi reads a copy pulled from the Pi.
RESULTS = os.environ.get("RESULTS_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
CHARTS = os.path.join(RESULTS, "charts")


def read_csv(name):
    path = name if os.path.isabs(name) or os.path.exists(name) else os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    os.makedirs(CHARTS, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, name), dpi=150)
    plt.close(fig)
    print("wrote", os.path.join(CHARTS, name))


def plot_budget(kpis):
    """p99 as % of the deadline vs block size, one line per model, one panel per tag."""
    tags = sorted({r["tag"] for r in kpis})
    fig, axes = plt.subplots(1, len(tags), figsize=(5.5 * len(tags), 4), sharey=True, squeeze=False)
    for ax, tag in zip(axes[0], tags):
        series = defaultdict(list)
        for r in kpis:
            if r["tag"] == tag:
                series[r["model"]].append((int(r["block"]), float(r["p99_pct_of_deadline"])))
        for model, pts in sorted(series.items()):
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=model)
        ax.axhline(100, color="crimson", ls="--", lw=1, label="deadline")
        ax.set_xscale("log", base=2)
        ax.set_xticks([64, 128, 256, 512], ["64\n1.33ms", "128\n2.67ms", "256\n5.33ms", "512\n10.67ms"])
        ax.set_title(f"p99 block time vs deadline ({tag})")
        ax.set_xlabel("block size (samples) / deadline")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("p99 as % of deadline")
    axes[0][-1].legend(fontsize=8)
    save(fig, "p99_vs_deadline.png")


def plot_missed(kpis):
    tags = sorted({r["tag"] for r in kpis})
    fig, axes = plt.subplots(1, len(tags), figsize=(5.5 * len(tags), 4), sharey=True, squeeze=False)
    for ax, tag in zip(axes[0], tags):
        series = defaultdict(list)
        for r in kpis:
            if r["tag"] == tag:
                series[r["model"]].append((int(r["block"]), float(r["missed_pct"])))
        for model, pts in sorted(series.items()):
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=model)
        ax.set_xscale("log", base=2)
        ax.set_xticks([64, 128, 256, 512], ["64", "128", "256", "512"])
        ax.set_title(f"missed deadlines ({tag})")
        ax.set_xlabel("block size (samples)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("% of blocks over deadline")
    axes[0][-1].legend(fontsize=8)
    save(fig, "missed_deadlines.png")


def _grouped_bars(rows, prefix, ylabel, title, fname, log=True):
    tags = [k[len(prefix):] for k in rows[0] if k.startswith(prefix)]
    labels = [f"{r['model']}\n{r['block']}" for r in rows]
    x = list(range(len(rows)))
    w = 0.8 / len(tags)
    fig, ax = plt.subplots(figsize=(max(6, 0.25 * len(rows) * len(tags) + 2), 4))
    for j, t in enumerate(tags):
        ax.bar([i - 0.4 + w * (j + 0.5) for i in x], [float(r[prefix + t]) / 1000 for r in rows], w, label=t)
    ax.scatter(x, [float(r["deadline_ms"]) for r in rows], color="crimson", marker="_", s=300,
               label="deadline", zorder=3)
    ax.set_xticks(x, labels, fontsize=7, rotation=90)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    save(fig, fname)


def plot_compare(rows):
    if not rows:
        return
    _grouped_bars(rows, "p99_us_", "p99 block time (ms)", "p99 block time by scheduling arm", "rt_comparison.png")
    if any(float(r[k]) > 0 for r in rows for k in r if k.startswith("wake_p99_us_")):
        _grouped_bars(rows, "wake_p99_us_", "p99 wake-up latency (ms)", "p99 wake-up latency by scheduling arm",
                      "wake_latency.png")


def plot_memory(rows):
    if not rows:
        return
    rows = [r for r in rows if r["tag"] == "baseline"] or rows
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh([r["model"] for r in rows], [float(r["peak_pss_mb"]) for r in rows])
    ax.set_xlabel("peak PSS (MB)")
    ax.set_title("memory per model")
    ax.grid(alpha=0.3, axis="x")
    save(fig, "memory_pss.png")


def plot_blocks(path):
    rows = [r for r in read_csv(path) if r["warmup"] == "0"]
    if not rows:
        return
    t = [int(r["start_ns"]) / 1e9 for r in rows]
    e = [int(r["elapsed_ns"]) / 1e3 for r in rows]
    d = int(rows[0]["deadline_ns"]) / 1e3
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.5))
    a1.plot(t, e, lw=0.4)
    a1.axhline(d, color="crimson", ls="--", lw=1)
    a1.set_xlabel("time (s)")
    a1.set_ylabel("block time (µs)")
    a1.set_title(os.path.basename(path))
    a2.hist(e, bins=200, log=True)
    a2.axvline(d, color="crimson", ls="--", lw=1)
    a2.set_xlabel("block time (µs)")
    a2.set_title("distribution")
    save(fig, "blocks_" + os.path.basename(path).replace(".csv", ".png"))


def plot_sustained(rows, suffix=""):
    if not rows:
        return
    m = [int(r["minute"]) for r in rows]
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(m, [float(r["p99_us"]) for r in rows], marker=".", label="p99 block time")
    ax1.set_xlabel("minute")
    ax1.set_ylabel("p99 block time (µs)")
    ax2 = ax1.twinx()
    temps = [(mm, float(r["temp_c_max"])) for mm, r in zip(m, rows) if r["temp_c_max"] not in ("", "None")]
    if temps:
        ax2.plot([p[0] for p in temps], [p[1] for p in temps], color="orange", label="SoC temp")
    ax2.set_ylabel("temp (°C)")
    ax1.set_title("30-minute sustained run")
    fig.legend(loc="upper left")
    save(fig, f"sustained{suffix}.png")


def plot_thermal(rows, suffix=""):
    if not rows:
        return
    t0 = float(rows[0]["unix_time"])
    t = [(float(r["unix_time"]) - t0) / 60 for r in rows]
    fig, ax1 = plt.subplots(figsize=(8, 3.5))
    ax1.plot(t, [float(r["temp_c"]) if r["temp_c"] != "na" else None for r in rows], color="orange")
    ax1.set_ylabel("temp (°C)")
    ax1.set_xlabel("minutes")
    ax2 = ax1.twinx()
    ax2.plot(t, [int(r["scaling_cur_freq_khz"]) / 1000 if r["scaling_cur_freq_khz"] != "na" else None
                 for r in rows], lw=0.8)
    ax2.set_ylabel("CPU MHz")
    ax1.set_title("temperature and CPU frequency")
    save(fig, f"thermal{suffix}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", action="append", default=[], help="per-block CSV(s) to plot as timeline")
    args = ap.parse_args()
    kpis = read_csv("kpi_by_block.csv")
    if kpis:
        plot_budget(kpis)
        plot_missed(kpis)
    plot_compare(read_csv("rt_comparison.csv"))
    plot_memory(read_csv("kpi_memory.csv"))
    for path in sorted(glob.glob(os.path.join(RESULTS, "thermal_sustained*.csv"))):
        tag = os.path.basename(path)[len("thermal_"):-4]
        suffix = "" if tag == "sustained" else "_" + tag[len("sustained_"):]
        plot_sustained(read_csv(f"sustained_per_minute{suffix}.csv"), suffix)
        plot_thermal(read_csv(path), suffix)
    for b in args.blocks:
        plot_blocks(b)


if __name__ == "__main__":
    main()
