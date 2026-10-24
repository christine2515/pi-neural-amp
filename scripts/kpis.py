#!/usr/bin/env python3
"""Aggregate results/*.csv into KPI tables. Standard library only, so it runs on the Pi.

Outputs (all derived from CSVs bench and the samplers wrote):
  results/kpi_by_block.csv        per tag/model/block, aggregated over repeats
  results/kpi_smallest_block.csv  per tag/model: smallest block with zero missed deadlines
  results/kpi_memory.csv          per tag/model: peak PSS/RSS and mean CPU% from mem samples
  results/rt_comparison.csv       with --compare A B: p99 and missed deadlines, A vs B
  results/sustained_per_minute.csv  per-minute p99 for the 30-minute run, joined with thermal data
  results/boot_summary.csv        boot-to-first-block stats over reboots
  results/kpi_tables.md           the same tables as markdown, for the README

Usage: python3 scripts/kpis.py [--compare baseline rt]
"""
import argparse
import csv
import glob
import os
import statistics
from collections import defaultdict

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def fmt(v, nd=1):
    if v is None or v == "":
        return "–"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def md_table(rows, fields, headers=None):
    headers = headers or fields
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(fields)]
    for r in rows:
        out.append("| " + " | ".join(fmt(r[f]) for f in fields) + " |")
    return "\n".join(out)


def load_summaries():
    """Rows from results/summary_<tag>.csv, with the tag attached."""
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "summary_*.csv"))):
        tag = os.path.basename(path)[len("summary_"):-len(".csv")]
        for r in read_csv(path):
            r["tag"] = tag
            rows.append(r)
    return rows


def by_block(summaries):
    groups = defaultdict(list)
    for r in summaries:
        if r["tag"] in ("boot", "sustained"):
            continue
        groups[(r["tag"], r["model"], int(r["block"]))].append(r)
    out = []
    for (tag, model, block), rs in sorted(groups.items()):
        f = lambda k: [float(r[k]) for r in rs]
        missed = sum(int(r["missed"]) for r in rs)
        measured = sum(int(r["measured_blocks"]) for r in rs)
        out.append({
            "tag": tag, "model": model, "block": block,
            "deadline_us": float(rs[0]["deadline_us"]),
            "deadline_ms": float(rs[0]["deadline_us"]) / 1000,
            "runs": len(rs),
            "p50_us": statistics.median(f("p50_us")),
            "p99_us": statistics.median(f("p99_us")),
            "p99_us_worst_run": max(f("p99_us")),
            "max_us": max(f("max_us")),
            "missed": missed,
            "measured_blocks": measured,
            "missed_pct": 100.0 * missed / measured if measured else 0.0,
            "p99_pct_of_deadline": 100.0 * statistics.median(f("p99_us")) / float(rs[0]["deadline_us"]),
            "dsp_load_pct": statistics.median(f("dsp_load_pct")),
            "governor": rs[0]["governor"], "paced": rs[0]["paced"], "rt": rs[0]["rt"],
        })
    return out


def smallest_block(kpis):
    groups = defaultdict(list)
    for r in kpis:
        groups[(r["tag"], r["model"])].append(r)
    out = []
    for (tag, model), rs in sorted(groups.items()):
        ok = sorted(r["block"] for r in rs if r["missed"] == 0)
        best = ok[0] if ok else None
        row = next((r for r in rs if r["block"] == best), None)
        out.append({
            "tag": tag, "model": model,
            "smallest_glitch_free_block": best if best is not None else "none",
            "latency_ms": row["deadline_ms"] if row else None,
            "p99_us_at_that_block": row["p99_us"] if row else None,
            "blocks_tested": " ".join(str(r["block"]) for r in sorted(rs, key=lambda r: r["block"])),
        })
    return out


def memory():
    out = []
    groups = defaultdict(list)
    for path in glob.glob(os.path.join(RESULTS, "mem", "*", "*.csv")):
        tag = os.path.basename(os.path.dirname(path))
        model = os.path.basename(path).split("__")[0]
        groups[(tag, model)].append(read_csv(path))
    for (tag, model), runs in sorted(groups.items()):
        samples = [s for run in runs for s in run]
        if not samples:
            continue
        cpus = [float(s["cpu_pct"]) for s in samples if s["cpu_pct"] not in ("na", "")]
        out.append({
            "tag": tag, "model": model, "runs": len(runs),
            "peak_pss_mb": max(int(s["pss_kb"]) for s in samples) / 1024,
            "median_pss_mb": statistics.median(int(s["pss_kb"]) for s in samples) / 1024,
            "peak_rss_mb": max(int(s["rss_kb"]) for s in samples) / 1024,
            "mean_cpu_pct": statistics.mean(cpus) if cpus else None,
        })
    return out


def compare(kpis, a, b):
    idx = {(r["tag"], r["model"], r["block"]): r for r in kpis}
    out = []
    for (tag, model, block), ra in sorted(idx.items()):
        if tag != a or (b, model, block) not in idx:
            continue
        rb = idx[(b, model, block)]
        out.append({
            "model": model, "block": block, "deadline_ms": ra["deadline_ms"],
            f"p99_us_{a}": ra["p99_us"], f"p99_us_{b}": rb["p99_us"],
            "p99_change_pct": 100.0 * (rb["p99_us"] - ra["p99_us"]) / ra["p99_us"] if ra["p99_us"] else None,
            f"max_us_{a}": ra["max_us"], f"max_us_{b}": rb["max_us"],
            f"missed_{a}": ra["missed"], f"missed_{b}": rb["missed"],
        })
    return out


def sustained():
    blocks = os.path.join(RESULTS, "blocks", "sustained", "sustained.csv")
    summ = os.path.join(RESULTS, "summary_sustained.csv")
    thermal = os.path.join(RESULTS, "thermal_sustained.csv")
    if not (os.path.exists(blocks) and os.path.exists(summ)):
        return []
    s = read_csv(summ)[-1]
    # bench's unix_time is written at the end of the run; reconstruct each block's wall time from it.
    rows = read_csv(blocks)
    end_offset_s = (int(rows[-1]["start_ns"]) + int(rows[-1]["elapsed_ns"])) / 1e9
    t_start = float(s["unix_time"]) - end_offset_s
    minutes = defaultdict(list)
    for r in rows:
        if r["warmup"] == "1":
            continue
        minutes[int(int(r["start_ns"]) / 60e9)].append((int(r["elapsed_ns"]), int(r["deadline_ns"])))
    temps = read_csv(thermal) if os.path.exists(thermal) else []
    out = []
    for m in sorted(minutes):
        ts = sorted(e for e, _ in minutes[m])
        lo, hi = t_start + 60 * m, t_start + 60 * (m + 1)
        win = [t for t in temps if lo <= float(t["unix_time"]) < hi]
        num = lambda k: [float(t[k]) for t in win if t[k] not in ("na", "")]
        temp, freq = num("temp_c"), num("scaling_cur_freq_khz")
        out.append({
            "minute": m, "blocks": len(ts),
            "p50_us": ts[len(ts) // 2] / 1e3,
            "p99_us": ts[min(len(ts) - 1, int(0.99 * len(ts)))] / 1e3,
            "max_us": ts[-1] / 1e3,
            "missed": sum(1 for e, d in minutes[m] if e > d),
            "temp_c_max": max(temp) if temp else None,
            "freq_mhz_min": min(freq) / 1000 if freq else None,
            "throttled_any": int(any(t["throttled_now"] == "1" or t["freq_capped_now"] == "1" for t in win)) if win else None,
        })
    return out


def boot():
    path = os.path.join(RESULTS, "boot_first_block.csv")
    if not os.path.exists(path):
        return []
    rows = read_csv(path)
    v = [float(r["first_block_boottime_s"]) for r in rows]
    return [{
        "boots": len(v), "median_s": statistics.median(v), "min_s": min(v), "max_s": max(v),
        "median_model_load_ms": statistics.median(float(r["model_load_ms"]) for r in rows),
    }]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()

    md = ["<!-- generated by scripts/kpis.py; do not edit by hand -->"]
    kpis = by_block(load_summaries())
    if kpis:
        f = ["tag", "model", "block", "deadline_ms", "runs", "p50_us", "p99_us", "p99_us_worst_run", "max_us",
             "missed", "measured_blocks", "missed_pct", "p99_pct_of_deadline", "dsp_load_pct", "governor", "paced", "rt"]
        write_csv(os.path.join(RESULTS, "kpi_by_block.csv"), kpis, f)
        md += ["## Block time by model and block size", md_table(kpis, f[:13])]
        sb = smallest_block(kpis)
        f = ["tag", "model", "smallest_glitch_free_block", "latency_ms", "p99_us_at_that_block", "blocks_tested"]
        write_csv(os.path.join(RESULTS, "kpi_smallest_block.csv"), sb, f)
        md += ["## Smallest glitch-free block", md_table(sb, f)]
    mem = memory()
    if mem:
        f = ["tag", "model", "runs", "peak_pss_mb", "median_pss_mb", "peak_rss_mb", "mean_cpu_pct"]
        write_csv(os.path.join(RESULTS, "kpi_memory.csv"), mem, f)
        md += ["## Memory and CPU", md_table(mem, f)]
    if args.compare:
        a, b = args.compare
        cmp_rows = compare(kpis, a, b)
        if cmp_rows:
            f = list(cmp_rows[0].keys())
            write_csv(os.path.join(RESULTS, "rt_comparison.csv"), cmp_rows, f)
            md += [f"## {a} vs {b}", md_table(cmp_rows, f)]
            with open(os.path.join(RESULTS, "rt_comparison.md"), "w") as fh:
                fh.write(md_table(cmp_rows, f) + "\n")
    sus = sustained()
    if sus:
        f = list(sus[0].keys())
        write_csv(os.path.join(RESULTS, "sustained_per_minute.csv"), sus, f)
        md += ["## Sustained run, per minute", md_table(sus, f)]
    bt = boot()
    if bt:
        f = list(bt[0].keys())
        write_csv(os.path.join(RESULTS, "boot_summary.csv"), bt, f)
        md += ["## Boot to first processed block (CLOCK_BOOTTIME, excludes firmware)", md_table(bt, f)]

    with open(os.path.join(RESULTS, "kpi_tables.md"), "w") as fh:
        fh.write("\n\n".join(md) + "\n")
    print("\n\n".join(md))


if __name__ == "__main__":
    main()
