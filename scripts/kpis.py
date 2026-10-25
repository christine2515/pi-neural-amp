#!/usr/bin/env python3
"""Aggregate results/*.csv into KPI tables. Standard library only, so it runs on the Pi.

Outputs (all derived from CSVs bench and the samplers wrote):
  results/kpi_by_block.csv        per tag/model/block, aggregated over repeats
  results/kpi_smallest_block.csv  per tag/model: smallest block with zero missed deadlines
  results/kpi_memory.csv          per tag/model: peak PSS/RSS and mean CPU% from mem samples
  results/rt_comparison.csv       with --compare A B [C...]: p99, wake-up latency, misses per arm vs A
  results/sustained_per_minute.csv  per-minute p99 for the 30-minute run, joined with thermal data
  results/boot_summary.csv        boot-to-first-block stats over reboots
  results/live_by_config.csv      live runs: xruns, process time, jitter per period x periods
  results/live_smallest_buffer.csv  live: smallest xrun-free period x periods per model
  results/live_latency_summary.csv  round-trip latency per period x periods (loopback test)
  results/kpi_tables.md           the same tables as markdown, for the README

Usage: python3 scripts/kpis.py [--compare baseline rt rt_isol]
"""
import argparse
import csv
import glob
import os
import statistics
from collections import defaultdict

# RESULTS_DIR=results-pi reads a copy pulled from the Pi.
RESULTS = os.environ.get("RESULTS_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


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
            # Wake-up latency (paced runs only; summaries from before it existed read as 0).
            "wake_p99_us": statistics.median(float(r.get("wake_p99_us") or 0) for r in rs),
            "wake_max_us": max(float(r.get("wake_max_us") or 0) for r in rs),
            "missed_with_wake": sum(int(r.get("missed_with_wake") or 0) for r in rs),
            "governor": rs[0]["governor"], "paced": rs[0]["paced"], "rt": rs[0]["rt"],
            "isolated_cpus": rs[0].get("isolated_cpus", "na"), "kernel": rs[0].get("kernel", "na"),
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
        # Stricter: also counts blocks where a late wake-up plus process time overran the deadline.
        ok_wake = sorted(r["block"] for r in rs if r["missed"] == 0 and r["missed_with_wake"] == 0)
        out.append({
            "tag": tag, "model": model,
            "smallest_glitch_free_block": best if best is not None else "none",
            "latency_ms": row["deadline_ms"] if row else None,
            "p99_us_at_that_block": row["p99_us"] if row else None,
            "smallest_block_incl_wake": ok_wake[0] if ok_wake else "none",
            "blocks_tested": " ".join(str(r["block"]) for r in sorted(rs, key=lambda r: r["block"])),
        })
    return out


def memory_self(summaries):
    """Memory each bench run reported about itself (pss_kb, input_kb columns)."""
    groups = defaultdict(list)
    for r in summaries:
        if r.get("pss_kb") not in (None, "", "-1"):
            groups[(r["tag"], r["model"])].append(r)
    out = []
    for (tag, model), rs in sorted(groups.items()):
        pss = [int(r["pss_kb"]) for r in rs]
        inp = [int(r["input_kb"]) for r in rs]
        out.append({
            "tag": tag, "model": model, "runs": len(rs),
            "pss_mb_max": max(pss) / 1024, "input_mb": max(inp) / 1024,
            # what's left once the decoded input file is taken out: model + buffers + libraries
            "pss_minus_input_mb": max(p - i for p, i in zip(pss, inp)) / 1024,
            "peak_rss_mb": max(int(r["peak_rss_kb"]) for r in rs) / 1024,
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


def compare(kpis, tags):
    """One row per model/block present in every tag; changes are relative to the first tag."""
    idx = {(r["tag"], r["model"], r["block"]): r for r in kpis}
    base = tags[0]
    out = []
    for (tag, model, block), r0 in sorted(idx.items()):
        if tag != base or any((t, model, block) not in idx for t in tags):
            continue
        row = {"model": model, "block": block, "deadline_ms": r0["deadline_ms"]}
        for metric in ("p99_us", "max_us", "missed", "wake_p99_us", "missed_with_wake"):
            for t in tags:
                row[f"{metric}_{t}"] = idx[(t, model, block)][metric]
        for t in tags[1:]:
            rt = idx[(t, model, block)]
            row[f"p99_change_pct_{t}"] = 100.0 * (rt["p99_us"] - r0["p99_us"]) / r0["p99_us"] if r0["p99_us"] else None
            row[f"wake_p99_change_pct_{t}"] = (
                100.0 * (rt["wake_p99_us"] - r0["wake_p99_us"]) / r0["wake_p99_us"] if r0["wake_p99_us"] else None)
        out.append(row)
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


def live():
    """Per live configuration (label prefix + mode/model + period x periods), aggregated over runs."""
    path = os.path.join(RESULTS, "live_summary.csv")
    if not os.path.exists(path):
        return [], []
    groups = defaultdict(list)
    for r in read_csv(path):
        if r["mode"] == "latency_test":
            continue
        tag = r["label"].split("__")[0]
        groups[(tag, r["model"], int(r["period"]), int(r["periods"]))].append(r)
    rows = []
    for (tag, model, period, periods), rs in sorted(groups.items()):
        rows.append({
            "tag": tag, "model": model, "period": period, "periods": periods,
            "io_buffer_ms": float(rs[0]["io_buffer_ms"]), "runs": len(rs),
            "seconds": sum(float(r["seconds"]) for r in rs),
            "xruns": sum(int(r["xruns"]) for r in rs),
            "proc_p99_us": statistics.median(float(r["proc_p99_us"]) for r in rs),
            "deadline_us": float(rs[0]["deadline_us"]),
            "cycle_jitter_p99_us": statistics.median(float(r["cycle_jitter_p99_us"]) for r in rs),
            "cpu_pct": statistics.median(float(r["cpu_pct"]) for r in rs),
            "pss_mb": max((int(r.get("pss_kb") or -1) for r in rs), default=-1) / 1024,
        })
    best = defaultdict(list)
    for r in rows:
        if r["xruns"] == 0:
            best[(r["tag"], r["model"])].append(r)
    smallest = []
    for (tag, model) in sorted({(r["tag"], r["model"]) for r in rows}):
        ok = sorted(best.get((tag, model), []), key=lambda r: r["io_buffer_ms"])
        smallest.append({
            "tag": tag, "model": model,
            "smallest_xrun_free": f"{ok[0]['period']}x{ok[0]['periods']}" if ok else "none",
            "io_buffer_ms": ok[0]["io_buffer_ms"] if ok else None,
        })
    return rows, smallest


def latency():
    path = os.path.join(RESULTS, "live_latency.csv")
    if not os.path.exists(path):
        return []
    groups = defaultdict(list)
    for r in read_csv(path):
        groups[(int(r["period"]), int(r["periods"]), int(r["rate"]))].append(float(r["roundtrip_ms"]))
    return [{
        "period": p, "periods": n, "clicks": len(v),
        "roundtrip_ms_median": statistics.median(v), "roundtrip_ms_min": min(v), "roundtrip_ms_max": max(v),
        "io_buffer_ms": 1000.0 * p * n / rate,
    } for (p, n, rate), v in sorted(groups.items())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs="+", metavar="TAG", help="baseline tag first, then arms to compare")
    args = ap.parse_args()

    md = ["<!-- generated by scripts/kpis.py; do not edit by hand -->"]
    kpis = by_block(load_summaries())
    if kpis:
        f = ["tag", "model", "block", "deadline_ms", "runs", "p50_us", "p99_us", "p99_us_worst_run", "max_us",
             "missed", "measured_blocks", "missed_pct", "p99_pct_of_deadline", "dsp_load_pct", "wake_p99_us",
             "wake_max_us", "missed_with_wake", "governor", "paced", "rt", "isolated_cpus", "kernel"]
        write_csv(os.path.join(RESULTS, "kpi_by_block.csv"), kpis, f)
        md += ["## Block time by model and block size", md_table(kpis, f[:13] + ["wake_p99_us", "missed_with_wake"])]
        sb = smallest_block(kpis)
        f = ["tag", "model", "smallest_glitch_free_block", "latency_ms", "p99_us_at_that_block",
             "smallest_block_incl_wake", "blocks_tested"]
        write_csv(os.path.join(RESULTS, "kpi_smallest_block.csv"), sb, f)
        md += ["## Smallest glitch-free block", md_table(sb, f)]
    mem_self = memory_self(load_summaries())
    if mem_self:
        f = list(mem_self[0].keys())
        write_csv(os.path.join(RESULTS, "kpi_memory_self.csv"), mem_self, f)
        md += ["## Memory (bench self-reported; input_mb is the decoded test file)", md_table(mem_self, f)]
    mem = memory()
    if mem:
        f = ["tag", "model", "runs", "peak_pss_mb", "median_pss_mb", "peak_rss_mb", "mean_cpu_pct"]
        write_csv(os.path.join(RESULTS, "kpi_memory.csv"), mem, f)
        md += ["## Memory and CPU", md_table(mem, f)]
    if args.compare:
        cmp_rows = compare(kpis, args.compare)
        if cmp_rows:
            f = list(cmp_rows[0].keys())
            write_csv(os.path.join(RESULTS, "rt_comparison.csv"), cmp_rows, f)
            md += ["## " + " vs ".join(args.compare), md_table(cmp_rows, f)]
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

    lv, lv_small = live()
    if lv:
        f = list(lv[0].keys())
        write_csv(os.path.join(RESULTS, "live_by_config.csv"), lv, f)
        md += ["## Live: xruns by configuration", md_table(lv, f)]
        f = list(lv_small[0].keys())
        write_csv(os.path.join(RESULTS, "live_smallest_buffer.csv"), lv_small, f)
        md += ["## Live: smallest xrun-free buffer", md_table(lv_small, f)]
    lat = latency()
    if lat:
        f = list(lat[0].keys())
        write_csv(os.path.join(RESULTS, "live_latency_summary.csv"), lat, f)
        md += ["## Round-trip latency (loopback cable)", md_table(lat, f, None)]

    with open(os.path.join(RESULTS, "kpi_tables.md"), "w") as fh:
        fh.write("\n\n".join(md) + "\n")
    print("\n\n".join(md))


if __name__ == "__main__":
    main()
