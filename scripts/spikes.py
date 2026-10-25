#!/usr/bin/env python3
"""Characterize block-time spikes in per-block CSVs (results/blocks/<tag>/*.csv). Standard library only.

A spike is a measured block slower than FACTOR x the run's median. Spikes within 50 ms are one event.
Reports the event count, rate and typical spacing: a steady spacing points at a periodic disturber
(a sampler, a timer, a daemon) rather than the model.

Usage: python3 scripts/spikes.py results-pi/blocks/baseline/*.csv [--factor 1.3]
Appends rows to $RESULTS_DIR/spikes.csv (default results/).
"""
import argparse
import csv
import os
import statistics
import time

RESULTS = os.environ.get("RESULTS_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def analyze(path, factor):
    rows = [r for r in csv.DictReader(open(path)) if r["warmup"] == "0"]
    t = [int(r["start_ns"]) / 1e9 for r in rows]
    e = [int(r["elapsed_ns"]) / 1e3 for r in rows]
    med = statistics.median(e)
    events = []
    for ti, ei in zip(t, e):
        if ei > factor * med and (not events or ti - events[-1] > 0.05):
            events.append(ti)
    gaps = [b - a for a, b in zip(events, events[1:])]
    dur = t[-1] - t[0] if len(t) > 1 else 0.0
    return {
        "unix_time": int(time.time()),
        "file": os.path.relpath(path),
        "factor": factor,
        "median_us": round(med, 1),
        "max_us": round(max(e), 1),
        "spike_events": len(events),
        "events_per_s": round(len(events) / dur, 2) if dur else 0,
        "median_gap_s": round(statistics.median(gaps), 3) if gaps else "",
        "gap_iqr_s": f"{statistics.quantiles(gaps, n=4)[0]:.3f}-{statistics.quantiles(gaps, n=4)[2]:.3f}"
        if len(gaps) >= 4 else "",
        "duration_s": round(dur, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--factor", type=float, default=1.3)
    args = ap.parse_args()
    out = os.path.join(RESULTS, "spikes.csv")
    rows = [analyze(f, args.factor) for f in args.files]
    exists = os.path.exists(out)
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if not exists:
            w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['file']}: median {r['median_us']} us, max {r['max_us']} us, {r['spike_events']} events "
              f"({r['events_per_s']}/s), gap {r['median_gap_s']} s [{r['gap_iqr_s']}]")


if __name__ == "__main__":
    main()
