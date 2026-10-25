#!/usr/bin/env python3
"""Analysis for docs/low-latency-plan.md. Standard library only.

  fit   TAG [TAG ...]   per model: least-squares fit of mean block time t(N) = a + b*N over block sizes
                        (a = fixed cost per process() call, b = cost per sample), plus p50/p99 per block.
  ticks CSV [CSV ...]   per-block CSVs: spikes (> $FACTOR x median, default 1.5), their dominant spacing (a 10 ms
                        spacing = the 100 Hz kernel tick), and the mean excess of a spike block.

Reads $RESULTS_DIR (default results-pi/). Appends rows to $RESULTS_DIR/lowlat_fit.csv / lowlat_ticks.csv.
Usage:
  python3 scripts/lowlat_analysis.py fit x_smallN_burst x_fs_full_p2
  python3 scripts/lowlat_analysis.py ticks results-pi/blocks/x_smallN_burst/*__8__r1.csv
"""
import collections
import csv
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("RESULTS_DIR") or os.path.join(HERE, "..", "results-pi")


def append(name, header, rows):
    path = os.path.join(RESULTS, name)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)


def fit(tags):
    out = []
    for tag in tags:
        rows = list(csv.DictReader(open(os.path.join(RESULTS, f"summary_{tag}.csv"))))
        by_model = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rows:
            by_model[r["model"]][int(r["block"])].append(r)
        for model, blocks in by_model.items():
            ns = sorted(blocks)
            means = [statistics.mean(float(r["mean_us"]) for r in blocks[n]) for n in ns]
            if len(ns) >= 2:
                mx, my = statistics.mean(ns), statistics.mean(means)
                b = sum((x - mx) * (y - my) for x, y in zip(ns, means)) / sum((x - mx) ** 2 for x in ns)
                a = my - b * mx
            else:
                a, b = float("nan"), float("nan")
            print(f"{tag} {model}: t(N) = {a:.1f} us + {b:.3f} us x N")
            for n, m in zip(ns, means):
                p50 = statistics.median(float(r["p50_us"]) for r in blocks[n])
                p99 = max(float(r["p99_us"]) for r in blocks[n])
                missed = sum(int(r["missed"]) for r in blocks[n])
                total = sum(int(r["measured_blocks"]) for r in blocks[n])
                dl = float(blocks[n][0]["deadline_us"])
                print(f"   N={n:4d} mean {m:8.1f}  p50 {p50:8.1f}  p99(worst run) {p99:8.1f}  "
                      f"deadline {dl:7.1f}  p50/deadline {100 * p50 / dl:5.1f}%  missed {missed}/{total}  "
                      f"fixed share {100 * a / m:4.1f}%")
                out.append([int(time.time()), tag, model, n, round(m, 2), p50, p99, dl, missed, total,
                            round(a, 2), round(b, 4)])
    append("lowlat_fit.csv", ["unix_time", "tag", "model", "block", "mean_us", "p50_us", "p99_us_worst_run",
                              "deadline_us", "missed", "blocks", "fit_fixed_us", "fit_per_sample_us"], out)


def ticks(paths, factor=float(os.environ.get("FACTOR", "1.5"))):
    out = []
    for p in paths:
        rows = [r for r in csv.DictReader(open(p)) if r["warmup"] == "0"]
        e = [int(r["elapsed_ns"]) / 1e3 for r in rows]
        t = [int(r["start_ns"]) / 1e9 for r in rows]
        dl = int(rows[0]["deadline_ns"]) / 1e3
        med = statistics.median(e)
        spikes = [i for i, x in enumerate(e) if x > factor * med]
        gaps = collections.Counter(round(t[b] - t[a], 3) for a, b in zip(spikes, spikes[1:]) if t[b] - t[a] > 0.002)
        top_gap, top_count = gaps.most_common(1)[0] if gaps else (0, 0)
        excess = statistics.mean(e[i] - med for i in spikes) if spikes else 0.0
        missed = sum(1 for x in e if x > dl)
        dur = t[-1] - t[0]
        print(f"{os.path.basename(p)}: median {med:.1f} us, deadline {dl:.1f}, spikes>{factor}x {len(spikes)} "
              f"({len(spikes) / dur:.1f}/s), top spacing {top_gap} s ({top_count}x), mean excess {excess:.1f} us, "
              f"missed {missed}")
        out.append([int(time.time()), p, len(e), round(med, 2), dl, factor, len(spikes), round(len(spikes) / dur, 2),
                    top_gap, top_count, round(excess, 2), missed])
    append("lowlat_ticks.csv", ["unix_time", "file", "blocks", "median_us", "deadline_us", "factor", "spikes",
                                "spikes_per_s", "top_spacing_s", "top_spacing_count", "mean_excess_us", "missed"], out)


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("fit", "ticks"):
        print(__doc__)
        sys.exit(2)
    (fit if sys.argv[1] == "fit" else ticks)(sys.argv[2:])
