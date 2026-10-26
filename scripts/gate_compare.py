#!/usr/bin/env python3
"""Noise in silence, gated vs ungated, from `live --record` WAVs (left = dry input before the gate,
right = amp output). For each file, take the N quietest 0.1 s windows by dry input level and report the amp
output's audible (AC) level and its DC offset there, plus the amp level while playing. Needs numpy.

Usage: python3 scripts/gate_compare.py results-pi/recordings/live_full_pipe_64x2.wav \
           results-pi/recordings/live_full_pipe_64x2_gate50.wav
Writes $RESULTS_DIR/gate_compare.csv (default results/).
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hum_check  # noqa: E402  (WAV reader)

RESULTS = os.environ.get("RESULTS_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
QUIET_WINDOWS = 10


def db(v):
    return float(20 * np.log10(v + 1e-12))


def main():
    rows = []
    for path in sys.argv[1:]:
        sr, x = hum_check.read(path)
        w = sr // 10
        n = len(x) // w
        dry = x[: n * w, 0].reshape(n, w)
        amp = x[: n * w, 1].reshape(n, w)
        dry_rms = np.sqrt((dry ** 2).mean(1))
        q = np.argsort(dry_rms)[:QUIET_WINDOWS]
        ac = np.sqrt(((amp[q] - amp[q].mean(1, keepdims=True)) ** 2).mean(1))
        dc = np.abs(amp[q].mean(1))
        playing = dry_rms > 10 ** (-35 / 20)
        amp_rms = np.sqrt((amp ** 2).mean(1))
        rows.append({
            "file": os.path.basename(path), "seconds": round(len(x) / sr, 1),
            "quiet_windows": QUIET_WINDOWS,
            "quiet_dry_rms_dbfs": round(db(dry_rms[q].mean()), 1),
            "quiet_amp_ac_dbfs": round(db(ac.mean()), 1),
            "quiet_amp_dc_dbfs": round(db(dc.mean()), 1),
            "playing_windows": int(playing.sum()),
            "playing_amp_rms_dbfs": round(db(amp_rms[playing].mean()), 1) if playing.any() else "",
        })
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "gate_compare.csv")
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
