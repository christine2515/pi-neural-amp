#!/usr/bin/env python3
"""Hum analysis for short silent recordings from the Scarlett (arecord S32_LE stereo or float WAVs).

For each file: broadband RMS, and the level of the 60 Hz mains series (60/120/180/240 Hz) and the
2.4-2.9 kHz buzz cluster (60 Hz harmonics that the amp model boosts), all in dBFS on the chosen channel.
Appends rows to $RESULTS_DIR/hum_check.csv (default results/).

Usage: python3 scripts/hum_check.py --channel 1 --label A_vol0 results/hum/A.wav [more.wav ...]
Needs numpy.
"""
import argparse
import csv
import os
import struct
import time

import numpy as np

RESULTS = os.environ.get("RESULTS_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def read(path):
    with open(path, "rb") as f:
        d = f.read()
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(d):
        cid, size = d[pos:pos + 4], struct.unpack("<I", d[pos + 4:pos + 8])[0]
        body = d[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            tag, ch, sr = struct.unpack("<HHI", body[:8])
            bits = struct.unpack("<H", body[14:16])[0]
            fmt = (tag, ch, sr, bits)
        elif cid == b"data":
            data = body
        pos += 8 + size + (size & 1)
    tag, ch, sr, bits = fmt
    if tag == 3:
        x = np.frombuffer(data, dtype="<f4")
    elif bits == 32:
        x = np.frombuffer(data, dtype="<i4") / 2147483648.0
    elif bits == 16:
        x = np.frombuffer(data, dtype="<i2") / 32768.0
    else:
        raise SystemExit(f"{path}: unsupported format")
    return sr, x.reshape(-1, ch)


def db(p):
    return 10 * np.log10(p + 1e-30)


def analyze(path, channel):
    sr, x = read(path)
    s = x[:, channel].astype(np.float64)
    s = s - s.mean()
    win = np.hanning(len(s))
    X = np.abs(np.fft.rfft(s * win)) ** 2 / (np.sum(win ** 2) * len(s) / 2)  # ≈ power per bin, sine-calibrated
    fr = np.fft.rfftfreq(len(s), 1 / sr)

    def tone(f):  # peak power within ±2 Hz, as dBFS of a sine
        m = (fr > f - 2) & (fr < f + 2)
        return db(X[m].max() / 2 * 2)

    def band(lo, hi):
        m = (fr >= lo) & (fr < hi)
        return db(X[m].sum() / 2 * 2)

    return {
        "file": os.path.basename(path), "seconds": round(len(s) / sr, 1),
        "rms_dbfs": round(10 * np.log10(np.mean(s ** 2) + 1e-30), 1),
        "hum60_dbfs": round(tone(60), 1), "hum120_dbfs": round(tone(120), 1), "hum180_dbfs": round(tone(180), 1),
        "hum_series_dbfs": round(db(sum(10 ** (tone(f) / 10) for f in (60, 120, 180, 240, 300))), 1),
        "buzz_2k4_2k9_dbfs": round(band(2400, 2900), 1),
        "hf_2k_8k_dbfs": round(band(2000, 8000), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--channel", type=int, default=1)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    out = os.path.join(RESULTS, "hum_check.csv")
    os.makedirs(RESULTS, exist_ok=True)
    rows = []
    for f in args.files:
        r = analyze(f, args.channel)
        r = {"unix_time": int(time.time()), "label": args.label, "channel": args.channel, **r}
        rows.append(r)
        print(", ".join(f"{k}={v}" for k, v in r.items()))
    new = not os.path.exists(out)
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
