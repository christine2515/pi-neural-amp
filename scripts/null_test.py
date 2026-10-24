#!/usr/bin/env python3
"""Null test: how different are two renders of the same input? Standard library only.

Use it to check that an optimization (e.g. -ffast-math) doesn't change the sound:
    ./build/bench ... --render results/render_ref.wav
    ./build-fast/bench ... --render results/render_fast.wav
    python3 scripts/null_test.py results/render_ref.wav results/render_fast.wav --label fastmath

Appends a row to results/null_tests.csv: max abs difference, RMS difference, and the residual level
relative to the reference in dB (-120 dB or lower is far below audibility; 16-bit noise is about -96 dB).
"""
import argparse
import csv
import math
import os
import struct
import time

# RESULTS_DIR=results-pi reads a copy pulled from the Pi.
RESULTS = os.environ.get("RESULTS_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def read_wav(path):
    """Mono/first-channel samples as floats from PCM 16/24/32-bit or IEEE float32 WAV."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a WAV file")
    pos, fmt, samples = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            tag, channels, rate = struct.unpack("<HHI", body[:8])
            bits = struct.unpack("<H", body[14:16])[0]
            if tag == 0xFFFE:  # extensible: real format code is the start of the sub-format GUID
                tag = struct.unpack("<H", body[24:26])[0]
            fmt = (tag, channels, rate, bits)
        elif cid == b"data":
            samples = body
        pos += 8 + size + (size & 1)
    if fmt is None or samples is None:
        raise ValueError(f"{path}: missing fmt or data chunk")
    tag, channels, rate, bits = fmt
    width = bits // 8
    frame = width * channels
    n = len(samples) // frame
    out = []
    for i in range(n):
        s = samples[i * frame:i * frame + width]
        if tag == 3 and bits == 32:
            out.append(struct.unpack("<f", s)[0])
        elif tag == 1 and bits == 16:
            out.append(struct.unpack("<h", s)[0] / 32768.0)
        elif tag == 1 and bits == 24:
            v = int.from_bytes(s, "little", signed=True)
            out.append(v / 8388608.0)
        elif tag == 1 and bits == 32:
            out.append(struct.unpack("<i", s)[0] / 2147483648.0)
        else:
            raise ValueError(f"{path}: unsupported format tag={tag} bits={bits}")
    return rate, out


def db(x):
    return 20 * math.log10(x) if x > 0 else float("-inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=os.path.join(RESULTS, "null_tests.csv"))
    args = ap.parse_args()

    rate_a, a = read_wav(args.reference)
    rate_b, b = read_wav(args.candidate)
    if rate_a != rate_b:
        raise SystemExit(f"sample rates differ: {rate_a} vs {rate_b}")
    n = min(len(a), len(b))
    if len(a) != len(b):
        print(f"warning: lengths differ ({len(a)} vs {len(b)}); comparing first {n} samples")
    diff = [a[i] - b[i] for i in range(n)]
    max_abs = max((abs(d) for d in diff), default=0.0)
    rms_diff = math.sqrt(sum(d * d for d in diff) / n) if n else 0.0
    rms_ref = math.sqrt(sum(x * x for x in a[:n]) / n) if n else 0.0
    rel_db = db(rms_diff / rms_ref) if rms_ref > 0 else float("-inf")

    row = {
        "unix_time": int(time.time()), "label": args.label,
        "reference": os.path.basename(args.reference), "candidate": os.path.basename(args.candidate),
        "samples": n, "max_abs_diff": f"{max_abs:.3e}", "max_abs_diff_dbfs": f"{db(max_abs):.1f}",
        "rms_diff": f"{rms_diff:.3e}", "residual_db_rel_ref": f"{rel_db:.1f}",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    exists = os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(", ".join(f"{k}={v}" for k, v in row.items()))


if __name__ == "__main__":
    main()
