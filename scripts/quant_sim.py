#!/usr/bin/env python3
"""Fixed-point (int16 / int8 / int32) simulation of the NAM A2 model, to get the null-test residual
of an integer kernel before writing one. Needs NumPy (use a venv if the system python's is broken).

    python3 scripts/quant_sim.py                      # Lite + Full, all schemes, 10 s of audio/di.wav
    python3 scripts/quant_sim.py --wav-dir /tmp/q     # also write float + quantized renders (WAV)
    python3 scripts/quant_sim.py --check-render ref.wav --check-slim 1   # float sim vs a bench render

What it models (see docs/beyond-fpu.md, question 3):
  - Weights: per-layer, per-output-channel scale (arbitrary real scale, max-abs calibrated), rounded to
    `wbits` signed integers. Activation scales of the inputs are folded into the weights first.
  - Activations: power-of-two scales (per tensor or per channel), max-abs calibrated on the whole
    calibration file, rounded and saturated to `abits`.
  - Conv and 1x1 accumulate exactly in integers (int32 in the kernel; the script reports the largest
    accumulator seen so int32 overflow is visible as headroom < 0 bits).
  - Bias and input mixin are added in accumulator units (int32; mixin as cond x a Q31 weight, rounded);
    the input (cond) is `cbits` wide at full scale [-1, 1), as an audio interface delivers it.
  - LeakyReLU on the int32 accumulator, slope 0.01 as Q31 (default) or Q15 (vqrdmulh.s16, 328/32768).
  - Requantize accumulator -> activation with rounding (vqrdmulh/vqrshrn style; multiplier precision
    is treated as exact).
  - Residual stream: `resid=16` keeps it in the activation format (rounded and saturated every layer);
    `resid=32` keeps 16 extra fractional bits (int32) and only the conv input/history is narrowed.
  - Head sum: int32 with power-of-two scales (exact); head conv (1% of the MACs) in float.
Reference = the same network in float64. Residual = 20*log10(rms(diff)/rms(ref)), as scripts/null_test.py.
Results append to results-pi/quant_sim.csv.
"""
import argparse
import csv
import json
import math
import os
import struct
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
KS = [6] * 14 + [15, 15] + [6] * 7
DS = [1, 3, 7, 17, 41, 101, 239] * 2 + [1, 13] + [1, 3, 7, 17, 41, 101, 239]
HEAD_K = 16
SLOPE = 0.01
PREWARM = 1 + sum((k - 1) * d for k, d in zip(KS, DS)) + HEAD_K - 1


# ----------------------------------------------------------------------------------------------- I/O
def read_wav(path):
    with open(path, "rb") as f:
        data = f.read()
    pos, fmt, body = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        chunk = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            tag, ch, rate = struct.unpack("<HHI", chunk[:8])
            bits = struct.unpack("<H", chunk[14:16])[0]
            if tag == 0xFFFE:
                tag = struct.unpack("<H", chunk[24:26])[0]
            fmt = (tag, ch, rate, bits)
        elif cid == b"data":
            body = chunk
        pos += 8 + size + (size & 1)
    tag, ch, rate, bits = fmt
    if tag == 3 and bits == 32:
        x = np.frombuffer(body, dtype="<f4")
    elif tag == 1 and bits == 16:
        x = np.frombuffer(body, dtype="<i2") / 32768.0
    elif tag == 1 and bits == 24:
        b = np.frombuffer(body, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = np.where(v >= 1 << 23, v - (1 << 24), v) / 8388608.0
    elif tag == 1 and bits == 32:
        x = np.frombuffer(body, dtype="<i4") / 2147483648.0
    else:
        raise SystemExit(f"{path}: unsupported WAV format")
    return rate, np.asarray(x, dtype=np.float64)[::ch]


def write_wav_f32(path, x, rate=48000):
    d = np.asarray(x, dtype="<f4").tobytes()
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + len(d)) + b"WAVE")
        f.write(b"fmt " + struct.pack("<IHHIIHH", 16, 3, 1, rate, rate * 4, 4, 32))
        f.write(b"data" + struct.pack("<I", len(d)) + d)


# ------------------------------------------------------------------------------------------- model
def load_model(path, channels):
    d = json.load(open(path))
    subs = d["config"]["submodels"] if d["architecture"] == "SlimmableContainer" else [{"model": d}]
    for s in subs:
        m = s["model"]
        if m["config"]["layers"][0]["channels"] == channels:
            w = np.array(m["weights"], dtype=np.float64)
            break
    else:
        raise SystemExit(f"no submodel with {channels} channels")
    C, pos = channels, 0

    def take(n):
        nonlocal pos
        v = w[pos:pos + n]
        pos += n
        return v

    p = {"C": C, "rech": take(C), "layers": []}
    for K in KS:
        conv = take(C * C * K).reshape(C, C, K)  # [out, in, tap]
        p["layers"].append({
            "W": [conv[:, :, k].copy() for k in range(K)], "b": take(C), "mix": take(C),
            "L": take(C * C).reshape(C, C), "lb": take(C)})
    p["hw"] = take(C * HEAD_K).reshape(C, HEAD_K)
    p["hb"] = take(1)[0]
    p["hscale"] = take(1)[0]
    assert pos == len(w), (pos, len(w))
    return p


def shift(x, n):
    """x delayed by n samples (zeros before the start), along the last axis."""
    if n == 0:
        return x
    out = np.zeros_like(x)
    out[..., n:] = x[..., :-n]
    return out


def head(p, hs, dtype=np.float64):
    y = np.full(hs.shape[1], p["hb"], dtype=dtype)
    hw = p["hw"].astype(dtype)
    for k in range(HEAD_K):
        y += hw[:, k] @ shift(hs, HEAD_K - 1 - k)
    return y * dtype(p["hscale"])


def forward_float(p, u, dtype=np.float64, stats=None):
    u = u.astype(dtype)
    x = p["rech"].astype(dtype)[:, None] * u[None, :]
    hs = np.zeros_like(x)
    for li, (L, K, D) in enumerate(zip(p["layers"], KS, DS)):
        if stats is not None:
            stats["x"].append(np.abs(x).max(axis=1))
        z = np.zeros_like(x)
        for k in range(K):
            z += L["W"][k].astype(dtype) @ shift(x, (K - 1 - k) * D)
        z += L["b"].astype(dtype)[:, None] + L["mix"].astype(dtype)[:, None] * u[None, :]
        a = np.where(z >= 0, z, z * dtype(SLOPE))
        if stats is not None:
            stats["a"].append(np.abs(a).max(axis=1))
        hs += a
        x = x + L["L"].astype(dtype) @ a + L["lb"].astype(dtype)[:, None]
    if stats is not None:
        stats["hs"] = np.abs(hs).max()
    return head(p, hs, dtype)


# ------------------------------------------------------------------------------------- fixed point
def pow2_scale(maxabs, bits, headroom_bits=0):
    """Smallest power-of-two step s with maxabs / s <= 2^(bits-1) - 1 (per element of maxabs)."""
    lim = 2.0 ** (bits - 1) - 1
    m = np.maximum(np.asarray(maxabs, dtype=np.float64), 1e-30)
    return 2.0 ** (np.ceil(np.log2(m / lim)) + headroom_bits)


class Counter:
    def __init__(self):
        self.sat = 0
        self.acc_max = 0.0  # largest |int accumulator| seen (conv and 1x1)
        self.hs_max = 0.0


def quant(v, bits, cnt):
    lim = 2.0 ** (bits - 1)
    r = np.round(v)
    cnt.sat += int(np.count_nonzero((r < -lim) | (r > lim - 1)))
    return np.clip(r, -lim, lim - 1)


def qweights(W, bits):
    """Per-output-row scale; W is (out, in) or list of taps. Returns (int weights, row scales)."""
    Ws = W if isinstance(W, list) else [W]
    m = np.max([np.abs(w).max(axis=1) for w in Ws], axis=0)
    s = np.maximum(m, 1e-30) / (2.0 ** (bits - 1) - 1)
    q = [np.round(w / s[:, None]) for w in Ws]
    return (q if isinstance(W, list) else q[0]), s


def forward_fixed(p, u, stats, sc, cnt):
    """sc: dict(wbits, abits, cbits, per_channel, resid, slope, headroom, qlayers). None bits = keep float.
    qlayers (optional set): quantize activations/weights only in these layers (sensitivity sweep)."""
    C = p["C"]
    hr = sc.get("headroom", 0)
    pc = sc["per_channel"]
    qlayers = sc.get("qlayers")
    slope = {"q31": round(SLOPE * 2 ** 31) / 2 ** 31, "q15": round(SLOPE * 2 ** 15) / 2 ** 15, "exact": SLOPE}[sc["slope"]]
    extra = 16 if sc["resid"] == 32 else 0  # extra fractional bits kept in the residual stream

    def bits_at(li, key):
        return sc[key] if (qlayers is None or li in qlayers) else None

    def act_scale(maxabs, ab):
        if ab is None:
            return np.ones(C)
        return pow2_scale(maxabs if pc else np.full(C, maxabs.max()), ab, hr)

    # Input: the interface's full scale, samples in [-1, 1).
    cb = sc["cbits"]
    u_used = u.copy() if cb is None else quant(u * 2.0 ** (cb - 1), cb, cnt) * 2.0 ** -(cb - 1)

    xr = p["rech"][:, None] * u_used[None, :]  # residual stream, real values on its grid
    ab0 = bits_at(0, "abits")
    if ab0 is not None:
        g = act_scale(stats["x"][0], ab0)[:, None] / 2.0 ** extra
        xr = quant(xr / g, ab0 + extra, cnt) * g
    hs = np.zeros_like(xr)
    s_h = None
    for li, (L, K, D) in enumerate(zip(p["layers"], KS, DS)):
        ab, wb = bits_at(li, "abits"), bits_at(li, "wbits")
        # conv input (history) in the activation format
        sx = act_scale(stats["x"][li], ab)
        xq = xr if ab is None else quant(xr / sx[:, None], ab, cnt)
        # weights with the input scales folded in, per-output-channel scale
        Wf = [w * sx[None, :] for w in L["W"]]
        Wq, sw = (Wf, np.ones(C)) if wb is None else qweights(Wf, wb)
        acc = np.zeros_like(xr)
        for k in range(K):
            acc += Wq[k] @ shift(xq, (K - 1 - k) * D)
        if wb is not None and ab is not None:
            cnt.acc_max = max(cnt.acc_max, float(np.abs(acc).max()))
        # bias + mixin, in accumulator units (int32; mixin = cond x Q31 weight, rounded)
        if wb is None:
            acc += L["b"][:, None] + L["mix"][:, None] * u_used[None, :]
            acc = np.where(acc >= 0, acc, acc * slope)
        else:
            acc += np.round(L["b"] / sw)[:, None] + np.round(L["mix"][:, None] * u_used[None, :] / sw[:, None])
            acc = np.where(acc >= 0, acc, np.round(acc * slope))  # LeakyReLU on the int32 accumulator
        z = acc * sw[:, None]  # real value
        sa = act_scale(stats["a"][li], ab)
        aq = z if ab is None else quant(z / sa[:, None], ab, cnt)
        hs += aq * sa[:, None]
        if ab is not None:
            s_h = sa.min() if s_h is None else min(s_h, sa.min())
        # 1x1 + bias, activation scales folded into the weights
        Lf = L["L"] * sa[None, :]
        if wb is None:
            r = Lf @ aq + L["lb"][:, None]
        else:
            Lq, sl = qweights(Lf, wb)
            acc2 = Lq @ aq + np.round(L["lb"] / sl)[:, None]
            if ab is not None:
                cnt.acc_max = max(cnt.acc_max, float(np.abs(acc2).max()))
            r = acc2 * sl[:, None]
        xr = xr + r
        abn = bits_at(li + 1, "abits") if li + 1 < len(KS) else ab
        if abn is not None:
            nxt = act_scale(stats["x"][li + 1] if li + 1 < len(KS) else stats["x"][li], abn)
            g = nxt[:, None] / 2.0 ** extra
            xr = quant(xr / g, abn + extra, cnt) * g
    if s_h is not None:
        cnt.hs_max = float(np.abs(hs).max() / s_h)
    return head(p, hs)


def rtz32(v):
    """Round float64 values to float32 toward zero (the QPU's rounding mode)."""
    f = v.astype(np.float32)
    over = np.abs(f.astype(np.float64)) > np.abs(v)
    f[over] = np.nextafter(f[over], np.float32(0))
    return f.astype(np.float64)


def forward_rtz(p, u):
    """float32 with every multiply and add rounded toward zero, op by op (VideoCore IV QPU semantics)."""
    u = rtz32(u)
    x = rtz32(p["rech"][:, None] * u[None, :])
    hs = np.zeros_like(x)
    r = lambda w: rtz32(np.asarray(w, dtype=np.float64))
    for L, K, D in zip(p["layers"], KS, DS):
        z = np.repeat(r(L["b"])[:, None], x.shape[1], axis=1)
        for k in range(K):
            xs = shift(x, (K - 1 - k) * D)
            W = r(L["W"][k])
            for j in range(x.shape[0]):
                z = rtz32(z + rtz32(W[:, j:j + 1] * xs[j:j + 1, :]))
        z = rtz32(z + rtz32(r(L["mix"])[:, None] * u[None, :]))
        a = np.where(z >= 0, z, rtz32(z * np.float64(np.float32(SLOPE))))
        hs = rtz32(hs + a)
        Lw = r(L["L"])
        acc = np.repeat(r(L["lb"])[:, None], x.shape[1], axis=1)
        for j in range(x.shape[0]):
            acc = rtz32(acc + rtz32(Lw[:, j:j + 1] * a[j:j + 1, :]))
        x = rtz32(x + acc)
    return head(p, hs)  # head in float64 (1% of the MACs)


def lowrank(p, r):
    """Each layer's dilated conv as a rank-r factorization of its (out x in*K) weight matrix:
    conv into r channels, then an r -> out 1x1. Conv MACs per frame: r*(in*K) + out*r instead of out*in*K."""
    q = {**p, "layers": []}
    for L, K in zip(p["layers"], KS):
        Wcat = np.concatenate(L["W"], axis=1)  # (out, in*K)
        U, S, Vt = np.linalg.svd(Wcat, full_matrices=False)
        Wr = (U[:, :r] * S[:r]) @ Vt[:r]
        C = p["C"]
        q["layers"].append({**L, "W": [Wr[:, k * C:(k + 1) * C] for k in range(K)]})
    return q


SCHEMES = [
    # Diagnostics: one source of error at a time.
    ("input 16-bit only (rest float)", dict(wbits=None, abits=None, cbits=16, per_channel=False, resid=16, slope="exact")),
    ("W16 only (acts float)", dict(wbits=16, abits=None, cbits=None, per_channel=False, resid=16, slope="exact")),
    ("A16 only (weights float, input 24-bit)", dict(wbits=None, abits=16, cbits=24, per_channel=True, resid=32, slope="exact")),
    # Candidate int16 kernels (vmlal.s16: int16 x int16 -> int32).
    ("W16A16 per-tensor, resid16", dict(wbits=16, abits=16, cbits=24, per_channel=False, resid=16, slope="q31")),
    ("W16A16 per-channel, resid16", dict(wbits=16, abits=16, cbits=24, per_channel=True, resid=16, slope="q31")),
    ("W16A16 per-tensor, resid32", dict(wbits=16, abits=16, cbits=24, per_channel=False, resid=32, slope="q31")),
    ("W16A16 per-channel, resid32", dict(wbits=16, abits=16, cbits=24, per_channel=True, resid=32, slope="q31")),
    ("W16A16 per-channel, resid32, input 16-bit", dict(wbits=16, abits=16, cbits=16, per_channel=True, resid=32, slope="q31")),
    ("W16A16 per-channel, resid32, Q15 slope", dict(wbits=16, abits=16, cbits=24, per_channel=True, resid=32, slope="q15")),
    ("W16A16 per-channel, resid32, 1 bit headroom", dict(wbits=16, abits=16, cbits=24, per_channel=True, resid=32, slope="q31", headroom=1)),
    # int32-safe: the W16A16 accumulators exceed int32 on this file, so give up bits.
    ("W13A16 per-channel, resid32 (int32-safe)", dict(wbits=13, abits=16, cbits=24, per_channel=True, resid=32, slope="q31")),
    ("W16A13 per-channel, resid32 (int32-safe)", dict(wbits=16, abits=13, cbits=24, per_channel=True, resid=32, slope="q31")),
    ("W14A15 per-channel, resid32 (int32-safe)", dict(wbits=14, abits=15, cbits=24, per_channel=True, resid=32, slope="q31")),
    # Mixed precision: the first layers (most sensitive, see the layer sweep) stay float.
    ("W16A16 per-channel, resid32, layers 0-2 float", dict(wbits=16, abits=16, cbits=24, per_channel=True, resid=32, slope="q31", qlayers=set(range(3, 23)))),
    ("W16A16 per-channel, resid32, layers 0-6 float", dict(wbits=16, abits=16, cbits=24, per_channel=True, resid=32, slope="q31", qlayers=set(range(7, 23)))),
    ("W13A16 per-channel, resid32 (int32-safe), layers 0-6 float", dict(wbits=13, abits=16, cbits=24, per_channel=True, resid=32, slope="q31", qlayers=set(range(7, 23)))),
    # Narrower formats (vmlal.s8 = int8 x int8 -> int16 runs 2x faster than vmlal.s16).
    ("W12A12 per-channel, resid32", dict(wbits=12, abits=12, cbits=24, per_channel=True, resid=32, slope="q31")),
    ("W8A16 per-channel, resid32", dict(wbits=8, abits=16, cbits=24, per_channel=True, resid=32, slope="q31")),
    ("W16A8 per-channel, resid32", dict(wbits=16, abits=8, cbits=24, per_channel=True, resid=32, slope="q31")),
    ("W8A8 per-channel, resid32 (vmlal.s8)", dict(wbits=8, abits=8, cbits=24, per_channel=True, resid=32, slope="q31")),
]


def residual(ref, cand):
    d = ref - cand
    rms_ref = math.sqrt(float(np.mean(ref * ref)))
    rms_d = math.sqrt(float(np.mean(d * d)))
    mx = float(np.abs(d).max())
    return (20 * math.log10(rms_d / rms_ref) if rms_d > 0 else -math.inf,
            20 * math.log10(mx) if mx > 0 else -math.inf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models/a2_fender_super_reverb.nam"))
    ap.add_argument("--input", default=os.path.join(ROOT, "audio/di.wav"))
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--channels", default="3,8")
    ap.add_argument("--hot-db", type=float, default=None,
                    help="also evaluate on input +N dB (default: raise the whole file to -0.1 dBFS peak, the loudest "
                         "input that doesn't clip at the interface)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results-pi/quant_sim.csv"))
    ap.add_argument("--wav-dir", default="")
    ap.add_argument("--calib-peak-dbfs", type=float, default=None,
                    help="calibrate on the input normalized to this peak (e.g. 0 = a full-scale player), not as recorded")
    ap.add_argument("--schemes", default="", help="comma-free substring filter on scheme names")
    ap.add_argument("--check-render", default="", help="bench --render WAV of the first --seconds, to validate the float sim")
    ap.add_argument("--check-slim", default="1")
    ap.add_argument("--layer-sweep", default="W16A16 per-channel, resid32",
                    help="scheme to apply to one layer at a time (others float); empty = skip")
    args = ap.parse_args()

    rate, audio = read_wav(args.input)
    n = int(args.seconds * rate)
    pad = np.zeros(PREWARM)
    rows = []
    for C in [int(c) for c in args.channels.split(",")]:
        p = load_model(args.model, C)
        name = {3: "lite", 8: "full"}.get(C, f"c{C}")
        t0 = time.time()
        stats = {"x": [], "a": []}
        cal = audio
        if args.calib_peak_dbfs is not None:
            cal = audio * (10 ** (args.calib_peak_dbfs / 20) / np.abs(audio).max())
        forward_float(p, np.concatenate([pad, cal]), np.float64, stats)  # calibration: whole file
        print(f"[{name}] calibration on {len(audio) / rate:.1f} s: {time.time() - t0:.1f} s; "
              f"max|x| per layer {min(s.max() for s in stats['x']):.3g}..{max(s.max() for s in stats['x']):.3g}, "
              f"max|a| {min(s.max() for s in stats['a']):.3g}..{max(s.max() for s in stats['a']):.3g}, "
              f"max|head sum| {stats['hs']:.3g}")
        u = np.concatenate([pad, audio[:n]])
        ref = forward_float(p, u)[PREWARM:]
        f32 = forward_float(p, u, np.float32)[PREWARM:].astype(np.float64)
        if args.check_render and str({3: "0.25", 8: "1"}[C]) == args.check_slim:
            _, r = read_wav(args.check_render)
            m = min(len(r), n)
            rd, mx = residual(r[:m], ref[:m])
            print(f"[{name}] float64 sim vs bench render: {rd:.1f} dB rel, max diff {mx:.1f} dBFS")
            rows.append(dict(model=name, scheme="float64 sim vs C++ bench render (validation)", input_gain_db=0,
                             residual_db=f"{rd:.1f}", max_abs_diff_dbfs=f"{mx:.1f}"))
        rd, mx = residual(ref, f32)
        rows.append(dict(model=name, scheme="float32 (numpy) vs float64", input_gain_db=0,
                         residual_db=f"{rd:.1f}", max_abs_diff_dbfs=f"{mx:.1f}"))
        print(f"[{name}] float32: {rd:.1f} dB")
        if args.wav_dir:
            os.makedirs(args.wav_dir, exist_ok=True)
            write_wav_f32(os.path.join(args.wav_dir, f"{name}_float64.wav"), ref, rate)
        for sname, sc in SCHEMES:
            if args.schemes and args.schemes not in sname:
                continue
            cnt = Counter()
            q = forward_fixed(p, u, stats, sc, cnt)[PREWARM:]
            rd, mx = residual(ref, q)
            hb = 31 - math.log2(cnt.acc_max) if cnt.acc_max > 0 else float("nan")
            hh = 31 - math.log2(cnt.hs_max) if cnt.hs_max > 0 else float("nan")
            row = dict(model=name, scheme=sname, input_gain_db=0, residual_db=f"{rd:.1f}",
                       max_abs_diff_dbfs=f"{mx:.1f}", saturations=cnt.sat,
                       acc_int32_headroom_bits=f"{hb:.1f}", headsum_int32_headroom_bits=f"{hh:.1f}")
            if args.calib_peak_dbfs is not None:
                row["scheme"] += f" (calibrated at {args.calib_peak_dbfs:g} dBFS peak)"
            rows.append(row)
            print(f"[{name}] {sname:45s} {rd:7.1f} dB rel  max {mx:7.1f} dBFS  sat {cnt.sat}  acc headroom {hb:.1f} b")
            if args.wav_dir and sname == "W16A16 per-channel, resid32":
                write_wav_f32(os.path.join(args.wav_dir, f"{name}_w16a16_pc_r32.wav"), q, rate)
        if args.calib_peak_dbfs is None and not args.schemes:
            # QPU float semantics (round toward zero on every op).
            rd, mx = residual(ref, forward_rtz(p, u)[PREWARM:])
            rows.append(dict(model=name, scheme="fp32 round-toward-zero per op (VideoCore IV QPU)", input_gain_db=0,
                             residual_db=f"{rd:.1f}", max_abs_diff_dbfs=f"{mx:.1f}"))
            print(f"[{name}] fp32 RTZ (QPU): {rd:.1f} dB")
            # Fewer MACs: low-rank conv weights (float).
            for r in range(1, C):
                rd, mx = residual(ref, forward_float(lowrank(p, r), u)[PREWARM:])
                macs = sum(r * (C * K) + C * r for K in KS) + len(KS) * (C * C + C) + C * HEAD_K + C
                full = sum(C * C * K for K in KS) + len(KS) * (C * C + C) + C * HEAD_K + C
                rows.append(dict(model=name, scheme=f"low-rank conv r={r} ({macs} vs {full} MAC/sample, float)",
                                 input_gain_db=0, residual_db=f"{rd:.1f}", max_abs_diff_dbfs=f"{mx:.1f}"))
                print(f"[{name}] low-rank r={r}: {macs}/{full} MAC/sample, {rd:.1f} dB")
        # Sensitivity: the scheme in one layer only, the rest float.
        if args.layer_sweep and args.calib_peak_dbfs is None:
            base = dict(dict(SCHEMES)[args.layer_sweep], cbits=None)
            for li in range(len(KS)):
                cnt = Counter()
                q = forward_fixed(p, u, stats, dict(base, qlayers={li}), cnt)[PREWARM:]
                rd, mx = residual(ref, q)
                rows.append(dict(model=name, scheme=f"{args.layer_sweep}, layer {li} only (K={KS[li]}, D={DS[li]})",
                                 input_gain_db=0, residual_db=f"{rd:.1f}", max_abs_diff_dbfs=f"{mx:.1f}",
                                 saturations=cnt.sat))
                print(f"[{name}] layer {li:2d} only: {rd:.1f} dB")
        # Hotter input than the calibration saw (calibration stays at the nominal level).
        if args.calib_peak_dbfs is not None:
            continue
        hot_db = args.hot_db if args.hot_db is not None else -0.1 - 20 * math.log10(np.abs(audio).max())
        g = 10 ** (hot_db / 20)
        uh = np.concatenate([pad, g * audio[:n]])
        refh = forward_float(p, uh)[PREWARM:]
        for sname in ("W16A16 per-channel, resid32", "W16A16 per-channel, resid32, 1 bit headroom"):
            cnt = Counter()
            q = forward_fixed(p, uh, stats, dict(SCHEMES)[sname], cnt)[PREWARM:]
            rd, mx = residual(refh, q)
            rows.append(dict(model=name, scheme=sname, input_gain_db=f"{hot_db:.1f}", residual_db=f"{rd:.1f}",
                             max_abs_diff_dbfs=f"{mx:.1f}", saturations=cnt.sat))
            print(f"[{name}] +{hot_db:.1f} dB input, {sname}: {rd:.1f} dB rel, sat {cnt.sat}")

    fields = ["unix_time", "model", "scheme", "input_gain_db", "seconds", "residual_db", "max_abs_diff_dbfs",
              "saturations", "acc_int32_headroom_bits", "headsum_int32_headroom_bits"]
    exists = os.path.exists(args.out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({"unix_time": int(time.time()), "seconds": args.seconds, **r})
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
