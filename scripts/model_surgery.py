#!/usr/bin/env python3
"""Accuracy cost of model-side shortcuts for A2-Full, without retraining (docs/low-latency-plan.md, Q6).

Writes modified copies of an A2 SlimmableContainer .nam (only the Full submodel's weights change):
  rank<r>     every 8x8 dilated-conv tap matrix replaced by its best rank-r approximation (SVD).
              A rank-r tap costs 16r MACs instead of 64, so only r <= 3 saves arithmetic.
  l1rank<r>   same for the 8x8 layer1x1 matrices only.
  prune<c>    the c weakest channels (smallest summed |weight| of their conv/1x1 rows and columns)
              zeroed everywhere: what dropping them from an 8-channel kernel would sound like
              without retraining.
Render each with bench --slim 1 and compare against the unmodified Full render with null_test.py.
Needs numpy (Mac only; nothing here runs on the Pi).

Usage: python3 scripts/model_surgery.py models/a2_fender_super_reverb.nam OUTDIR rank2 rank3 prune1
"""
import copy
import json
import os
import sys

import numpy as np

KS = [6] * 14 + [15, 15] + [6] * 7
C = 8


def layout(weights_len):
    """Offsets of each Full tensor in the flat weight list (NAM loader order)."""
    off = 0
    parts = {"rechannel": (off, C)}
    off += C
    layers = []
    for k in KS:
        L = {}
        L["conv"] = (off, C * C * k); off += C * C * k  # order: out i, in j, tap k
        L["conv_b"] = (off, C); off += C
        L["mixin"] = (off, C); off += C
        L["l1x1"] = (off, C * C); off += C * C  # order: out i, in j
        L["l1x1_b"] = (off, C); off += C
        layers.append(L)
    parts["layers"] = layers
    parts["head"] = (off, C * 16); off += C * 16
    off += 2  # head bias, head scale
    assert off == weights_len, (off, weights_len)
    return parts


def low_rank(m, r):
    u, s, vt = np.linalg.svd(m)
    return (u[:, :r] * s[:r]) @ vt[:r]


def apply(w, how):
    w = np.array(w, dtype=np.float64)
    P = layout(len(w))
    if how.startswith("rank") or how.startswith("l1rank"):
        r = int(how.split("rank")[1])
        for li, L in enumerate(P["layers"]):
            k = KS[li]
            if how.startswith("rank"):
                o, n = L["conv"]
                t = w[o:o + n].reshape(C, C, k)  # [i, j, k]
                for kk in range(k):
                    t[:, :, kk] = low_rank(t[:, :, kk], r)
                w[o:o + n] = t.reshape(-1)
            else:
                o, n = L["l1x1"]
                w[o:o + n] = low_rank(w[o:o + n].reshape(C, C), r).reshape(-1)
    elif how.startswith("prune"):
        c = int(how[len("prune"):])
        score = np.zeros(C)
        for li, L in enumerate(P["layers"]):
            k = KS[li]
            o, n = L["conv"]
            t = np.abs(w[o:o + n].reshape(C, C, k))
            score += t.sum(axis=(1, 2)) + t.sum(axis=(0, 2))
            o, n = L["l1x1"]
            m = np.abs(w[o:o + n].reshape(C, C))
            score += m.sum(axis=0) + m.sum(axis=1)
        drop = list(np.argsort(score)[:c])
        o, n = P["rechannel"]
        w[o + np.array(drop)] = 0
        for li, L in enumerate(P["layers"]):
            k = KS[li]
            o, n = L["conv"]
            t = w[o:o + n].reshape(C, C, k)
            t[drop, :, :] = 0
            t[:, drop, :] = 0
            w[o:o + n] = t.reshape(-1)
            for key in ("conv_b", "mixin", "l1x1_b"):
                o, n = L[key]
                w[o + np.array(drop)] = 0
            o, n = L["l1x1"]
            m = w[o:o + n].reshape(C, C)
            m[drop, :] = 0
            m[:, drop] = 0
            w[o:o + n] = m.reshape(-1)
        o, n = P["head"]
        h = w[o:o + n].reshape(C, 16)
        h[drop, :] = 0
        w[o:o + n] = h.reshape(-1)
        print(f"  {how}: zeroed channels {sorted(int(d) for d in drop)} (scores {np.round(score, 1).tolist()})")
    else:
        raise SystemExit(f"unknown variant {how}")
    return w.tolist()


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(2)
    src, outdir, variants = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(outdir, exist_ok=True)
    base = json.load(open(src))
    subs = base["config"]["submodels"]
    full = max(range(len(subs)), key=lambda i: subs[i]["model"]["config"]["layers"][0]["channels"])
    assert subs[full]["model"]["config"]["layers"][0]["channels"] == C
    for v in variants:
        d = copy.deepcopy(base)
        d["config"]["submodels"][full]["model"]["weights"] = apply(subs[full]["model"]["weights"], v)
        out = os.path.join(outdir, f"{os.path.splitext(os.path.basename(src))[0]}__{v}.nam")
        json.dump(d, open(out, "w"))
        print("wrote", out)


if __name__ == "__main__":
    main()
