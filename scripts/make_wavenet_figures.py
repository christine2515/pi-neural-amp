#!/usr/bin/env python3
"""Figures (and the numbers quoted) for learning/10-wavenet-explained.md.

    <venv>/bin/python scripts/make_wavenet_figures.py            # all figures -> learning/img/
    <venv>/bin/python scripts/make_wavenet_figures.py --only 4,8 # a subset
    <venv>/bin/python scripts/make_wavenet_figures.py --check-render lite.wav --check-slim 0.25

Needs NumPy + matplotlib (use a venv: the system python's NumPy is broken on Christine's Mac).
The model loading and the float64 forward pass come from scripts/quant_sim.py, which matched a C++
`bench --render` to about -130 dB. Everything this script prints is what the doc quotes.

Figures (PNG, 150 dpi, light background, Okabe-Ito-like categorical palette from the dataviz skill):
  wn01_block_diagram       input -> rechannel -> 23 layers -> head sum -> head conv -> output
  wn02_layer_internals     one layer: dilated taps, bias, mixin, LeakyReLU, head sum, 1x1 + residual
  wn03_dilated_timeline    which past samples one output touches; three stacked layers
  wn04_receptive_field     per-layer look-back and cumulative receptive field (6,347 samples)
  wn05_leaky_relu          the activation, with the slope-0.01 region zoomed
  wn06_mac_breakdown       MACs per sample by step, Lite vs Full
  wn07_ring_buffer         per-layer history buffer, ring mode 1 (pow2 + mirror) vs mode 0 (rewind)
  wn08_real_audio_overview 1 s of audio/di.wav: input, Full activations at depth, head sum, output
  wn09_real_audio_zoom     25 ms zoom, Lite: input, activations, head sum, output
  wn10_level_transfer      sine level in vs out (Lite, Full) + real-audio blocks
  wn11_small_signal        impulse response, energy vs time vs receptive field, frequency response
  wn12_optimizations       register tile, two-core pipeline, frame split
"""
import argparse
import json
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import quant_sim as q  # noqa: E402

ROOT = os.path.join(HERE, "..")
MODEL = os.path.join(ROOT, "models/a2_fender_super_reverb.nam")
DI = os.path.join(ROOT, "audio/di.wav")
OUT = os.path.join(ROOT, "learning/img")
RATE = 48000

# Palette (dataviz skill reference palette, light mode).
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#6250d6", "#e34948"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#dcdbd6", "#ffffff"
TINT = {BLUE: "#dce9f9", ORANGE: "#fbe1d6", AQUA: "#d3f0e5", YELLOW: "#fbecc8", VIOLET: "#e3e0f7",
        MAGENTA: "#fae5ed", GREEN: "#d4ead4", RED: "#f9dddd", "gray": "#f0efec"}
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#f7f7f5", "#86b6ef", "#2a78d6", "#123f7a"])

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


# ------------------------------------------------------------------------------------------ model
def load_config():
    d = json.load(open(MODEL))
    cfg = {}
    for s in d["config"]["submodels"]:
        m = s["model"]
        la = m["config"]["layers"][0]
        cfg[la["channels"]] = dict(K=la["kernel_sizes"], D=la["dilations"], head_k=la["head"]["kernel_size"],
                                   n_weights=len(m["weights"]), head_scale_json=m["config"]["head_scale"],
                                   head_scale_weight=m["weights"][-1], max_value=s["max_value"])
        assert la["kernel_sizes"] == q.KS and la["dilations"] == q.DS and la["head"]["kernel_size"] == q.HEAD_K
        assert all(a["type"] == "LeakyReLU" and a["negative_slope"] == q.SLOPE for a in la["activation"])
    return cfg


def forward_capture(p, u):
    """quant_sim.forward_float, keeping every layer's input x, activation a, and the head sum."""
    x = p["rech"][:, None] * u[None, :]
    hs = np.zeros_like(x)
    xs, acts = [], []
    for L, K, D in zip(p["layers"], q.KS, q.DS):
        xs.append(x)
        z = np.zeros_like(x)
        for k in range(K):
            z += L["W"][k] @ q.shift(x, (K - 1 - k) * D)
        z += L["b"][:, None] + L["mix"][:, None] * u[None, :]
        a = np.where(z >= 0, z, z * q.SLOPE)
        acts.append(a)
        hs = hs + a
        x = x + L["L"] @ a + L["lb"][:, None]
    return dict(x=xs, a=acts, hs=hs, y=q.head(p, hs), x_out=x)


def macs(C):
    conv = sum(K * C * C for K in q.KS)
    return dict(conv=conv, l1x1=len(q.KS) * C * C, mixin=len(q.KS) * C, head=q.HEAD_K * C, rech=C)


def params(C):
    return C + sum(K * C * C + C + C + C * C + C for K in q.KS) + q.HEAD_K * C + 1 + 1


# --------------------------------------------------------------------------------------- helpers
def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + ".png")
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {os.path.relpath(path, ROOT)}")


def box(ax, x, y, w, h, text, color=BLUE, fs=11, bold=False, tint=True, ls="-", tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=TINT.get(color, color) if tint else SURF, ec=color, lw=1.5, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc)


def arrow(ax, xy0, xy1, color=INK2, lw=1.4, style="-|>", ls="-", rad=0.0, ms=12):
    ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def diagram_axes(fig, rect=None, xlim=(0, 10), ylim=(0, 10)):
    ax = fig.add_axes(rect or [0, 0, 1, 1])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return ax


# ----------------------------------------------------------------------------------- figure 1
def fig_block_diagram():
    fig = plt.figure(figsize=(10.5, 10))
    ax = diagram_axes(fig, xlim=(0, 10.5), ylim=(0, 10))
    ax.text(0.1, 9.75, "A2 WaveNet: the whole model, one block of N frames", fontsize=14, fontweight="bold")
    ax.text(0.1, 9.42, "Shapes are channels × frames.  Lite: C = 3.  Full: C = 8.  "
            "Same topology, same 23 layers, same dilations.", fontsize=10.5, color=INK2)

    cx, bw = 3.9, 3.0  # main column
    box(ax, cx, 8.55, bw, 0.55, "input  u   (1 × N)", "gray", bold=True)
    box(ax, cx, 7.65, bw, 0.6, "rechannel: x = r · u\n(1→C, no bias)   C × N", BLUE)
    arrow(ax, (cx + bw / 2, 8.55), (cx + bw / 2, 8.25))

    groups = [("layers 0–6", "K = 6,  d = 1, 3, 7, 17, 41, 101, 239", 6.3),
              ("layers 7–13", "K = 6,  d = 1, 3, 7, 17, 41, 101, 239", 4.95),
              ("layers 14–15", "K = 15,  d = 1, 13", 3.6),
              ("layers 16–22", "K = 6,  d = 1, 3, 7, 17, 41, 101, 239", 2.25)]
    gh = 0.95
    prev = 7.65
    for name, sub, y in groups:
        col = ORANGE if "14" in name else BLUE
        box(ax, cx, y, bw, gh, "", col)
        ax.text(cx + bw / 2, y + gh * 0.66, name, ha="center", va="center", fontsize=11.5, fontweight="bold")
        ax.text(cx + bw / 2, y + gh * 0.3, sub, ha="center", va="center", fontsize=9.5, color=INK2)
        arrow(ax, (cx + bw / 2, prev), (cx + bw / 2, y + gh))
        prev = y
    ax.text(cx + bw / 2 + 0.1, 3.6 + gh + 0.12, "x  (C × N)  residual stream", fontsize=9, color=INK2)
    # last layer output x goes nowhere (unused)
    ax.text(cx + bw / 2, 2.25 - 0.28, "final x is not used", ha="center", fontsize=9, color=MUTED, style="italic")

    # mixin bus (left)
    bx = 2.2
    ax.plot([bx, bx], [8.82, 2.25 + gh / 2], color=GREEN, lw=2.2, solid_capstyle="round")
    arrow(ax, (cx, 8.82), (bx, 8.82), color=GREEN, style="-")
    for _, _, y in groups:
        arrow(ax, (bx, y + gh / 2), (cx, y + gh / 2), color=GREEN, lw=1.6)
    ax.text(bx - 0.12, 5.6, "input mixin: the dry input u\nis added inside EVERY layer\n(weights m, C × 1)",
            ha="right", va="center", fontsize=9.5, color=GREEN)

    # head-sum bus (right)
    hx = 8.0
    ax.plot([hx, hx], [6.3 + gh / 2, 1.25], color=VIOLET, lw=2.2, solid_capstyle="round")
    for _, _, y in groups:
        arrow(ax, (cx + bw, y + gh / 2), (hx, y + gh / 2), color=VIOLET, lw=1.6)
    ax.text(hx + 0.15, 5.6, "skip path:\neach layer's activation a\nis added into the\nhead sum  s = Σ a\n(C × N)",
            ha="left", va="center", fontsize=9.5, color=VIOLET)

    box(ax, 6.0, 0.62, 3.2, 0.62, "head conv: K = 16, d = 1, C → 1\n+ bias, × head_scale (0.011273)",
        VIOLET, fs=10)
    arrow(ax, (hx, 1.25), (hx, 1.24))
    box(ax, 6.0, -0.15, 3.2, 0.5, "output  y   (1 × N)", "gray", bold=True)
    arrow(ax, (7.6, 0.62), (7.6, 0.35))
    ax.set_ylim(-0.3, 10)
    save(fig, "wn01_block_diagram")


# ----------------------------------------------------------------------------------- figure 2
def fig_layer_internals():
    fig = plt.figure(figsize=(12, 6.6))
    ax = diagram_axes(fig, xlim=(0, 12), ylim=(0, 6.6))
    ax.text(0.1, 6.35, "One layer (layer i, kernel K, dilation d), for a block of N frames", fontsize=14,
            fontweight="bold")
    ax.text(0.1, 6.02, "Lite C = 3, Full C = 8.  Bottleneck = C here, so every matrix is C × C.", fontsize=10.5,
            color=INK2)

    box(ax, 0.1, 3.4, 1.7, 0.9, "layer input x\nC × N", BLUE, bold=True)
    box(ax, 2.3, 3.0, 2.3, 1.7, "", BLUE)
    ax.text(3.45, 4.42, "history ring buffer", ha="center", fontsize=10.5, fontweight="bold")
    ax.text(3.45, 4.05, "C × (lookback + N)", ha="center", fontsize=10)
    ax.text(3.45, 3.55, "lookback = (K − 1) · d", ha="center", fontsize=10, color=INK2)
    ax.text(3.45, 3.2, "(5 to 1,195 frames)", ha="center", fontsize=9, color=INK2)
    arrow(ax, (1.8, 3.85), (2.3, 3.85))

    # taps
    ty = [5.3, 4.55, 3.0, 2.25]
    labels = ["tap 0:  x(t − (K−1)d)", "tap 1:  x(t − (K−2)d)", "tap K−2:  x(t − d)", "tap K−1:  x(t)"]
    wn = ["W_0", "W_1", "W_(K−2)", "W_(K−1)"]
    for y, lab, w in zip(ty, labels, wn):
        box(ax, 5.1, y - 0.25, 2.25, 0.5, lab + "\n", ORANGE, fs=9.5)
        ax.text(6.22, y - 0.12, f"× {w}  (C × C)", ha="center", fontsize=9, color=INK2)
        arrow(ax, (4.6, 3.85), (5.1, y), color=ORANGE, lw=1.1)
        arrow(ax, (7.35, y), (7.9, 3.85), color=ORANGE, lw=1.1)
    ax.text(6.22, 3.78, "⋮", ha="center", va="center", fontsize=16, color=ORANGE)

    box(ax, 7.9, 3.4, 1.0, 0.9, "Σ\nC × N", ORANGE, bold=True)
    box(ax, 9.3, 3.4, 1.15, 0.9, "+ bias b\n(C)", ORANGE, fs=10)
    arrow(ax, (8.9, 3.85), (9.3, 3.85))
    box(ax, 9.3, 1.95, 1.15, 0.9, "+ m · u\nmixin (C × 1)", GREEN, fs=9.5)
    arrow(ax, (9.875, 3.4), (9.875, 2.85))
    ax.text(10.55, 1.62, "u = raw input\n(1 × N)", fontsize=9, color=GREEN, ha="left")
    arrow(ax, (10.5, 1.75), (10.3, 1.95), color=GREEN)
    box(ax, 9.3, 0.55, 1.15, 0.9, "LeakyReLU\n(0.01)", RED, fs=10)
    arrow(ax, (9.875, 1.95), (9.875, 1.45))
    ax.text(8.6, 1.12, "a  (C × N)", ha="center", fontsize=10, fontweight="bold")

    # head sum
    box(ax, 10.85, 3.4, 1.1, 0.9, "head sum\ns += a", VIOLET, fs=10)
    arrow(ax, (10.45, 1.0), (11.4, 3.4), color=VIOLET, rad=0.35)
    ax.text(11.4, 4.45, "skip path\n(to the head)", ha="center", fontsize=9, color=VIOLET)

    # 1x1 + residual
    box(ax, 5.6, 0.55, 2.3, 0.9, "1×1 conv:  L · a + c\n(C × C, bias C)", BLUE, fs=10)
    arrow(ax, (9.3, 1.0), (7.9, 1.0))
    box(ax, 2.6, 0.55, 2.3, 0.9, "x_next = x + L·a + c\n(residual add)  C × N", BLUE, fs=10, bold=False)
    arrow(ax, (5.6, 1.0), (4.9, 1.0))
    arrow(ax, (0.95, 3.4), (2.6, 1.0), color=BLUE, rad=0.25)
    ax.text(0.1, 2.3, "identity path:\nx itself", fontsize=9, color=BLUE)
    ax.text(3.75, 0.25, "→ next layer's input", ha="center", fontsize=9.5, color=INK2)
    save(fig, "wn02_layer_internals")


# ----------------------------------------------------------------------------------- figure 3
def fig_dilated_timeline():
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), gridspec_kw=dict(height_ratios=[1, 2.1]))
    # (a) one layer, d = 3, K = 6
    ax = axes[0]
    K, d = 6, 3
    ts = np.arange(-22, 1)
    ax.scatter(ts, np.zeros_like(ts), s=28, color=GRID, zorder=2)
    taps = [-(K - 1 - k) * d for k in range(K)]
    ax.scatter(taps, np.zeros(K), s=60, color=ORANGE, zorder=3)
    ax.scatter([0], [1], s=80, color=BLUE, zorder=3)
    for k, t in enumerate(taps):
        ax.plot([t, 0], [0, 1], color=ORANGE, lw=1.2, zorder=1)
        ax.text(t, -0.32, f"W{k}", ha="center", fontsize=9, color=INK2)
    ax.text(0.5, 1.0, "output at t", va="center", fontsize=10)
    ax.text(-22.4, 0.0, "layer input x", va="center", ha="right", fontsize=10)
    ax.set_xlim(-27, 3)
    ax.set_ylim(-0.6, 1.4)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("time (samples relative to t)")
    ax.set_title("(a) One dilated causal layer, K = 6, d = 3: one output reads 6 inputs spaced 3 apart, "
                 "all in the past", fontsize=11)
    ax.text(-15, 1.15, "look-back = (K − 1)·d = 15 samples", fontsize=10, color=ORANGE, ha="center")

    # (b) three stacked layers d = 1, 3, 7 (the model's first three)
    ax = axes[1]
    ds = [1, 3, 7]
    span = sum((K - 1) * dd for dd in ds)  # 5 + 15 + 35 = 55
    lo = -span - 3
    levels = {0: {0}}
    # level index: 3 = output of layer 2, 2 = output of layer 1, 1 = output of layer 0, 0 = input
    reach = {3: {0}}
    for lev, dd in zip([3, 2, 1], [7, 3, 1]):
        nxt = set()
        for t in reach[lev]:
            for k in range(K):
                nxt.add(t - (K - 1 - k) * dd)
        reach[lev - 1] = nxt
    names = {0: "input u (via rechannel)", 1: "after layer 0 (d = 1)", 2: "after layer 1 (d = 3)",
             3: "after layer 2 (d = 7)"}
    for lev in range(4):
        ts = np.arange(lo, 1)
        ax.scatter(ts, np.full_like(ts, lev), s=10, color=GRID, zorder=2)
        r = sorted(reach[lev])
        ax.scatter(r, np.full(len(r), lev), s=22, color=[BLUE, AQUA, ORANGE, VIOLET][lev], zorder=3)
        ax.text(lo - 1.5, lev, names[lev], ha="right", va="center", fontsize=10)
    for lev, dd, col in zip([3, 2, 1], [7, 3, 1], [VIOLET, ORANGE, AQUA]):
        for t in reach[lev]:
            for k in range(K):
                ax.plot([t, t - (K - 1 - k) * dd], [lev, lev - 1], color=col, lw=0.6, alpha=0.45, zorder=1)
    ax.annotate("", xy=(-span, -0.55), xytext=(0, -0.55),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.2))
    ax.text(-span / 2, -0.85, f"receptive field of 3 layers = 1 + 5·1 + 5·3 + 5·7 = {span + 1} samples, "
            "every one of them reached (no holes)", ha="center", fontsize=10)
    ax.set_xlim(lo - 30, 3)
    ax.set_xticks([-50, -40, -30, -20, -10, 0])
    ax.set_ylim(-1.1, 3.4)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("time (samples relative to t)")
    ax.set_title("(b) Stacking layers: the dependency tree of ONE output after three layers (residual paths "
                 "omitted)", fontsize=11)
    fig.tight_layout()
    save(fig, "wn03_dilated_timeline")
    return span + 1


# ----------------------------------------------------------------------------------- figure 4
def fig_receptive_field():
    lb = [(K - 1) * D for K, D in zip(q.KS, q.DS)]
    cum = np.cumsum(lb)
    total = 1 + cum[-1] + q.HEAD_K - 1
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.4), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.25]))
    idx = np.arange(len(lb) + 1)
    lab = [str(i) for i in range(len(lb))] + ["head"]
    cols = [ORANGE if K == 15 else BLUE for K in q.KS] + [VIOLET]
    ax = axes[0]
    vals = lb + [q.HEAD_K - 1]
    ax.bar(idx, vals, color=cols, width=0.72)
    for i, (v, D, K) in enumerate(zip(vals, q.DS + [1], q.KS + [16])):
        ax.text(i, v + 25, f"{v}", ha="center", fontsize=8.5, color=INK2)
        ax.text(i, -150, f"d={D}", ha="center", fontsize=7.5, color=INK2, rotation=90, va="top")
    ax.set_ylabel("look-back of this layer\n(K − 1)·d  [samples]")
    ax.set_title("Per-layer look-back: the ×~2.4 dilation ladder (1, 3, 7, 17, 41, 101, 239) three times, "
                 "plus two K = 15 layers", fontsize=11)
    ax.set_ylim(-420, 1700)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=BLUE), Rectangle((0, 0), 1, 1, color=ORANGE),
                       Rectangle((0, 0), 1, 1, color=VIOLET)],
              labels=["K = 6", "K = 15", "head, K = 16"], loc="upper left", frameon=False, ncol=3)
    ax = axes[1]
    cum_all = np.concatenate([1 + cum, [total]])
    ax.step(idx, cum_all, where="mid", color=BLUE, lw=2)
    ax.scatter(idx, cum_all, s=18, color=cols, zorder=3)
    for g in (6, 13, 15, 22):
        ax.text(g, cum_all[g] + 230, f"{cum_all[g]:,}", ha="center", fontsize=9, color=INK2)
    ax.axhline(total, color=VIOLET, lw=1, ls="--")
    ax.text(0.2, total + 150, f"receptive field = {total:,} samples = {1000 * total / RATE:.1f} ms at 48 kHz",
            fontsize=11, color=VIOLET, fontweight="bold")
    ax.set_ylabel("samples one output depends on\n(1 + Σ look-backs so far)")
    ax.set_ylim(0, total * 1.17)
    ax.set_xticks(idx)
    ax.set_xticklabels(lab, fontsize=9)
    ax.set_xlabel("layer index")
    fig.tight_layout()
    save(fig, "wn04_receptive_field")
    return lb, total


# ----------------------------------------------------------------------------------- figure 5
def fig_leaky_relu(example_z=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    z = np.linspace(-2, 2, 801)
    f = np.where(z >= 0, z, 0.01 * z)
    ax = axes[0]
    ax.plot(z, f, color=RED, lw=2)
    ax.plot(z, z, color=MUTED, lw=1, ls=":")
    ax.text(1.05, 1.45, "identity (y = z)", color=MUTED, fontsize=9.5)
    ax.set_xlabel("z (pre-activation)")
    ax.set_ylabel("LeakyReLU(z)")
    ax.set_title("LeakyReLU(z) = z if z ≥ 0, else 0.01·z")
    ax.text(-1.9, 0.45, "negative side: not zero,\njust 100× smaller\n(slope 0.01)", fontsize=10, color=INK2)
    ax.add_patch(Rectangle((-1, -0.012), 1.2, 0.022, fill=False, ec=ORANGE, lw=1.5))
    ax.text(-0.95, -0.35, "zoomed at right", color=ORANGE, fontsize=9.5)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.6, 2)
    ax = axes[1]
    z2 = np.linspace(-1, 0.2, 400)
    ax.plot(z2, np.where(z2 >= 0, z2, 0.01 * z2), color=RED, lw=2)
    ax.set_xlim(-1, 0.2)
    ax.set_ylim(-0.012, 0.01)
    ax.set_xlabel("z")
    ax.set_title("Zoom: the slope-0.01 region (note the y scale)")
    ax.annotate("z = −1 → −0.01", xy=(-1, -0.01), xytext=(-0.8, -0.0105), fontsize=10,
                arrowprops=dict(arrowstyle="->", color=INK2))
    ax.text(-0.2, 0.0068, "kink at 0:\nslope jumps 0.01 → 1", fontsize=10, color=INK2, ha="center")
    if example_z is not None:
        for zz in example_z:
            if -1 <= zz <= 0.2:
                yy = zz if zz >= 0 else 0.01 * zz
                ax.scatter([zz], [yy], s=40, color=BLUE, zorder=4)
        ax.text(-0.98, 0.0045, "dot: the worked example's channel 1\n(z = −0.0262 → a = −0.00026)", fontsize=9,
                color=BLUE)
    fig.tight_layout()
    save(fig, "wn05_leaky_relu")


# ----------------------------------------------------------------------------------- figure 6
def fig_mac_breakdown():
    steps = [("dilated convs", "conv", BLUE), ("1×1 residual", "l1x1", ORANGE), ("input mixin", "mixin", AQUA),
             ("head (K = 16)", "head", VIOLET), ("rechannel", "rech", YELLOW)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axes[0]
    for row, (C, name) in enumerate([(8, "Full (C = 8)"), (3, "Lite (C = 3)")]):
        m = macs(C)
        left = 0
        for lab, key, col in steps:
            v = m[key]
            ax.barh(row, v, left=left, color=col, height=0.55, edgecolor=SURF, linewidth=2)
            if v > 600:
                ax.text(left + v / 2, row, f"{v:,}", ha="center", va="center", fontsize=9.5, color="white",
                        fontweight="bold")
            left += v
        ax.text(left + 150, row, f"{left:,} MAC/sample", va="center", fontsize=10.5, fontweight="bold")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Full (C = 8)", "Lite (C = 3)"])
    ax.set_xlim(0, 15000)
    ax.set_xlabel("multiply-accumulates per output sample")
    ax.set_title("MACs per sample, by step")
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    for row, C in enumerate([8, 3]):
        m = macs(C)
        tot = sum(m.values())
        left = 0
        for lab, key, col in steps:
            v = 100 * m[key] / tot
            ax.barh(row, v, left=left, color=col, height=0.55, edgecolor=SURF, linewidth=2)
            if v > 6:
                ax.text(left + v / 2, row, f"{v:.0f}%", ha="center", va="center", fontsize=9.5, color="white",
                        fontweight="bold")
            left += v
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Full", "Lite"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of MACs (%)")
    ax.set_title("Share: the dilated convs dominate")
    ax.grid(axis="y", visible=False)
    handles = [Rectangle((0, 0), 1, 1, color=c) for _, _, c in steps]
    fig.legend(handles, [s[0] for s in steps], loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "wn06_mac_breakdown")


# ----------------------------------------------------------------------------------- figure 7
def fig_ring_buffer():
    K, d, N = 6, 3, 8  # an illustrative layer: look-back 15, block 8 (the real block is 128)
    lookback = (K - 1) * d
    pow2 = 1
    while pow2 < lookback + N:
        pow2 <<= 1
    cols1 = pow2 + N
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.6), gridspec_kw=dict(height_ratios=[1.25, 1]))

    # ---- mode 1
    ax = axes[0]
    wp_before = 28
    wp_after = (wp_before + N) & (pow2 - 1)
    new_cols = [(wp_before + i) & (pow2 - 1) for i in range(N)]
    for c in range(cols1):
        if c >= pow2:
            fc, ec, hatch = TINT[VIOLET], VIOLET, "///"
        elif c in new_cols:
            fc, ec, hatch = ORANGE, ORANGE, None
        else:
            fc, ec, hatch = TINT[BLUE], BLUE, None
        ax.add_patch(Rectangle((c, 0), 0.92, 1, fc=fc, ec=ec, hatch=hatch, lw=0.8))
        if c % 4 == 0 or c == cols1 - 1:
            ax.text(c + 0.46, -0.35, str(c), ha="center", fontsize=8, color=INK2)
    ax.text(cols1 + 0.6, 0.5, f"hatched = mirror of\ncols 0–{N - 1}, refreshed\nevery block", va="center",
            fontsize=9.5, color=VIOLET)
    ax.text(0, 1.15, "orange = this block's N = 8 new frames: cols 28–31, then wrapped to cols 0–3", fontsize=9.5,
            color=ORANGE)
    # tap reads
    for j in range(K):
        base = (wp_after - N - j * d) & (pow2 - 1)
        y = -1.0 - 0.45 * j
        ax.add_patch(Rectangle((base, y), N - 0.08, 0.3, fc=TINT[GREEN], ec=GREEN, lw=1))
        ax.text(base - 0.4, y + 0.15, f"taps_back {j}: cols {base}–{base + N - 1}", ha="right", va="center",
                fontsize=8.5, color=GREEN)
    ax.text(0, 1.9, f"Ring mode 1 (default upstream): pow2_size = next_pow2(look-back + N) = {pow2}, storage = "
            f"{pow2} + N = {cols1} columns of C floats", fontsize=11, fontweight="bold")
    ax.text(0, -4.0, "Every tap reads N contiguous columns starting at (write_pos − N − j·d) & mask. A read that "
            "runs past col 31 lands in the mirror,\nso the inner loop never checks for wrap-around. Cost: one "
            "extra memcpy of N columns per layer per block.", fontsize=9.5, color=INK2)
    ax.set_xlim(-12, cols1 + 9)
    ax.set_ylim(-4.3, 2.4)
    ax.axis("off")

    # ---- mode 0
    ax = axes[1]
    hc = 2 * lookback + N  # 38
    wp = 33  # next block would overflow: 33 + 8 > 38
    for c in range(hc):
        if wp - lookback <= c < wp:
            fc, ec = TINT[GREEN], GREEN
        elif c < wp:
            fc, ec = "#f0efec", MUTED
        else:
            fc, ec = SURF, MUTED
        ax.add_patch(Rectangle((c, 0), 0.92, 1, fc=fc, ec=ec, lw=0.8))
        if c % 4 == 0 or c == hc - 1:
            ax.text(c + 0.46, -0.35, str(c), ha="center", fontsize=8, color=INK2)
    ax.annotate("", xy=(lookback / 2, -0.75), xytext=(wp - lookback / 2, -0.75),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.6, connectionstyle="arc3,rad=0.25"))
    ax.text((wp + lookback / 2) / 2 - 2, -1.75, f"rewind: memmove the last look-back = {lookback} columns to the "
            f"front, write_pos = {lookback}", ha="center", fontsize=9.5, color=RED)
    ax.text(wp + 0.5, 1.25, f"write_pos = {wp}; {wp} + N > {hc} → rewind first", fontsize=9.5, color=INK2)
    ax.text(0, 1.9, f"Ring mode 0 (this project's choice, iteration 4): linear buffer of 2·look-back + N = {hc} "
            "columns", fontsize=11, fontweight="bold")
    ax.text(0, -2.85, "Reads are plain (write_pos − N − j·d). A rewind copies look-back columns about every "
            "look-back/N blocks: at most ~N columns per block on average,\nfar fewer for short look-backs (layer 0 "
            "at N = 128: 5 vs mode 1's 128). Measured: Lite −4.0% p50, bit-identical output.", fontsize=9.5,
            color=INK2)
    ax.set_xlim(-12, cols1 + 9)
    ax.set_ylim(-3.2, 2.4)
    ax.axis("off")
    ax.text(-12, 0.5, "green = the look-back\nthe next block needs", fontsize=9, color=GREEN, va="center")
    fig.suptitle(f"One layer's history buffer, illustrated with K = {K}, d = {d} (look-back {lookback}) and "
                 f"N = {N}; each cell is one frame = C floats", fontsize=12, x=0.02, ha="left", y=1.0)
    fig.tight_layout()
    save(fig, "wn07_ring_buffer")


def ring_memory(N=128):
    out = {}
    for C in (3, 8):
        m1 = m0 = 0
        for K, D in zip(q.KS, q.DS):
            lb = (K - 1) * D
            p2 = 1
            while p2 < lb + N:
                p2 <<= 1
            m1 += C * (p2 + N) * 4
            m0 += C * (2 * lb + N) * 4
        p2 = 1
        while p2 < 15 + N:
            p2 <<= 1
        m1 += C * (p2 + N) * 4
        m0 += C * (2 * 15 + N) * 4
        out[C] = (m1, m0)
    return out


# ----------------------------------------------------------------------------------- figure 8/9
def real_segment(start_s, dur_s):
    rate, audio = q.read_wav(DI)
    assert rate == RATE
    t0 = int(start_s * RATE)
    pre = q.PREWARM
    u = audio[t0 - pre:t0 + int(dur_s * RATE)]
    return u, pre


def fig_real_overview(start_s=19.25, dur_s=1.0):
    u, pre = real_segment(start_s, dur_s)
    p = q.load_model(MODEL, 8)
    r = forward_capture(p, u)
    sl = slice(pre, None)
    t = start_s + np.arange(len(u) - pre) / RATE
    win = 120  # 2.5 ms
    nwin = (len(u) - pre) // win

    def agg(m):
        return m[:, sl][:, :nwin * win].reshape(m.shape[0], nwin, win).mean(axis=2)

    depths = [0, 7, 14, 22]
    fig, axes = plt.subplots(len(depths) + 3, 1, figsize=(11, 12.5), sharex=True,
                             gridspec_kw=dict(height_ratios=[1] + [1] * len(depths) + [1, 1]))
    ext = [t[0], t[0] + nwin * win / RATE, -0.5, 7.5]
    axes[0].plot(t, u[sl], color=BLUE, lw=0.5)
    axes[0].set_ylim(-0.8, 0.8)
    axes[0].set_ylabel("input u")
    axes[0].set_title("Full model (C = 8) on 1 s of audio/di.wav: what each depth looks like", fontsize=12)
    for ax, li in zip(axes[1:], depths):
        A = agg(r["a"][li])
        vmax = np.percentile(A, 99.5)
        im = ax.imshow(A, aspect="auto", origin="lower", extent=ext, cmap=SEQ, vmin=0, vmax=vmax,
                       interpolation="nearest")
        ax.set_ylabel(f"layer {li}\nchannel")
        ax.set_yticks([0, 7])
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
        cb.ax.tick_params(labelsize=8)
        K, D = q.KS[li], q.DS[li]
        ax.text(1.0, 1.02, f"activation a, mean over 2.5 ms (K={K}, d={D})", transform=ax.transAxes,
                ha="right", fontsize=9, color=INK2)
    ax = axes[len(depths) + 1]
    H = agg(r["hs"])
    im = ax.imshow(H, aspect="auto", origin="lower", extent=ext, cmap=SEQ, vmin=0,
                   vmax=np.percentile(H, 99.5), interpolation="nearest")
    ax.set_ylabel("head sum s\nchannel")
    ax.set_yticks([0, 7])
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
    cb.ax.tick_params(labelsize=8)
    ax.text(1.0, 1.02, "s = Σ over 23 layers of a, mean over 2.5 ms", transform=ax.transAxes, ha="right",
            fontsize=9, color=INK2)
    ax = axes[-1]
    ax.plot(t, r["y"][sl], color=ORANGE, lw=0.5)
    ax.set_ylim(-0.8, 0.8)
    ax.set_ylabel("output y")
    ax.set_xlabel("time in audio/di.wav (s)")
    # keep line panels the same width as the heatmap panels
    for a in (axes[0], axes[-1]):
        cb = fig.colorbar(im, ax=a, pad=0.01, fraction=0.03)
        cb.ax.set_visible(False)
    fig.tight_layout()
    save(fig, "wn08_real_audio_overview")
    stats = {}
    for li in range(23):
        A = r["a"][li][:, sl]
        pos = (A > 0).mean(axis=1)
        stats[li] = dict(always_on=int((pos > 0.99).sum()), always_off=int((pos < 0.01).sum()),
                         switching=int(((pos >= 0.01) & (pos <= 0.99)).sum()))
    return r, sl, stats


def fig_real_zoom(start_s=19.25, dur_s=1.0, zoom=(19.62, 19.645)):
    u, pre = real_segment(start_s, dur_s)
    p = q.load_model(MODEL, 3)
    r = forward_capture(p, u)
    i0 = pre + int((zoom[0] - start_s) * RATE)
    i1 = pre + int((zoom[1] - start_s) * RATE)
    t = 1000 * (np.arange(i0, i1) - i0) / RATE
    depths = [0, 7, 14, 22]
    cols = [BLUE, ORANGE, AQUA]
    fig, axes = plt.subplots(len(depths) + 3, 1, figsize=(11, 12.5), sharex=True)
    axes[0].plot(t, u[i0:i1], color=INK2, lw=1.4)
    axes[0].set_ylabel("input u")
    axes[0].set_title(f"Lite model (C = 3), 25 ms zoom at {zoom[0]:.2f} s: traces per channel", fontsize=12)
    for ax, li in zip(axes[1:], depths):
        for c in range(3):
            ax.plot(t, r["a"][li][c, i0:i1], color=cols[c], lw=1.4, label=f"ch {c}")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_ylabel(f"layer {li}\nactivation a")
    axes[1].legend(loc="upper right", ncol=3, frameon=False)
    ax = axes[len(depths) + 1]
    for c in range(3):
        ax.plot(t, r["hs"][c, i0:i1], color=cols[c], lw=1.4)
    ax.set_ylabel("head sum s")
    ax = axes[-1]
    ax.plot(t, r["y"][i0:i1], color=ORANGE, lw=1.6, label="output y")
    ax.plot(t, u[i0:i1], color=INK2, lw=1.0, ls="--", label="input u (for scale)")
    ax.legend(loc="upper right", ncol=2, frameon=False)
    ax.set_ylabel("output y")
    ax.set_xlabel(f"time (ms after {zoom[0]:.2f} s)")
    fig.tight_layout()
    save(fig, "wn09_real_audio_zoom")
    return r


# ----------------------------------------------------------------------------------- figure 10
def level_sweep(p, f=110.0, levels=range(-60, 1, 3)):
    n = int(0.3 * RATE)
    t = np.arange(q.PREWARM + n) / RATE
    rows = []
    for db in levels:
        u = 10 ** (db / 20) * np.sin(2 * np.pi * f * t)
        y = q.forward_float(p, u)[-int(0.1 * RATE):]
        y = y - y.mean()
        ui = u[-int(0.1 * RATE):]
        rows.append((db, 20 * np.log10(np.sqrt(np.mean(ui ** 2))), 20 * np.log10(np.sqrt(np.mean(y ** 2)))))
    return np.array(rows)


def fig_level_transfer():
    P = {C: q.load_model(MODEL, C) for C in (3, 8)}
    sweeps = {C: level_sweep(P[C]) for C in P}
    # real audio: 20 ms block RMS, first 20 s, Full
    rate, audio = q.read_wav(DI)
    n = 20 * RATE
    u = np.concatenate([np.zeros(q.PREWARM), audio[:n]])
    y = q.forward_float(P[8], u)[q.PREWARM:]
    blk = int(0.02 * RATE)
    nb = n // blk
    ri = 20 * np.log10(np.sqrt(np.mean(audio[:nb * blk].reshape(nb, blk) ** 2, axis=1)) + 1e-12)
    yo = y[:nb * blk].reshape(nb, blk)
    ro = 20 * np.log10(np.sqrt(np.mean((yo - yo.mean(axis=1, keepdims=True)) ** 2, axis=1)) + 1e-12)
    keep = ri > -70

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    ax.scatter(ri[keep], ro[keep], s=6, color=MUTED, alpha=0.35, label="real audio, 20 ms blocks (Full, first 20 s)",
               lw=0)
    for C, col, name in ((3, BLUE, "Lite"), (8, ORANGE, "Full")):
        s = sweeps[C]
        ax.plot(s[:, 1], s[:, 2], color=col, lw=2, marker="o", ms=4, label=f"{name}: 110 Hz sine, steady state")
    xs = np.array([-65, 0])
    g0 = sweeps[8][0, 2] - sweeps[8][0, 1]
    ax.plot(xs, xs + g0, color=INK2, lw=1, ls=":")
    ax.text(-62, -62 + g0 + 3.5, f"slope 1 (linear, +{g0:.1f} dB)", fontsize=9.5, color=INK2, rotation=33)
    ax.set_xlabel("input level (dBFS RMS)")
    ax.set_ylabel("output level (dBFS RMS, DC removed)")
    ax.set_title("Level in vs level out")
    ax.set_xlim(-66, 0)
    ax.set_ylim(-60, 0)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax = axes[1]
    f = 110.0
    tt = np.arange(q.PREWARM + int(0.2 * RATE)) / RATE
    for db, col, ls in ((-40, BLUE, "-"), (-6, ORANGE, "-")):
        u = 10 ** (db / 20) * np.sin(2 * np.pi * f * tt)
        yy = q.forward_float(P[8], u)[-int(2 / f * RATE):]
        yy = yy - yy.mean()
        yy = yy / np.abs(yy).max()
        ax.plot(1000 * np.arange(len(yy)) / RATE, yy, color=col, lw=1.8, ls=ls, label=f"{db} dBFS peak input")
    ax.set_xlabel("time (ms), two periods of 110 Hz")
    ax.set_ylabel("output, normalized to peak 1")
    ax.set_title("Output waveform shape (Full)")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "wn10_level_transfer")
    return sweeps


# ----------------------------------------------------------------------------------- figure 11
def small_signal(p, amp=1e-3):
    n = 2 * q.PREWARM + 100
    u = np.zeros(n)
    i0 = q.PREWARM
    base = q.forward_float(p, u)
    u2 = u.copy()
    u2[i0] = amp
    h = (q.forward_float(p, u2) - base)[i0:] / amp
    u3 = u.copy()
    u3[i0] = -amp
    h3 = (q.forward_float(p, u3) - base)[i0:] / (-amp)
    lin_err = np.max(np.abs(h - h3)) / np.max(np.abs(h))
    return h, lin_err, base[-1]


def fig_small_signal():
    res = {}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for C, col, name in ((3, BLUE, "Lite"), (8, ORANGE, "Full")):
        p = q.load_model(MODEL, C)
        h, lin_err, dc = small_signal(p)
        e = np.cumsum(h ** 2)
        e /= e[-1]
        H = np.abs(np.fft.rfft(h, 1 << 16))
        fr = np.fft.rfftfreq(1 << 16, 1 / RATE)
        res[C] = dict(h=h, lin_err=lin_err, dc=dc, e=e, H=H, fr=fr)
        tm = 1000 * np.arange(len(h)) / RATE
        axes[0].plot(tm[:240], h[:240], color=col, lw=1.4, label=name)
        axes[1].plot(tm[1:], 100 * e[1:], color=col, lw=1.8, label=name)
        axes[2].semilogx(fr[1:], 20 * np.log10(H[1:]), color=col, lw=1.6, label=name)
    axes[0].set_xlabel("time after the impulse (ms)")
    axes[0].set_ylabel("output per unit input")
    axes[0].set_title("Small-signal impulse response")
    axes[0].legend(frameon=False)
    ax = axes[1]
    ax.set_xscale("log")
    ax.axvline(1000 * 6347 / RATE, color=VIOLET, ls="--", lw=1)
    ax.text(1000 * 6347 / RATE * 0.93, 8, "receptive field\n132 ms", ha="right", color=VIOLET, fontsize=9.5)
    ax.set_xlim(0.05, 200)
    ax.set_ylim(0, 102)
    ax.set_xlabel("time after the impulse (ms, log)")
    ax.set_ylabel("% of the response's energy so far")
    ax.set_title("Where in the 132 ms the energy is")
    ax.legend(frameon=False, loc="center right")
    ax = axes[2]
    ax.set_xlim(20, 20000)
    ax.set_ylim(-60, 30)
    ax.set_xlabel("frequency (Hz, log)")
    ax.set_ylabel("gain (dB)")
    ax.set_title("Small-signal frequency response")
    ax.legend(frameon=False, loc="lower left")
    fig.text(0.01, -0.04, "Input: one impulse of amplitude 0.001 after silence, minus the silence output. At this "
             "level the response is linear (±impulse agree to ~1e-13), so this is the model's 'quiet' "
             "filter, not its distortion.", fontsize=9, color=INK2)
    fig.tight_layout()
    save(fig, "wn11_small_signal")
    return res


# ----------------------------------------------------------------------------------- figure 12
def fig_optimizations():
    fig = plt.figure(figsize=(13, 9.6))
    # (a) register tile
    ax = diagram_axes(fig, [0.0, 0.62, 0.5, 0.38], xlim=(0, 10), ylim=(0, 10))
    ax.text(0.1, 9.5, "(a) Full's register tile (a2_full_neon_tile.patch)", fontsize=12, fontweight="bold")
    for o in range(8):
        for f in range(4):
            ax.add_patch(Rectangle((1.2 + f * 1.0, 1.0 + (7 - o) * 0.85), 0.92, 0.78,
                                   fc=TINT[BLUE] if o < 4 else TINT[AQUA], ec=BLUE if o < 4 else AQUA, lw=1))
    for f in range(4):
        ax.text(1.66 + f, 7.95, f"t+{f}", ha="center", fontsize=9.5)
    for o in range(8):
        ax.text(1.05, 1.39 + (7 - o) * 0.85, f"out {o}", ha="right", va="center", fontsize=8.5)
    ax.text(3.2, 8.5, "8 outputs × 4 frames = 32 sums\n= 8 q-registers, live across ALL K taps", ha="center",
            fontsize=9.5)
    ax.text(5.6, 7.7, "for tap k in 0..K−1:\n  for input pair j, j+1:\n    load W_k columns j, j+1\n"
            "      (4 q-regs)\n    load x_j, x_j+1 at 4 frames\n    16 × VMLA (lane form)\n"
            "then bias, mixin, LeakyReLU,\nhead sum, 1×1 in registers", fontsize=9.5, va="top", family="monospace")
    ax.text(5.6, 2.9, "Each weight load feeds 4 frames;\nno z buffer round-trips.\nIteration 6: Full 1.91× "
            "(6,712 →\n3,511 µs/block); Lite's lanes\nkernel 1.17× (991 → 849 µs).", fontsize=9.5, color=INK2, va="top")

    # (b) pipeline
    ax = fig.add_axes([0.56, 0.66, 0.42, 0.3])
    ax.set_title("(b) Two-core pipeline, split at layer 13 (a2_pipeline.patch)", fontsize=12)
    for n in range(4):
        ax.barh(1, 0.98, left=n, color=BLUE, height=0.5, edgecolor=SURF)
        ax.text(n + 0.49, 1, f"block {n}\nlayers 0–12", ha="center", va="center", fontsize=8.5, color="white")
        if n >= 1:
            ax.barh(0, 0.95, left=n, color=ORANGE, height=0.5, edgecolor=SURF)
            ax.text(n + 0.47, 0, f"block {n - 1}\n13–22 + head", ha="center", va="center", fontsize=8.5,
                    color="white")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["worker\n(core 2)", "audio thread\n(core 3)"])
    ax.set_xticks(range(5))
    ax.set_xlabel("audio callback number")
    ax.set_xlim(0, 4)
    ax.grid(False)
    ax.text(0, -0.75, "Same math, output one block late. Iteration 8: Full 3,511 → 1,785 µs (1.97×).\n"
            "Iteration 10: split 13 is best (91 vs 88 cost units; ~140 µs per layer off balance).",
            fontsize=9.5, color=INK2, va="top")
    ax.set_ylim(-0.4, 1.4)

    # (c) frame split
    ax = diagram_axes(fig, [0.03, 0.03, 0.94, 0.55], xlim=(0, 20), ylim=(0, 10))
    ax.text(0.1, 9.6, "(c) Frame split across cores, one barrier per layer (a2_frame_split.patch)", fontsize=12,
            fontweight="bold")
    cols = [BLUE, ORANGE, AQUA]
    nl = 6
    for li in range(nl):
        y = 8.2 - li * 1.25
        lab = f"layer {li}" if li < nl - 1 else "… layer 22"
        ax.text(1.6, y + 0.3, lab, ha="right", va="center", fontsize=10)
        for c in range(3):
            ax.add_patch(Rectangle((1.9 + c * 3.0, y), 2.9, 0.6, fc=TINT[cols[c]], ec=cols[c], lw=1.2))
            ax.text(1.9 + c * 3.0 + 1.45, y + 0.3, f"core {3 - c}: frames {c * 16}–{c * 16 + 15}", ha="center",
                    va="center", fontsize=9)
        ax.plot([1.85, 10.85], [y - 0.17, y - 0.17], color=RED, lw=1.5, ls="--")
    ax.text(11.1, 8.2 - (nl - 1) * 1.25 - 0.17, "barrier", color=RED, fontsize=9.5, va="center")
    ax.text(11.2, 8.0, "Why a barrier after every layer?", fontsize=10.5, fontweight="bold")
    ax.text(11.2, 7.6, "Frame 20 of layer 8 (d = 3) reads layer 8's input at\nframes 20, 17, 14, 11, 8, 5. "
            "Frames 5–15 are\nproduced by another core, so it must wait until\nthat core has finished "
            "layer 7.\n\nThis is the sequential dependency chain: layer i+1\nneeds layer i's output, "
            "including at EARLIER frames\nthat belong to other threads.\n\nMeasured (Full, block 128, burst): "
            "3,461 µs on 1\ncore → 1,770 µs on 2, 943 µs on 4. Bit-identical\noutput, no added delay "
            "(docs/low-latency-plan.md).", fontsize=9.5, va="top", color=INK2)
    ax.text(1.9, 0.2, "Example: N = 48 frames on 3 cores (whole 4-frame tiles per thread)", fontsize=9.5,
            color=INK2)
    save(fig, "wn12_optimizations")


# ----------------------------------------------------------------------------------- worked example
def worked_example(start_s=19.25, dur_s=1.0):
    """Lite, layer 0, one time step: every number, printed."""
    u, pre = real_segment(start_s, dur_s)
    p = q.load_model(MODEL, 3)
    r = forward_capture(p, u)
    L = p["layers"][0]
    K, D = q.KS[0], q.DS[0]
    x = r["x"][0]
    # pick a frame where the input is moderate and one channel is negative after the mixin
    z_all = np.zeros_like(x)
    for k in range(K):
        z_all += L["W"][k] @ q.shift(x, (K - 1 - k) * D)
    z_all += L["b"][:, None] + L["mix"][:, None] * u[None, :]
    cand = [t for t in range(pre, len(u)) if (z_all[:, t] < 0).any() and 0.05 < abs(u[t]) < 0.2]
    t = cand[0]
    print(f"\n=== Worked example: Lite, layer 0 (K={K}, d={D}), frame {t - pre} of the segment at "
          f"{start_s} s (= sample {int(start_s * RATE) + t - pre} of di.wav)")
    print("rechannel r =", np.array2string(p["rech"], precision=4))
    print("input u(t-5..t) =", np.array2string(u[t - 5:t + 1], precision=4))
    for k in range(K):
        lag = (K - 1 - k) * D
        print(f"tap {k} (lag {lag}): x = {np.array2string(x[:, t - lag], precision=4)}  W_{k} = "
              f"{np.array2string(L['W'][k], precision=4).replace(chr(10), '')}  W·x = "
              f"{np.array2string(L['W'][k] @ x[:, t - lag], precision=4)}")
    conv = sum(L["W"][k] @ x[:, t - (K - 1 - k) * D] for k in range(K))
    print("sum over taps =", np.array2string(conv, precision=4))
    print("bias b =", np.array2string(L["b"], precision=4))
    print("mixin m =", np.array2string(L["mix"], precision=4), " m*u(t) =",
          np.array2string(L["mix"] * u[t], precision=4))
    z = conv + L["b"] + L["mix"] * u[t]
    a = np.where(z >= 0, z, 0.01 * z)
    print("z =", np.array2string(z, precision=4), " a = LeakyReLU(z) =", np.array2string(a, precision=5))
    print("1x1 L =", np.array2string(L["L"], precision=4).replace("\n", ""), " c =",
          np.array2string(L["lb"], precision=4))
    r1 = L["L"] @ a + L["lb"]
    print("L·a + c =", np.array2string(r1, precision=4), " x_next = x + that =",
          np.array2string(x[:, t] + r1, precision=4))
    print("check vs forward pass: a", np.array2string(r["a"][0][:, t], precision=5), " x_next",
          np.array2string(r["x"][1][:, t], precision=4))
    print("final output y(t) for this frame (all 23 layers + head):", f"{r['y'][t]:.5f}")
    return z


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated figure numbers (1-12)")
    ap.add_argument("--check-render", default="", help="a C++ `bench --render` WAV of audio/di.wav")
    ap.add_argument("--check-slim", default="1", help="the --slim used for that render (0.25 = Lite, 1 = Full)")
    args = ap.parse_args()
    only = {int(s) for s in args.only.split(",") if s}

    def want(n):
        return not only or n in only

    cfg = load_config()
    print("=== Config (from the .nam file)")
    for C, c in sorted(cfg.items()):
        m = macs(C)
        print(f"C={C}: {c['n_weights']} weights (counted {params(C)}), MAC/sample {sum(m.values())} {m}, "
              f"slimmable max_value {c['max_value']}, head_scale JSON {c['head_scale_json']} / trailing weight "
              f"{c['head_scale_weight']}")
    lite, full = q.load_model(MODEL, 3), q.load_model(MODEL, 8)
    print(f"head bias: Lite {lite['hb']:.4f}, Full {full['hb']:.4f}")
    print("Lite == first 3 channels of Full? rechannel", np.allclose(lite["rech"], full["rech"][:3]),
          "| layer 0 tap 0", np.allclose(lite["layers"][0]["W"][0], full["layers"][0]["W"][0][:3, :3]))
    # reachable lags with K=6 and the dilation ladder: contiguous?
    reach = {0}
    for D in [1, 3, 7, 17, 41, 101, 239]:
        reach = {r + j * D for r in reach for j in range(6)}
    print(f"lags reachable through one 7-layer group: {len(reach)} distinct of 0..{max(reach)} "
          f"(contiguous: {reach == set(range(max(reach) + 1))})")
    mem = ring_memory(128)
    for C, (m1, m0) in mem.items():
        print(f"history memory at N=128, C={C}: ring mode 1 {m1 / 1024:.1f} KiB, mode 0 {m0 / 1024:.1f} KiB")
    for li, (K, D) in enumerate(zip(q.KS, q.DS)):
        if li in (0, 6, 15, 22):
            lb = (K - 1) * D
            p2 = 1
            while p2 < lb + 128:
                p2 <<= 1
            print(f"  layer {li}: look-back {lb}, mode 1 cols {p2}+128, mode 0 cols {2 * lb + 128}")

    if args.check_render:
        C = 3 if float(args.check_slim) < 0.5 else 8
        rate, audio = q.read_wav(DI)
        n = 10 * RATE
        ref = q.forward_float(q.load_model(MODEL, C), np.concatenate([np.zeros(q.PREWARM), audio[:n]]))[q.PREWARM:]
        _, r = q.read_wav(args.check_render)
        rd, mx = q.residual(r[:n], ref)
        print(f"=== float64 forward pass vs {args.check_render} (C={C}, first 10 s): {rd:.1f} dB rel, "
              f"max diff {mx:.1f} dBFS")

    z = worked_example()
    if want(1):
        fig_block_diagram()
    if want(2):
        fig_layer_internals()
    if want(3):
        rf3 = fig_dilated_timeline()
        print(f"3-layer receptive field (d = 1, 3, 7): {rf3}")
    if want(4):
        lb, total = fig_receptive_field()
        print(f"look-backs {lb}; sum {sum(lb)}; receptive field 1 + {sum(lb)} + {q.HEAD_K - 1} = {total} "
              f"= {1000 * total / RATE:.2f} ms")
    if want(5):
        fig_leaky_relu(z)
    if want(6):
        fig_mac_breakdown()
    if want(7):
        fig_ring_buffer()
    if want(8):
        r, sl, stats = fig_real_overview()
        y, u = r["y"][sl], None
        print("=== Full on 19.25-20.25 s: per-layer channel behaviour (of 8): always on (>99% positive) / "
              "always off (<1%) / switching")
        print("  " + "  ".join(f"L{li}:{s['always_on']}/{s['always_off']}/{s['switching']}"
                               for li, s in stats.items()))
        hs = r["hs"][:, sl]
        print(f"  head sum range {hs.min():.2f}..{hs.max():.2f}, mean {hs.mean():.2f}; output mean {y.mean():.5f}, "
              f"output peak {np.abs(y).max():.3f}")
    if want(9):
        r3 = fig_real_zoom()
    if want(10):
        sweeps = fig_level_transfer()
        for C, s in sweeps.items():
            print(f"=== level sweep C={C} (peak dBFS, in RMS, out RMS):")
            print("  " + "  ".join(f"{a:.0f}:{b:.1f}->{c:.1f}" for a, b, c in s))
    if want(11):
        res = fig_small_signal()
        for C, d in res.items():
            e, h = d["e"], d["h"]
            fr, H = d["fr"], d["H"]
            pts = {f: 20 * np.log10(H[np.argmin(abs(fr - f))]) for f in (50, 80, 100, 200, 500, 1000, 3000, 5000,
                                                                      8000, 12000)}
            pk = int(np.argmax(np.abs(h)))
            print(f"=== small signal C={C}: linearity err {d['lin_err']:.1e}, silence output {d['dc']:.6f}, "
                  f"peak at {pk} samples ({1000 * pk / RATE:.2f} ms); energy 50/90/99/99.9% by "
                  + "/".join(f"{1000 * np.argmax(e >= x) / RATE:.2f}" for x in (0.5, 0.9, 0.99, 0.999)) + " ms; "
                  + ", ".join(f"{f} Hz {v:.1f} dB" for f, v in pts.items()))
    if want(12):
        fig_optimizations()


if __name__ == "__main__":
    main()
