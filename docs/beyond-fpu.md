# Beyond the FPU: integer math, soft float, and the VideoCore IV GPU

Can NAM A2 on the Pi 2 get more arithmetic per second than the Cortex-A7's floating-point unit gives?
This looks at (a) integer and fixed-point math on the CPU, including software-emulated floats, and
(b) the VideoCore IV GPU's QPUs. **Measured** numbers come from `results-pi/int_peak.csv`
(`src/int_peak.cpp`, run on the Pi) and `results-pi/quant_sim.csv` (`scripts/quant_sim.py`, run on the
Mac). Everything else is labeled **estimate** or cited.

## Correction first: "1 flop per cycle"

Iteration 7 measured about **1 float lane-op per cycle per core** for every FP form. For `vmla`/`vfma`
a lane-op is a multiply-accumulate (MAC), which is **2 FLOP**. So the A7 does about **0.97 MAC =
1.9 FLOP per cycle per core** on MAC-shaped code, and about 1 FLOP per cycle with separate `vmul` +
`vadd`. Four cores at 900 MHz: about 3.5 GMAC/s = 7 GFLOPS peak. The kernels reach less than that: Full
runs at 0.47 MAC/cycle, Lite at 0.28.

## Summary

Current reference: the f32 NEON peak is 0.97 MAC/cycle/core, and the Full kernel runs at 0.47 MAC/cycle
(single core). Accuracy is the null-test residual of Full against float (the project bar is about −100 dB).

| Option | Throughput vs float | Accuracy (Full) | Latency | Effort | Verdict |
|---|---|---|---|---|---|
| Soft-float emulation on the integer core | **Measured** 0.013 MAC/cycle (libgcc), 0.019 (minimal, non-IEEE emulation): **75× / 52× slower** | exact (libgcc) | none added | none | **Dismiss** |
| Scalar integer `smlad` (dual 16-bit MAC) | **Measured** 1.77 MAC/cycle peak. Doesn't overlap with NEON (measured), so no gain when added to NEON | same as int16 | none | high | **No**: NEON int16 does the same and better |
| NEON int16 → int32 (`vmlal.s16`) | **Measured** 1.88 MAC/cycle peak (1.94× f32). Conv-tile instruction mix: **1.64 vs 0.80** (2.05×). Whole Full kernel: **estimate 0.64–0.89 MAC/cycle (1.4–1.9× today), ceiling ~1.5** | **−71 to −75 dB**. **−65 to −70 dB** once made int32-safe and robust to a full-scale input | none added | 1–2 weeks | **Fails the −100 dB bar.** Only if a must-have needs it |
| NEON int16 in mixed precision (layers 0–6 float, 7–22 int16) | **Estimate** 1.25–1.5× over today's Full kernel | **−82 to −83 dB** | none added | 2 weeks | Best integer accuracy found. Still fails the bar; needs a listening test |
| NEON 16-bit accumulators (`vmla.i16`) / int8 (`vmlal.s8`) | **Measured 3.76 MAC/cycle (3.9×)** | W8A8 **−25 dB**; 16-bit accumulators overflow | — | — | **Dismiss** (audibly wrong) |
| VideoCore IV QPUs | **Cited** 24 GFLOPS peak = 12 GMAC/s, about 3.4× all four A7 cores' float peak. Realistic sgemm 8 GFLOPS (cited) | **−107 dB** (simulated round-toward-zero fp32) | **Estimate** 0.2–0.6 ms per 128-sample block, including ~100 µs dispatch (cited) | weeks, plus sudo, a config change and a reboot | **Blocked** on this setup, and risky for real time. **Don't pursue** |
| 3-stage CPU pipeline (3 cores) | **Estimate** ~1.2 ms per call at 128 (≈45% load) vs 1.77 ms | bit-identical | +2 blocks (5.3 ms at 128) | 1–2 days | Viable if the latency is acceptable |
| Group the loads in the float kernel | **Measured** penalty: a non-`vmla` NEON instruction after `vmla.f32` costs ~4 extra cycles. Same conv mix: grouped 0.80 vs clang's interleaved order 0.59 MAC/cycle | bit-identical (same arithmetic order) | none | hours to 1 day | **Try first**: cheap and accuracy-neutral |
| Low-rank conv weights (no retraining) | rank 4 = 7,520 vs 11,776 MAC/sample (1.57× fewer) | **−10 dB** (rank 4), −22 dB (rank 7) | none | — | **Dismiss** without retraining |

**Recommendation.** The FP ceiling is real, but it isn't today's bottleneck. Full runs at 48% of f32
peak, and the A2 model needs ~−100 dB to count as unchanged. In order:

1. **Fix the float kernel's instruction order first** (measured cause, bit-identical output).
2. **Use more cores before fewer bits:** the pipeline already doubles throughput at no accuracy cost.
3. **Treat int16 as an experiment with a known accuracy cost.** The best scheme found is −82 dB. It
   won't pass −100 dB with any int16 design simulated here, so only build it if a goal needs it, such as
   Full on one core with no pipeline latency.
4. **Leave the GPU alone.**

The test plan is at the end.

---

## 1. Software float emulation

**Question.** If FP is the limit, could the integer pipeline emulate floats faster?

**Method.** `int_peak` times 8 independent multiply-adds per iteration (8 M in total) three ways:
libgcc's `__aeabi_fmul` + `__aeabi_fadd`, which is exactly what `-mfloat-abi=soft` code calls (here
through the PLT into `libgcc_s`); a minimal hand-written emulation in C (normal numbers only,
truncation, no NaN/Inf/denormal handling, so a lower bound for any emulation); and the same C loop with
hardware VFP floats.

| Measured (`results-pi/int_peak.csv`) | cycles per MAC | MACs per cycle |
|---|---|---|
| libgcc soft float (`__aeabi_fmul` + `__aeabi_fadd`) | **74.7** | 0.013 |
| minimal emulation, not IEEE-correct | **53.9** | 0.019 |
| hardware VFP, same C loop | 1.25 | 0.80 |
| NEON `vmla.f32` q peak (iteration 7) | 1.03 | 0.97 |

**Verdict: dismissed.** Emulation is **52–75× slower** than the hardware MAC. Even a single-cycle
integer multiplier needs ~50 instructions to unpack, align, add, normalize and repack a float. The
FPU is the fast path for floats. The only way the integer side helps is to stop using floats, i.e.
fixed point (section 2).

## 2. Integer and fixed-point throughput on the A7

`src/int_peak.cpp` uses the same method as `neon_peak` (8-instruction inline-asm loop bodies, 20 M
iterations, `--rt` on CPU 3, cycles = time × 900 MHz). Throughput uses 8 independent accumulators,
latency uses a dependent chain. Full table: `results-pi/int_peak.csv`.

| Instruction | lanes | cycles / instr | **MACs (ops) / cycle** | vs f32 `vmla` q (0.97) |
|---|---|---|---|---|
| `vmla.i16` q (16×16→16, wraps) | 8 | 2.13 | **3.76** | 3.9× |
| `vmla.i16` d | 4 | 1.13 | 3.55 | 3.7× |
| `vmlal.s8` (8×8→16) | 8 | 2.13 | **3.76** | 3.9× |
| `vmlal.s16` (16×16→32), by vector or by scalar | 4 | 2.13 | **1.88** | 1.94× |
| `vqdmlal.s16` (saturating, doubling) | 4 | 2.13 | 1.88 | 1.94× |
| `vmla.i32` q | 4 | 4.13 | 0.97 | 1.0× |
| `vmull.s16` | 4 | 2.13 | 1.88 (mul) | |
| `vqdmulh.s16` / `vqrdmulh.s16` q (incl. by scalar) | 8 | 4.13 | 1.94 (mul) | |
| `vqrdmulh.s32` q | 4 | 4.13 | 0.97 (mul) | |
| `vmul.i16`, `vadd.i16`, `vqadd.s16`, `vmax.s16` q | 8 | 2.13 | 3.76 (ops) | |
| `vadd.i32` q | 4 | 2.13 | 1.88 (ops) | |
| `vqrshrn.s32 #15` (narrow 32→16) | 4 | 1.13 | 3.55 (ops) | |
| scalar `mla`, `smlabb`, `smlawb` | 1 | 1.13 | 0.89 | 0.9× |
| scalar `smlad` (two 16×16 MACs) | 2 | 1.13 | **1.77** | 1.8× |
| latency: `vmla.i16` q / `vmlal.s16` / `vadd.i16` q chain | | 4.0 | | |
| latency: `vqrdmulh.s16` q chain | | 4.13 | | |
| latency: `mla` / `smlad` chain | | 1.13 | (accumulator forwarding) | |

**What the A7's NEON does** (inferred from these numbers): the integer unit processes 64 bits per cycle,
so 4 × int16 or 2 × int32 lanes per cycle. Multiplies of 16-bit lanes, and anything producing 32-bit
lanes, cost twice that (`vmlal.s16`, `vqrdmulh.s16`: 2 lanes of 32-bit work per cycle). 32-bit
multiplies run at the float rate. So:

- **16-bit math with 16-bit results is 3.9× f32**, but a 16-bit accumulator can't hold a 48- to
  120-term dot product of 16-bit values (it wraps). Int8 inputs with 16-bit sums (`vmlal.s8`) have the
  same rate, but the accuracy is −25 dB (section 3).
- **The useful form, 16×16→32 (`vmlal.s16`), is 1.94× f32.** This is the realistic integer ceiling.
- **The scalar integer core doesn't add throughput on top of NEON.** `smlad` alone does 1.77 MAC/cycle,
  but interleaved with NEON the two run back to back, not in parallel. Measured: `vmla.i16` q ×4 +
  `smlad` ×4 takes 13.0 cycles = 4 × 2.13 + 4 × 1.13. `vmla.f32` q ×4 + `smlad` ×4 takes 21.0 cycles =
  4 × 4.13 + 4 × 1.13.

**Loads, measured with MACs:**

| Mix | MACs / cycle |
|---|---|
| `vmlal.s16` + `vld1.16` d, 1:1 | 1.23 (= 2.13 + 1.13 cycles per pair: no overlap) |
| `vmlal.s16` ×6 + `vld1.16` d ×2 | 1.60 |
| `vmla.i16` q + `vld1.16` q, 1:1 | 1.88 (a q load costs ~2.1 cycles) |
| **conv-tile mix, int16**: 2 + 1 q loads + 16 `vmlal.s16` by scalar (64 MACs) | **1.64** |
| **conv-tile mix, f32**: 4 q + 4 d loads + 16 `vmla.f32` q by scalar (64 MACs), loads grouped | **0.80** |
| same f32 mix with loads interleaved between MACs (clang's order today) | **0.59** |

**A float finding along the way (measured).** After a `vmla.f32`, any NEON instruction other than
another `vmla.f32` costs about **4 extra cycles**. That covers loads, `vadd.f32`, `vmul.f32` and integer
NEON ops; scalar ARM instructions are exempt.

| f32 mix | cycles per 8 instructions | no-penalty sum |
|---|---|---|
| `vmla.f32` q + `vld1` q, 1:1 | 41 | 25 |
| `vmla.f32` q + `vadd.f32` q, 1:1 | 49 | 33 |
| `vmla.f32` q + `vmul.f32` q, 1:1 | 49 | 33 |
| `vmla.f32` ×6 + `vld1` q ×2, interleaved | 37 | 29 |
| `vmla.f32` ×6 + `vld1` q ×2, **grouped** | **33** | 29 |
| `vmla.f32` q + ARM `add`, 1:1 | 21 | 21 |

Integer NEON has no such penalty: `vmla.i16` + `vadd.i16` interleaved runs at full rate.
The current `a2_full_tile<6,4>` (`build-armhf-nhead`, `llvm-objdump`) interleaves `vld1.32` between
`vmla.f32` in its inner loop, so it pays this repeatedly. Iteration 3 (44% of sampled cycles on loads)
and the gap between the kernel's 0.47 and the mix's 0.80 MAC/cycle are consistent with that. It is the
cheapest lever in this document (section 6).

## 3. Accuracy of fixed point for A2

**Method.** `scripts/quant_sim.py` (NumPy) re-implements A2 (rechannel, 23 dilated layers with bias,
mixin, LeakyReLU, head sum and 1×1 residual, head conv + scale) from `models/a2_fender_super_reverb.nam`.
It runs a float64 reference and integer-exact fixed-point variants on the first 10 s of `audio/di.wav`
(with the model's receptive field of silence first, like `prewarm`). The residual is
`20·log10(rms(diff)/rms(ref))`, the same as `null_test.py`. Checks:

- the float64 simulation vs a C++ `bench --render` of Full: **−130.2 dB** (the simulation is the model);
- `null_test.py` on the simulation's WAVs reproduces its number (−74.9 dB for the main scheme).

**Fixed-point design simulated** (the parts that map to NEON):

- **Weights** are int16 (or fewer bits), scaled per output channel. The input activation scales are
  folded into the weights.
- **Activations** use power-of-two scales (per tensor or per channel), max-abs calibrated on the whole
  69 s file.
- **Accumulation** is exact integer (int32 in the kernel; the script reports overflow headroom).
- **Bias and mixin** are added in accumulator units (mixin as cond × a Q31 weight).
- **LeakyReLU** runs on the int32 accumulator (slope as Q31, or Q15 for a `vqrdmulh.s16` variant).
- **Requantization** to int16 rounds, then saturates.
- **Residual stream** is either int16 (`resid16`) or int16 + 16 fractional bits in int32 (`resid32`;
  only the conv input/history is narrowed).
- **Head sum** is int32 (exact, power-of-two scales). The head conv (1% of MACs) stays in float.
- **Input** is 24-bit unless noted, at full scale [−1, 1), as an interface delivers it.

**Results** (`results-pi/quant_sim.csv`, residual relative to the float output, dB):

| Scheme | Lite | Full | Notes |
|---|---|---|---|
| float32 vs float64 (floor) | −133.2 | −135.3 | |
| **input quantized to 16 bit, everything else float** | **−75.3** | **−75.3** | what a 16-bit interface alone does |
| int16 weights only | −81.3 | −86.5 | |
| int16 activations only | −71.9 | −75.0 | |
| W16A16 per-tensor, resid16 | −68.0 | −70.9 | |
| W16A16 per-channel, resid16 | −69.6 | −72.9 | |
| W16A16 per-tensor, resid32 | −70.2 | −73.6 | |
| **W16A16 per-channel, resid32** | **−71.7** | **−74.9** | best all-int16 scheme, but its int32 accumulators overflow (below) |
| … with LeakyReLU slope Q15 instead of Q31 | −71.8 | −74.8 | the slope format doesn't matter |
| … with 16-bit input | −70.1 | −72.1 | |
| … with 1 bit of headroom on activations | −65.7 | −68.9 | |
| W13A16 (int32-safe) | −65.7 | −70.1 | |
| W14A15 (int32-safe) | −65.1 | −68.4 | |
| W16A13 (int32-safe) | −54.0 | −56.0 | activations need the bits more than weights |
| **mixed: layers 0–2 float, rest W16A16** | −79.0 | **−82.4** | |
| **mixed: layers 0–6 float, rest W16A16** | −82.8 | **−83.4** | best integer scheme found |
| mixed: layers 0–6 float, rest W13A16 (int32-safe) | −72.4 | −74.4 | |
| W12A12 | −46.9 | −49.8 | |
| W8A16 | −37.6 | −41.9 | |
| W16A8 | −22.4 | −25.7 | |
| W8A8 (`vmlal.s8`) | −22.5 | −25.5 | |
| fp32, round toward zero on every op (QPU semantics, section 5) | −111.7 | −107.4 | |

**Robustness** (calibration matters):

| | Lite | Full |
|---|---|---|
| W16A16 pc resid32, calibrated as recorded, input raised +2.5 dB (peak −0.1 dBFS) | **−47.8** (3,883 saturations) | **−43.8** (7,184) |
| same with 1 bit of headroom | −66.6 (0 saturations) | −70.1 (0) |
| W16A16 pc resid32, calibrated on the file normalized to 0 dBFS peak, nominal input | −69.0 | −71.3 |
| same, layers 0–6 float | −82.6 | −82.6 |

**Overflow.** With full-range int16 weights and activations, Full's conv accumulators reach 2^33.6, i.e.
**2.6 bits past int32** (Lite 1.3 bits), on this file. A `vmlal.s16` kernel must either give up 3 bits
(W13A16: −70 dB) or split the tap sum (extra instructions). `vqdmlal` saturates instead of wrapping, but
that's still an error.

**Which layers matter** (W16A16 in one layer only, the rest float; Full): layer 0 −79.5, layer 1 −79.2,
layer 2 −85.7, layer 7 −87.7 dB. All other layers are −92 to −105 dB. The first dilation cycle
dominates: its errors pass through all later layers. Lite is similar (layers 0–2: −76 to −81 dB).

**Verdict.**
- **No int16 scheme reaches −100 dB.** The practical all-int16 kernel lands around **−65 to −70 dB**
  (int32-safe, with headroom for a full-scale player), and the mixed-precision kernel around **−77 to
  −83 dB**.
- **Is that audible?** Probably not on its own. For scale, feeding the *float* model a 16-bit input
  already gives −75 dB against a 24-bit input, so a −70 dB integer kernel is a 16-bit interface's worth
  of error. But it is clearly **not** "unchanged" by the project's −100 dB rule, and it varies with
  input level and model, so it would need a blind listening test (ABX) on high-gain and clean patches.
- The roadmap's R2-9 prediction (−70 to −90 dB, miss the gate) holds.
- **Int8 is out** (−25 dB, audible).

## 4. Projected speed of an int16 kernel (estimate)

Built from the measured instruction costs above and today's Full kernel profile.

- **Inner loop.** The int16 conv-tile mix does 64 MACs in 39 cycles (1.64 MAC/cycle) vs 80 cycles for
  f32 (0.80): **2.05×** on the MAC part.
- **Today's f32 kernel** runs at 0.47 MAC/cycle, i.e. 59% of its own mix rate. For a 4-frame × 1-layer
  tile (K=6: 1,792 MACs incl. 1×1) that's ~3,810 cycles, of which ~2,240 are the MAC mix and **~1,570
  are everything else**: tap loop, ring writes, bias/mixin/LeakyReLU/head-sum/1×1 epilogue, stores and
  stalls.
- **Extra int16 work per tile** (from measured costs; **estimate ~130 cycles**):
  - requantize: 8 × `vqrdmulh.s32` ≈ 33 cycles;
  - narrow: 8 × `vqrshrn` ≈ 9;
  - LeakyReLU in int16: 4 × (`vqrdmulh.s16` + `vmax.s16`) ≈ 25;
  - residual rescale + add: ≈ 50;
  - narrow history: ≈ 9.
- **Range:**
  - *Pessimistic* (non-MAC overhead unchanged): 1,093 + 1,570 + 130 ≈ 2,790 cycles → **0.64 MAC/cycle,
    1.37×**.
  - *Realistic-good* (half the non-MAC overhead goes, because int16 halves history/weight bytes and
    loads): ≈ 2,000 cycles → **0.89 MAC/cycle, 1.9×**.
  - *Ceiling* (MAC mix + int16 extras only): ≈ 1,220 cycles → **1.47 MAC/cycle, 3.1×**.
- **Full per sample:** 27.4 µs today single core → **14–20 µs** (69–94% load on one core), or ~35–47%
  per core pipelined. **Mixed precision** (layers 0–6 float = 27% of MACs): **1.25–1.5×** overall.

So int16 could make Full fit on **one** core without the pipeline's +1 block, if it lands in the upper
half of the range, at −65 to −83 dB. For context, the current two-core build already runs Full at 66%
load, bit-exact.

## 5. VideoCore IV GPU (QPUs)

**Peak.** 12 QPUs (3 slices × 4), each 16-way SIMD built as 4 physical lanes over 4 cycles, with an
add ALU and a multiply ALU that issue together, at 250 MHz: 12 × 4 × 2 × 250 M = **24 GFLOPS**
([VC4CL performance notes](https://github.com/doe300/VC4CL/wiki/Performance);
[Wikipedia](https://en.wikipedia.org/wiki/VideoCore) gives 250 MHz for BCM2836 and the
4-way-multiplexed 16-way SIMD). The ALUs are separate, so 24 GFLOPS = **12 GMAC/s**: about 3.4× all
four A7 cores' float peak (3.5 GMAC/s), and ~13× one core. Measured on this Pi (read-only debugfs,
`/sys/kernel/debug/dri/0/v3d_ident`): **QPUs 12, slices 3, TMUs 6, semaphores 16**. Best published
realistic number: [py-videocore](https://github.com/nineties/py-videocore) reports **8 GFLOPS sgemm**
(large matrices, i.e. a third of peak).

**Float semantics.** Add/sub/mul are single precision, correctly rounded **toward zero** (not nearest);
NaN is not supported; division and special functions are approximate
([VC4CL numerical compliance](https://github.com/doe300/VC4CL/wiki/NumericalCompliance)).
Simulated round-toward-zero on every op: **Full −107 dB, Lite −112 dB** (`quant_sim.csv`). It passes the
−100 dB bar, but with less margin than the CPU's −130 dB.

**How user code runs on the QPUs, and what this Pi allows** (checked read-only on the Pi, 2026-10-25):

| Path | What it needs | This Pi |
|---|---|---|
| Firmware mailbox `execute_qpu` (GPU_FFT/hello_fft, py-videocore, VC4CL "mailbox" mode) | `/dev/vcio` + VC memory (`mem_alloc`/`mem_lock`, then map it via `/dev/mem`, or VCSM-CMA); firmware owning V3D, i.e. **no** `vc4-kms-v3d` / `vc4-fkms-v3d` overlay ([py-videocore README](https://raw.githubusercontent.com/Idein/py-videocore/master/README.rst): "does not work with the CPU-side OpenGL stack", switch to the legacy driver) | `/dev/vcio` is **`crw------- root root`** (root only); `/dev/mem` is `root:kmem` (ctu not in kmem); `config.txt` line 26 is **`dtoverlay=vc4-kms-v3d`** |
| VC4CL "register poking" | direct V3D register access (`/dev/mem`, root) | blocked |
| Open vc4 DRM/KMS driver (`/dev/dri/renderD128`) | GL ES 2.0 via Mesa; the kernel driver has **no compute job interface** ("isn't supported by this driver", [kernel vc4 docs](https://docs.kernel.org/gpu/vc4.html)) | render node accessible (ctu is in `render`/`video`), but **no Mesa installed** and no compute path anyway |
| VCSM-CMA memory (`/dev/vcsm-cma`) | video group | accessible; memory alone doesn't run code |

Also on the Pi: `vcgencmd get_mem gpu` = **76M**; CMA 256 MB (190 MB free); kernel
6.18.50+rpt-rpi-v7; Pi 2 Model B Rev 1.1.

**QPU "hello world": not attempted.** It needs root for `/dev/vcio` (and `/dev/mem` for most loaders),
and under the KMS driver it would compete with the kernel for V3D. Both are outside the no-sudo /
no-config-change rule.

**Dispatch and memory cost vs the deadline (estimates unless cited):**

- **Mailbox round trip ~100 µs** per call: GPU_FFT's documented ARM↔GPU latency
  ([Raspberry Pi blog, GPU_FFT](https://www.raspberrypi.com/news/accelerating-fourier-transforms-using-the-gpu/)),
  measured on the 700 MHz Pi 1. That's 4% of the 2.67 ms budget at 128 samples, **15% at 32**, before
  any compute.
- **Project-specific red flag (measured, iteration 12):** `vcgencmd` mailbox calls once per second
  stalled the audio cores by up to ~5 ms. A GPU model would make a mailbox call every block (375/s at
  128). Register-poking dispatch avoids the mailbox but needs root and is "less compatible"
  (VC4CL).
- **Memory.** GPU buffers must be VC-visible: uncached on the ARM side, or flushed/invalidated per
  block. Per-block I/O is tiny (128 floats each way), and the 23 history buffers would live on the GPU
  side. The bigger cost is **23 sequential layers**: each needs every QPU's results written out (VPM/DMA)
  and read back (TMU) by all QPUs, behind a barrier on the 16 hardware semaphores.
  **Estimate 5–20 µs per layer** for sync + memory round trip → 0.1–0.5 ms.

**Parallelism.**
- At 128 samples one Full layer has 8 channels × 128 frames = 1,024 independent outputs, each a
  48-MAC (K=6) or 120-MAC (K=15) dot product, vs 192 SIMD lanes: ~5 outputs per lane. That's enough
  within a layer, but tiny per launch: 1.5 M MACs per block in total, ~65 k per layer.
- At 32 samples it drops to 256 outputs per layer (1.3 per lane). Sync dominates.
- **Estimate per 128-block: 0.2–0.6 ms** (dispatch + 23 × (compute ~1–5 µs + sync 5–20 µs)). It
  fits the 2.67 ms budget on paper; at 32 samples it would be ~50% of the 0.67 ms budget.
- Published workloads that do well on VC4 are large and batched: FFT batches (GPU_FFT ~10× a 700 MHz
  ARM), sgemm (8 GFLOPS), GoogLeNet image classification ("about 3 times faster" than a Pi 3's four
  A53s, [CNX Software](https://www.cnx-software.com/2017/04/30/gpu-accelerated-object-recognition-on-raspberry-pi-3-raspberry-pi-zero/)).
  I found no published real-time, small-block audio or WaveNet work on VC4.

**What it would take** (all Christine's call, none done):
1. `sudo` edit of `config.txt`: remove `dtoverlay=vc4-kms-v3d` (the HDMI console falls back to the
   firmware framebuffer; fine headless), raise `gpu_mem` (py-videocore's tests want ≥128 MB), reboot.
2. Run as root or add a udev rule for `/dev/vcio`.
3. Check that the current firmware on trixie still serves the `execute_qpu` mailbox tag with no KMS
   overlay (**unverified**; GPU_FFT/py-videocore date from the Raspbian/Buster era).
4. Toolchain: [py-videocore](https://github.com/nineties/py-videocore) (Python assembler, "work in
   progress"), vc4asm, or VC4C/[VC4CL](https://github.com/doe300/VC4CL) (OpenCL 1.2 embedded profile,
   "still in development").
5. Write a 23-layer QPU kernel with GPU-side ring buffers, per-layer barriers and a CPU-side fallback.

**Effort estimate: 3–6 weeks**, unmaintained tools, and a real-time risk from firmware-mediated dispatch.

**Verdict: don't pursue.** The main blocker is that the GPU belongs to the vc4 KMS kernel driver, which
has no compute interface, and the only compute path (the firmware mailbox) needs root, a config change
and a reboot. Even then it adds ~0.1 ms dispatch per block through the same mailbox that already caused
multi-ms stalls in this project, to speed up a workload the two-core CPU pipeline already handles at 66%
load.

## 6. Other CPU-side ideas the FP limit suggests

- **Group the float kernel's loads (try first).** Measured: after `vmla.f32`, any other NEON instruction
  costs ~4 extra cycles. clang 16 interleaves `vld1.32` between `vmla.f32` in `a2_full_tile<6,4>`. On the
  conv-tile mix, grouped loads run at 0.80 vs 0.59 MAC/cycle interleaved.
  - Change: issue each pair's weight and input loads back to back before its 16 MACs, via `asm volatile`
    scheduling barriers or a small inline-asm inner block.
  - **Estimate:** a 10–25% faster Full kernel (the mix itself is 1.36× faster grouped).
  - Arithmetic order is unchanged, so the output should be bit-identical.
- **More cores.** The 2-stage pipeline scaled 1.97×. A **3-stage** split (layers ≈ 0–8 / 9–15 / 16–22 +
  head, rebalanced with `--pipeline-split`-style tests) would cut Full to **~1.2 ms per 128-block
  (estimate ~45% load per core)** for +2 blocks (5.3 ms) of delay. **4 stages: ~0.9 ms, +3 blocks**,
  and nothing left for the OS and IRQs.
  - More cores also mean more shared-L2 and bus contention (iteration 9's tail spikes).
  - This is better as a fallback for Full at 64 samples than as a default.
- **Mixed precision.** The layer sweep shows layers 0–2 (and 7) carry most of the int16 error. Keeping
  them float and the rest int16 gives −82 dB (section 3). Inside a layer, int16 weights with float
  activations (−86.5 dB) don't help speed: NEON has no int16 × f32 MAC.
- **Fewer MACs.**
  - **Low-rank** factorization of each layer's 8 × 8K conv matrix, without retraining: rank 4
    (1.57× fewer MACs) gives −10 dB, and rank 7 still gives −22 dB. The weights are effectively full
    rank.
  - **Unstructured pruning** doesn't help NEON (dense 8-wide math).
  - **Structured pruning** (fewer channels) is what the slimmable Lite submodel already is.
  - All of these change the model and would need **retraining or fine-tuning**, not post-hoc surgery.
    A retrained narrower model (e.g. 6 channels, 0.56× MACs) is a model-choice question for Christine,
    not a kernel trick.

## Test plan (what to run next, with the repo's tools)

**Step 1: float kernel with grouped loads (bit-identical expected).**
1. New patch `patches/a2_full_neon_tile_grouped.patch` on top of the `nhead` stack. Build into a new
   dir:
   `BUILD_DIR=build-armhf-ngroup ./scripts/cross_build.sh -DBENCH_A2_RING_MODE=0 -DBENCH_NAM_PATCHES="…;patches/a2_full_neon_tile_grouped.patch"`
2. Check the `llvm-objdump` of `a2_full_tile<6,4>`: count `vmla.f32`→`vld1` switches before and after.
3. Accuracy: render 10 s with `nhead` and `ngroup` on the Pi;
   `python3 scripts/null_test.py render_nhead_slim1.wav render_ngroup_slim1.wav --label ngroup`. Expect
   max diff 0.
4. Speed: `VARIANT=ngroup TAG=ngroup SLIM=full REPEATS=3 EXTRA="--pipeline 8" ./scripts/remote_sweep.sh`,
   plus an unpipelined run, compared with `clangref`.
5. Profile: `TAG=perf_ngroup BENCH=./build-ngroup/bench ./scripts/perfstat.sh` (IPC, MACs/cycle vs
   `perf_neon`). Adopt if p50 improves beyond run-to-run noise (~1%).

**Step 2 (only if a goal needs it): int16 Full, mixed precision.**
1. Extend `quant_sim.py` to a second, louder DI (`audio/raw/`) and a clean patch. Fix the calibration
   at a 0 dBFS-peak input with 1 bit of headroom. Gate: residual ≤ −80 dB on both, no saturation at a
   −0.1 dBFS peak.
2. Kernel: `vmlal.s16` by-scalar tile (8 out × 4 frames, int32 accumulators). Requantize with
   `vqrdmulh.s32`, then `vqrshrn`. LeakyReLU as `vmax(x, vqrdmulh(x, 0.01·2^15))`. int32 residual
   stream, int16 history rings, float layers 0–6. Handle the 2.6-bit accumulator overflow by splitting
   the tap sum (taps 0–2, 3–5 into separate int32 sums) rather than dropping weight bits.
3. Verify the C++ kernel against `quant_sim.py` (they should agree to ~−130 dB), then
   `null_test.py` vs `nhead` on the Pi (expect ≈ −80 dB). `remote_sweep.sh` unpipelined and pipelined,
   `perfstat.sh` for MACs/cycle (target ≥ 0.64). ABX listening test before any adoption.

**Step 3 (optional): 3-stage pipeline.** Generalize `a2_pipeline.patch` to N stages, then
`remote_sweep.sh` with `MODE=paced` at 64/128 and `--rt`, and a 30-minute `sustained.sh` run with
`thermal_logger.py` (three busy cores may change the thermal picture: iteration 12 saw 44 °C with two).

## What I couldn't verify

- QPU dispatch latency on this Pi, kernel version and firmware: no QPU code was run (root needed). The
  ~100 µs figure is GPU_FFT's, from a Pi 1.
- Whether `execute_qpu` still works on current firmware with the KMS overlay removed.
- QPU denormal behavior (VC4CL's compliance page lists the heading but says nothing).
- The A7's NEON datapath width from an ARM document. The 64-bit-per-cycle integer and 32-bit-per-cycle
  float behavior is inferred from `int_peak` and `neon_peak` timings.
- The int16 kernel speedup (section 4) is an estimate from instruction-mix measurements, not a kernel.
- Accuracy numbers are for one model and one DI (calibrated on the same file, in-sample). Other models
  or players may need more headroom.

## Reproduce

```
# integer peak (Mac cross build into a scratch dir, run on the Pi)
cmake -S . -B /tmp/build-int -DCMAKE_TOOLCHAIN_FILE=$PWD/cmake/armhf-toolchain.cmake && cmake --build /tmp/build-int --target int_peak
scp /tmp/build-int/int_peak pi:~/pi-neural-amp/build-int/
ssh pi 'cd ~/pi-neural-amp && ./build-int/int_peak --rt --cpu 3 --out results/int_peak.csv'
# quantization (Mac, NumPy)
python3 scripts/quant_sim.py                                   # all schemes + layer sweep + hot input
python3 scripts/quant_sim.py --calib-peak-dbfs 0 --schemes "W16A16 per-channel, resid32"
```
