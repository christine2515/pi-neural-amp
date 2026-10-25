# 05: CPU performance on ARMv7 / Cortex-A7

## SIMD and NEON basics

- **SIMD** = one instruction operates on several values (lanes). NEON is ARM's SIMD unit.
- **Registers:** 16 **q** registers (128-bit = 4 floats), which alias 32 **d** registers (64-bit = 2 floats).
- `vmla.f32` = multiply-accumulate. `vfma` = fused version (one rounding).
- **ARMv7 NEON isn't IEEE-754** (it flushes denormals to zero), so GCC and clang **won't auto-vectorize
  float code unless you allow it** (`-ffast-math`). Measured: Lite's code had **0** NEON float
  instructions by default and **231** with fast-math.

## What the Cortex-A7 can actually do (measured, `neon_peak`)

| Instruction (f32) | Cycles each | Float ops / cycle |
|---|---|---|
| q (4 lanes): vmla / vfma / vmul / vadd | 4.13 | **0.97** |
| d (2 lanes): vmla / vfma | 2.13 | 0.94 |
| scalar VFP vmla / vfma | 1.13 | 0.89 |
| vmla dependent chain | 5.01 | latency **5 cycles** |

**Every FP form peaks at ~1 float lane-op per cycle.** A MAC is 2 FLOPs, so that's ~1 MAC per cycle per
core. **NEON doesn't multiply arithmetic throughput on this core.** It only helps by cutting the
*other* instructions (loads, index math, loop overhead).

The explanation that "the A7's NEON datapath is 64 bits wide" is widely reported but **unverified**:
no ARM document stating it was found. The numbers above are measured; the reason behind them is a
hypothesis.

**What this implies for Full:** 11,776 MACs at a perfect 1 MAC/cycle ≈ 13.1 µs per sample vs a
20.8 µs budget. So even a perfect single core reaches only ~63% load (arithmetic, not measured).

## In-order cores, latency and stalls

- The A7 issues instructions **in order**. If an instruction needs a result that isn't ready (a load,
  or a VMLA with 5-cycle latency), the pipeline waits.
- **Hide latency** with independent work: several accumulators, each updated far apart. The kernels
  space dependent MACs 8 instructions apart; the head kernel uses 2 accumulators.
- The NEON Full kernel profile: **~44%** of sampled cycles on loads (`vld1`/`vldr`), 41% on `vmla`.

## Register pressure and spills

- 16 q registers is tight. An 8×8 weight matrix alone fills all 16.
- **Spill** = the compiler runs out of registers and stores values to the stack, adding loads and stores.
- **GCC 14 vs clang 16 on the same Full kernel:** 581 instructions and **100** stack references
  vs 388 and **13**. Full ran **+26% slower** with GCC (Lite ran 6% faster).

## Why "obvious" vectorization lost

**Fast-math on Lite** (compiler vectorizes across frames):
- Instructions **−55%**, but cycles **+11%**; IPC 0.89 → 0.36. Lite became **10% slower**.
- Hot: `vld3` (de-interleaving 3-channel data, multi-cycle), `vorr` copies and stack loads (spills).
- Also: LLVM models `cortex-a7` as slow at VMLA, so clang emitted `vmul` + `vadd` instead.

**Eigen `lazyProduct` on Full** (skip GEMM packing):
- Instructions **−20%** (packing gone), but L1 loads **+18%** and IPC 0.47 → 0.36: **6% slower**.
- Hot: `vld1` broadcasts. Every weight was reloaded for each of the 128 columns.
- **Lesson:** Eigen's packing was paying for itself by enabling register reuse.

## What won: register blocking

- **Idea:** keep a tile of accumulators in registers across **all taps** and the layer's tail (bias,
  mixin, LeakyReLU, head sum, 1×1), so each weight load serves several frames.
- **Full kernel:** 8 output channels × 4 frames = 8 q accumulators, explicit `vmlaq_lane_f32`, an
  `asm volatile` fence, and `target("no-slowfpvmlx")` (clang) to force real VMLA.
- **Lite kernel:** 3 channels in lanes, 4-frame tiles, **no `vld3`**.
- **Result:** Full **1.91×** (instructions 21.4 G → 6.6 G), Lite **1.17×**. Lite output bit-identical;
  Full −133 dB.

## Reading IPC correctly

After the NEON kernel, Full's IPC **fell** from 0.47 to **0.27** while it got 1.9× faster. Most
instructions are now 4-cycle q ops, so low IPC is expected. Judge by **MACs per cycle** (0.25 → 0.47)
and wall time, not IPC alone.

## Smaller wins and non-wins

- **Ring buffer mode 0 vs 1:** mode 1 copies a mirrored tail every block (`memcpy` 5% of Lite). Mode 0:
  Lite p50 −4%, max −18%, bit-identical. Full unchanged.
- **Head conv as one dot product:** the 16 taps are contiguous in memory, so each output is one
  dot product (48 or 128 floats). −1.1% Full, −1.4% Lite.
- **Global VMLA flag:** no effect; the hot loops already used VMLA.

## Key takeaways

> - On the A7, NEON ≈ scalar for FP throughput (~1 lane-op/cycle). Vectorize only to cut overhead.
> - In-order core: hide latency with independent accumulators; avoid spills.
> - Fewer instructions can be slower (vld3, spills, reloads). Measure cycles.
> - Register blocking (reuse weights across frames) is what made Full fast.

## Check yourself

1. Why did fast-math cut Lite's instructions 55% yet make it slower?
2. What does a VMLA latency of 5 cycles mean for how you order accumulator updates?
3. Why can IPC drop while performance improves?
4. Why does ARMv7 need permission to auto-vectorize float math?
