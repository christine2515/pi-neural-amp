# Optimizations summary: every change tried, with its measured effect and status

> **2026-10-25.** One row per optimization, grouped by category. Every number is copied
> from [experiment-report.md](../experiment-report.md), [low-latency-plan.md](../low-latency-plan.md),
> [beyond-fpu.md](../beyond-fpu.md) or a CSV in [`results-pi/`](../../results-pi/); the tag or file is in each
> row. **Estimate** = arithmetic or a cited source, not a Pi measurement. For how the best build runs
> live, see [state-of-the-union.md](state-of-the-union.md).

Conditions unless a row says otherwise: Pi 2, clang 16 cross build, `performance` governor, block 128
(deadline 2,667 µs), burst, 3 runs × 26,025 blocks. Accuracy = null-test residual vs the previous build
(`null_tests.csv`, `x_null_tests.csv`); the project bar is ≈ −100 dB.

**Status:** ✅ adopted · ❌ rejected (why) · ⏳ pending (Claude can run it) · 🧑 needs Christine
(sudo, reboot, kernel, hardware, cable) · ➖ measured, informs design only.

## 1. Compute kernels

| Optimization | What it does | Measured effect (source) | Accuracy | Status |
|---|---|---|---|---|
| Profile first (`perf stat/record`) | Find where time goes before changing code | Not memory-bound (>98.9% L1 hits); Full 59% Eigen GEMM + **22% repacking**; Lite all scalar (`perf/perf_base/`) | n/a | ➖ |
| `-ffast-math` on `a2_fast.cpp` | Let clang auto-vectorize Lite | Lite p50 1,032 → **1,140 µs (+10.5%)**; instructions −55% but IPC 0.89 → 0.36 (`kpi_by_block.csv` `a2fm`) | not tested | ❌ slower: `vld3` de-interleave, q-register spills, clang's slow-VMLA model |
| Eigen `lazyProduct` | Skip GEMM packing for Full's 8×8 products | Full p50 6,696 → **7,099 µs (+6.0%)**; L1 loads +18% (`lazy`) | not tested | ❌ weights reloaded per column; packing was paying for itself |
| Register-blocked NEON tile, Full (`a2_full_neon_tile`) | 8 out × 4 frames of accumulators in q registers across all taps; fused bias/activation/skip/1×1 | Full p50 6,712 → **3,511 µs (1.91×)** (`ring0_full` → `neon`) | −132.8 dB (summation order) | ✅ |
| NEON lanes kernel, Lite (`a2_lite_neon_lanes`) | 3 output channels in lanes, 4-frame tiles, no `vld3` | Lite p50 991 → **849 µs (1.17×)** (`ring0` → `neon`) | bit-identical | ✅ |
| NEON output head (`a2_head_neon`) | Head conv as one 16×C dot product, two accumulators | Full (pipelined) 1,786 → **1,767 µs (−1.1%)**; Lite 855 → **843 µs (−1.4%)** (`clangref` → `head`) | −129.9 dB Full, −135.7 dB Lite | ✅ |
| A7 FP ceiling microbenchmark (`neon_peak`) | Measure cycles per FP instruction form | **~1 float op/cycle in every form** (q 4.13, d 2.13, scalar 1.13 cycles); VMLA latency 5 (`neon_peak.csv`) | n/a | ➖ explains the two losses above |
| Load-cost microbenchmark (`kernel_peak`) | Cost of NEON loads next to VMLA | Loads don't overlap with NEON math: +2/+3/+6 cycles per d/q/32 B load (`x_kernel_peak*.csv`) | n/a | ➖ |
| Conv inner loop in inline asm | Software-pipelined loads, fewer load instructions | **+4.8%** on the conv, **+1.0%** on the whole tile, in L1 (`x_kernel_peak.csv`); ~−3% Full in-model (**estimate**) | bit-identical | ⏳ ready in `src/kernel_peak.cpp`, not a patch |
| Epilogue polish (mixin `vmla`, bias-first accumulators, 1×1 from registers) | Cut q-ops after the tap loop (epilogue = 24% of the tile, measured) | ~5–8% Full combined (**estimate**, low-latency-plan §4.3) | bit-identical / ~−130 dB | ⏳ |
| Group float loads before VMLAs | Avoid the ~4-cycle penalty after `vmla.f32` | Conv-tile instruction mix 0.59 → 0.80 MAC/cycle in isolation (measured, beyond-fpu §6); 10–25% Full (**estimate**) | bit-identical | ⏳ (overlaps with the asm-conv row; that measured only +1% on the full tile) |
| Software prefetch (`pld`) | Prefetch long-dilation taps | ≤7% upper bound (in-L1 vs in-model gap), likely 2–3% (**estimate**) | bit-identical | ⏳ |
| NEON int16 (`vmlal.s16`) | Fixed-point instead of float | Peak 1.88 vs 0.97 MAC/cycle (`int_peak.csv`); kernel 1.4–1.9× (**estimate**) | **−65 to −75 dB** (`quant_sim.csv`) | ❌ fails the −100 dB bar |
| Mixed precision (layers 0–6 float, 7–22 int16) | Keep the sensitive layers float | 1.25–1.5× (**estimate**) | −82 to −83 dB | ❌ still fails the bar; would need a listening test |
| int8 / 16-bit accumulators | Lower precision still | 3.76 MAC/cycle peak | W8A8 −25 dB; 16-bit accumulators overflow | ❌ audibly wrong |
| Soft-float emulation | Emulate floats on the integer core | **52–75× slower** than hardware VFP (`int_peak.csv`) | exact (libgcc) | ❌ |

## 2. Parallelism (spare cores)

| Optimization | What it does | Measured effect (source) | Accuracy | Status |
|---|---|---|---|---|
| Two-stage pipeline (`a2_pipeline`) | Layers 0–12 on the audio thread, 13–22 + head on a worker core; two atomics, spin-wait | Full p50 3,511 → **1,785 µs (1.97×)**, load 132% → 67% (`neon` → `pipe_full`); **+1 block of delay** | bit-identical after a 1-block shift | ✅ |
| Pipeline split point | Balance the two stages | Split 13: 1,781 µs; 12/14: ~1,920 µs; ~140 µs per layer off balance (`summary_pipe_split.csv`) | bit-identical | ✅ split 13 is the default |
| Pipelined Lite | Same, for Lite | Clean at 16 samples paced `--rt` (0 / 416,400), but 16 + 16 = 32 samples total delay, same as unpipelined 32 (`pipe_lite_paced_rt`) | bit-identical | ❌ buys headroom, not latency |
| Frame split (`a2_frame_split`) | Each layer's frames split over 2–4 cores, one barrier per layer, **no extra block** | Full 128: 2 cores **1,770 µs** (= pipeline, without its delay), 4 cores **943 µs**; paced 3 cores at 48: 13 / 138,800 missed, 2.0 ms digital delay (`summary_x_fs*.csv`, `summary_x_fsp*.csv`) | bit-identical (`x_null_tests.csv`) | ⏳ offline only: needs `--fsplit` flags in `live` (diff in low-latency-plan §7) and a live run; below ~2 ms it needs isolated cores 🧑 |
| 4 threads on the stock kernel | Use all four cores | Single **41–51 ms** stalls (RT throttling with a FIFO spinner on every core, inference) | — | ❌ keep core 0 for the OS |
| 3-stage pipeline | Three cores in series | ~1.2 ms per 128-block, +2 blocks (**estimate**, beyond-fpu §6) | bit-identical (expected) | ⏳ superseded by frame split |

## 3. Memory and buffers

| Optimization | What it does | Measured effect (source) | Accuracy | Status |
|---|---|---|---|---|
| Ring mode 0 (`-DBENCH_A2_RING_MODE=0`) | Linear history buffer with rare rewinds, instead of mirroring every block | Lite p50 1,032 → **991 µs (−4.0%)**, max −18%; Full neutral (+0.2%) (`ring0`, `ring0_full`) | bit-identical | ✅ |
| Ring mode 1, dirty-only mirror (`a2_ring_mirror_dirty`) | Mirror only the columns written this block | 0 ± 2% vs `ring0` (**estimate**); copies 36% of mode 1 (Mac check) | bit-identical (Mac) | ⏳ never run on the Pi; low value |
| Pipeline worker stack (`a2_pipeline_stack`) | Worker as a pthread with a 256 KB stack, so `mlockall` doesn't lock an 8 MB default stack | Full pipelined live PSS **13,778 → 5,842 kB**; p99 1,071 vs 1,079 µs (`live_summary.csv` `memfix_build`, `memfix_build-nhead2`) | unchanged code path | ✅ |
| Huge pages / cache partitioning | — | ≈0 (dTLB misses negligible); A7 L2 has no partitioning (**estimate**) | — | ❌ not applicable |

## 4. Build and compiler

| Optimization | What it does | Measured effect (source) | Accuracy | Status |
|---|---|---|---|---|
| clang 16 cross build on the Mac | Fast iteration; the kernels were tuned against its register allocator | Reference build for all results | — | ✅ |
| GCC 14.2 native | Compare compilers | Lite **−6.3%** (801 µs); Full pipelined **+26%** (2,249 µs): 100 vs 13 stack refs in the tile (`gcc14`) | not tested | ❌ Full needs clang |
| Global VMLA flag (`-slowfpvmlx` off) | Let clang emit VMLA everywhere | Lite +0.7%, Full +0.5%: noise (`vmla`) | not tested | ❌ no effect; hot loops already use VMLA |
| Per-file `-O2` vs `-O3` | Smaller code | ≈0 (**estimate**; I-cache misses < 0.1% of cycles, measured) | — | ❌ skipped |
| A2 fast path vs generic WaveNet | NAM's hand-optimized A2 path | Not measured on the Pi (RUNBOOK §4c) | — | ⏳ optional, for the record |

## 5. OS and scheduling

| Optimization | What it does | Measured effect (source) | Accuracy | Status |
|---|---|---|---|---|
| `performance` governor | Fixed 900 MHz | `ondemand` gave Lite max 3,373 µs and 7 misses (`baseline_ondemand`) | n/a | ✅ |
| `--rt` (SCHED_FIFO 80, pinned CPU 3, `mlockall`) | Fix wake-up latency | Wake-up p99 **73–80 → 16–24 µs**; Lite misses at 16 samples 72 → 4; compute unchanged (`rt_comparison.csv`); Full paced 64: 33 → 1 missed (`pipe_full_paced*`) | n/a | ✅ |
| Worker at the same FIFO priority on CPU 2 | Pipeline worker can't be preempted by normal tasks | Part of all pipelined `--rt` rows | n/a | ✅ |
| `isolcpus` / `nohz_full` / `irqaffinity=0` | Keep the tick, IRQs and housekeeping off the audio cores | Not measured; targets the 100 Hz tick (+40–150 µs) and an every-2 s 200–400 µs event (measured, low-latency-plan §3.4) | n/a | 🧑 sudo + reboot; `nohz_full` needs a kernel rebuild |
| PREEMPT_RT kernel | Bound kernel latency | Not measured | n/a | 🧑 |
| USB IRQ affinity | `dwc_otg` lands on all four cores today (measured) | Not measured | n/a | 🧑 |
| Overclock 900 → 1,000 MHz | More compute | ≤10% (**estimate**) | n/a | 🧑 |
| FTZ in the audio thread | Flush denormals in scalar code | ≈0 (**estimate**; NEON already flushes) | changes only denormals | ⏳ low priority |

## 6. Audio I/O (live, Scarlett Solo, `--rt --cpu 3`)

| Optimization | What it does | Measured effect (source) | Accuracy | Status |
|---|---|---|---|---|
| Small periods, 2 periods | Minimum in-device buffering | Lite **32 × 2 = 1.33 ms**, 0 underruns; Full pipelined **64 × 2 + 64 = 4.0 ms**, **1 in 450,000** periods over 10 min (`small_live`, `lsw_*`, `live10_full_pipe_64x2`) | n/a | ✅ |
| 16-sample periods | Go below 32 | bypass 16 × 2: 45,082 / 90,000 underruns; Lite 16 × 3: 15 / 60,000; 16 × 4: 1 / 60,000 (`small_live`, `lsw_lite16_*`) | n/a | ❌ USB floor at 2 periods; 16 × 4 is no shorter than 32 × 2 |
| ALSA mmap (`--mmap`) | Skip the read/write copy | Lite unchanged; Full pipelined **2 periods fails** (688 / 15,000 at 64; 7,517 / 7,500 at 128); 3 periods clean (`lsw_*_mmap`) | n/a | ❌ keep read/write access |
| Noise gate (`--gate -50`) | Mute input below −50 dBFS before the model (4 dB hysteresis, 80 ms hold, 1 ms attack, 100 ms release; no look-ahead) | Amp noise in silence **−58.3 → −88.5 dBFS (−30 dB)**; playing 0.0 dB change offline (`gate_compare.csv`); 2 underruns in that 30 s take (`rec_full_gate50`) | no added latency; playing unchanged | ✅ (gated take's dry input was also 5.4 dB quieter) |
| I2S ADC/DAC instead of USB | Remove the USB floor | Round trip ~1.2–1.8 ms (**estimate**) | n/a | 🧑 hardware |

## 7. Measurement hygiene

| Fix | What it does | Measured effect (source) | Status |
|---|---|---|---|
| Memory sampler off; PSS self-reported | Sampler forked `awk`/`sed` every 0.2 s | It raised Lite p99 **1,092 → 1,386 µs (+27%)**, spikes every 0.254 s (`ctl_sampler`, `spikes.csv`) | ✅ |
| Fork-free thermal logger (`thermal_logger.py`) | Old logger called `vcgencmd` (firmware mailbox) every second | 30 min Full: overruns **66 → 7**, stalls > 4 ms 8 → 0, max 7,279 → 2,702 µs (`summary_sustained_full_pipe*.csv`) | ✅ |
| Governor control | Isolates `ondemand` from the sampler | See section 5 | ✅ |
| Paced mode + wake-up timing | Measure like an audio callback | Separates scheduler lateness from compute (`wake_p99_us` in `kpi_by_block.csv`) | ✅ |
| Null test on every adopted change | Prove the sound didn't change | `null_tests.csv`, `x_null_tests.csv` | ✅ |
| `live` timing arrays scale with `--duration` | Per-period arrays preallocated | 10-min run showed 12.8 MB PSS (≈7 MB of arrays); model footprint stays 5.8 MB | ➖ quote memory from short runs |

## 8. Model-level

| Option | What it does | Measured effect (source) | Accuracy vs Full | Status |
|---|---|---|---|---|
| Lite submodel (`--slim 0.25`) | The file's own 3-channel model | 1,731 vs 11,776 MAC/sample; 843 vs 1,767 µs | −14.3 dB (`x_model_surgery_null.csv`) | ✅ a choice, not an optimization |
| Low-rank conv taps (SVD, no retraining) | Fewer MACs | rank 3 saves 25% of conv; rank 2 saves 50% | −15.9 / −11.1 dB (`x_model_surgery_null.csv`) | ❌ as far from Full as Lite already is |
| Channel pruning (no retraining) | 8 → 7 or 6 channels | ~23% / ~44% saved | −6.0 / −5.3 dB | ❌ |
| Retrained smaller model | — | Not measurable here | — | 🧑 model choice |

## 9. Hardware offload

| Option | What it does | Measured / found (source) | Accuracy | Status |
|---|---|---|---|---|
| VideoCore IV QPUs | 12 QPUs, 24 GFLOPS peak (cited) | No compute path: `vc4-kms-v3d` overlay owns V3D, `/dev/vcio` root-only, KMS driver has no compute job interface (checked on the Pi, beyond-fpu §5) | −107 dB (simulated) | ❌ blocked, and risky for real time |
| I2S codec | See section 6 | — | — | 🧑 |

## Cumulative before / after (block 128 unless noted)

| Step (tag) | Full p50 / p99 | Full load | Lite p50 / p99 |
|---|---|---|---|
| Baseline (`baseline`) | 6,696 / 6,925 µs | 251% | 1,032 / 1,092 µs |
| + ring mode 0 (`ring0`, `ring0_full`) | 6,712 / 6,940 | 252% | 991 / 1,067 |
| + NEON kernels (`neon`) | 3,511 / 3,732 | 132% | 849 / 921 |
| + two-core pipeline (`pipe_full`) | 1,785 / 1,928 | 67% | (not pipelined) |
| + NEON head (`head`) | **1,767 / 1,910** | **66%** | **843 / 917** |
| Total | **3.79×** | | **1.22×** |
| Live, best config (`live10_full_pipe_64x2`, `small_live`) | 64 × 2: 967 / 1,084 µs vs 1,333 µs deadline; 1 underrun in 450,000 | | 32 × 2: 236 / 303 µs vs 667 µs; 0 / 45,000 |
| Live memory (PSS) | 5,842 kB (`memfix_build-nhead2`) | | 5,582 kB (`first_live`) |

`pipeline_stack` doesn't change compute (p99 1,071 vs 1,079 µs live), so `head` is the last compute row.

## Best build

```
BUILD_DIR=build-armhf-nhead2 ./scripts/cross_build.sh -DBENCH_A2_RING_MODE=0 \
  "-DBENCH_NAM_PATCHES=patches/a2_full_neon_tile.patch;patches/a2_lite_neon_lanes.patch;patches/a2_pipeline.patch;patches/a2_head_neon.patch;patches/a2_pipeline_stack.patch"
```

clang 16, `-O3 -mcpu=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=hard`, NAM Core v0.6.0 + the five patches
(order matters: head and stack apply after the pipeline). Recorded in `build_flags` of the
`memfix_build-nhead2` and `live10_full_pipe_64x2` rows of `live_summary.csv`. Not in this build:
`a2_frame_split`, `a2_ring_mirror_dirty`, `a2_full_lazyproduct`.
