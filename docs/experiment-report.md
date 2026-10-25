# Experiment report: NAM A2 on a Raspberry Pi 2

Running log of optimization iterations, following [optimization-roadmap.md](optimization-roadmap.md).
Every number below comes from a CSV in `results-pi/` (pulled from the Pi) via `scripts/kpis.py` or
the analysis commands shown. Estimates are labeled as estimates.

## Setup (fixed for all iterations unless an iteration says otherwise)

| | |
|---|---|
| Board | Raspberry Pi 2 Model B, 4× Cortex-A7 @ 900 MHz, 1 GB, Raspbian 13 (trixie) 32-bit, kernel 6.18.50+rpt-rpi-v7 (voluntary preemption) |
| Governor | `performance` (all 4 cores, 900 MHz), throttle flags 0x0 at start |
| Build | clang 16 cross-compiled on the Mac (`scripts/cross_build.sh`), `-O3 -mcpu=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=hard`, NAM Core v0.6.0 with A2 fast path |
| Model | `a2_fender_super_reverb.nam` (A2 SlimmableContainer): Lite (3 channels, 1,731 MAC/sample) and Full (8 channels, 11,776 MAC/sample) |
| Input | `audio/di.wav`: 69.4 s dry guitar, mono 48 kHz float32 |
| Block | 128 samples (deadline 2,667 µs) unless noted |
| Mode | `burst` (blocks back to back) unless noted; 100 warm-up blocks discarded; 26,025 measured blocks per run |
| Runs | `scripts/remote_sweep.sh` from the Mac: deploys the variant, runs `sweep.sh` detached on the Pi, one SSH session polls every 30 s, results pulled to `results-pi/` |
| Memory | self-reported by `bench` after the loop (PSS from `/proc/self/smaps_rollup`); the external sampler is off |

"Missed" = blocks whose `process()` time exceeded the deadline. "Load" = mean block time / deadline.

## Baseline (tag `baseline`)

Clang build, `performance` governor, sampler off, block 128, burst, 3 runs × 26,025 blocks per model.
Source: `results-pi/summary_baseline.csv` → `kpi_by_block.csv`, `kpi_memory_self.csv`.

| Model | p50 | p99 (median of runs) | p99 worst run | max | missed | load |
|---|---|---|---|---|---|---|
| **Lite** | 1,032 µs | **1,092 µs** | 1,099 µs | 1,705 µs | **0 / 78,075** | **38.9%** |
| **Full** | 6,696 µs | 6,925 µs | 6,935 µs | 7,845 µs | 78,075 / 78,075 | **251.0%** |

- **Lite runs in real time at 128 samples with 2.4× p99 headroom** (2,667 / 1,092).
- **Full cannot run in real time at any block size** as built: it needs ≥2.5× less time per sample.
  Bigger blocks don't help because load is per sample (~52 µs/sample vs a 20.8 µs budget).
- **Memory:** peak PSS 16.9 MiB for either model, of which 12.7 MiB is the decoded test file held by
  `bench`. The rest, ~4.0–4.2 MiB, is the model (both submodels are loaded), buffers and libraries.
  `live`, which streams audio, is where the resume's memory number should come from.

### Control 1: governor (`baseline_ondemand`, earlier run)

Same build, `ondemand` governor, **and the external memory sampler on** (the Pi still had the older
default). Lite p50 1,039 µs, p99 1,421 µs, **max 3,373 µs, 7 missed of 104,100**. Full p99 7,716 µs.
The misses and the >2.67 ms maxima appear only in this run, not in the sampler-on control below, so they
are attributed to the `ondemand` governor. That attribution holds only because control 2 isolates the sampler.

### Control 2: memory sampler (`ctl_sampler`)

`performance` governor, **sampler on** (`MEM_SAMPLE=1`), Lite only, 3 runs.
Spike analysis with `scripts/spikes.py` (spike = block > 1.3× the run's median, events 50 ms apart merged):

| Lite, block 128 | p99 | max | spike events / s | median gap between events |
|---|---|---|---|---|
| sampler off (`baseline`) | 1,092 µs | 1,705 µs | 0.04–0.15 | no period |
| sampler on (`ctl_sampler`) | **1,386 µs** | 1,777 µs | **3.98–4.06** | **0.253–0.254 s** |

**Verdict: the sampler causes the periodic spikes and inflates Lite's p99 by 27%** (1,092 → 1,386 µs).
It wakes every 0.2 s and starts `awk`/`sed`/`date`; with its own run time the cycle is ~0.25 s. Those
processes run on other cores but share the 512 KB L2 and memory bus. Fix (now default): `bench`/`live`
read their own PSS after the loop; the sampler is opt-in for time-series only.

## Iterations

_Each iteration: hypothesis → change → result → verdict._

### Iteration 1: profile before changing code (`perf_base`)

`scripts/perfstat.sh`: `perf stat` (user-space counters, 3 groups of ≤4 events, so no multiplexing) and
`perf record` (2 kHz sampling) on 20 s of audio per model. Output: `results-pi/perf/perf_base/`.
These runs execute under perf, so their timings are not used as results.

| Counter (user space) | Lite | Full |
|---|---|---|
| cycles | 7.46 G | 45.53 G |
| instructions | 6.63 G | 21.38 G |
| **IPC** | **0.89** | **0.47** |
| L1D load miss rate | 1.09% | 0.72% |
| L2 (LLC) loads / misses | 49.9 M / 1.06 M (2.1%) | 198 M / 14.9 M (7.5%) |
| branch miss rate | 3.3% | 4.4% |

Derived (processed samples = 20 s × 48 kHz + 100 warm-up blocks = 972,800; MACs from the model
structure; cycles also include model load and warm-up, so these are slight overestimates of cost):

| | Lite | Full |
|---|---|---|
| MACs | 1.68 G | 11.46 G |
| **instructions per MAC** | 3.9 | 1.9 |
| **MACs per cycle** | 0.23 | 0.25 |

Hotspots (`perf report`, self time):

| Lite | | Full | |
|---|---|---|---|
| `A2FastModel<3>::process` (all Lite kernels, inlined) | 89.4% | Eigen `gebp_kernel` (GEMM inner kernel) | **59.2%** |
| `memcpy` (most likely the ring buffers' mirror writes; not attributed by call stack) | 5.1% | Eigen `gemm_pack_rhs` + `gemm_pack_lhs` (repacking inputs) | **22.0%** |
| Eigen GEMM (Full's warm-up at load) | 3.2% | `A2FastModel<8>::process` (non-GEMM work) | 13.8% |
| | | GEMM driver + blocking heuristic | 1.3% |
| | | `memcpy` + `memset` | 3.1% |

**What it says:**
1. **Neither model is memory-bound.** >98.9% of L1D loads hit; Full's 7.5% L2 miss rate applies to a
   small stream (198 M L2 loads vs 12.4 G L1 loads). The roadmap's memory items drop down the list.
2. **Full is overhead-bound inside Eigen.** 22% of its time only repacks 8×N inputs for a general
   matrix multiply, 179 times per block (156 conv taps + 23 1×1 convs). IPC 0.47 on an in-order core with NEON says the GEMM kernel itself
   is stalling too. → iteration 3 (`lazyProduct`) targets exactly the 23% of packing + setup.
3. **Lite is scalar.** 3.9 instructions per MAC and no vector code (confirmed by the Mac codegen check).
   → iteration 2 (vectorize Lite's loops).
4. **`memcpy` is 5% of Lite**, most likely the ring mirror. → ring mode 0 test (iteration 4) will tell.

### Iteration 2: vectorize Lite with `-ffast-math` on `a2_fast.cpp` (`a2fm`)

**Hypothesis.** Lite's loops compile to scalar VFP code because ARMv7 NEON isn't IEEE-754 and the
compiler won't vectorize float math without permission. Allowing it (`-ffast-math`, only on
`a2_fast.cpp`) should turn on 4-wide NEON across frames and cut Lite's time. Estimate from the roadmap:
1.3–2× faster.

**Change.** `cross_build.sh -DBENCH_A2_FASTMATH=ON` (build `build-armhf-fm`). The binary has 625 NEON f32
q-register arithmetic instructions vs 386 in the base build (`llvm-objdump`). Recorded in `build_flags`
as `a2_flags=[-ffast-math]`.

**Result** (block 128, burst, 3 runs × 26,025 blocks):

| | Lite p50 | Lite p99 | Lite max | Full p50 | Full p99 |
|---|---|---|---|---|---|
| baseline | 1,032 µs | 1,092 µs | 1,705 µs | 6,696 µs | 6,925 µs |
| `a2fm` | **1,140 µs (+10.5%)** | **1,195 µs (+9.4%)** | 1,515 µs | 6,739 µs (+0.6%) | 6,958 µs (+0.5%) |

Profile of the `a2fm` Lite build (`results-pi/perf/perf_a2fm/`, 20 s of audio):

| Lite | baseline | `a2fm` |
|---|---|---|
| instructions | 6.63 G | **2.95 G (−55%)** |
| cycles | 7.46 G | **8.26 G (+11%)** |
| IPC | 0.89 | **0.36** |

Hottest instructions in the vectorized Lite code (`perf annotate`): `vld3.32` de-interleaving loads,
`vorr` register-to-register copies, and `vldmia`/`add rX, sp, #…` loads from the stack.

**Verdict: refuted. Don't adopt.** The compiler did vectorize (fewer than half the instructions),
but on the Cortex-A7 the vector code is slower:
- the A7's NEON datapath is 64 bits wide, so each 128-bit (q-register) operation issues as two halves
  (widely reported for ARM's in-order A-series NEON units, but I found no ARM document stating it for the
  A7; consistent with the IPC drop, not measured directly here);
- Lite stores 3 channels interleaved per frame, so vectorizing across frames needs `vld3`/`vst3`
  de-interleaving, which is multi-cycle on this core;
- 9 weights + accumulators + inputs exceed ARMv7's 16 q-registers, so the loop spills to the stack
  (the `vorr` copies and `sp`-relative loads).
- found later (round-2 research): LLVM marks `cortex-a7` as slow at VMLA and schedules it with the
  Cortex-A8 model, so clang emitted separate `vmul` + `vadd` instead of multiply-accumulate. This likely
  contributed too ([LLVM ARMProcessors.td](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/ARM/ARMProcessors.td)).
The roadmap's 1.3–2× estimate assumed vector instructions would be roughly as cheap as scalar ones;
on this core they aren't. A hand-written kernel would have to change the data layout (channel-major,
so no de-interleaving) and keep the working set inside 16 q-registers to win. That's the roadmap's
item 5, now with a concrete reason. No null test was run since the variant isn't adopted.

### Iteration 3: Full without Eigen's GEMM packing (`lazy`)

**Hypothesis.** 23% of Full's time is Eigen repacking inputs and setting up a general matrix multiply
for each 8×8 × 8×128 product (iteration 1). `lazyProduct` asks Eigen for a plain coefficient-based
product instead, with no packing. If packing is pure overhead, Full should get ~20% faster.

**Change.** `patches/a2_full_lazyproduct.patch` (two lines in `a2_fast.cpp`: the per-tap conv product
and the 1×1 product), applied to a build-dir copy of NAM (`-DBENCH_NAM_PATCHES=…`, build
`build-armhf-lazy`, `patches=[a2_full_lazyproduct]` in `build_flags`).

**Result** (Full, block 128, burst, 3 runs):

| Full | p50 | p99 | max | load |
|---|---|---|---|---|
| baseline | 6,696 µs | 6,925 µs | 7,845 µs | 251.0% |
| `lazy` | **7,099 µs (+6.0%)** | **7,320 µs (+5.7%)** | 8,253 µs | 266.3% |

Profile (`results-pi/perf/perf_lazy/`, 20 s of audio):

| Full | baseline | `lazy` |
|---|---|---|
| instructions | 21.38 G | **17.16 G (−20%)** |
| cycles | 45.53 G | **48.33 G (+6%)** |
| IPC | 0.47 | **0.36** |
| L1D loads | 12.38 G | **14.57 G (+18%)** |
| hotspots | GEMM kernel 59%, packing 22% | Eigen assignment loops 69% (no packing), A2 code 27% |

Hottest instructions: `vld1.32 {dN[]-dM[]}` (load one weight and broadcast it to all lanes) and
`vld1.32` column loads, roughly evenly spread.

**Verdict: refuted. Don't adopt.** Packing disappeared, as intended (−20% instructions), but the
coefficient-based product re-loads every weight for every one of the 128 columns instead of keeping
weights in registers. The A7 is now load-bound (more loads, lower IPC). Eigen's packing was paying for
itself after all. What Full needs is a register-blocked kernel: keep a tile of weights in NEON
registers across all columns of the block. With ARMv7's 16 q-registers, all 64 weights (16 q-regs)
can't stay resident with room for inputs and accumulators, so the tile has to be smaller, e.g. 4
output channels × 8 inputs (8 q-regs) + 4 accumulators + inputs. That's the roadmap's fused NEON
kernel (item 4), and iterations 2–3 now give its design constraints from measurements.

### Iteration 4: ring-buffer mode 0 (`ring0`)

**Hypothesis.** ~5% of Lite's time is `memcpy` (iteration 1), most likely ring mode 1 writing the
mirrored tail of each layer's history buffer every block. Mode 0 (a linear buffer that rewinds with a
`memmove` only when it reaches the end) should do less copying on average, possibly with rare spikes
at each rewind.

**Change.** `-DBENCH_A2_RING_MODE=0` (build `build-armhf-ring0`; `a2_flags=[-DNAM_A2_RING_MODE=0]`).

**Result** (Lite, block 128, burst, 3 runs):

| Lite | p50 | p99 (median) | p99 worst run | max | missed |
|---|---|---|---|---|---|
| baseline | 1,032 µs | 1,092 µs | 1,099 µs | 1,705 µs | 0 |
| `ring0` | **991 µs (−4.0%)** | **1,067 µs (−2.3%)** | 1,069 µs | **1,405 µs (−18%)** | 0 |

No periodic spikes from the rewinds (`spikes.py`: 0–1 events per run).

**Accuracy.** Rendered 10 s with both builds on the Pi; `null_test.py`: **bit-identical** for Lite and
Full (max difference 0). Ring mode only changes where history lives, not the arithmetic.

**Full** (`ring0_full`, 3 runs): p50 6,712 µs vs 6,696 (+0.2%), p99 6,940 vs 6,925 µs, max 7,810 vs
7,845 µs. No measurable change: Full's `memcpy` share was only 2.5%, and its time is in Eigen.

**Verdict: adopt.** First positive result, and it's free: a compile flag, identical output, faster on
average and in the tail for Lite, neutral for Full. **`ring0` is the reference build from here on.**

### Iteration 5: Lite's smallest buffer, paced, with and without `--rt` (`lite_paced`, `lite_paced_rt`)

**Question.** Back-to-back mode says Lite has 2.4× headroom at 128. How small can the block go in a
realistic callback pattern (sleep until each deadline, `--paced`), and what does real-time
scheduling buy? `--paced` also measures **wake-up latency**: how late the thread wakes after each sleep.
A block counts as "missed with wake" when wake lateness + process time exceeds the deadline.

**Setup.** `ring0` build, Lite, blocks 16/32/64/128, 2 runs each (each run = the 69 s file in real time).
`--rt` = SCHED_FIFO priority 80 + pinned to CPU 3 + `mlockall`. Stock kernel (PREEMPT_VOLUNTARY), no
isolcpus.

| Block (deadline) | p50 | p99 | max | wake p99 | wake max | missed (process) | missed incl. wake |
|---|---|---|---|---|---|---|---|
| 16 (333 µs) | 136 | 179 | 390 | 73 | 313 | 4 / 416,400 | **72** |
| 16, `--rt` | 135 | 173 | 343 | **16** | 154 | 2 / 416,400 | **4** |
| 32 (667 µs) | 260 | 304 | 473 | 76 | 290 | 0 | 0 |
| 32, `--rt` | 258 | 299 | 564 | **24** | 179 | 0 | 0 |
| 64 (1,333 µs) | 504 | 562 | 962 | 78 | 258 | 0 | 0 |
| 64, `--rt` | 502 | 555 | 810 | **19** | 139 | 0 | 0 |
| 128 (2,667 µs) | 998 | 1,062 | 1,278 | 80 | 213 | 0 | 0 |
| 128, `--rt` | 997 | 1,078 | 1,237 | **20** | 107 | 0 | 0 |

(µs. Source: `results-pi/summary_lite_paced*.csv` → `kpi_by_block.csv`.)

**Findings.**
- **Smallest glitch-free block for Lite: 32 samples (0.67 ms)**, with or without `--rt`, 0 misses in
  208,200 blocks. At 16 samples it fails both ways.
- **`--rt` cuts wake-up latency ~4× at p99** (73–80 → 16–24 µs) and roughly halves the worst case. At
  16 samples that turns 72 deadline misses into 4. It doesn't change compute time (p50 within 1%).
- **Per-sample cost rises at small blocks** (p50 per sample: 8.5 µs at 16 vs 7.8 µs at 128): fixed
  per-call overhead. At 16 samples, compute p99 alone is 52% of the deadline before scheduling noise.
- What's left at 16 + `--rt` (4 misses, 154 µs worst wake) is what isolcpus / PREEMPT_RT target.
- The model's 0.67 ms block is far below the Scarlett's USB floor (Reddit reports ~2–3 ms round
  trip), so the live round-trip test will likely show the interface, not the model, sets Lite's latency.

### Iteration 6: register-blocked NEON kernels for Full and Lite (`neon`)

**Hypothesis.** Iterations 1–3 showed Full loses its time to Eigen's per-call packing and to weight
reloads, and Lite to scalar code plus, with fast-math, `vld3` and spills. Round-2 research also found
clang models `cortex-a7` as slow at VMLA and splits it into `vmul` + `vadd`. A hand-written kernel that
keeps accumulators in registers across all taps, uses VMLA explicitly, and never de-interleaves should win.

**Change.** `patches/a2_full_neon_tile.patch` (Full: 8 output channels × 4 frames per tile, all taps +
bias + mixin + LeakyReLU + head sum + 1×1 fused, 8 q accumulators) and `patches/a2_lite_neon_lanes.patch`
(Lite: 3 output channels in lanes, 4-frame tiles, no `vld3`), stacked on `ring0`. Build `build-armhf-nboth`.

**Result** (block 128, burst, 3 runs):

| | p50 | p99 | max | load | vs `ring0` |
|---|---|---|---|---|---|
| Full, `ring0` | 6,712 µs | 6,940 µs | 7,810 µs | 252% | — |
| **Full, `neon`** | **3,511 µs** | **3,732 µs** | 4,491 µs | **132%** | **1.91× faster** |
| Lite, `ring0` | 991 µs | 1,067 µs | 1,405 µs | 37.4% | — |
| **Lite, `neon`** | **849 µs** | **921 µs** | 1,246 µs | **32.1%** | **1.17× faster** |

**Accuracy (on the Pi).** 10 s renders vs `ring0`: **Lite bit-identical**; **Full −132.8 dB** relative
(max sample difference −131.4 dBFS), from a different summation order. Far below audibility.

**Profile** (`results-pi/perf/perf_neon/`, 20 s of audio):

| | Full `ring0`-era baseline | Full `neon` | Lite baseline | Lite `neon` |
|---|---|---|---|---|
| instructions | 21.38 G | **6.55 G** | 6.63 G | **2.05 G** |
| cycles | 45.53 G | 24.24 G | 7.46 G | 6.12 G |
| IPC | 0.47 | 0.27 | 0.89 | 0.34 |
| MACs per cycle | 0.25 | **0.47** | 0.23 | 0.28 |

Full now spends 92% of its time in the two kernel instantiations (`a2_full_tile<6,4>` 78%,
`<15,4>` 15%). Lite: `a2_lite_tile4` 83%, `memcpy` 4%.

**Verdict: adopt.** Full nearly halved, but it still needs **1.32×** more to run in real time.
Lower IPC is expected now: most remaining instructions are 4-cycle NEON q operations (iteration 7).

### Iteration 7: what the Cortex-A7 can actually do (`neon_peak`)

**Why.** Instruction counts fell much more than cycles in iterations 2 and 6, so the cost per
instruction matters. I couldn't find a citable A7 timing table, so I measured it.

**Method.** `src/neon_peak.cpp`: for each instruction form, a loop of 8 instructions written in inline
asm (so the compiler can't substitute), 160 M instructions, `--rt` on CPU 3, cycles = time × 900 MHz.
"Independent" = 8 different accumulators (throughput); "chain" = all on one accumulator (latency).
Source: `results-pi/neon_peak.csv`.

| Instruction (f32) | cycles / instr | float ops / cycle |
|---|---|---|
| `vmla` q (by scalar or by vector), `vfma` q, `vmul` q, `vadd` q | 4.13 | **0.97** |
| `vmla` d, `vfma` d | 2.13 | **0.94** |
| scalar VFP `vmla` s, `vfma` s | 1.13 | **0.89** |
| `vmla` q / d dependent chain | 5.01 | (latency 5 cycles) |
| `vadd` q dependent chain | 4.13 | |

**Finding: on the Cortex-A7, every FP form peaks at ~1 float operation per cycle.** A 4-lane q
instruction takes 4 cycles, a 2-lane d instruction 2, scalar 1. NEON doesn't multiply arithmetic
throughput here (the NEON/VFP unit appears to process one 32-bit lane per cycle). What vectorizing
*can* still do is cut loads, address math and loop overhead around the MACs. That explains why
iteration 2 (more instructions per useful MAC from `vld3` and spills) lost, and why iteration 6 won
(fewer instructions per MAC).

**What it means for Full** (from these measurements; ceilings are arithmetic, not measured):
- Single-core ceiling: 11,776 MACs per sample at ~1 MAC/cycle ≈ 11,800 cycles ≈ **13.1 µs per sample**
  vs the 20.8 µs budget, i.e. **~63% load at best** on one core.
- `neon` runs at 0.47 MACs/cycle (27.4 µs per sample, 132% load). Real time needs ≥0.63 MACs/cycle,
  so the kernel must spend ≥75% of cycles issuing MACs (today ~48%). That's possible in principle but
  tight: loads, the bias/activation/1×1 tail and the head all compete for the same in-order pipeline.
- The other lever is **more cores**: the A7 has four, and three sit idle. A two-stage pipeline
  (roadmap R2-3) doubles available MAC throughput for +1 block of latency.
