# Experiment report: NAM A2 on a Raspberry Pi 2

Running log of optimization iterations, following [optimization-roadmap.md](optimization-roadmap.md).
Every number below comes from a CSV in `results-pi/` (pulled from the Pi) via `scripts/kpis.py` or
the analysis commands shown. Estimates are labeled as estimates.

## Summary (so far)

| | First baseline | Best so far | How |
|---|---|---|---|
| **Full**, block 128, p50 | 6,696 µs (251% load, misses every block) | **1,767 µs (66% load)** | NEON kernel (1.91×) + two-core pipeline (1.97×) + NEON head (1.01×) = **3.79×** |
| **Full**, smallest clean block (paced) | none | **128 samples**: 7 marginal overruns (≤35 µs) in 30 min, no stalls; 256 clean | + 1 block pipeline delay, `--rt` |
| **Full**, 30 min sustained | — | p50 1,786 / p99 1,922 µs, flat; 44 °C peak, never throttled | |
| **Lite**, block 128, p50 | 1,032 µs (39% load) | **843 µs (32%)** | ring mode 0 + NEON kernel + NEON head (1.22×) |
| **Lite**, smallest clean block (paced, `--rt`) | — | **32 samples (0.67 ms)**, or 16 pipelined (same total delay) | |
| Memory (bench PSS minus test file) | ~4 MiB | — | measured, not optimized |

What didn't work, and why (measured): compiler auto-vectorization of Lite (−10%: `vld3`, spills, and
clang's slow-VMLA model for the A7), Eigen `lazyProduct` for Full (−6%: weights reloaded per column).
What explains everything: on the Cortex-A7, **every FP form peaks at ~1 float op per cycle**, so
vectorization only pays by cutting non-arithmetic instructions, and the second core is the big lever.
Measurement lesson: the memory sampler (+27% Lite p99) and the thermal logger (stalls up to 7 ms) were
the biggest tail-latency sources found; both are now fixed (self-reported memory, fork-free logger).

Also tried and measured: a global clang VMLA flag (no effect) and a native GCC 14 build (Lite −6%,
Full +26% from register spills in the hand kernel; clang kept). **Best build: `nhead`** = ring mode 0 +
`a2_full_neon_tile` + `a2_lite_neon_lanes` + `a2_pipeline` + `a2_head_neon`, clang 16.
Remaining Claude-runnable item: int16 Full (R2-9, research; expected to miss the −100 dB accuracy bar).

Waiting on Christine: core isolation (sudo + reboot), PREEMPT_RT kernel, Scarlett connected (live,
mmap, round-trip latency), I2S hardware.

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
  `live`, which streams audio, is where a real-world memory number should come from.

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

### Iteration 8: two-core pipeline for Full (`pipe_full`)

**Hypothesis.** Iteration 7 puts a single-core ceiling of ~63% load on Full (at a perfect 1 MAC/cycle),
and the `neon` kernel reaches 0.47 MACs/cycle (132% load). Squeezing another 1.32× out of one in-order
core is uncertain (44% of the kernel's sampled cycles sit on loads). Three of the four cores are idle.
Splitting the 23 layers across two cores should roughly halve the per-call time, for one block of
extra latency.

**Change.** `patches/a2_pipeline.patch` (on top of both NEON patches and `ring0`; build
`build-armhf-npipe`). Stage A = input rechannel + layers 0–12 on the calling (audio) thread; stage B =
layers 13–22 + head on a worker thread. Two buffer sets alternate: during call *n*, the worker runs
stage B of block *n−1* while the audio thread runs stage A of block *n*, then the audio thread waits
for the worker and returns block *n−1*'s output. Handoff is two atomics (no locks, no syscalls); the
worker spin-waits with `yield` and is pinned/prioritized at model load (`bench --pipeline 8`,
`--pipeline-cpu`, and with `--rt` it gets the same SCHED_FIFO priority). Prewarm runs unpipelined.

**Accuracy.** Output is exactly one block late (128 samples = 2.67 ms). After a 128-sample shift,
the pipelined render is **bit-identical** to the unpipelined `neon` render, on the Mac and on the Pi
(`null_test.py --shift 128`).

**Result** (Full, block 128, burst, 3 runs):

| Full | p50 | p99 | max | missed | load | CPU |
|---|---|---|---|---|---|---|
| `neon` (1 core) | 3,511 µs | 3,732 µs | 4,491 µs | 100% | 132% | 100% |
| **`pipe_full` (2 cores)** | **1,785 µs** | **1,928 µs** | 2,734 µs | **1 / 78,075** | **67%** | 200% |

**Verdict: adopt for Full.** 1.97× faster per call than `neon`, which is almost perfect two-way
scaling (the split puts 91 vs 88 cost units on each side). **Full now runs in real time at 128
samples** in back-to-back mode, with 1.4× p99 headroom, for +2.67 ms latency and a second core spinning.
The single miss (2.73 ms) shows the tail needs `--rt` and core isolation; paced runs follow.
Overall Full speedup vs the first baseline: 6,696 → 1,785 µs = **3.75×**.

### Iteration 9: pipelined Full in paced mode, with and without `--rt` (`pipe_full_paced*`)

`npipe` build, `--pipeline 8`, paced, blocks 64/128/256, 2 runs each. With `--rt` the audio thread is
SCHED_FIFO 80 on CPU 3 and the worker SCHED_FIFO 80 on CPU 2. Throttle flags 0x0, 40 °C afterwards.

| Block (deadline) | p50 | p99 | max | wake p99 | missed (process) | missed incl. wake | load |
|---|---|---|---|---|---|---|---|
| 64 (1,333 µs) | 906 | 1,001 | 1,685 | 100 | 14 / 104,100 | **33** | 68.6% |
| 64, `--rt` | 901 | 989 | 1,438 | **27** | 1 / 104,100 | **1** | 68.2% |
| 128 (2,667 µs) | 1,796 | 1,914 | 2,433 | 100 | 0 / 52,050 | **0** | 67.8% |
| 128, `--rt` | 1,789 | 1,901 | 2,766 | 59 | 2 / 52,050 | **2** | 67.5% |
| 256 (5,333 µs) | 3,610 | 3,779 | 4,733 | 140 | 0 / 26,024 | **0** | 67.8% |
| 256, `--rt` | 3,591 | 3,753 | 4,609 | 40 | 0 / 26,024 | **0** | 67.4% |

(µs. Source: `results-pi/summary_pipe_full_paced*.csv` → `kpi_by_block.csv`.)

**Findings.**
- **Full is usable live at 128–256 samples.** At 256: 0 misses in both arms. At 128: 0 misses without
  `--rt` and 2 with it (2 of 52,050 = 0.004%; the two misses are compute spikes to 2.77 ms, not
  late wake-ups). With 2 runs each, 0 vs 2 is within noise; neither arm is reliably clean at 128 yet.
- **At 64 samples, `--rt` matters:** 33 misses → 1, mostly by cutting wake-up p99 from 100 to 27 µs.
- **Remaining misses are rare compute spikes** (max 2.4–2.8 ms against a p99 of 1.9 ms). The likely
  sources are interrupts and other tasks on cores 2/3 and contention in the shared L2 between the two
  stages. Those are what `isolcpus` + IRQ affinity (needs sudo + a reboot) and PREEMPT_RT address.
- Load is flat at ~68% across block sizes: the pipeline's per-block sync cost is small.

**Status vs the goal.** Full: real time at 128 with a ~1-in-25,000 miss rate; clean at 256 (5.3 ms
buffer + 5.3 ms pipeline delay). Lite: clean at 32 samples. Next levers for Full's tail: core
isolation (Christine), a 3-stage pipeline (more latency), or more single-core kernel work.

### Iteration 10: where to split the pipeline (`pipe_split`)

Full, `--pipeline 8 --pipeline-split S`, block 128, burst, 30 s of audio, 2 runs per split
(`results-pi/summary_pipe_split.csv`):

| Split (first stage-B layer) | p50 | p99 | max | missed / 22,500 |
|---|---|---|---|---|
| 11 | 2,057 µs | 2,212 µs | 2,468 µs | 0 |
| 12 | 1,918 µs | 2,064 µs | 2,905 µs | 2 |
| **13** | **1,781 µs** | **1,933 µs** | 2,751 µs | 1 |
| 14 | 1,923 µs | 2,067 µs | 2,974 µs | 2 |
| 15 | 2,198 µs | 2,362 µs | 3,179 µs | 7 |

**13 is optimal**, matching the cost model (layers 0–12 vs 13–22 + head ≈ 91 vs 88 units: the
two kernel-15 layers sit at indices 14–15). Each layer moved off balance costs ~140 µs per block.
Kept as the default.

### Iteration 11: pipelined Lite at 16 samples (`pipe_lite_paced_rt`)

`npipe` build (NEON Lite + pipeline), `--pipeline 3 --rt --cpu 3`, paced, 2 runs each.

| Lite, paced, `--rt` | p50 | p99 | max | missed incl. wake |
|---|---|---|---|---|
| 16, unpipelined (`lite_paced_rt`, scalar Lite) | 135 | 173 | 343 | 4 / 416,400 |
| **16, pipelined** | **60** | **82** | 310 | **0 / 416,400** |
| 32, unpipelined (`lite_paced_rt`, scalar Lite) | 258 | 299 | 564 | 0 / 208,200 |
| 32, pipelined | 115 | 140 | 417 | 0 / 208,200 |

(µs. The unpipelined rows are the `ring0` scalar build; the pipelined rows also include the NEON
Lite kernel, so the speedup mixes both changes.)

**Finding.** Pipelined Lite is clean at 16 samples with p99 at 25% of the deadline. But the
pipeline adds one block, so 16 + 16 = 32 samples of total delay, the same as unpipelined 32. **For
Lite, the pipeline buys headroom, not lower latency.** Single-core NEON Lite at 32 samples is the
simpler configuration with the same latency.

### Iteration 12: 30 minutes sustained, and the thermal logger as a disturber (`sustained_full_pipe`, `ctl_nologger`)

**Run 1** (`sustained_full_pipe`): Full, `npipe` build, `--pipeline 8 --rt --cpu 3`, block 128, paced,
1,800 s, with `thermal_logger.sh` at 1 Hz (which called `vcgencmd measure_clock` and `vcgencmd
get_throttled` every second). Chart: `results-pi/charts/thermal_full_pipe.png`.

- **Thermal:** 38.5 °C idle → **44.9 °C** peak, plateau after ~4 minutes; CPU at 900 MHz the whole time;
  throttle / capping / under-voltage flags never set. The Pi 2 handles two busy cores without a heatsink.
- **Performance stable:** per-minute p50 1,792 → 1,800 µs (+0.5% over 30 min), p99 2,047–2,065 µs.
- **But 66 compute overruns in 675,000 blocks** (2.2 per minute), 8 of them > 4 ms, max 7,279 µs. Each
  long stall also left the schedule behind, so 3,027 blocks (0.45%) started late while catching up. A
  real ALSA stream would see each stall as one underrun and restart, so 66 is the better estimate of
  audible glitches.

**Control** (`ctl_nologger`): identical, 600 s, **no logger**:

| 10 minutes, same config | compute overruns | stalls > 4 ms | max | late incl. catch-up |
|---|---|---|---|---|
| with logger (run 1, first 600 s) | 26 | 6 | 6,954 µs | 1,035 |
| **no logger** | **0 / 225,000** | **0** | **2,644 µs** | **1** |

**Verdict: the thermal logger caused the stalls.** `vcgencmd` talks to the VideoCore firmware through
a mailbox. Calling it every second stalled the audio cores (2 and 3) for up to ~5 ms. This is the second
time monitoring has disturbed the measurement (the memory sampler in Control 2), so the rule is now:
**nothing that forks or talks to firmware may run periodically during a timed run.**
**Fix, attempt 1** (run 2, `sustained_full_pipe2`): the shell logger reading only sysfs (temperature,
CPU frequency, hwmon under-voltage alarm) and calling `vcgencmd get_throttled` only at start and exit.
Overruns fell from 66 to 14 in 30 minutes, but **late wake-ups of 1.5–2.9 ms every ~2.1 s** remained,
i.e. on every other logger tick (the loop takes ~1.07 s). The logger still forks `cat`/`cut`/`date`
each tick and reads cpufreq and hwmon files that may go through the firmware.

**Fix, attempt 2:** `scripts/thermal_logger.py`, one Python process that never forks. Each second it
reads only the temperature register from sysfs. Frequency, under-voltage alarm and `get_throttled` are
read only at start and exit (the "has occurred" bits are sticky, so the exit reading covers the run).

| 10 minutes, Full pipelined, `--rt`, 128, paced | overruns | wake-ups > 1 ms | late incl. catch-up | max | wake max |
|---|---|---|---|---|---|
| no logger (`ctl_nologger`) | 0 | 0 | 1 | 2,644 µs | 224 µs |
| **Python logger (`ctl_pylogger`)** | **0** | **0** | **0** | 2,610 µs | 252 µs |
| shell logger, sysfs only (run 2, first 600 s) | 3 | 741 | 858 | 3,494 µs | 4,867 µs |
| shell logger, `vcgencmd` every second (run 1, first 600 s) | 26 | 826 | 1,035 | 6,954 µs | 5,142 µs |

**The Python logger is indistinguishable from no logger.** The 30-minute run is repeated with it
(run 3, below). The shell logger is removed from the repo.

**Run 3** (`sustained_full_pipe3`): the same 30-minute configuration with `thermal_logger.py`.
Chart: `results-pi/charts/sustained_full_pipe3.png`.

| 30 min, Full pipelined, `--rt`, 128, paced | run 1 (vcgencmd logger) | run 2 (sysfs shell logger) | **run 3 (Python logger)** |
|---|---|---|---|
| blocks | 675,000 | 675,000 | 675,000 |
| p50 / p99 (whole run) | 1,796 / 2,056 µs | 1,790 / 2,039 µs | **1,786 / 1,922 µs** |
| compute overruns | 66 | 14 | **7** (each 0–35 µs over) |
| stalls > 4 ms | 8 | 4 | **0** |
| max block time | 7,279 µs | 7,093 µs | **2,702 µs** |
| wake-ups > 1 ms / wake max | many / 5.1 ms | 741 in 10 min / 4.9 ms | **0 / 195 µs** |
| late incl. catch-up | 3,027 | 2,352 | **7** |
| temperature (idle → peak) | 38.5 → 44.9 °C | 37.9 → 44.9 °C | 39.0 → 44.4 °C |
| throttle / capping / under-voltage | never | never | never (0x0 at start and exit) |

**Verdict.** Over 30 minutes the pipelined Full model holds p50 1,784–1,789 µs and p99
1,917–1,932 µs in every minute. There are no thermal effects; the Pi 2 plateaus at ~44 °C with two
busy cores. What remains is 7 marginal overruns (≤35 µs over the 2,667 µs deadline, ~1 every
4 minutes). With a third ALSA period they would most likely be absorbed by the extra buffer, but that is
an inference until `live` runs on the Scarlett. Isolating cores 2–3 from the OS is the measured-next
lever. **Lesson for the write-up:** two of the three biggest "tail latency problems" in this project
were caused by the measurement tools (the memory sampler and the thermal logger), not the model.

### Iteration 13: let clang emit VMLA everywhere (`vmla`)

**Hypothesis.** Round-2 research found LLVM treats `cortex-a7` as slow at VMLA and splits multiply-adds into
`vmul` + `vadd`. The NEON kernels already override this per function (`target("no-slowfpvmlx")`). Turning
it off for the whole build (`-Xclang -target-feature -Xclang -slowfpvmlx`) should help the remaining
scalar and Eigen code. Estimate (roadmap R2-4): 0–15% on scalar code.

**Change.** Same as `neon` (both NEON patches, `ring0`) plus the flag via `CMAKE_CXX_FLAGS` (build
`build-armhf-nvmla`; `build_flags` now records `cxxflags=[…]`). `llvm-objdump`: 841 `vmla.f32` vs 654,
and 416 `vmul.f32` vs 606. The flag took effect.

**Result** (block 128, burst, 3 runs, unpipelined):

| | Lite p50 / p99 | Full p50 / p99 |
|---|---|---|
| `neon` | 849 / 921 µs | 3,511 / 3,732 µs |
| `vmla` | 855 / 925 µs | 3,527 / 3,739 µs |

**Verdict: no effect (+0.5–0.7%, within run-to-run noise). Don't adopt.** The hot loops are the hand
kernels, which already use VMLA. The code the flag changed accounts for a few percent of the time, and
iteration 7 showed VMLA and separate VMUL + VADD have about the same throughput on the A7 anyway
(both ~1 float op per cycle). The split mainly costs extra instructions, which matters only in code
that's already instruction-bound.

### Iteration 14: head convolution as one NEON dot product (`head`)

**Hypothesis.** The output head (kernel 16, `Channels` → 1) ran as a scalar triple loop with a masked
index computation per tap. Its 16 taps for one frame are 16 consecutive history columns, contiguous in
memory in both ring modes, so each output is one dot product of 16 × `Channels` floats (48 for Lite,
128 for Full). Estimate: the head is 2.8% (Lite) / 1.1% (Full) of the MACs, so ≤1–2%.

**Change.** `patches/a2_head_neon.patch` (after the pipeline patch): flattened weights, two NEON
accumulators (to cover VMLA's 5-cycle latency, iteration 7), scalar fallback off ARMv7. Build
`build-armhf-nhead` = `npipe` + head.

**Result** (block 128, burst, 3 runs; reference `clangref` = `npipe` measured in the same session):

| | Lite p50 / p99 | Full (pipelined) p50 / p99 | Full missed |
|---|---|---|---|
| `clangref` | 855 / 932 µs | 1,786 / 1,938 µs | 2 / 78,075 |
| **`head`** | **843 / 917 µs (−1.4%)** | **1,767 / 1,910 µs (−1.1%)** | 4 / 78,075 |

**Accuracy (Pi).** vs `neon`: Lite −135.7 dB, Full −129.9 dB relative (summation order).

**Verdict: adopt.** Small, as predicted, and free. `nhead` (ring0 + Full tile + Lite lanes + pipeline +
head) is the new best build.

### Iteration 15: GCC 14 native build vs clang 16 cross build (`gcc14`)

**Question.** Is clang (cross-compiled for convenience) costing performance? Built natively on the Pi with
Raspbian's GCC 14.2 and the same three patches + `ring0` (`build-gcc`; build time 12 min 24 s with `-j2`).

| | Lite p50 / p99 | Full (pipelined) p50 / p99 | Full missed / load |
|---|---|---|---|
| clang 16 (`clangref`) | 855 / 932 µs | **1,786 / 1,938 µs** | 2 / 67.6% |
| GCC 14 (`gcc14`) | **801 / 885 µs (−6.3%)** | 2,249 / 2,398 µs (**+26%**) | 30 / 84.7% |

**Why Full is slower with GCC** (`objdump` of `a2_full_tile<6,4>` on the Pi):

| | instructions | `vmla` | stack references (`sp`) |
|---|---|---|---|
| clang 16 | 388 | 128 | 13 |
| GCC 14 | 581 | 136 | **100** |

GCC spills the kernel's register-blocked tile (8 q accumulators + weights + inputs, sized to ARMv7's 16
q registers) to the stack, adding ~50% more instructions, mostly loads and stores. The kernel was written
and checked against clang's register allocator (`asm volatile` fences, `target("no-slowfpvmlx")` is
clang-only). The Lite kernel is small enough that GCC's scheduling wins there.

**Verdict.** Keep clang for Full (the configuration that needs the speed). GCC would buy Lite 6%,
which it doesn't need (32% load). A kernel written in inline asm or with GCC-specific register pressure
tuning could make the two compilers equivalent; not worth it for this project.


## Live: Scarlett Solo 3rd gen on the Pi (`live`, `nhead` build)

**Setup.** Scarlett Solo 3rd gen (`hw:USB`), USB 2.0 high speed (480M), synchronous endpoints (no
implicit feedback), 125 µs packet interval, S32_LE, 2 in / 2 out, 48 kHz. Guitar in input 2 (INST),
which is **capture channel 1**. Level check (90 s `arecord`, `results/x_level_check.wav` on the Pi):
guitar peaks −6.8 to −2.7 dBFS, 0 clipped samples; input 1 (empty) at a −91 dBFS noise floor.
**Power warning:** `vcgencmd get_throttled` = `0x50000` (under-voltage and throttling occurred since
boot; the kernel logged "Voltage normalised" as the Solo connected). Use a stronger supply or a powered hub.
No headphones yet (the Solo's headphone jack is ¼"), so these runs measure underruns and timing, not sound.

All runs: `--rt --cpu 3`, 2 periods, 30 s (`results-pi/live_summary.csv`, labels `first_live`, `small_live`):

| Config | ALSA buffer | Underruns | Model time p50 / p99 / max | CPU | PSS |
|---|---|---|---|---|---|
| bypass, 128 × 2 | 5.33 ms | 0 / 11,250 periods | 0.8 / 1.4 / 2.2 µs | 2% | 4.4 MB |
| Lite, 128 × 2 | 5.33 ms | 0 / 11,250 | 893 / 988 / 1,134 µs | 36% | 5.6 MB |
| Full pipelined, 128 × 2 | 5.33 ms + 128-sample pipeline | 0 / 11,250 | 1,875 / 2,062 / 2,314 µs | 174% | 13.8 MB* |
| bypass, 32 × 2 | 1.33 ms | 0 / 45,000 | — | 8% | |
| **bypass, 16 × 2** | 0.67 ms | **45,082 / 90,000** | — | | |
| Lite, 64 × 2 | 2.67 ms | 0 / 22,500 | 448 / 509 / 681 µs | 38% | |
| **Lite, 32 × 2** | **1.33 ms** | **0 / 45,000** | 236 / 303 / 435 µs (deadline 667) | 44% | |
| **Full pipelined, 64 × 2** | 2.67 ms + 64-sample pipeline | **0 / 22,500** | 969 / 1,085 / 1,340 µs (deadline 1,333) | 178% | |

\*PSS is inflated by `mlockall` locking the pipeline worker's default 8 MB stack (see live-testing-todo).

**Findings.**
- **Live works at the offline limits.** Lite runs at 32 × 2 (1.33 ms ALSA buffer) and Full pipelined at
  64 × 2 with no underruns in 30 s each. Live model time is ~6% above `bench` (Full 1,875 vs 1,767 µs at
  128), likely sample-format conversion and USB interrupt work on the audio core.
- **The USB path stops between 16 and 32 frames** with 2 periods: bypass at 16 × 2 underruns on every
  other period with no model at all. That's the interface/driver floor at 2 periods (16 × 3 untested).
- Period arrival jitter (p99) is ~25% of the period at every size (170 µs at 32, 340 µs at 64, 675 µs at
  128); interpretation: wake-ups follow USB transfer completion, not exact period boundaries. Not yet investigated.
- Digital delay of the best configs: Lite 2 × 32 = 64 samples (1.33 ms); Full 2 × 64 + 64 = 192 samples
  (4.0 ms). The round trip adds the Solo's converters and USB, measured next with the loopback test
  (needs an RCA → ¼" cable: the Solo's line outputs are RCA).

### Live: background hum diagnosis (`scripts/hum_check.py`, `results-pi/hum_check.csv`)

The silent parts of the live recordings had an audible buzz: 60 Hz mains hum and its harmonics,
including a cluster at 2.4–2.9 kHz that the amp model boosts (~+14 dB at that level). 10 s silent
recordings of the guitar input (capture channel 1), one change at a time. Earlier countdown-based
attempts were discarded: the changes hadn't been made before recording, so all showed identical levels.

| Step | Setup | Hum, 60 Hz series | Buzz 2.4–2.9 kHz | Total |
|---|---|---|---|---|
| B | guitar unplugged, normal gain | −107.7 dBFS | −109.6 | −89.7 |
| E | unplugged, gain at minimum | −118.6 | −120.4 | −101.1 |
| A | plugged in, guitar volume 0 | **−126.2** | −122.1 | −105.1 |
| C | volume partly up, hand on strings | −69.5 | −92.4 | −66.8 |
| C2 | same, not touching | −68.7 | −90.2 | −66.2 |
| D | same, guitar moved/turned away from electronics | −75.8 | −94.7 | −69.9 |
| (ref) | earlier recording's quiet passage, another position | ~−89 | ~−94 | −72 |

**Verdict.** The hum comes from the guitar's **single-coil pickups** (Squier Strat) picking up 60 Hz
magnetic fields. It is position-dependent: 7 dB from moving the guitar, ~20 dB between positions
overall. The Solo, cable, Pi, USB and power are clean: −126 dBFS with the guitar volume at 0, and the
unplugged floor scales with the gain knob, so nothing is injected after the preamp. Touching the
strings made no difference. Fixes, in order: face away from power bricks and screens; Strat
selector positions 2/4 (hum-cancelling if the middle pickup is reverse-wound, reverse-polarity,
unverified for this guitar); a noise gate before the model (no added latency); noiseless pickups or
shielding.

### Live sweeps: smallest clean buffer, mmap, periods (`lsw_*` labels in `results-pi/live_summary.csv`)

Scarlett Solo, `--rt --cpu 3`, guitar on channel 1, 20 s per configuration; underruns / periods:

| Config | 32 × 2 | 32 × 3 | 64 × 2 | 64 × 3 | 128 × 2 | 128 × 3 |
|---|---|---|---|---|---|---|
| Lite | **0** / 30,000 | 0 | 0 | 0 | 0 | 0 |
| Lite + `--mmap` | 0 | 0 | 0 | 0 | 0 | 0 |
| Full pipelined | — | — | **0** / 15,000 | 0 | 0 | 0 |
| Full pipelined + `--mmap` | — | — | **688** / 15,000 | 0 | **7,517** / 7,500 | 0 |

Lite at 16 frames: 16 × 3 → 15 / 60,000; 16 × 4 → 1 / 60,000 (bypass at 16 × 3 and 16 × 4: 0). With 3+
periods the USB path handles 16-frame periods; with 2 it doesn't (earlier: bypass 16 × 2 underran every
other period).

**Findings.**
- **Smallest clean buffers, live:** Lite **32 × 2 = 1.33 ms**; Full pipelined **64 × 2 = 2.67 ms** + one
  64-sample pipeline block = **4.0 ms** of in-device buffering (clean in this 20 s run; earlier 30 s takes
  at 64 × 2 had 0, 1 and 2 underruns, so "near-zero" until the 10-minute run below).
- **ALSA mmap (Reddit idea): no benefit.** Lite's timing and underruns are unchanged with mmap. Pipelined Full
  with mmap and **2 periods fails** (688 underruns at 64, every period at 128) while 3 periods is clean.
  Cause not investigated: the model times are the same, so it's an mmap + 2-period start/refill interaction,
  not compute. Keep the default read/write access.
- Live model times match offline `bench` (Lite 32: p99 294 µs; Full pipelined 64: p99 1,074 µs).

### Live memory: the pipeline worker's stack (`patches/a2_pipeline_stack.patch`)

Under `--rt`, `mlockall(MCL_FUTURE)` locked the pipeline worker's default 8 MB thread stack into RAM, so
pipelined Full showed 13.8 MB PSS vs Lite's 5.6 MB (5,582 kB). The patch starts the worker as a pthread with a 256 KB
stack. Full pipelined, 64 × 3, 15 s each (`memfix_*`): **PSS 13,778 → 5,842 kB**, p99 1,071 vs 1,079 µs,
0 underruns both. Adopted (build `nhead2`, deployed to the Pi's `build/`). **Live memory: Lite 5.6 MB,
Full pipelined 5.8 MB** (whole process: model, ALSA buffers, libraries).

### Live noise gate (`live --gate`, `src/common.h` NoiseGate)

Threshold chosen from data: hum peaks −56 to −53 dBFS at the worst guitar positions vs a median playing
peak of −13 dBFS → open at −50 dBFS, close at −54 (4 dB hysteresis), 80 ms hold, 1 ms attack ramp,
100 ms release; no look-ahead, so **no added latency**. Full pipelined 64 × 2, 30 s takes; amp output in
the 10 quietest 0.1 s windows:

| Take | Audible noise (AC) | DC offset (inaudible, model bias) |
|---|---|---|
| without gate | −58.3 dBFS | −70.0 dBFS |
| **with `--gate -50`** | **−88.5 dBFS (−30 dB)** | −70.3 dBFS |

Offline on the same recording, playing passages were unchanged (0.0 dB). The gated take had 2 underruns in
30 s (max 1,992 µs).

### Live, 10 minutes: Full pipelined at 64 × 2 (`live10_full_pipe_64x2`)

`nhead2` build, `--pipeline 8 --rt --cpu 3`, 64 × 2 periods (2.67 ms ALSA buffer + 64-sample pipeline
= 4.0 ms in-device buffering), 600 s, fork-free thermal logger. Per-period data: `results-pi/live10_periods.csv`.

- **1 underrun in 450,000 periods** (10 minutes).
- Model time p50 967 µs, p99 1,084 µs, max 1,709 µs (deadline 1,333 µs). **43 periods (0.01%) took
  longer than one period**, mostly by < 100 µs. The 2-period buffer absorbed all but one of them, so a
  compute overrun isn't automatically an audible glitch.
- Temperature ~39.5 → 46.5 °C (the log's 37.9 °C minimum is from before the run window); no new throttling (the start reading's 0x50000 is the earlier under-voltage
  event at plug-in, which is sticky since boot).
- PSS 12.8 MB in this run: `live`'s preallocated per-period timing arrays grow with `--duration`
  (450,000 × 16 bytes ≈ 7 MB here). The model's own footprint is the 5.8 MB measured in the 15 s run.

### Live: period alignment and frame split (2026-10-25)

**Period-arrival jitter explained** (`live10_periods.csv`): at 64-frame periods (1.333 ms), periods arrive
in a 1.5 / 1.5 / 1.0 ms pattern (62% / 28%), summing to exactly 4.000 ms per 3 periods. The USB driver
delivers in 0.5 ms chunks. Periods that are multiples of 0.5 ms (24/48/96 frames) arrive evenly:
read-to-read p1/p99 489/514 µs at 24 × 3, 984/1,018 at 48 × 2 (`quant__*` rows). Lite 24 × 3 (1.5 ms) is clean;
24 × 2 underruns every other period (the USB floor at 2 periods).

**2 s kernel event:** the `raspberrypi_hwmon` mailbox poll (every 2.08 s, `x_mbox_events.csv`) is ruled out:
2 of 22 polls fell near a spike, the same as chance. Spikes cluster at 10 ms (100 Hz tick). Identifying the
2.000 s event needs a kernel trace (docs/sudo-session.md step 2). USB IRQs land on all 4 cores
(~2,000/s per core); moving them is step 1 there.

**Frame split live** (`patches/a2_frame_split.patch`, 3 threads on cores 3/2/1, `NAM_A2_FSPLIT*` env, no
pipeline, build `nfs`): 20 s runs with 0 underruns at 48 × 2 (2.0 ms), 32 × 3 (2.0 ms), 64 × 2 and 48 × 3.
**10 minutes at 48 × 2 (`live10_full_fsplit3_48x2`): 1 underrun in 600,000 periods**; p50/p99/max
595/769/1,772 µs (deadline 1,000); 226 periods over deadline were absorbed by the buffer; 43.3 → 49.2 °C,
no new throttling. **Full's in-device buffering drops from 4.0 ms (pipeline, 64 × 2) to 2.0 ms** at the same
underrun rate, using 3 cores (CPU 267%).

### Open-item follow-ups (2026-10-25)

**mmap with 2 periods** (`mmapdbg_*`): sampling `/proc/asound/card2/pcm0p/sub0/status` mid-run (Lite, 64 × 2):
read/write keeps `appl_ptr − hw_ptr` at 60–126 frames; mmap keeps ~64–68 and dropped to 2, and its
`hw_ptr` reset to near 0 twice in 8 s (stream restarts; 3 underruns this run, 0 in the earlier 20 s sweep).
So with mmap less audio is queued ahead of the USB endpoint (`lowlatency=Y` in snd-usb-audio), leaving less
slack for slow blocks, which is why it fails first with Full. Mechanism observed; the driver-level root cause
is a hypothesis (the low-latency playback path behaves differently for mmap commits). Decision: keep
read/write access.

**Frame split stalls** (`summary_x_fs_stalls.csv`, 60 s paced, block 32):
- 4 threads (cores 3, 2, 1, 0): p50 294 µs, 38 misses (0.04%), 13 stalls > 5 ms (max 7.4 ms) in two clusters.
  `dmesg` has **no "RT throttling activated"** message since boot (2026-10-24 17:38), which covers the
  low-latency agent's 4-core runs (02:47), so the earlier 41–51 ms stalls were **not RT throttling**.
  Most likely cause (not traced): the worker on core 0, which also runs the OS and the USB IRQs (~2,000/s per
  core); every layer barrier waits for it.
- 2 threads at 32: no slow stretches in 60 s, but 330 misses (0.37%) at 76% load; simply too tight.
- Decision: 3 threads on cores 1–3, core 0 left to Linux (as used live).

**Declined, with reasons:**
- Kernel load scheduling (roadmap R2-5/agent estimate 5–8%): a hand-scheduled asm conv measured +1% on the
  whole tile (docs/low-latency-plan.md); not worth the maintenance.
- Mixed compilers (GCC for Lite, clang for Full): GCC's Lite gain is 6% (`gcc14`) on a model already at
  ~32–40% load; Full is what limits latency, and clang wins there by 26%.
- Pipeline-stack patch on top of frame split: not needed; frame split already creates its workers with a
  256 KB stack.
