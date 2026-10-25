# Optimization roadmap: NAM A2 on a Raspberry Pi 2

What to measure or change next, ranked. Builds on `docs/nam-core-optimizations.md`; where that
file already gives a command, this one links to it rather than repeating it.

**Labels used throughout**

- **Measured (Pi)**: your Pi 2 numbers, as given (native GCC 12, `-O3 -mcpu=cortex-a7
  -mfpu=neon-vfpv4`, no fast-math, block 128, back to back, 48 kHz, 2,667 us deadline).
- **Checked (Mac)**: an experiment I ran for this doc: clang cross-compile to armv7 assembly, or
  reading the vendored source. Section 6 lists them. None of these is a Pi timing.
- **Estimate**: a guess, with its reasoning written next to it. Nothing labeled "estimate" is a
  measurement, and none of it should go in the README until a script produces it.

Baseline numbers used below (Measured, Pi):

| | p50 | p99 | max | missed | load | MAC/sample | MAC/cycle |
|---|---|---|---|---|---|---|---|
| Lite (C=3) | 964 us | 1,257 us | 1,581 us | 0 | 37% | 1,731 | ~0.25 |
| Full (C=8) | 6,664 us | 6,867 us | 7,658 us | 100% | 250% | 11,776 | ~0.25 |

Arithmetic you'll see again (derived from the table, not new measurements):
- To get Full's **p99** under the deadline you need 6,867 / 2,667 = **2.6x**. To run Full at a 70% load
  (p50 ≤ 1,867 us, leaving room for the tail) you need 6,664 / 1,867 = **3.6x**.
- 3.6x on 0.25 MAC/cycle means ~0.9 MAC/cycle sustained on one Cortex-A7 at 900 MHz.

---

## 1. Summary: what to do next, in order

1. **Measure first (half a day, no code).** Pin the governor, then run `perf stat`/`perf record` on
   Lite and Full, plus a tiny NEON peak-throughput microbenchmark and an `objdump` of the GCC build
   (section 5). Every ranking below depends on two unknowns: whether Full is compute-bound or
   memory-bound, and what one Cortex-A7 core can actually sustain in NEON fp32 MACs. Both are cheap to
   measure and change what's worth building.
2. **Turn on NEON for Lite, in `a2_fast.cpp` only (≈1 hour + a null test).** Checked (Mac): Lite's hot
   loops compile to 0 NEON f32 ops today, and clang refuses them with "loop not vectorized due to
   unsafe FP support". Per-loop pragmas or a per-file flag fix it without fast-math for the whole build
   (item 1). Lite already passes at 128, so this buys smaller buffers (32–64 frames) and headroom, not
   correctness.
3. **Find Lite's real smallest buffer end to end.** Run paced, `--rt`, `isolcpus` arms at 16–128
   frames, and measure the Scarlett's own floor with `live --bypass` and `latency_test.sh` (items 7, 8,
   15). On a Pi 2 + USB, the interface floor may be what sets the latency, not the model. Measure it
   before working on the kernel side.
4. **Full: replace the per-tap Eigen GEMM.** Try the 2-line `lazyProduct` patch first (item 3,
   minutes). Then decide from the profile whether to write a fused NEON layer kernel (item 4, days).
   Checked (Mac): Full makes 179 Eigen GEMM calls per block. Each one packs its inputs and computes
   blocking sizes, for an 8×8 × 8×N product. That overhead is the most plausible reason Eigen's NEON
   kernel is no faster per MAC than Lite's scalar code.
5. **Full: two-core pipeline, only if 4 lands short of ~3.6x (item 9).** It adds one block of latency.
   It's the only lever that scales past what one A7 core can do.

Why this order: (1) is cheap and decides between (4) and the memory items. (2) is the cheapest real
speedup and has a clean test (null test plus sweep). (3) is the product number (smallest glitch-free
buffer) and may show the USB path is the limit. (4) and (5) are the only routes to Full, and they cost
the most.

### Decisions so far (Christine)

- **PREEMPT_RT (item 13): planned for later.** I'll build my own `rpi-7.1.y`+ kernel with
  `PREEMPT_RT=y`. Before that, take the stock-kernel `compare_rt.sh` and `live_sweep.sh` numbers
  as the baseline to compare against. Check the dwc_otg FIQ issue and the USB IRQ thread priority
  when it boots. Cross-compile it rather than building on the Pi.
- **ALSA mmap (item 14): in the plan.** Add `--mmap` to `live` and run it as its own `live_sweep.sh`
  arm (`TAG=live_mmap`), even though the expected effect is small. It's cheap, and "no change" is
  a result worth reporting.
- **Compiler (item 2): clang 16 cross-build for everything.** Faster to iterate. The one comparison
  run (Pi, block 128, one run each) had GCC 14.2 7% faster on Lite p50 and within ~1% on Full,
  kept in `results-pi/`. Revisit GCC only if a final number needs the last few percent.
- **I2S converters (item 16): a consideration for later.** Decide after the USB floor is measured
  (`live --bypass` + `latency_test.sh`, item 15). If the Scarlett path sets the latency floor, an
  I2S ADC/DAC HAT plus a guitar buffer is the next hardware step.

---

## 2. Ranked table

Effect columns are **estimates** unless marked. "≈0" means no expected change in compute time.

| # | Name | Category | Expected effect, Lite | Expected effect, Full | Effort | Risk |
|---|---|---|---|---|---|---|
| 1 | NEON auto-vectorization of the Lite loops (pragma or per-file unsafe-math) | compiler/build | 1.3–2x on p50 (est.) | ≈0 | 1 h | low (null test) |
| 2 | Clang cross build vs GCC native build, on the Pi | compiler/build | unknown, ± | unknown, ± | 30 min | low |
| 3 | Full: `lazyProduct` instead of Eigen GEMM per tap | NAM kernel code | n/a | unknown sign; removes 179 GEMM setups/block (est.) | 30 min | low |
| 4 | Full: fused NEON layer kernel (8×4 register tile across all K taps + tail) | NAM kernel code | n/a | 2.5–4x (est., wide) | 2–4 days | medium |
| 5 | Lite: hand NEON kernel, taps fused, `vld3q_f32` | NAM kernel code | further 1.2–2x over #1 (est.) | n/a | 1–2 days | medium |
| 6 | Ring mode 0 vs 1; mirror only `num_frames` columns | memory layout | ≤5% mean, maybe better max (est.) | ≤3% mean (est.) | 15 min / 1 h | low |
| 7 | Smaller blocks for Lite (16/32/64) | audio I/O | latency down; per-sample cost up (unknown) | n/a | 30 min | low |
| 8 | `--rt` + `isolcpus` + keep the audio thread off core 0 | scheduling/OS | tail (wake-up) only | tail only | 1 h + reboots | low |
| 9 | Two-core layer pipeline | multi-core | not needed | ~1.8x per core (est.), +1 block latency | 3–5 days | medium-high |
| 10 | Flush-to-zero in the audio thread | compiler/build | ≈0 normally; removes denormal spikes if any | same | 1 h | low |
| 11 | Full: shrink history traffic (fp16 history or smaller blocks) | memory layout | n/a | only if perf shows L2 refills matter | 1–2 days | medium (accuracy) |
| 12 | OS de-noising (disable services) / DietPi | scheduling/OS | tail only | tail only | 1–2 h | low |
| 13 | PREEMPT_RT kernel | scheduling/OS | tail only, matters at ≤32 frames | n/a while Full misses | 1–3 days | high |
| 14 | ALSA mmap access in `live` | audio I/O | <1% of the deadline (est.) | same | 1–2 h | low |
| 15 | USB path hygiene: HS link check, no Ethernet traffic, 2 vs 3 periods | audio I/O | xruns/latency floor | same | 1 h | low |
| 16 | I2S ADC/DAC HAT instead of USB | audio I/O | removes the USB latency floor | same | hardware + 1 day | medium |
| 17 | Overclock 900 → 1000 MHz | scheduling/OS (firmware) | ≤11% (upper bound, est.) | ≤11% | 30 min | medium (stability, thermal) |
| 18 | A2 fast path vs generic WaveNet on the Pi | NAM kernel code | sanity check | sanity check | 30 min (binaries exist) | low |
| 19 | Model choice: ship Lite; Full only if 4 + 9 succeed | model choice | — | — | decision | — |
| 20 | Int16 fixed-point NEON for Full | NAM kernel code | n/a | unknown; research | 1–2 weeks | high (accuracy) |
| — | JACK / Carla / PipeWire layers | audio I/O | negative (adds a layer) | same | — | — |

---

## 3. Items

### 1. NEON auto-vectorization of the Lite loops, scoped to `a2_fast.cpp`

**Hypothesis.** Lite runs at ~0.25 MAC/cycle because the three Lite frame loops (`a2_fast.cpp`
lines 538, 565, 598) are scalar VFP: each MAC is a separate `vmul.f32 s` + `vadd.f32 s`. Making them
4-wide NEON should cut Lite's layer time noticeably.

**What I checked (Mac, clang armv7 `-mcpu=cortex-a7 -mfpu=neon-vfpv4 -O3`, Lite `process()` with the
layer kernels inlined):**

| Variant | NEON f32 q-ops | scalar f32 ops | vld3/vst3 |
|---|---|---|---|
| default | 0 | 340 | 0 / 0 |
| `-ffast-math` | 237 | 304 | 40 / 24 |
| `#pragma clang loop vectorize(enable)` on the 3 Lite loops, **no fast-math** | 234 | 340 (scalar tails) | 40 / 22 |
| any one of: `-funsafe-math-optimizations`, `-ffinite-math-only`, `-fdenormal-fp-math=preserve-sign`, `-ffp-contract=fast`, `-fassociative-math …` | 0 | 340 | 0 |
| `-funsafe-math-optimizations -ffinite-math-only` | 237 | 304 | 40 / 24 |

- Clang's remark for each Lite loop is `loop not vectorized: loop not vectorized due to unsafe FP
  support`. On ARMv7 the loop vectorizer needs either the *full* fast-math flag set on the
  instructions or an explicit `vectorize(enable)` hint. The narrower flags don't do it.
- The vectorized loops use `vld3`/`vst3` (de-interleave the channel-interleaved history), as expected.
- Neither build fuses multiply-add in the Lite loops (still `vmul` + `vadd`, no `vmla`/`vfma`), and
  these loops contain no reductions. So the pragma build does the same arithmetic in the same order
  per lane as the scalar build. The one semantic difference is that NEON flushes denormals to zero.
  **Prediction (testable): a pragma build should null-test at exactly 0 against the default build
  unless denormals occur.** `-ffast-math` additionally lets the compiler reassociate scalar code
  elsewhere (for example the 1×1 residual sum `lb0 + lw00*a0 + lw10*a1 + lw20*a2`), which is a
  plausible source of the −135 dB residual you saw on the Mac.
- GCC: its manual says NEON float code is not auto-vectorized without
  `-funsafe-math-optimizations` ([GCC ARM options][gcc-arm]). I couldn't run GCC here; check the Pi
  build with `objdump` (section 5).

**Estimate of the gain.** Upper bound ~2x on the vectorized arithmetic. Reasoning: community timing
tables show the A7's NEON unit processing 64 bits per cycle, so a 4-lane q-op issues over 2 cycles
(2 lanes/cycle) vs about 1 scalar VFP op/cycle ([hardwarebug A7 timings][a7-timings]; scraped table,
medium confidence). The `vld3` loads and scalar tails eat into that, so **1.3–2x on Lite p50
(estimate)**. Full is unchanged: its GEMM kernel is already NEON (section 6).

**How to test.** Two options, both need a change (I didn't make it):
- *Per-file flag (CMake change):* new option `BENCH_A2_UNSAFE_MATH` that does
  `set_source_files_properties(${NAM_DIR}/NAM/wavenet/a2_fast.cpp PROPERTIES COMPILE_OPTIONS
  "-funsafe-math-optimizations;-ffinite-math-only")` and appends `a2_unsafe=ON` to `BUILD_FLAGS_STR`, so
  the summary CSV records it. On GCC, `-funsafe-math-optimizations` alone should do (the manual's
  rule). On clang you need both flags (table above).
- *Pragma (vendored-code patch):* add `#pragma clang loop vectorize(enable)` (clang) /
  `#pragma GCC ivdep` + function attribute `__attribute__((optimize("unsafe-math-optimizations")))`
  (GCC) before the three loops. This patches `external/`, so keep it as a patch file.

Then, cross-built (seconds each):

    BUILD_DIR=build-armhf      ./scripts/cross_build.sh
    BUILD_DIR=build-armhf-vec  ./scripts/cross_build.sh -DBENCH_A2_UNSAFE_MATH=ON   # new option
    PI_USER=<you> BUILD_DIR=build-armhf     DEST=build-clang     ./scripts/deploy.sh
    PI_USER=<you> BUILD_DIR=build-armhf-vec DEST=build-clang-vec ./scripts/deploy.sh

On the Pi:

    ./scripts/set_governor.sh performance
    BENCH=./build-clang/bench     TAG=clang     MODE=paced BLOCKS="32 64 128" ./scripts/sweep.sh
    BENCH=./build-clang-vec/bench TAG=clang_vec MODE=paced BLOCKS="32 64 128" ./scripts/sweep.sh
    python3 scripts/kpis.py --compare clang clang_vec
    M=models/a2_fender_super_reverb.nam
    ./build-clang/bench     --model $M --slim 0.25 --input audio/di.wav --block 128 --render results/ref_lite.wav --summary-out results/summary_render.csv
    ./build-clang-vec/bench --model $M --slim 0.25 --input audio/di.wav --block 128 --render results/vec_lite.wav --summary-out results/summary_render.csv
    python3 scripts/null_test.py results/ref_lite.wav results/vec_lite.wav --label clang_vec_lite

(Confirm `--slim 0.25` selects Lite with `./build/bench --model $M --print-slim-sizes`.)

**Confirms:** Lite p50 drops by ≥1.3x at 128 and the null test is ≤ −120 dB (exactly 0 for the
pragma variant). **Refutes:** <10% change. Then either the A7's NEON doesn't beat scalar for this
access pattern (`vld3` cost), or Lite is memory-bound. `perf stat` L1D refills (section 5) tell you
which.

**Pitfalls.** Compare clang with clang. Don't compare a cross build against the GCC native baseline
(`build_flags` shows `AppleClang`). Use a DI with near-silence in it for the null test, since
denormals only appear there. Global `-ffast-math` (the existing `BENCH_FAST_MATH`) also compiles
`NAM/linear.cpp` and `NAM/wav.cpp`, which rely on `std::isfinite` checks that clang may fold away
under `-ffinite-math-only`. That's one more reason to scope the flag to one file.

### 2. Clang cross build vs GCC native build, on the Pi

**Hypothesis.** The two compilers schedule the Lite loops and inline Eigen differently. For example,
clang splits every multiply-add into `vmul` + `vadd` on Cortex-A7 (Checked, Mac). GCC may emit
`vmla`/`vfma`, which changes both speed and rounding. You've only measured GCC.

**Test.** Deploy `build-armhf` (already built) to `build-clang` as above. Then:

    BENCH=./build/bench       TAG=gcc   MODE=burst BLOCKS=128 ./scripts/sweep.sh
    BENCH=./build-clang/bench TAG=clang MODE=burst BLOCKS=128 ./scripts/sweep.sh
    python3 scripts/kpis.py --compare gcc clang

Also count the fused ops in each (section 5, `objdump`).

**Confirms/refutes.** Any p50 difference >5% beyond run-to-run spread (3 repeats) means the compiler
choice matters, and the faster one becomes the baseline for every later item. **Pitfall:** pick one
compiler for the README headline and say which.

### 3. Full: `lazyProduct` instead of a GEMM per tap

**Hypothesis.** Full spends a large share of its time in Eigen GEMM overhead, not in MACs.
Checked (Mac): `W * input_block` (8×8 × 8×N, line 670) and the 1×1 product (line 678) dispatch at
runtime to `general_matrix_matrix_product::run`, because 8 + 8 + N ≥ 20
(`EIGEN_GEMM_TO_COEFFBASED_THRESHOLD`, `GeneralMatrixMatrix.h:388`). Each call computes blocking
sizes and packs W and the 8×N input into scratch. That's 21×(6+1) + 2×(15+1) = **179 GEMM calls per
block**, each re-packing data it just read. Its micro-kernel (`gebp_kernel<…,12,4>`) is NEON
(260 `vmla.f32 q`) in both default and fast-math builds. So Full's poor MAC/cycle is not a
vectorization problem. (Your earlier note's 52 → 60 count covered only `process()`, not this kernel.)

Also checked: the block comment at line 514 describes a "frame-tiled T=4 register tile" for Full,
but the code below it calls Eigen GEMM per tap. The comment is stale.

**Change (2 lines in `external/`, as a patch):**
`ztile.noalias() += W.lazyProduct(input_block);` and
`lin_block.noalias() += l1x1_mat.lazyProduct(ztile);`. Checked (Mac): this compiles to a vectorized
coefficient loop (16 `vfma.f32 q` per output column, 34 `vld1`), with no GEMM call or packing.

**Estimate.** Unknown sign. It removes the packing and setup, but reloads W for every column
(34 loads : 16 FMAs). If `perf report` (section 5) shows `gemm_pack_*` + blocking setup + `memcpy`
above ~20% of Full's samples, expect a gain. If `gebp_kernel` dominates, expect a loss.

**Test.** Build `-DCMAKE_CXX_FLAGS=-DNAM_A2_FULL_LAZY` (if you gate the patch with a macro) into
`build-armhf-lazy`. Sweep with `TAG=clang_lazy BLOCKS=128 MODE=burst`, compare to `clang`, and run
the null test on `--slim 1`. **Pitfall:** `vfma` is fused and `gebp`'s `vmla` is not, so expect a
small nonzero null residual. It should still be far below −96 dB; check it.

### 4. Full: fused NEON layer kernel

**Hypothesis.** A kernel that keeps an 8-output × 4-frame accumulator tile (8 q-registers) in
registers across all K taps, then applies bias, mixin, LeakyReLU, head-sum and the 1×1 residual
before storing, cuts loads and stores by several times. Per tile and tap it loads W columns (2 q-regs
per input channel) and uses `vmlaq_lane_f32` against input values. Arithmetic per tile at K=6 is 384
`vmla.f32 q` against ~144 q-loads. GEMM-per-tap stores and reloads the accumulator 7 times per layer
instead.

**Estimate (wide).** 2.5–4x on Full, i.e. 0.6–1.0 MAC/cycle. Reasoning: if a 4-lane q-MAC issues
every 2 cycles (64-bit NEON datapath, medium confidence), peak is ~2 MAC/cycle. A careful small-matrix
kernel on an in-order core reaching 30–50% of peak is a guess, not data. **The NEON peak
microbenchmark in section 5 replaces this guess with a ceiling.** If the measured peak is ~1
MAC/cycle rather than 2, this item alone can't reach 3.6x, and item 9 becomes required.

**Test.** Write it behind a macro in `a2_fast.cpp` (patch). Before timing anything, verify
correctness with a null test against the Eigen build at `--slim 1` (≤ −120 dB). Then sweep
`BLOCKS="64 128 256"` burst + paced. Also run `perf stat` cycles/instructions/L1D refills, so the
README can show where the MAC/cycle went.

**Pitfalls.** ARMv7 has 16 q-registers. 8 accumulators + 2 W + input d-regs fits, but spills will
show up as `vstr`/`vldr` in `objdump`; check for them. The history is channel-interleaved, so a
4-frame tile reads 4 × 8 contiguous floats per tap, which is good. The kernel isn't portable; keep
the Eigen path as the fallback.

### 5. Lite: hand NEON kernel with fused taps

**Hypothesis.** Item 1's vectorized loops still store and reload `z` (3×N floats) for every tap, using
`vld3`/`vst3`. A hand kernel keeps 3 accumulators (one per output channel, lanes = 4 frames) in
registers across all K taps. It `vld3q_f32`-loads each tap's 4 frames and does 9 `vmlaq_n_f32` per tap.

**Estimate.** A further 1.2–2x over item 1, from removing the per-tap `z` traffic. That traffic is
about K−1 extra load/store pairs per frame per layer (est.). **Test:** same as item 1, comparing
against `clang_vec`. **Pitfall:** `vld3` is comparatively slow on small cores. If perf annotate shows
time piling up on `vld3`, try a planar (channel-major) history layout. That's a bigger change to the
ring code.

### 6. Ring buffer mode, and the mirror copy

**Hypothesis.** In mode 1, `_ring_write` copies **`mbs` columns** of mirror every block (line 429),
however many frames were written. Checked (Mac, arithmetic on the code): at block 128 that's
23 × 128 × C floats = 34.5 KiB/block for Lite and 92 KiB/block for Full, on top of the tap reads.
Mode 0 has no mirror, but memmoves `max_lookback` columns when a layer's linear buffer wraps, which
shows up as spikes in p99/max.

**Estimate.** Mean effect ≤5% for Lite, ≤3% for Full (est.: one memcpy of tens of KiB against a
~1 ms / ~6.7 ms block). The interesting part is max/p999, not the mean.

**Test.** `BUILD_DIR=build-armhf-ring0 ./scripts/cross_build.sh -DCMAKE_CXX_FLAGS=-DNAM_A2_RING_MODE=0`,
deploy, sweep with `KEEP_BLOCKS=1`, compare p99/p999/max against `clang`. Plot the per-block CSVs
with `plot.py --blocks` to see the periodic spikes. A second variant (patch): copy only the columns
just written into the mirror region (`min(num_frames, mbs - wp)` …). **Pitfall:** `build_flags` in the
summary CSV doesn't include `CMAKE_CXX_FLAGS`, so the ring mode lives only in your TAG. Name tags
carefully, or add the macro to `BUILD_FLAGS_STR`.

### 7. Smaller blocks for Lite

**Hypothesis.** Lite at 37% load at 128 should also pass at 64 and possibly 32 without any code
change. Estimate: if per-block time scales linearly, Lite p99 at 64 is ~630 us against a 1,333 us
deadline. Fixed per-block costs (23 ring writes, 23 kernel dispatches, mirror copies) make small
blocks proportionally more expensive. How much is unknown.

**Test.** `sweep.sh` runs both variants. To run only Lite, call bench directly:

    for b in 16 32 64 128; do for r in 1 2 3; do
      ./build/bench --model models/a2_fender_super_reverb.nam --slim 0.25 --input audio/di.wav \
        --block $b --paced --label lite_small__${b}__r$r --summary-out results/summary_lite_small.csv \
        --blocks-out results/blocks/lite_small/${b}__r$r.csv; done; done

Then compare µs per sample (`p50_us / block`) across block sizes. **Confirms** a fixed overhead if
µs/sample rises steeply below 64. A new `SLIM=` filter in `sweep.sh` would make this one line.

**Pitfall.** Paced mode, not burst: at small blocks, wake-up latency (`wake_p99_us`) is a larger
share of the deadline.

### 8. `--rt`, `isolcpus`, and keeping off core 0

**Hypothesis.** Compute won't change, but the tail will: wake-up latency and preemption by other
tasks. It matters most at small blocks.

**Test.** RUNBOOK §4 and §4b (`compare_rt.sh` → baseline / rt / rt_isol). Before that, look at where
interrupts land: `cat /proc/interrupts` (per-CPU columns for `dwc_otg`, the USB controller) during a
`live` run. My understanding, medium confidence: on the BCM2836, peripheral interrupts are delivered
to a single core (core 0), so `irqaffinity=` may have no effect for USB. Whatever the answer, keep
the audio thread on core 3 (`--cpu 3`, the scripts' default) and confirm in `/proc/interrupts` that
core 3's USB count doesn't grow.

**Confirms:** `wake_p99_us` and `missed_with_wake` drop in `rt_comparison.csv`. **Pitfalls:**
`--rt` with `MODE=burst` hits the RT throttle (`sched_rt_runtime_us`, see `compare_rt.sh`). Don't
set `sched_rt_runtime_us=-1` to get around it: a runaway SCHED_FIFO thread then locks up the box.
Check that `nohz_full` is actually available (`cat /sys/devices/system/cpu/nohz_full`) before adding
it to the cmdline. The Raspberry Pi `bcm2709_defconfig` sets `CONFIG_NO_HZ=y` (idle), and I found no
`NO_HZ_FULL` in it.

### 9. Two-core pipeline for Full

**Hypothesis.** Split the 23 layers into two stages of roughly equal MACs on cores 2 and 3, with an
SPSC ring passing `layer_in`, `head_sum` and `cond` (N × (2C+1) floats per block). Each core then does
about half the work. Estimate ~1.8x per core: the split isn't perfectly even, and there's handoff
cost. The cost is one extra block of latency (+2.67 ms at 128, +1.33 ms at 64).

**Arithmetic.** With item 4 at 2.5x, Full p50 ≈ 2,666 us (est.), right at the deadline. Split in
two, that's ≈1,480 us per core (est.), about 55% of the deadline at 128. Without item 4 (just this
pipeline): ≈3,700 us per core (est.), which still misses. **So item 9 helps only after item 4 works,
or with 3+ stages.**

**Test.** New code (not in NAM): a pipelined wrapper in `bench`/`live`, or a split-model API in a
patch. Measure per-stage block times separately. The current harness times one `process()` call,
so you'd need a `--pipeline 2` flag that reports each stage's p99 and the end-to-end latency.
**Pitfalls.** Synchronize lock-free and without syscalls in the audio path (spin on an atomic with a
bounded wait, since a futex is a syscall). Two SCHED_FIFO threads on isolated cores means isolating
two cores (`isolcpus=2,3`). Watch `thermal_logger.py`: two cores pegged on a Pi 2 is a different
thermal regime from one.

### 10. Flush-to-zero in the audio thread

Already item 2 in `nam-core-optimizations.md`. Two additions. After item 1, the vectorized loops
already flush (NEON), but the scalar tails and the head don't. Set FPSCR.FZ once in `bench`/`live`
(new `--ftz` flag, ~5 lines, `vmrs`/`vmsr`). Test on a DI with long near-silence and compare `max_us`.
**Confirms** only if max drops on the silent stretch. Expect ≈0 on normal playing.

### 11. Full: shrink history traffic

Checked (Mac, arithmetic on the code): the *live* history window (lookback + N columns per layer)
is ~294 KiB for Full at block 128 and ~110 KiB for Lite. The *allocated* pow2 rings are larger
(~560 / ~210 KiB by my count, close to your 548 / 206 KiB). Data outside the live window is dead, so
the live set, not the allocation, is what has to stay in cache. 294 KiB fits in the 512 KB L2, but
the L2 is shared and its replacement isn't perfect LRU. Distinct bytes the taps read per block:
~238 KiB (Full) / ~89 KiB (Lite) at 128, against a 32 KB L1D.

**Do this only if** section 5 shows L2 refills are a meaningful share of Full's cycles. Options:
smaller blocks (Full's live set at 64 is ~246 KiB), or fp16 history (the A7 supports `vcvt.f16.f32`
with `-mfpu=neon-vfpv4`; the conversion costs ALU slots and adds rounding, so null-test it).
**Pitfall:** fp16 history changes the model's numerics at every layer. Expect a residual well above
fast-math's; it has to stay below −96 dB.

### 12. OS de-noising / DietPi

**Hypothesis.** Background daemons cause occasional wake-up delays. Fewer services means a cleaner
tail. DietPi on a Pi uses Raspberry Pi's own kernel packages
([DietPi forum][dietpi-kernel]; `linux-image-rpi-v7` is the 32-bit meta-package
([package listing][rpi-v7-pkg])). So DietPi vs Pi OS Lite on the same Pi is a **userland** difference:
fewer services, same kernel, same drivers, same USB stack. You can get most of it on Pi OS Lite by
disabling timers and daemons.

**Test.** First, `systemctl list-units --type=service --state=running` and `systemctl list-timers`.
Disable what you don't need (apt-daily timers, man-db, triggerhappy, avahi if unused). Then rerun the
`rt` arm under a new tag (a `compare_rt.sh` arm needs a new name, e.g. `ARMS=rt` with `TAG` changed by
hand, since the script maps arm → tag). Compare `wake_max_us`. **Confirms** if `wake_max_us` drops.
If `isolcpus` already flattened it, this adds nothing. **Pitfall:** don't change OS and isolation in
the same step.

### 13. PREEMPT_RT kernel

**Applies to Pi 2? Only with a custom kernel, and only matters at small blocks.**
- Mainline PREEMPT_RT landed in 6.12 for x86, arm64 and RISC-V ([Wikipedia][rt-wiki],
  [Heise][rt-heise]). **32-bit ARM was not included.** Arm32 support came with the
  "ARM: Allow to enable RT" change, reported as merged for **Linux 7.1** ([Phoronix][rt-arm71]).
- Checked (GitHub): in the Raspberry Pi kernel tree, `arch/arm/Kconfig` has `select ARCH_SUPPORTS_RT`
  in `rpi-7.1.y` and `rpi-7.3.y`, but not in `rpi-6.12.y`. The default branch is currently
  `rpi-6.18.y`, which also lacks it. `bcm2709_defconfig` (the Pi 2 32-bit config) sets
  `CONFIG_PREEMPT_VOLUNTARY=y` in 6.12 and 6.18 ([raspberrypi/linux][rpi-linux]). I found no prebuilt
  RT kernel package for the Pi 2. Expect to build `rpi-7.1.y` or later with `PREEMPT_RT=y` yourself.
- A Pi 2 specific hazard: OSADL found a PREEMPT_RT BCM2709 kernel **wouldn't boot** until the USB
  driver's FIQ was disabled (`dwc_otg.fiq_enable=0 dwc_otg.fiq_fsm_enable=0`), and proposed a patch
  ([OSADL, 2016][osadl]). That FIQ code is what makes USB audio work well on dwc_otg. Turning it off
  could make the Scarlett worse. I don't know whether current kernels still need the workaround; test
  it.
- What changes vs the Pi 5: the Pi 5 is arm64, where RT has been mainline since 6.12, and its USB
  goes through the RP1 chip, not dwc_otg plus a hub. On the Pi 2 both points go against you.

**Hypothesis to test first, without building a kernel.** The stock kernel's paced `wake_p99_us`
plus model p99 already fits the deadline at the block sizes you care about. Lite at 128 has
~1,400 us of slack at p99, so RT can only matter at 32 frames or below. **Decide from
`rt_comparison.csv`:** build RT only if `missed_with_wake` > 0 at your target block size in the
`rt_isol` arm while `missed` = 0.

**Test, if you go ahead.** Same `compare_rt.sh` arms on the RT kernel. `bench` records the
preemption model in the `kernel` column (`KernelInfo()`), and the RT run should show `PREEMPT_RT`.
Then `live_sweep.sh` and `latency_test.sh`, to check the USB path still works with any FIQ change.
**Pitfalls:** threaded IRQs under RT, including the USB IRQ thread, need priorities. If the audio
thread at FIFO 80 outranks the USB IRQ thread (default FIFO 50), it can starve the very transfers it
waits on. Building a kernel natively on a Pi 2 takes hours (my expectation, not measured).

### 14. ALSA mmap

**Applies? Yes, but the expected effect is tiny for `live`.** `live.cpp` opens `hw:` with
`SND_PCM_ACCESS_RW_INTERLEAVED` and calls `snd_pcm_readi`/`writei`, which copy one period between the
ALSA buffer and the user buffer. Estimate: at 128 frames and 2–4 channels of 32-bit samples that's
1–2 KiB per direction per period, a few microseconds against a 2,667 us deadline, so <1%. In the
Reddit setups, mmap mattered for JACK/Pipedal configurations. In a direct `hw:` client like `live`
there's not much for it to save. With USB, `snd-usb-audio` already copies between URBs and the ring
buffer either way.

**Test.** New `--mmap` flag in `live`: `SND_PCM_ACCESS_MMAP_INTERLEAVED` +
`snd_pcm_mmap_readi`/`snd_pcm_mmap_writei`, same loop. Run `live_sweep.sh` with
`EXTRA="--mmap"`, `TAG=live_mmap`, and compare xruns at 32/64. **Confirms** only if xruns at a given
period size drop. Expect no change. Low priority.

### 15. USB path hygiene (Scarlett on a Pi 2)

**Facts that differ from a Pi 5.** On the Pi 2, every external USB port **and** the Ethernet sit
behind one on-board LAN951x hub/Ethernet chip, on a single USB 2.0 link to the SoC's dwc_otg
controller ([Raspberry Pi docs][rpi-docs]: "LAN951x" is the on-board hub/Ethernet). SSH or rsync
traffic shares the bus with your audio. The Pi 5's USB and Ethernet hang off RP1 instead.

**Tests (all existing tools):**
1. `lsusb -t`. The Scarlett should show `480M` (high-speed). A full-speed (`12M`) device behind the
   hub needs split transactions, which is the hardest case for dwc_otg.
2. `cat /proc/asound/card*/stream0` while `live` runs, to see the endpoint's packet interval.
3. Floor: `live --bypass` and `./scripts/latency_test.sh` at 32/64/128 × 2/3 periods. The gap
   between measured round trip and period × periods is converter + USB latency. Reddit reported a
   USB floor around 16–32 samples / 2–3 ms (3.5 ms with a Scarlett on a Pi 5). Find yours before
   optimizing anything below it.
4. Interference: rerun one `live_sweep.sh` configuration while `iperf3` or an rsync pushes Ethernet
   traffic, and compare xruns. Run sessions from `tmux` and detach, so SSH output isn't streaming.
5. Power: `vcgencmd get_throttled` (logged by `thermal_logger.py`). The RUNBOOK already suggests a
   powered hub if the Scarlett crackles.

### 16. I2S ADC/DAC HAT

**Applies? Yes.** The Pi 2 has the same 40-pin header with I2S as later models. An I2S codec
(Reddit: PCM5102A DAC + PCM1808 ADC, or a HiFi HAT with input and output) takes USB and dwc_otg out
of the audio path entirely. Reddit reported 1.9 ms round trip with I2S vs 3.5 ms with a Scarlett,
on a Pi 5. On a Pi 2 the gain could be larger, given the shared hub, but that's unmeasured.
**Pitfalls:** a guitar needs a high-impedance buffer or DI before a line-level ADC (as the thread
says), with shielding. You lose the Scarlett's preamp. Device-tree overlay setup varies by board.
**Test:** the same `latency_test.sh` / `live_sweep.sh` with `DEVICE=hw:<i2s card>`.

### 17. Overclock 900 → 1000 MHz

Estimate: at most +11% (1000/900), and less wherever time goes to memory (DRAM doesn't speed up
with the core clock). It doesn't change the ranking: it can't close Full's 2.6x gap. **Test:**
`arm_freq=1000` in `config.txt` (plus whatever `over_voltage` your board needs). Then a
30-minute `sustained.sh` with `thermal_logger.py` running, checking that `throttled_ever` stays 0.
**Pitfalls:** stability (run sustained before believing any sweep), SD-card corruption on a crash,
and it's a per-board result.

### 18. A2 fast path vs generic WaveNet on the Pi

`build-armhf-generic` already exists (RUNBOOK §4c). Deploy it to `DEST=build-clang-generic` and run
`TAG=clang_generic`. **Why bother:** the fast path was tuned on an M1. If the generic path isn't
slower on the A7, the fast path's design assumptions don't hold there, and that's worth a line in
the README. Expect the fast path to win (est.).

### 19. Model choice

**Lite is the product on a Pi 2.** It passes at 128 today with 63% headroom (Measured), and items
1, 7 and 8 aim at its latency. Full needs 2.6–3.6x and only items 4 and 9 together plausibly get
there (est.). The "A2 models" from the Reddit thread are the same architecture family, so the Pi 5's
"<20% CPU" doesn't transfer. The thread doesn't say which submodel it ran. The Pi 5 has ~11× the
single-core score (`docs/hardware-comparison.md`).

### 20. Int16 fixed-point NEON for Full (last resort)

Integer NEON MACs (`vmlal.s16`) pack twice as many lanes per 64-bit datapath beat as fp32. In
principle that's ~2x the fp32 peak (est., unverified on A7), but it needs per-layer scaling and
saturation, and the null test will show real quantization error. Only consider it if items 4 and 9
together leave Full short.

### Not recommended: JACK, Carla, PipeWire

**Applies? No.** In the Reddit build they hosted the plugin. `live` already talks to `hw:` directly
with linked capture/playback streams, which is the minimum path. A sound server adds a process, a
context switch and its own buffering. Add one only if you need routing, and measure it with
`latency_test.sh`.

---

## 4. Reddit ideas: do they apply to a Pi 2 + USB Scarlett?

Source: r/NAM_NeuralAmpModeler, Pi 5 at 1.9 ms → 1.1 ms round trip ([thread][reddit]), as
summarized in the request. Pi 2 = 4× Cortex-A7, 900 MHz, ARMv7. Pi 5 = 4× Cortex-A76, 2.4 GHz, arm64.

| Idea | Applies to Pi 2 + Scarlett? | What changes on a Cortex-A7 vs Pi 5 | Item |
|---|---|---|---|
| PREEMPT_RT + kernel tuning | Only with a self-built kernel | Arm64 RT is mainline since 6.12, but arm32 only from 7.1, and the Pi kernel's arm32 RT option starts at `rpi-7.1.y`. Known dwc_otg FIQ conflict on BCM2709. Helps the tail only, not compute. | 13 |
| DietPi instead of Pi OS | Partly: same RPi kernel, fewer services | No kernel, driver or compiler difference. You can replicate it by disabling services. | 12 |
| `isolcpus` for the audio thread | Yes (`isolcpus.sh`) | Peripheral IRQs probably all land on core 0 on BCM2836 (medium confidence), so isolating core 3 is the right choice. Check `/proc/interrupts`. | 8 |
| JACK / Carla | No | Extra layer. `live` is already a direct ALSA client. | — |
| ALSA mmap | Yes, negligible | Copy saved is ~KiB per period. Same order of cost on either board relative to the deadline (est.). | 14 |
| New A2 models | Already using A2 | Lite fits (37%). Full needs 2.6–3.6x; the Pi 5's "<20% CPU" reflects ~11× faster cores. | 19 |
| Buffers as low as 4 samples | No (USB + per-block overhead) | USB packetization floor plus per-block fixed cost on a slow core. Measure the floor first, then try 16–32. | 7, 15 |
| USB floor ~16–32 samples / 2–3 ms | Yes, probably worse | Pi 2's USB and Ethernet share one LAN951x hub behind dwc_otg. The Pi 5 has RP1. Measure with `--bypass` + `latency_test.sh`. | 15 |
| I2S HAT with input and output | Yes (40-pin I2S) | Removes the shared USB path, likely a bigger win on the Pi 2 than on the Pi 5 (est.). Needs a guitar buffer. | 16 |
| Guitar buffer + shielding | Yes, for I2S | Not needed with the Scarlett's INST input. | 16 |
| 64 × 3 periods with an RME UCX (Pi 5) | Try 64 × 3 | Whether the Pi 2's USB path sustains it is the question; `live_sweep.sh` already covers 2 vs 3 periods. | 15 |

---

## 5. Measure first (on the Pi, before any code change)

Run these on the current GCC build (`./build/bench`). Then run the same on `build-clang` if you take
item 2. `M=models/a2_fender_super_reverb.nam`; check the slim values with
`./build/bench --model $M --print-slim-sizes` (assumed below: Lite `0.25`, Full `1`).

**0. Conditions (2 min).** Every number below is meaningless if the clock moves.

    grep -m1 "CPU part" /proc/cpuinfo                 # 0xc07 = Cortex-A7
    ./scripts/set_governor.sh performance              # Pi 2 defconfig default is ondemand
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq   # expect 900000
    vcgencmd get_throttled                             # expect 0x0
    uname -v                                           # preemption model
    dmesg | grep -i "perfevents"                       # PMU driver + number of counters
    sudo apt install linux-perf                        # if perf is missing; version mismatch warnings are usually harmless

The Linux driver names the PMU `armv7_cortex_a7`. Generic events map like this: `cycles`, `instructions`,
`cache-references` = L1D access (0x04), `cache-misses` = L1D refill (0x03), `LLC-loads` = L2 access
(0x16), `LLC-load-misses` = L2 refill (0x17). Raw events: `r13` mem access, `r19` bus access, `r05`
DTLB refill, `r10` branch mispredict ([arm_v7_pmu.c][pmu]). The A7 has a small number of event
counters (the `dmesg` line says how many; I believe 4). Keep each `perf stat` run at that many events
or fewer, so counts aren't multiplexed estimates (perf prints a `(xx%)` when they are).

**1. Counters, 60 s of audio each, Lite and Full (≈5 min):**

    for v in 0.25 1; do
      sudo perf stat -r 3 -e cycles,instructions,cache-references,cache-misses -- \
        ./build/bench --model $M --slim $v --input audio/di.wav --block 128 --duration 60 \
        --label perfstat_$v --summary-out results/summary_perfstat.csv 2> results/perfstat_l1_$v.txt
      sudo perf stat -r 3 -e LLC-loads,LLC-load-misses,r13,r19 -- \
        ./build/bench --model $M --slim $v --input audio/di.wav --block 128 --duration 60 \
        --label perfstat_$v --summary-out results/summary_perfstat.csv 2> results/perfstat_l2_$v.txt
    done

(`--duration 60` makes the timed loop dominate load and prewarm. For Full, 60 s of audio takes
~150 s at 250% load.) `perf stat` writes text, not CSV. To keep the "all measurements write CSV"
rule, add `-x,` and redirect, or write a small `scripts/perfstat.sh` wrapper that does.

**2. Profile (≈5 min):**

    cmake -S . -B build-prof -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-g && cmake --build build-prof -j2   # -g doesn't change codegen
    for v in 0.25 1; do
      sudo perf record -F 2000 -o results/perf_$v.data ./build-prof/bench --model $M --slim $v \
        --input audio/di.wav --block 128 --duration 30 --summary-out results/summary_perf.csv
      sudo perf report -i results/perf_$v.data --stdio --no-children --sort symbol | head -40 > results/perf_top_$v.txt
    done
    sudo perf annotate -i results/perf_1.data --stdio -s <hottest symbol> | head -120

Use a flat profile, as above. `perf record -g` relies on frame pointers, which an `-O3` ARMv7 build
doesn't keep, so call graphs come out broken. If you need callers, use `--call-graph dwarf` (much
larger files). RUNBOOK §5's `-g` command gives you the flat part anyway. Building natively takes a
long time, so you could instead cross-build with `-DCMAKE_CXX_FLAGS=-g` on the Mac and profile that
(clang).

**3. Codegen of the binary you actually run (1 min):**

    objdump -d -C --no-show-raw-insn build/bench | awk '/A2FastModel<3>::process/,/^$/' > /tmp/lite.s
    grep -cE 'v(mla|mul|add|fma)\.f32\s+q' /tmp/lite.s ; grep -cE 'v(mla|fma)\.f32\s+s' /tmp/lite.s ; grep -cE 'vmul\.f32\s+s' /tmp/lite.s

That gives NEON q-ops, fused scalar MACs, and plain scalar multiplies in GCC's Lite code. Clang's
are 0 / 0 / 163 (Checked, Mac).

**4. NEON peak microbenchmark (new tool, ~40 lines, cross-compile it):** `src/neon_peak.cpp`. Loop
over 8 independent `float32x4_t` accumulators doing `vmlaq_f32` (and a `vfmaq_f32` variant, plus a
scalar `float` variant with 8 accumulators). Time it with `clock_gettime` on core 3 and print MAC/cycle
assuming 900 MHz. Add a variant whose loop also does one `vld3q_f32` per 3 MACs. This puts a measured
ceiling on items 1, 4 and 5. Right now that ceiling is a guess (2 MAC/cycle).

**How the results reorder the list:**

| Observation | Reading | Move |
|---|---|---|
| Full: IPC (instructions/cycles) low (≲0.5) **and** L2 refills × ~64 B add up to a large share of time (e.g. refills × an assumed ~100-cycle miss ≳ 25% of cycles) | memory-bound | Items 11 and 6 move above 4. Try smaller blocks for Full first. |
| Full: `perf report` shows `gemm_pack_*`, blocking setup and `memcpy` together ≳ 20% | per-call overhead | Item 3 first (cheap), then 4. |
| Full: `gebp_kernel` ≳ 70% of samples, L2 refills small | compute-bound in Eigen's kernel | Item 4 is the main lever. Use the microbenchmark to size the expected win. |
| NEON peak ≈ 2 MAC/cycle | headroom exists | Items 4 + 9 can plausibly reach Full at 128. |
| NEON peak ≈ 1 MAC/cycle or less | thin headroom | Full needs item 9 with 3 stages or item 20. Consider declaring Lite the product (item 19). |
| Lite: L1D refill rate low, IPC well under 1 | latency-bound scalar chains | Item 1 should help a lot; then item 5. |
| Lite: L1D refills high | memory-bound even at 110 KiB | Items 1 and 5 help less. Look at item 6 and planar layout. |
| GCC objdump already shows NEON q-ops in Lite | GCC vectorizes without the flag | Item 1 drops (re-check which loops). |
| `memcpy` ≳ 10% in either profile | ring writes and mirror | Item 6 moves up. |

---

## 6. What I checked for this doc (Mac only, no Pi timings)

All with Apple clang 16, `--target=armv7a-linux-gnueabihf`, the Bookworm sysroot, the project's
flags (`-std=c++20 -O3 -mcpu=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=hard -DNAM_ENABLE_A2_FAST
-DNAM_SAMPLE_FLOAT`), compiling `a2_fast.cpp` to assembly and counting instructions per function.
Patched copies were made in a scratch directory. Nothing in `external/` was changed.

1. **Lite needs the full fast-math set or a vectorize pragma.** No single sub-flag of `-ffast-math`
   vectorizes it. `-funsafe-math-optimizations -ffinite-math-only` does (237 q-ops), and so does
   `#pragma clang loop vectorize(enable)` on the three loops without fast-math (234 q-ops). Clang's
   remark: "loop not vectorized due to unsafe FP support".
2. **No fused multiply-add in Lite**, default or fast-math. Clang emits `vmul` + `vadd` (163/165
   scalar, 111–114/117 vector).
3. **Full's GEMM is NEON already.** `gebp_kernel<float,…,12,4>` has 260 `vmla.f32 q` in the default
   build and 262 with fast-math. `process()` calls `general_matrix_matrix_product::run`; Eigen's
   threshold (8+8+N ≥ 20) sends every per-tap product to GEMM. Blocking scratch at N=128 is under
   `EIGEN_STACK_ALLOCATION_LIMIT` (128 KiB), so it's `alloca`'d, not `malloc`'d (read from
   `GeneralMatrixMatrix.h`). It still packs on every call.
4. **`lazyProduct` patch for Full** compiles to a vectorized loop with 16 `vfma.f32 q` + 34 `vld1`
   per column and no GEMM call.
5. **Footprints** (arithmetic on the code's allocation formulas): allocated history Lite 210 KiB /
   Full 560 KiB at block 128; live window 110 / 294 KiB; mirror copy per block 34.5 / 92 KiB.
6. **External facts checked:** Raspberry Pi kernel branches (`ARCH_SUPPORTS_RT` from `rpi-7.1.y`;
   Pi 2 defconfig `PREEMPT_VOLUNTARY`, `NO_HZ`, `ondemand`), BCM2836 device tree has an
   `arm,cortex-a7-pmu` node ([bcm2836.dtsi][dtsi]), and the A7 perf event map ([arm_v7_pmu.c][pmu]).

**Not verified:** A7 q-register FMA throughput (community data only), GCC 12's codegen for these
loops, whether current dwc_otg still conflicts with RT, and whether BCM2836 IRQs can be steered off
core 0.

## Sources

- [gcc-arm]: GCC 12 ARM options, `-mfpu` (NEON not IEEE, no FP auto-vectorization without `-funsafe-math-optimizations`): https://gcc.gnu.org/onlinedocs/gcc-12.2.0/gcc/ARM-Options.html
- [a7-timings]: Måns Rullgård, "Cortex-A7 instruction cycle timings" (community measurement): https://hardwarebug.org/2014/05/15/cortex-a7-instruction-cycle-timings/
- [rt-wiki]: PREEMPT_RT, Wikipedia: https://en.wikipedia.org/wiki/PREEMPT_RT
- [rt-heise]: Heise, "Real-time capable Linux 6.12 released": https://heise.de/-10061458
- [rt-arm71]: Phoronix, "With Linux 7.1 The Mainline Kernel Now Supports Real-Time 'RT' On ARM": https://www.phoronix.com/news/Linux-7.1-ARM-RT
- [rpi-linux]: Raspberry Pi kernel tree (`arch/arm/Kconfig`, `arch/arm/configs/bcm2709_defconfig` on `rpi-6.12.y`, `rpi-6.18.y`, `rpi-7.1.y`, `rpi-7.3.y`): https://github.com/raspberrypi/linux
- [osadl]: OSADL, PREEMPT_RT on BCM2709 and the dwc_otg FIQ (2016): https://www.osadl.org/Single-View.111+M5a2ac6f21e6.0.html
- [pmu]: Linux `drivers/perf/arm_v7_pmu.c` (Cortex-A7 event maps): https://github.com/torvalds/linux/blob/master/drivers/perf/arm_v7_pmu.c
- [dtsi]: Linux `arch/arm/boot/dts/broadcom/bcm2836.dtsi`: https://github.com/torvalds/linux/blob/master/arch/arm/boot/dts/broadcom/bcm2836.dtsi
- [rpi-docs]: Raspberry Pi hardware documentation (LAN951x on-board hub/Ethernet): https://www.raspberrypi.com/documentation/computers/raspberry-pi.html
- [dietpi-kernel]: DietPi forum, kernel packages come from the Raspberry Pi apt source: https://dietpi.com/forum/t/linux-headers/403
- [rpi-v7-pkg]: `linux-image-rpi-v7` package listing: https://aptpeek.dev/raspberrypi/bookworm/main/armhf/linux-image-rpi-v7/1:6.12.93-1+rpt1
- [reddit]: r/NAM_NeuralAmpModeler, "1.9ms round-trip latency using a Raspberry Pi 5": https://www.reddit.com/r/NAM_NeuralAmpModeler/comments/1t7w4ho/19ms_roundtrip_latency_using_a_raspberry_pi_5/

[gcc-arm]: https://gcc.gnu.org/onlinedocs/gcc-12.2.0/gcc/ARM-Options.html
[a7-timings]: https://hardwarebug.org/2014/05/15/cortex-a7-instruction-cycle-timings/
[rt-wiki]: https://en.wikipedia.org/wiki/PREEMPT_RT
[rt-heise]: https://heise.de/-10061458
[rt-arm71]: https://www.phoronix.com/news/Linux-7.1-ARM-RT
[rpi-linux]: https://github.com/raspberrypi/linux
[osadl]: https://www.osadl.org/Single-View.111+M5a2ac6f21e6.0.html
[pmu]: https://github.com/torvalds/linux/blob/master/drivers/perf/arm_v7_pmu.c
[dtsi]: https://github.com/torvalds/linux/blob/master/arch/arm/boot/dts/broadcom/bcm2836.dtsi
[rpi-docs]: https://www.raspberrypi.com/documentation/computers/raspberry-pi.html
[dietpi-kernel]: https://dietpi.com/forum/t/linux-headers/403
[rpi-v7-pkg]: https://aptpeek.dev/raspberrypi/bookworm/main/armhf/linux-image-rpi-v7/1:6.12.93-1+rpt1
[reddit]: https://www.reddit.com/r/NAM_NeuralAmpModeler/comments/1t7w4ho/19ms_roundtrip_latency_using_a_raspberry_pi_5/

---

## Round 2 (after Pi measurements, 2026-10-24)

Everything under "Measured (Pi)" comes from `docs/experiment-report.md` (baseline, iterations 1–4).
"Checked (Mac)" means I built or ran something on the Mac for this section: clang armv7 cross builds
with `llvm-objdump`, or native arm64 renders compared with `null_test.py`. None of that is a Pi
timing. "Estimate" means a guess, with the reasoning next to it.

Reference points (Measured, Pi, block 128, clang 16, `performance` governor):

| | p50 | p99 | load | cycles / sample | instr / sample | IPC |
|---|---|---|---|---|---|---|
| Lite, ring mode 1 (`baseline`) | 1,032 µs | 1,092 µs | 38.9% | 7,670 | 6,815 | 0.89 |
| Lite, ring mode 0 (`ring0`, adopted) | 991 µs | 1,067 µs | — | — | — | — |
| Full, ring mode 1 (`baseline`) | 6,696 µs | 6,925 µs | 251% | 46,800 | 21,980 | 0.47 |

Cycles and instructions per sample are the `perf_base` counts divided by the 972,800 processed
samples (iteration 1). They include model load and warm-up, so they slightly overstate per-sample cost.
Full has to get **≥2.6x** faster for its p99 to fit the 2,667 µs deadline.

### R2.1 Why the round-1 vectorization ideas lost on the A7

Two "obvious" changes made more vector code and ran slower (iteration 2: Lite +10%, IPC 0.89 → 0.36;
iteration 3: Full +6%, IPC 0.47 → 0.36). These are the reasons I can support, with how sure I am
of each:

1. **NEON can't dual-issue, and the core is in order.** *High confidence.* GCC's Cortex-A7
   scheduling model says the A7 "is in order and can dual-issue under limited circumstances", and
   "neon instructions cannot dual-issue" ([GCC cortex-a7.md][gcc-a7md]). ARM's launch material
   describes an in-order 8-stage pipeline ([New Electronics][ne-a7]). What follows: every `vorr`
   copy, stack spill, address add and de-interleaving load in NEON code takes its own issue slot,
   and nothing out of order hides a load-use stall. Both refuted variants added exactly those
   instructions (Measured: `vorr`, `sp`-relative `vldmia`, `vld3` and `vld1 {d[]}` were the hot
   instructions in `perf annotate`).
2. **The NEON datapath is 64 bits wide, so a q-register op costs about two d-register ops.**
   *Medium confidence.* It's widely reported and it fits iteration 2: instructions fell 55% while
   cycles rose 11%, so the cycles per instruction went up 2.5x. I couldn't find it in an ARM
   document this session. I found no instruction timings in the ARM material I could reach (the
   NEON MPE TRM is [ARM DDI 0462][a7-neon-trm]), and the author of [hardwarebug][a7-timings] says the
   public TRM omits them. GCC's model
   doesn't separate d and q forms: it books all non-multiply NEON ops as 2 cycles on both issue slots
   (`cortex_a7_both*2`) and NEON MLA as 1 cycle with 8-cycle latency and a 4-cycle accumulator
   bypass. The experiment report calls the 64-bit datapath "ARM's documented microarchitecture";
   I'd soften that to "widely reported" until there's a citation. A microbenchmark settles it
   (round 1, section 5, step 4: `vmla.f32 d` vs `q`, 8 independent accumulators).
3. **clang schedules the A7 as if it were an A8 and refuses to emit VMLA.** *High confidence
   (read from source, then Checked on the Mac).* In LLVM's `ARMProcessors.td`, `cortex-a7` uses
   `CortexA8Model` with `FeatureHasSlowFPVMLx`, `FeatureHasSlowFPVFMx`, `FeatureHasVMLxHazards` and
   `FeatureVMLxForwarding` ([LLVM ARMProcessors.td][llvm-armproc]). Checked (Mac, Apple clang 16 and
   Homebrew clang 23, `-mcpu=cortex-a7`): `vmlaq_lane_f32` compiles to `vmul.f32` + `vadd.f32`, two
   instructions per MAC. My first draft of the Full kernel also came out with every MAC regrouped per
   accumulator (the A8 model's VMLA forwarding makes chains on one accumulator look cheap), which kept
   all 64 weights live and spilled them: 253 instructions per tap loop, including 19 stack stores.
   With VMLA re-enabled for the kernel (`__attribute__((target("no-slowfpvmlx")))`) and a zero-cost
   `asm volatile("" : "+w"(acc)...)` fence after each input pair, the loop is 104 instructions with no
   spills. Iteration 2's `vorr` copies and stack loads look like the same mechanism hitting the
   auto-vectorized loop. That's an inference: I didn't re-derive that build's schedule.
4. **16 q-registers.** *High confidence (architecture).* Auto-vectorizing Lite across 4 frames needs
   9 broadcast weights (9 q) + 3 accumulators + 3 de-interleaved inputs + temporaries, more than 16,
   hence the spills. The fix is to choose a layout where weights are shared across frames by
   construction, not to vectorize harder.
5. **`vld3` is expensive.** *Medium-low confidence on cost; high that it was hot.* `vld3.32` was a
   hot instruction (Measured). GCC's model gives 3-register loads one more cycle of latency than
   1–2-register loads, but that's a compiler model, and `perf annotate` skid smears attribution
   onto neighbouring instructions. Either way, the round-2 Lite kernel doesn't use it.
6. **Weight reload.** *High confidence (Measured).* `lazyProduct` removed packing (−20%
   instructions) but raised L1D loads 18% and dropped IPC to 0.36. The hot instructions were
   `vld1.32 {dN[]}` broadcasts: each weight is re-broadcast for every output column. Eigen's GEMM
   pays packing to keep a register tile, and on this core that was the right trade. A kernel has to
   keep weights in registers across several frames *and* never spill.

The design rule that follows, and that both round-2 kernels obey: **≤14 live q-registers, weights
loaded once per tap and shared by 4 frames, inputs used as `d`-register lanes (`VMLA.F32 Qd, Qn,
Dm[x]`), no de-interleaving loads, VMLA rather than VMUL+VADD, and an explicit fence so clang's A8
model can't regroup the MACs.** VMLA is the non-fused multiply-accumulate (two roundings, the same
as the scalar `a += w * x` code); VFMA is the fused one ([LLVM commit][llvm-vmla]; ARMv7-A ARM
[DDI 0406][armv7arm]). So VMLA can match the reference bit for bit, and VFMA can't.

### R2.2 Ranked candidates

Ranked by expected payoff against the measured bottlenecks: Full's 2.6x gap first, then Lite headroom,
then the tail. All effect numbers are **estimates** unless marked. "Who" says whether Claude can run
the test alone (cross build on the Mac, `remote_sweep.sh`/`perfstat.sh`/renders over SSH, no sudo,
no reboot) or whether it needs Christine (sudo, reboot, kernel, hardware). This assumes the Pi still
has the `performance` governor. After a reboot, setting it again (`set_governor.sh`) needs sudo.

| # | Candidate | Model | Expected effect (estimate) | Effort | Risk | Who | Patch |
|---|---|---|---|---|---|---|---|
| R2-1 | Register-blocked NEON layer kernel (8 out × 4 frames, all taps + tail fused) | Full | **2–4x** (central ~3x) | done; Pi test 1 h | medium | Claude | `a2_full_neon_tile.patch` |
| R2-2 | NEON kernel with output channels in lanes, 4 frames/tile, taps fused, no `vld3` | Lite | **1.2–2.5x** | done; Pi test 1 h | low–medium | Claude | `a2_lite_neon_lanes.patch` |
| R2-3 | Two-core layer pipeline (after R2-1) | Full | 1.8–1.9x throughput, +1 block latency | 2–4 days | medium-high | Claude (code + run) | — (needs `src/`) |
| R2-4 | Compiler: VMLA re-enabled globally, clang 23, GCC 14 native, `-mtune` | both | 0–15% on scalar code; ≈0–5% after R2-1/2 | 1–3 h | low | Claude | — (flags) |
| R2-5 | Block-size re-sweep after R2-1/2 (per-block overhead vs cache) | both | picks the smallest safe block; 64 likely for Lite | 1 h | low | Claude | — |
| R2-6 | Ring mode 1 with dirty-only mirror refresh | both | vs mode 1: −3 to −5% Lite p50; vs `ring0`: 0 ± 2% | done; Pi test 30 min | low | Claude | `a2_ring_mirror_dirty.patch` |
| R2-7 | Head conv: vectorize + drop per-tap masking | both | Lite −3 to −8% after R2-2; Full ≈1% | 0.5 day | low | Claude | — |
| R2-8 | Software prefetch (`pld`) for long-dilation taps | Full | 0–3% now; 0–10% after R2-1 | 0.5 day | low | Claude | — |
| R2-9 | Int16 fixed-point NEON (`vmlal.s16`/`vqdmulh`) | Full | up to 1.5–2x over R2-1 if s16 MACs issue 2x faster (unverified) | 1–2 weeks | high (accuracy) | Claude | — |
| R2-10 | FTZ (FPSCR.FZ) in the audio thread | both | ≈0 normally; removes denormal stalls in scalar remainder/head | 1 h | low | Claude (code in `src/`) | — |
| R2-11 | Per-file `-O2` vs `-O3` (code size / I-cache) | both | ≈0 (Measured: I-cache misses negligible) | 30 min | low | Claude | — |
| R2-12 | `isolcpus=3` + `--rt` + IRQ-affinity check + USB FIQ settings | both | tail only (p99.9/max, wake-up) | 1–2 h + reboots | low–medium | **Christine** | — |
| R2-13 | PREEMPT_RT kernel (`rpi-7.1.y`+) | both | tail only; matters at ≤32–64 frames | 1–3 days | high | **Christine** | — |
| R2-14 | `live --mmap`, then I2S ADC/DAC instead of USB | both | <1% compute; I2S removes the USB latency floor | 1–2 h / hardware | low / medium | **Christine** (hardware) | — |

#### R2-1. Full: register-blocked NEON layer kernel — `patches/a2_full_neon_tile.patch`

**Hypothesis.** Full is overhead-bound in Eigen (Measured: 22% packing, `gebp` at IPC 0.47, 179 GEMM
calls per block), and a coefficient-based product is load-bound (Measured, iteration 3). A kernel
that holds an 8-output × 4-frame accumulator tile in 8 q-registers across all K taps, loads each
weight column once per tile, and fuses bias, mixin, LeakyReLU, head sum and the 1×1 residual removes
the packing, the per-tap stores and reloads of `z`, the broadcast loads, and all K + 1 Eigen calls (with their
setup) per layer.

**Change (the patch).** New `a2_full_tile<K, T>` in `a2_fast.cpp`, used for `Channels == 8` when
`__ARM_NEON` is defined (`-DNAM_A2_FULL_EIGEN` restores the Eigen path). Per input pair (j, j+1):
4 q-loads of W columns, T d-loads of inputs, 4T `vmlaq_lane_f32`. Registers: 8 accumulators + 4 W
+ 2 (inputs) = 14 q. The tap loop stays rolled (`#pragma clang loop unroll(disable)`) to keep code
small. The 1×1 reuses the same MAC routine on a 32-float stack scratch holding the activations.
`noinline` + `target("no-slowfpvmlx")` (clang/ARMv7 only) gets real VMLA. Frames left over from
`num_frames % 4` use a T=1 tile.

**Checked (Mac):**
- Builds cleanly, no warnings: `BUILD_DIR=<scratch> ./scripts/cross_build.sh
  -DBENCH_NAM_PATCHES=patches/a2_full_neon_tile.patch` (Apple clang 16). Also with Homebrew clang 23,
  stacked with the other two patches, and with `-DBENCH_A2_RING_MODE=0`.
- armv7 codegen (`llvm-objdump` on the linked `bench`): `a2_full_tile<6,4>` is 388 instructions.
  The tap loop is **104 instructions per tap per 4-frame tile: 64 `vmla.f32 q`, 32 NEON loads,
  8 other, no spills** (the only stack traffic is the intended 32-float scratch). That's 0.41
  instructions per MAC in the loop. Dynamic count per layer: K=6 → 908 instructions per 4 frames
  (1,792 MACs, 0.51/MAC); K=15 → 1,844 (4,096 MACs, 0.45/MAC). Baseline Full: 1.9 instructions/MAC
  (Measured). The mixin and 1×1 tail still has 16 `vmul` (clang's VMLx-hazard expansion splits some
  tail MACs). That's minor.
- **Null test (Mac arm64, 10 s of `di.wav`, block 128 and 127): Full residual −133.2 dB relative
  (max abs diff 2.68e-7, −131.4 dBFS); Lite bit-identical.** Below the −100 dB bar. The residual
  comes from summation order: one running sum over all taps vs Eigen's per-tap partial sums added
  into `z`. On the Mac both sides use fused `fmla`. On the Pi both use non-fused VMLA, so expect a
  residual of the same order (estimate). **Check it on the Pi before adopting.**

**Expected effect (estimate).** Instructions per sample: layers ≈ 21 × 232 + 2 × 466 ≈ 5,800 (Checked
counts above plus ~5 per frame of call overhead, est.), plus ~1,000–2,000 for ring writes, the head
and I/O (est.). That's ~7,000–8,000 instead of 21,980 (Measured), **≈ −65% (est.)**. Cycles depend
on what a `vmla.f32 q` costs on this core, which is the open question in R2.1 point 2:
- at 2 cycles per q-VMLA with loads not dual-issued: ~170–200 cycles per 256-MAC tap step,
  ≈ 1.3–1.5 MAC/cycle, ≈ 8–9 k cycles/sample of MACs;
- memory: the 204 L2 accesses and 15 L2 misses per sample (Measured counts) don't shrink with the
  kernel. At an assumed 10–20 cycles per L2 hit and 100–150 per miss (assumed, not measured),
  that's 3–6 k cycles/sample if none of it overlaps.

That totals 12–17 k cycles/sample vs 46.8 k, **2.8–3.9x**. If q-VMLA effectively costs 4 cycles,
it's ~21 k, **2.2x**. So the range is **2–4x, central ~3x**, against the **2.6x** needed. It may or
may not get Full's p99 under the deadline alone, which is why R2-3 stays on the list.

**Test (Claude can run all of it):**

    BUILD_DIR=build-armhf-r2full ./scripts/cross_build.sh -DBENCH_A2_RING_MODE=0 \
        -DBENCH_NAM_PATCHES=patches/a2_full_neon_tile.patch
    VARIANT=r2full TAG=r2_full SLIM=full BLOCKS="64 128 256" MODE=burst ./scripts/remote_sweep.sh
    RESULTS_DIR=results-pi python3 scripts/kpis.py --compare ring0_full r2_full
    ssh pi 'cd ~/pi-neural-amp && TAG=perf_r2full BENCH=./build-r2full/bench SLIMS=1 ./scripts/perfstat.sh'
    rsync -a pi:~/pi-neural-amp/results/perf/ results-pi/perf/
    # accuracy on the Pi (reference = ring0 build, same ring mode):
    ssh pi 'cd ~/pi-neural-amp && ./build-r2full/bench --model models/a2_fender_super_reverb.nam --slim 1 \
        --input audio/di.wav --duration 10 --block 128 --render results/render_r2full_slim1.wav \
        --summary-out results/summary_render.csv'
    rsync -a pi:~/pi-neural-amp/results/render_r2full_slim1.wav results-pi/
    RESULTS_DIR=results-pi python3 scripts/null_test.py results-pi/render_ring0_slim1.wav \
        results-pi/render_r2full_slim1.wav --label r2_full_pi

(`remote_sweep.sh` doesn't sync renders: it excludes `*.wav`, hence the explicit `rsync`. Check that
the existing `render_ring0_slim1.wav` used the same `--duration 10 --block 128`, or re-render it. `ring0_full`
is the existing Full ring-mode-0 tag in `results-pi/`; it isn't in the experiment report yet, so
summarize it first.) **Confirms:** Full p50 ≤ 2,500 µs at 128 (≥2.7x) and Pi null ≤ −120 dB. Then
read `perfstat` IPC and L2 misses to see whether the remainder is compute or memory. **Refutes:**
<1.5x. Then the A7's q-VMLA throughput is the limit. Run the d-vs-q microbenchmark and consider a
d-register variant (2 outputs per op, half the lanes, no double pumping).

**Risk.** The A7 surprised twice, so this is a test, not a promise. GCC 14 natively is untested
(no GCC on this Mac); the intrinsics and `"+w"` asm are GCC-compatible as far as I know.

#### R2-2. Lite: NEON kernel with channels in lanes — `patches/a2_lite_neon_lanes.patch`

**Hypothesis.** Lite is scalar, at 3.9 instructions per MAC (Measured). Iteration 2 lost because it
vectorized *across frames* of an interleaved layout (`vld3`, 9 broadcast weights, spills). Put the
3 *output channels* in lanes instead (one q-register per frame, lane 3 = zero padding). Then a tap's
three weight columns are 3 q-registers shared by 4 frames, and the 4 frames' inputs are 12
contiguous floats (3 plain `vld1q`). Each input value is a d-register lane of those loads, so no
de-interleaving is needed. The accumulators stay in registers across all K taps, so the `z` buffer
store/reload per tap disappears.

**Change (the patch).** Padded weight copies (`conv_w4`, `l1x1_w4`, … built once after weight
load), and `a2_lite_tile4<K>` used for each run of 4 frames. Leftover frames fall through to the
original scalar loops (now starting at `f_begin`). The arithmetic order per output is the same as
the scalar code: bias, then taps in order with input channels 0, 1, 2, then mixin, LeakyReLU
(`max(x, 0.01x)`, exact for slope < 1), head sum, and `lin += ((b + L0·a0) + L1·a1) + L2·a2`.
`-DNAM_A2_LITE_SCALAR` restores the scalar path.

**Checked (Mac):**
- Builds cleanly (clang 16 and 23, alone and stacked).
- armv7 codegen: tap loop **22 instructions per tap per 4 frames: 12 `vmla.f32 q`, 6 `vld1`, 4 other,
  no spills, no `vld3`**, i.e. 0.61 instructions per MAC. The scalar tap loop it replaces is 31
  instructions per tap per frame (9 `vmul` + 9 `vadd` + 9 loads/stores …), 3.4 per MAC. Whole tile:
  K=6 → ~246 instructions per 4 frames (264 MACs incl. mixin and 1×1); K=15 → ~444 (588 MACs).
  The tail uses `vtrn`/`vext` to pack 4×3 lanes into the stride-3 buffers.
- **Null test: bit-identical (residual −inf) for Lite and Full**, at block 128 and 127, both with
  clang's default FP contraction (fused `fmla` on both sides) and with `-ffp-contract=off` (unfused on
  both sides, closer to ARMv7 VMLA). The test can see rounding changes: the reference built with
  and without contraction differs by −131.3 dB.
- On the Pi, expect bit-identical output, except that ARMv7 Advanced SIMD always flushes denormals
  to zero while scalar VFP doesn't ([GCC ARM options][gcc-arm]: NEON is not fully IEEE 754). That
  only matters for values ≲1e-38.

**Expected effect (estimate).** Layer instructions ≈ 21 × 66 + 2 × 116 ≈ 1,630 per sample, plus ring
writes, the head (48 scalar MACs) and I/O. That's ~2,000–2,300 instead of 6,815 (Measured),
**≈ −67%**. At 2 cycles per q-VMLA, ≈ 34 cycles per tap per 4 frames plus the tail, ≈ 2,000 cycles/sample
for the layers, ~3,000 total vs 7,670: **~2.5x**. If the result runs at iteration 2's IPC of 0.36,
it's ~6,100 cycles: **1.25x**. **Range 1.2–2.5x**, so Lite p50 at 128 lands somewhere around
400–830 µs (est.). The payoff isn't passing at 128 (it already does). It's headroom for 64 or 32
frames (R2-5) and a lower CPU share.

**Test (Claude):**

    BUILD_DIR=build-armhf-r2lite ./scripts/cross_build.sh -DBENCH_A2_RING_MODE=0 \
        -DBENCH_NAM_PATCHES=patches/a2_lite_neon_lanes.patch
    VARIANT=r2lite TAG=r2_lite SLIM=lite BLOCKS="32 64 128" MODE=burst ./scripts/remote_sweep.sh
    RESULTS_DIR=results-pi python3 scripts/kpis.py --compare ring0 r2_lite
    ssh pi 'cd ~/pi-neural-amp && TAG=perf_r2lite BENCH=./build-r2lite/bench SLIMS=0.25 ./scripts/perfstat.sh'
    # null: render --slim 0.25 as in R2-1, compare against results-pi/render_ring0_slim0.25.wav

**Confirms:** Lite p50 at 128 ≤ 800 µs (≥1.25x vs `ring0`'s 991 µs), null = −inf or ≤ −140 dB.
**Refutes:** ≤5% gain. Then look at `perf annotate`: if the time sits on `vmla.f32 q`, q-ops are the
bottleneck. Fall back to a d-register variant (2 frames per tile, outputs (0,1) in one d-register and
output 2 in a second, 6 d-VMLA per tap).

#### R2-3. Full: two-core layer pipeline

**Hypothesis.** One A7 core may not reach 2.6x even with R2-1 (the low end of its range). Splitting
the 23 layers over two cores halves per-core work at the cost of one block of latency.

**Change.** Split at layer 13. Cost ∝ (K+1) per layer gives 13 × 7 = 91 units for layers 0–12 vs
7 + 2 × 16 + 7 × 7 = 88 for layers 13–22 (Checked, arithmetic). The handoff per block is `layer_in`,
`head_sum` and `cond`: 128 × (8 + 8 + 1) floats ≈ 8.5 KiB through the shared L2. It needs a NAM
API to run a layer range (a patch) and a pinned second SCHED_FIFO/normal thread in `bench`/`live`
with a lock-free SPSC handoff (code in `src/`, outside this round's scope).

**Expected effect (estimate).** Per-stage time ≈ 52% of the single-core R2-1 time plus <2% handoff,
so **1.8–1.9x throughput**. With R2-1 at its central 3x (≈2,230 µs at 128, est.), each stage is
≈1,200 µs, ~45% load. **+1 block latency** (2.67 ms at 128, 1.33 ms at 64). **Who:** Claude can
write and run it (`--pipeline 2`, no sudo; isolating two cores for it is Christine's step).
**Do it only if R2-1 lands below ~2.6x.**

#### R2-4. Compiler choices

**Hypothesis.** clang's A7 target is tuned for an A8 (R2.1 point 3). Re-enabling VMLA globally should
halve the arithmetic instructions in any remaining scalar code without changing results.

- **VMLA everywhere:** `-DCMAKE_CXX_FLAGS="-Xclang -target-feature -Xclang -slowfpvmlx"`. Checked
  (Mac): the unpatched Lite tap loops become 9 `vmla.f32 s` instead of 9 `vmul` + 9 `vadd`
  (63 scalar VMLA in Lite's `process`). Estimate: 0–15% on the unpatched Lite (latency chains, not
  instruction count, may dominate). Expect bit-identical output (VMLA isn't fused); null-test it.
  ≈0–5% once R2-1/R2-2 are in, since their kernels already use VMLA.
- **clang 23** (Homebrew): `-DCROSS_CLANG=/opt/homebrew/opt/llvm/bin/clang
  -DCROSS_CLANGXX=/opt/homebrew/opt/llvm/bin/clang++`. Checked: builds with all patches, same
  104/22-instruction inner loops. The A7 still uses `CortexA8Model` on LLVM `main`, so expect ≈0.
- **GCC 14 native on the Pi:** round 1's single-run comparison had GCC 7% faster on Lite p50
  (Measured, one run). GCC has a real A7 model ([GCC cortex-a7.md][gcc-a7md]). Build natively on the
  Pi (no sudo, slow) and run `sweep.sh` with `TAG=gcc14_*`. The patches haven't been compiled with
  GCC: a build failure is possible, but I expect it to compile.
- **`-mtune`:** clang accepts `-mtune=cortex-a7` silently; on ARM `-mcpu` already picks the model, so
  expect no change (untested on the Pi). **`-fno-trapping-math`:** round 1 showed narrower FP flags
  don't unlock vectorization, and the kernels no longer need it. **`__restrict` and
  `#pragma clang loop unroll`:** already used in the patches. In the Full kernel, a fully unrolled
  tap loop would blow up code size for K=15, so it's kept rolled on purpose.

**Pitfall.** `CMAKE_CXX_FLAGS` isn't recorded in `build_flags`, so put the flag in the `TAG` (e.g.
`TAG=r2_vmla`). `BUILD_FLAGS_STR` records the compiler ID and version, so clang 23 shows up.

#### R2-5. Block size, re-measured after R2-1/2

**Hypothesis.** Once the MACs are cheap, fixed per-block costs (24 ring writes, 23 × (1 + N/4) kernel
calls, head setup) and the history working set (Full's live window: ~246 KiB at 64 vs ~294 KiB at 128,
round 1 §11) decide the best block. Estimate: µs/sample flat from 128 down to 64, rising below 32.

**Test (Claude):** `BLOCKS="16 32 64 128 256" MODE=paced` for the R2 variants (paced, so wake-up
latency counts), then compare `p50_us / block` across sizes. **Confirms** a fixed overhead if
µs/sample at 32 is >20% above 128.

#### R2-6. Ring mode 1 with dirty-only mirror — `patches/a2_ring_mirror_dirty.patch`

**Hypothesis.** Mode 1 re-copies all `max_buffer_size` mirror columns of every layer every block
(Measured: `memcpy` 5.1% of Lite; `ring0` −4% p50, which removes it). Only columns this block wrote
inside `[0, mbs)` need refreshing.

**Change.** `_mirror_dirty()` copies `[wp, min(wp+first, mbs))` and the wrapped `[0, n−first)` into
the mirror. Same buffer contents after every block. **Checked (Mac):** builds cleanly; **bit-identical**
for Lite and Full at block 128 and 127 (127 exercises partial wraps). Copy volume (Checked,
arithmetic over the 23 layers + head): at block 128 an average of 1,096 instead of 3,072 columns per
block, **36%** (Lite 12.8 vs 36 KiB, Full 34 vs 96 KiB). Short-lookback layers (`pow2 = 256`) still
mirror every other block.

**Expected effect (estimate).** vs mode 1: Lite −3 to −5% p50 (≈ the `ring0` delta). vs `ring0`:
0 ± 2% mean, possibly a tighter max (no rewind `memmove`; `ring0`'s rewinds didn't show as spikes,
though, Measured). Its value is a constant-work mode-1 ring that stacks with R2-1/R2-2, and the share
it saves grows as compute shrinks (~41 µs of a ~500 µs Lite block after R2-2, est.).

**Test (Claude):** build *without* `BENCH_A2_RING_MODE` (mode 1), stacked with R2-2, and compare
against R2-2 in mode 0:

    BUILD_DIR=build-armhf-r2mirror ./scripts/cross_build.sh \
        "-DBENCH_NAM_PATCHES=patches/a2_lite_neon_lanes.patch;patches/a2_ring_mirror_dirty.patch"
    VARIANT=r2mirror TAG=r2_mirror SLIM=lite BLOCKS="64 128" MODE=burst KEEP_BLOCKS=1 ./scripts/remote_sweep.sh
    RESULTS_DIR=results-pi python3 scripts/kpis.py --compare r2_lite r2_mirror

#### R2-7. Head conv

The head (K=16, C→1) runs frame-outer with a masked column index per (frame, tap) and a scalar
C-long dot product. That's 48 MACs/sample for Lite (2.8% of MACs) at high overhead per MAC. Change:
tap-outer over the contiguous mirrored history, 4 frames per q-register (`vmlaq_n_f32` by weight),
no masks. Estimate: −3 to −8% of Lite after R2-2 (its share grows as the layers shrink), ≈1% Full.
Not written yet. Null test should be bit-identical if the per-output tap/channel order is kept.

#### R2-8. Prefetch

Measured: L1D miss rate <1.1%, Full 15 L2 misses per sample. Today that's ≲3% of Full's cycles at
assumed DRAM latencies, but it could be 10%+ of R2-1's cycle budget. Change: `__builtin_prefetch` of
the next tile's columns for the dilation-101/239 layers (their taps are 400–1,200 columns apart).
**Test:** `perfstat.sh` L2 misses + `remote_sweep` on top of R2-1. **Confirms** only if R2-1's
`perfstat` first shows L2 misses as a significant share.

#### R2-9. Int16 fixed point for Full (research)

`vmlal.s16 Qd, Dn, Dm[x]` does 4 MACs from a 64-bit d-register input. If the datapath really is
64-bit, it might issue at twice the rate of `vmla.f32 q` (unverified; the microbenchmark tells).
Pipeline: int16 weights/activations, int32 accumulators, `vqrdmulh`/`vqmovn` to rescale per layer,
LeakyReLU as compare/select. **Accuracy plan before any timing:** (1) per-layer max-abs calibration
on `di.wav` plus a second, louder DI; (2) a float simulation of the quantization in Python/C++ on
the Mac, to get the null residual before writing NEON; (3) null test vs float on both DIs, gate at
**−100 dB, which I expect it to miss (est. −70 to −90 dB for 16-bit activations through 23 residual
layers)**; (4) a listening check. Expect it to fail the gate. Do it only if R2-1 + R2-3 fall short.

#### R2-10. FTZ

NEON already flushes denormals. What's left is the scalar remainder frames, the head and the
rechannel. Set FPSCR.FZ once in the audio thread (`vmrs`/`vmsr`, a `--ftz` flag in `src/`).
Estimate ≈0 on normal playing. Possible removal of rare `max` spikes on near-silence decays. Changes
output only at denormal levels. Low priority. Bundle it with R2-7.

#### R2-11. Per-file `-O2` vs `-O3`

Measured: L1 I-cache misses are 219 k (Lite) and 2.27 M (Full) per 20 s run, against 7.5 G / 45.5 G
cycles, under 0.1% even at 20 cycles each (assumed). The round-2 kernels are small (388 and 136
instructions, Checked). **Estimate ≈0.** `-O3` is hard-coded in `CMakeLists.txt` (`OPT_FLAGS`), so a
per-file `-O2` arm needs a new CMake option. Skip it unless something else points at code size.

#### R2-12 to R2-14. OS, kernel and I/O (Reddit items, tail and latency, not compute)

All of these need Christine: they need sudo, a reboot, a kernel build, or hardware.
- **R2-12 `isolcpus=3` + `--rt` + IRQs:** `compare_rt.sh` arms `rt`/`rt_isol` (RUNBOOK §4/4b) on top
  of the R2 build with the best block size. Read `/proc/interrupts` during a `live` run to see where
  `dwc_otg` lands (root not needed to read it; `irqaffinity`/`/proc/irq/*/smp_affinity` writes need
  root). USB FIQ (`dwc_otg.fiq_fsm_enable`, `fiq_enable` on the cmdline) is a reboot experiment.
  Leave it at the default unless xruns point at USB. Expected: lower `wake_p99_us`/max, no change to
  p50.
- **R2-13 PREEMPT_RT:** unchanged from round 1 (item 13): `rpi-7.1.y`+ self-built, decide from
  `missed_with_wake` in `rt_isol`. It matters only once R2-2 makes 32–64-frame blocks viable.
- **R2-14 `live --mmap`, then I2S:** `--mmap` is planned (round 1, item 14). Expected <1% of the
  deadline. Claude can write it; running it needs the interface connected. I2S stays a hardware
  decision after the USB floor (`live --bypass` + `latency_test.sh`) is measured. Not tied to this
  round's compute work.

### R2.3 Suggested order

1. **R2-1 on the Pi** (Full, ring mode 0) with its null test and `perfstat`. It's the only item that
   can close most of Full's gap, and its result decides R2-3 and R2-8.
2. **R2-2 on the Pi** (Lite), then **R2-5** block sweep with R2-1 + R2-2 stacked
   (`-DBENCH_A2_RING_MODE=0 "-DBENCH_NAM_PATCHES=patches/a2_full_neon_tile.patch;patches/a2_lite_neon_lanes.patch"`;
   Checked: builds).
3. **R2-4 VMLA flag and GCC 14**: cheap, mostly for the record.
4. **R2-6** mirror vs `ring0`, then **R2-7** head.
5. **R2-3** only if R2-1 < 2.6x. **R2-9** only if R2-1 + R2-3 fall short.
6. **R2-12/13** (Christine) once the target block size is known.

### R2.4 What I checked for this section, and what I couldn't

Checked (Mac, 2026-10-24):
- All three patches apply with `patch -p1` to a copy of `external/NeuralAmpModelerCore/NAM` at
  `0b3d3c9`, alone and stacked in the order R2-1, R2-2, R2-6. `external/` itself is untouched.
- Cross builds (`scripts/cross_build.sh`, scratch `BUILD_DIR`s, Apple clang 16): each patch alone,
  all three stacked, R2-1+R2-2 with `-DBENCH_A2_RING_MODE=0`, and R2-1+R2-2 with Homebrew clang
  23.1.4: all build with 0 warnings. `build_flags` records the patch names.
- Codegen numbers above (`llvm-objdump` on the linked armv7 `bench`).
- Native arm64 renders (10 s of `audio/di.wav`, block 128; also block 127 for R2-1/R2-6), compared
  with `null_test.py`:

| Patch | Lite (`--slim 0.25`) | Full (`--slim 1`) |
|---|---|---|
| `a2_full_neon_tile` | bit-identical | **−133.2 dB** (max 2.68e-7, −131.4 dBFS) |
| `a2_lite_neon_lanes` | bit-identical (also with `-ffp-contract=off`) | bit-identical |
| `a2_ring_mirror_dirty` | bit-identical | bit-identical |
| all three stacked | bit-identical | −133.2 dB |

Not verified:
- **Any Pi timing of the patches.** Every speedup above is an estimate.
- Pi (ARMv7) numerics. Mac arm64 NEON uses fused `fmla` and doesn't flush denormals; the Pi uses
  VMLA with flush-to-zero. Run the null tests on the Pi renders.
- The A7's q- vs d-register NEON throughput and VMLA latency (no ARM doc found; GCC's model is a
  proxy).
- GCC 14 compiling the patches.
- DRAM/L2 latencies used in the R2-1 estimate (assumed values).

### Round 2 sources

- [gcc-a7md]: GCC Cortex-A7 pipeline description, `gcc/config/arm/cortex-a7.md` (in-order, limited dual-issue, NEON never dual-issues, MLA latency/bypass): https://github.com/gcc-mirror/gcc/blob/master/gcc/config/arm/cortex-a7.md
- [llvm-armproc]: LLVM `ARMProcessors.td` (`cortex-a7` uses `CortexA8Model`, `HasSlowFPVMLx`, `HasSlowFPVFMx`, `HasVMLxHazards`, `VMLxForwarding`): https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/ARM/ARMProcessors.td
- [llvm-vmla]: LLVM commit noting VMLA and friends are not fused (separate rounding steps), mirrored copy: https://git.applefritter.com/6502/llvm-6502/commit/48575f6ea7d5cd21ab29ca370f58fcf9ca31400b
- [armv7arm]: ARMv7-A/R Architecture Reference Manual (VMLA vs VFMA, Advanced SIMD flush-to-zero): https://developer.arm.com/documentation/ddi0406/latest/
- [a7-neon-trm]: Cortex-A7 NEON Media Processing Engine TRM, ARM DDI 0462 (no instruction timings): https://developer.arm.com/documentation/ddi0462/latest/
- [ne-a7]: New Electronics, "ARM unveils Cortex-A7 MPCore processor" (in-order 8-stage pipeline): https://newelectronics.co.uk/content/news/arm-unveils-cortex-a7-mpcore-processor
- Round 1 sources still used: [gcc-arm], [a7-timings].

[gcc-a7md]: https://github.com/gcc-mirror/gcc/blob/master/gcc/config/arm/cortex-a7.md
[llvm-armproc]: https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/ARM/ARMProcessors.td
[llvm-vmla]: https://git.applefritter.com/6502/llvm-6502/commit/48575f6ea7d5cd21ab29ca370f58fcf9ca31400b
[armv7arm]: https://developer.arm.com/documentation/ddi0406/latest/
[a7-neon-trm]: https://developer.arm.com/documentation/ddi0462/latest/
[ne-a7]: https://newelectronics.co.uk/content/news/arm-unveils-cortex-a7-mpcore-processor
