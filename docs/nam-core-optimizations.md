# NAM Core optimization candidates for the Pi 2

Investigation notes for the Fender A2 model (`a2_fender_super_reverb.nam`, Lite = 3 channels,
Full = 8 channels, 23 layers). Nothing here is a Pi measurement yet. Each item says what evidence
exists and what to run on the Pi to measure it. Ranked by expected payoff per hour of work.

## How the A2 model runs (per block of N samples)

`A2FastModel<Channels>::process` in `external/NeuralAmpModelerCore/NAM/wavenet/a2_fast.cpp`:

1. **Rechannel**: input sample → `Channels` values.
2. **23 layers**, each:
   - dilated convolution: kernel 6 (or 15), dilations 1…239, `Channels → Channels`,
     read from a per-layer ring buffer
   - + bias + input mixin, LeakyReLU(0.01)
   - add into the head accumulator
   - 1×1 conv residual back into the layer input
3. **Head**: kernel-16 convolution `Channels → 1`, scaled.

Two code paths, chosen at compile time by channel count:
- **Lite (3 ch)**: hand-unrolled scalar loops (3×3 matrix–vector per frame), written so the compiler
  can vectorize *across frames*.
- **Full (8 ch)**: Eigen matrix multiplies, one 8×8 × 8×N per tap.

The file's comments say it was tuned on an Apple M1 with clang. The Pi 2 is a different world:
32-bit ARMv7, Cortex-A7, 16 NEON q-registers, GCC 12, and NEON that is not fully IEEE-754.

## 1. Let the compiler use NEON: `-ffast-math` / `-funsafe-math-optimizations` (highest payoff, ~1 rebuild)

**Finding.** GCC's documented behaviour on ARM: with `-mfpu=neon`, floating-point code is *not*
auto-vectorized unless `-funsafe-math-optimizations` is on, because ARMv7 NEON flushes denormals and
isn't IEEE-754 compliant. Clang behaves the same way. Cross-compiling `a2_fast.cpp` for
`armv7a-linux-gnueabihf -mcpu=cortex-a7 -mfpu=neon-vfpv4 -O3` with clang and counting float
arithmetic instructions:

| A2 code | Flags | Vector f32 ops (q regs) | Scalar f32 ops |
|---|---|---|---|
| Lite `process` | `-O3` | **0** | 328 |
| Lite `process` | `-O3 -ffast-math` | **231** | 296 |
| Full `process` | `-O3` | 52 | 57 |
| Full `process` | `-O3 -ffast-math` | 60 | 57 |

So with the current build flags **the Lite model runs fully scalar on the Pi 2**. Full is mostly
vectorized already through Eigen's hand-written NEON code. (Counts are clang; the Pi builds with
GCC, which documents the same rule. Confirm with `objdump -d` on the Pi build.)

**Accuracy.** On the Mac, a `-ffast-math` build of Lite differs from the default build by a
residual of −135 dB relative to the signal (max sample difference −131 dBFS), measured with
`scripts/null_test.py`. 16-bit audio's noise floor is about −96 dB, so that's inaudible.
Re-run the null test on the Pi, since the arithmetic differs there.

**Run on the Pi:**

    cmake -S . -B build-fast -DCMAKE_BUILD_TYPE=Release -DBENCH_FAST_MATH=ON && cmake --build build-fast -j2
    BENCH=./build-fast/bench TAG=fastmath ./scripts/sweep.sh
    python3 scripts/kpis.py --compare baseline fastmath
    ./build/bench      --model models/a2_fender_super_reverb.nam --slim 0.25 --input audio/di.wav --block 128 --render results/ref_lite.wav
    ./build-fast/bench --model models/a2_fender_super_reverb.nam --slim 0.25 --input audio/di.wav --block 128 --render results/fast_lite.wav
    python3 scripts/null_test.py results/ref_lite.wav results/fast_lite.wav --label fastmath_lite

Faster alternative: cross-compile both variants on the Mac (seconds each), deploy them to separate
folders, and compare those two against each other, not against the Pi's GCC baseline:

    BUILD_DIR=build-armhf ./scripts/cross_build.sh
    BUILD_DIR=build-armhf-fast ./scripts/cross_build.sh -DBENCH_FAST_MATH=ON
    PI_USER=<you> BUILD_DIR=build-armhf DEST=build-clang ./scripts/deploy.sh
    PI_USER=<you> BUILD_DIR=build-armhf-fast DEST=build-clang-fast ./scripts/deploy.sh

Narrower GCC-only variant (only allows reassociation and FTZ, keeps NaN/inf handling):
`-DCMAKE_CXX_FLAGS=-funsafe-math-optimizations` instead of `BENCH_FAST_MATH=ON`.

**Expected:** a large Lite speedup, little change for Full. Story: "the vendor's fast path was tuned
for M1; on ARMv7 its hot loop compiled to scalar code; one flag, verified with a null test, fixed it."

## 2. Flush denormals to zero in the audio thread (cheap)

Scalar VFP code on ARMv7 can slow down on denormal floats (tiny values, e.g. reverb/feedback tails
or near-silence). Setting the FPSCR flush-to-zero bit once at the start of the audio thread is
standard practice in audio code. NEON already flushes, so this matters mostly for the default
(scalar) build. A `--ftz` flag in `bench`/`live` is a few lines. Measure on a DI with silence in it
(the untrimmed original) vs. trimmed.

## 3. Ring-buffer mode (compile flag, no code)

`a2_fast.cpp` has two history-buffer strategies, `NAM_A2_RING_MODE=1` (default: power-of-two ring
with mirrored tail, constant work) and `=0` (linear buffer with occasional memmove: cheaper on
average, periodic spikes). Chosen on M1. On the Pi's small caches the trade-off may differ, and the
spike shows up in p99/max, which is what you care about. Build with
`-DCMAKE_CXX_FLAGS=-DNAM_A2_RING_MODE=0` and compare p99 and max, not mean.

## 4. Hand-written NEON kernels (the DIY project)

If profiling after #1 still shows the layer convolutions dominating:
- **Lite:** vectorize across 4 frames with `float32x4_t`: each 3×3 tap becomes 9 `vmlaq_n_f32`
  over 4 frames. That's what `-ffast-math` asks the compiler to do; writing it by hand gives the
  same speed without changing math semantics anywhere else, and makes a clean write-up.
- **Full:** Eigen's general matrix-multiply kernel is tuned for large matrices and ARMv8. An 8×8
  weight matrix is 16 q-registers, exactly ARMv7's whole register file, so a hand kernel has to
  tile carefully (e.g. 4 output channels × 4 frames per pass). Only worth it if `perf` shows Eigen's
  `gebp` kernel at the top.
- **Head:** kernel-16 × `Channels` dot product per frame, written as a scalar loop. Small but runs
  every sample; easy to vectorize.

Measure each with the existing sweep plus a null test.

## 5. Use the other three cores (bigger change, latency trade-off)

The model runs on one core; the Pi 2 has four. Splitting the 23 layers into two stages on two
cores (a pipeline) roughly halves per-core work but adds one block of latency. That's a real
product trade-off: e.g. at 64 samples, one extra block is +1.33 ms. Synchronization has to be
lock-free (single-producer/single-consumer ring between the threads) to keep the audio-path rules.
Worth it only if Full can't meet 128 samples on one core after #1–#4.

## Not a fit: FFT convolution

FFT convolution beats direct convolution when kernels are long (hundreds to thousands of taps), as in
cabinet impulse responses. NAM Core already has a partitioned FFT path for *Linear* models
(`tools/BENCHMARK_LINEAR.md`, tuned on an Apple M1). The A2 WaveNet's kernels are 6–16 taps with
dilation, so each layer is a handful of multiply-adds per sample per channel. FFT setup and
transforms would cost far more than they save. The A2 model also models the cab inside the network,
so there's no separate IR to FFT-convolve. If you later add a separate IR cab after a Linear
or amp-only model, re-tuning that FFT dispatch table for the Pi 2 is a legitimate project.

## Before optimizing: profile on the Pi

Every item above is a hypothesis about where Pi time goes. Run RUNBOOK §5 (`perf record` on the
Lite and Full models at block 128) first, so each optimization is chosen by the profile and its
before/after is a sweep result, not a guess.
