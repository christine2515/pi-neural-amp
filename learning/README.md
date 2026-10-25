# Learning notes: neural amp on a Raspberry Pi 2

Study notes distilled from building and optimizing this project. Each topic is taught with the real
number or moment from the project that showed it. For the full experimental record, see
[../docs/experiment-report.md](../docs/experiment-report.md).

## Reading order

| # | File | What you'll learn |
|---|---|---|
| 1 | [01-real-time-audio.md](01-real-time-audio.md) | Blocks, deadlines, latency vs throughput, why "too slow" means dropouts |
| 2 | [02-nam-and-wavenet.md](02-nam-and-wavenet.md) | What NAM is, how the WaveNet runs, A2 Lite vs Full, counting MACs |
| 3 | [03-raspberry-pi-and-linux-rt.md](03-raspberry-pi-and-linux-rt.md) | Pi 2 hardware, governors, real-time scheduling, ALSA, USB vs I2S |
| 4 | [04-measuring-performance.md](04-measuring-performance.md) | Percentiles, perf, IPC, null tests, and the observer effect |
| 5 | [05-cpu-performance-on-arm.md](05-cpu-performance-on-arm.md) | NEON, in-order cores, register pressure, why "obvious" vectorization lost |
| 6 | [06-optimization-journey.md](06-optimization-journey.md) | Every iteration: hypothesis, change, result, lesson |
| 7 | [07-parallelism-and-pipelining.md](07-parallelism-and-pipelining.md) | The two-core pipeline that made Full real-time |
| 8 | [08-build-and-tooling.md](08-build-and-tooling.md) | Cross-compiling, CMake, patches, the remote workflow, rsync/SSH pitfalls |
| 9 | [09-open-questions-and-next-steps.md](09-open-questions-and-next-steps.md) | What's left, plus self-test questions with answers |
| 10 | [10-wavenet-explained.md](10-wavenet-explained.md) | Deep dive after 2: one sample through the A2 WaveNet, with shapes, a worked example, figures, and real-audio activations |
| 11 | [11-electric-guitars-and-hum.md](11-electric-guitars-and-hum.md) | How pickups work, why a Strat hums, the hum diagnosis, noise gates |

## Cheat sheet: the 20 things to remember

1. **Deadline = block ÷ sample rate.** At 48 kHz: 64 → 1.33 ms, 128 → 2.67 ms, 256 → 5.33 ms, 512 → 10.67 ms. Each sample is worth 20.8 µs.
2. **Too slow doesn't mean more latency; it means dropouts.** Full at 250% load produced only 40% of the audio needed. A bigger buffer only delays the first gap.
3. **Buffers absorb spikes, not average overload.** Extra periods help when average load < 100% and rare blocks spike over.
4. **Cost scales with channels².** Lite (3 ch) is 1,731 MAC/sample, Full (8 ch) 11,776. That's 6.8× predicted; the Pi measured 6.9×.
5. **Receptive field: 6,347 samples ≈ 132 ms.** Dilated convolutions see a long history cheaply.
6. **Measure the tail.** p99 and max decide glitches; p50 decides feasibility.
7. **Burst vs paced.** Back-to-back runs flatter the code (warm caches). Paced runs mimic a real audio callback and expose wake-up latency.
8. **Governor matters.** `ondemand` caused deadline misses (Lite max 3.4 ms); use `performance` for benchmarks.
9. **The observer effect is real.** A memory sampler raised Lite p99 by 27% (0.254 s spikes). A thermal logger calling `vcgencmd` caused stalls up to 7 ms. The fork-free Python logger matched having no logger at all.
10. **Profile before optimizing.** `perf` showed neither model was memory-bound (>98.9% L1 hits). Full lost 22% of its time to Eigen repacking data.
11. **On the Cortex-A7, every FP form peaks at ~1 float lane-op per cycle.** A 4-lane NEON q op takes ~4 cycles, d ~2, scalar ~1. VMLA latency is 5 cycles (measured with `neon_peak`).
12. **So vectorization only wins by cutting non-math instructions** (loads, index math, overhead), not by doing more math per cycle.
13. **Fewer instructions ≠ faster.** Fast-math cut Lite's instructions 55% but made it 10% slower (IPC 0.89 → 0.36): `vld3` de-interleaves, register spills, and clang's slow-VMLA model.
14. **Register blocking wins.** Keeping accumulators in registers across taps made Full 1.91× faster and Lite 1.17×.
15. **Spare cores are the big lever.** A two-stage pipeline gave 1.97× for +1 block of latency. Full went from 3,511 to 1,785 µs per 128-sample block.
16. **Balance the pipeline.** Splitting at layer 13 was optimal; each layer off-balance cost ~140 µs per block.
17. **`--rt` fixes wake-ups, not compute.** SCHED_FIFO cut wake-up p99 ~4× (73–80 → 16–24 µs) and left compute time unchanged.
18. **Compilers differ on hand kernels.** GCC 14 spilled the Full kernel (581 instructions, 100 stack refs vs clang's 388 / 13), so Full ran +26% slower; Lite ran 6% faster.
19. **Null tests prove you didn't change the sound.** Ring mode 0 was bit-identical; the NEON kernels were −130 to −133 dB. The bar is ~−100 dB, and 16-bit audio's noise floor is ~−96 dB.
20. **Final:** Full 6,696 → 1,767 µs (3.79×), real time at 128 samples on two cores, clean over 30 minutes at ~44 °C. Lite 1,032 → 843 µs, glitch-free at 32 samples.

## Key repo docs

- [../docs/experiment-report.md](../docs/experiment-report.md): every measurement, with tables
- [../docs/optimization-roadmap.md](../docs/optimization-roadmap.md): ranked ideas, round 1 and round 2
- [../docs/hardware-comparison.md](../docs/hardware-comparison.md): Pi 2 vs Pi 5 context
- [../RUNBOOK.md](../RUNBOOK.md): how to run everything
