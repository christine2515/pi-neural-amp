# 06: The optimization journey

All runs: Pi 2, block 128 at 48 kHz (deadline 2,667 µs), clang 16 cross build, `performance` governor,
unless noted. Full details: [../docs/experiment-report.md](../docs/experiment-report.md).

| # | Hypothesis | Change | Result | Lesson |
|---|---|---|---|---|
| 0 | Baseline | Stock A2 fast path | Full p50 6,696 µs (251% load, misses every block); Lite 1,032 µs (39%) | Full can't run live as built |
| C1/C2 | Spikes come from the governor or the sampler? | Controls changing one variable | Sampler → 0.254 s spikes (+27% p99); `ondemand` → >2.67 ms misses | Change one thing per run |
| 1 | Where does time go? | `perf stat/record` | Not memory-bound; Full 22% Eigen packing; Lite scalar | Profile first |
| 2 | Fast-math vectorizes Lite | `-ffast-math` on a2_fast.cpp | **Lite −10%** (IPC 0.36) | Vector ≠ faster on A7 (vld3, spills) |
| 3 | Remove GEMM packing | Eigen `lazyProduct` | **Full −6%** (+18% loads) | Packing enabled register reuse |
| 4 | Mirror memcpy wastes time | Ring mode 0 | Lite p50 −4%, max −18%, bit-identical; Full neutral | Cheap, free win |
| 5 | How small can Lite go? | Paced 16–128, ± `--rt` | Lite clean at **32**; `--rt` wake p99 ~4× better | `--rt` fixes wake-ups, not compute |
| 6 | Register-blocked NEON kernels | Patches for Full and Lite | **Full 1.91×, Lite 1.17×** | Design kernels for the core |
| 7 | What can the A7 do? | `neon_peak` asm microbenchmark | ~1 float op/cycle for every form; VMLA latency 5 | Measure instead of trusting folklore |
| 8 | Use a second core | Two-stage pipeline | **Full 1.97× → 1,785 µs (67% load)**, +1 block latency | Idle cores were the big lever |
| 9 | Paced pipelined Full | 64/128/256 ± `--rt` | Clean at 256; ~1 in 25,000 at 128 | Tail = OS interference |
| 10 | Best split point | Split 11–15 | **13** best; ~140 µs per layer off-balance | Balance stages by cost |
| 11 | Lite at 16 samples | Pipelined Lite | 0 misses at 16, but total delay = 32 | Pipeline buys headroom, not latency |
| 12 | 30-minute stability | Sustained run + loggers | `vcgencmd` logger: 66 overruns, 7 ms stalls → fork-free logger: 7 marginal overruns, 44 °C, no throttling | Monitoring perturbs timing |
| 13 | Global VMLA flag helps scalar code | clang `-slowfpvmlx` | No effect | Hot loops already optimal |
| 14 | Head conv as a dot product | NEON head patch | −1.1% Full, −1.4% Lite | Small but free |
| 15 | Is clang costing us? | GCC 14 native build | Lite −6%, **Full +26%** (spills) | Hand kernels are compiler-sensitive |

## Final numbers

| | Start | Best | How |
|---|---|---|---|
| Full p50, block 128 | 6,696 µs (251%) | **1,767 µs (66%)** | NEON kernel × pipeline × head = **3.79×** |
| Full sustained 30 min | n/a | p50 1,786 / p99 1,922 µs, 7 marginal overruns, 44 °C peak | + `--rt`, fork-free logger |
| Lite p50, block 128 | 1,032 µs | **843 µs** | ring mode 0 + NEON + head |
| Lite smallest clean block | n/a | **32 samples** (0.67 ms) | paced, ± `--rt` |
| Best build | | `nhead` = ring0 + full tile + lite lanes + pipeline + head, clang 16 | |

## The meta-lessons

1. **Measure before and after every change**, with a null test for sound.
2. **Negative results are results.** Two "obvious" optimizations lost, and understanding why (the
   microbenchmark) shaped what worked.
3. **Know your hardware's real limits** (~1 FP op/cycle) before choosing a strategy.
4. **The tools can be the bug:** the sampler and logger caused the worst tails.
5. **Write down the hypothesis first,** so a result can confirm or refute it.

## Key takeaways

> - The wins: kernel design for the core (1.9×), using a second core (2×), small free wins (ring0, head).
> - The losses: compiler auto-vectorization and Eigen `lazyProduct`. Both cut instructions but added cycles.
> - 3.79× on Full turned "can't run" into "real time at 128 samples".

## Check yourself

1. Which two changes produced almost all of Full's 3.79×?
2. Why was the `neon_peak` microbenchmark a turning point?
3. Why did adopting ring mode 0 need a null test even though it's "just a buffer layout"?
