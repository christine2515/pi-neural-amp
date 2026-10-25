# 04: Measuring performance honestly

## How `bench` times a block

```cpp
t0 = steady_clock::now();      // CLOCK_MONOTONIC: never jumps when the wall clock changes
model->process(in, out, 128);  // only this is timed
t1 = steady_clock::now();
blockNs[b] = t1 - t0;          // preallocated array; nothing written until the run ends
```

- **Audio-path rules:** no allocation, locks, logging or syscalls inside the timed loop.
- **Warm-up:** the first 100 blocks are discarded.
- **Every CSV row records** `host`, `kernel`, `governor`, `build_flags` (compiler, flags, patches),
  `isolated_cpus`. That caught a mix-up where two Mac rows ended up in the Pi's results after an rsync
  (see [08-build-and-tooling.md](08-build-and-tooling.md)).

## Statistics that matter

| Metric | Meaning | Why it matters |
|---|---|---|
| **p50** | Typical block | Feasibility: is the average under the deadline? |
| **p99** | 1-in-100 worst | Headroom against routine spikes |
| **max** | Worst block | One bad block = one click |
| **missed** | Blocks > deadline | Audible glitches (process time only) |
| **missed incl. wake** | wake lateness + compute > deadline | Realistic glitch count in paced mode |
| **DSP load** | mean time ÷ deadline | 250% = needs 2.5× faster |
| **CPU %** | CPU time ÷ wall time | 100% = one full core (not all four); the pipeline shows 200% |

**Catch-up cascades:** in the 30-minute run, 66 compute overruns caused **3,027** late blocks while
the schedule caught up. A real ALSA stream would register each stall as one underrun, so count the
overruns.

## Memory: PSS and RSS

- **RSS:** resident pages. **PSS:** proportional share (shared libraries split among the processes using them).
- **Lesson:** peak PSS was **16.9 MiB** for both Lite and Full, but **12.7 MiB** of it was the decoded
  69 s test file held by `bench`. The model plus buffers plus libraries came to ~4 MiB. Lite and Full
  matched because both submodels are always loaded. **Measure what you mean to measure:** a streaming
  `live` process is the honest memory number.
- Memory is now read by the program itself after the loop (`/proc/self/smaps_rollup`, `getrusage`),
  with no external process.

## perf: what the CPU is doing

- `perf stat -e cycles:u,instructions:u,...`: counters. The A7 has **4 counters**, so we ran events in
  groups of ≤4 to avoid multiplexing. `:u` = user space only (works with `perf_event_paranoid=2`).
- `perf record` + `perf report`: where time goes. `perf annotate`: which instructions are hot.
- **IPC** (instructions per cycle) on an in-order core:
  - High IPC but too slow → too many instructions.
  - Low IPC → stalls (memory, dependencies, or multi-cycle instructions).
  - Caveat: once most instructions are 4-cycle NEON q ops, a low IPC is expected (see [05](05-cpu-performance-on-arm.md)).
- **Project baseline profile:** L1D miss rate 0.72–1.09%, so **not memory-bound**. Full: Eigen GEMM
  kernel 59%, **packing 22%**. Lite: 89% in its scalar kernel, 5% `memcpy`.

## Null tests: proving you didn't change the sound

- Render the same input with two builds, subtract, and report the residual relative to the signal in dB
  (`scripts/null_test.py`).
- **Bar:** ≈ **−100 dB** relative. For scale, 16-bit audio's noise floor is ≈ −96 dB.
- **Results:** ring mode 0 was bit-identical (−∞). The NEON kernels were −130 to −133 dB (summation
  order). The pipeline was bit-identical after `--shift 128` (it outputs one block late).

## The observer effect (biggest lesson of the project)

**The memory sampler** (`mem_sampler.sh`, every 0.2 s, starting `awk`/`sed`/`date`):
- Lite spikes every **0.254 s**, ~4 per second; p99 **+27%** (1,092 → 1,386 µs).
- Cause: process starts on other cores share the L2 and memory bus.
- Fix: memory is self-reported, and the sampler is off by default.

**The thermal logger:**
- With `vcgencmd` every second: 66 overruns in 30 min, stalls up to **7.3 ms**.
- A 10-minute control **with no logger**: 0 overruns in 225,000 blocks.
- A sysfs-only shell version still caused late wake-ups every ~2.1 s (forks and firmware-backed reads).
- A **fork-free Python logger** reading only the temperature register: indistinguishable from no
  logger. Run 3 over 30 min: 7 marginal overruns (≤35 µs over), no stalls.

**Rule:** nothing that forks or talks to firmware may run periodically during a timed run.

## How controls isolate one variable

The `ondemand` run differed from the new baseline in **two** ways (governor + sampler). A control with
`performance` **+ sampler on** separated them: the sampler explained the 0.254 s spikes, and the
governor explained the >2.67 ms misses. Change one thing per run.

## Key takeaways

> - Report p50 + p99 + max + misses; the tail is what you hear.
> - Profile before optimizing; IPC and cache-miss rates tell you what kind of bottleneck you have.
> - Prove sound equivalence with null tests (−100 dB bar).
> - Your measurement tools can be the problem. Use controls to find out.

## Check yourself

1. Why were Lite and Full's PSS the same, and what would give a real memory number?
2. What does L1D miss rate < 1% rule out?
3. The 30-minute run showed 66 overruns but 3,027 late blocks. Which number estimates audible glitches?
4. Design a control to test whether X causes spikes.
