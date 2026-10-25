# 07: Parallelism and pipelining

## Why audio is hard to parallelize

- Each block depends on the previous one (layers keep history).
- Within a block, layer *n+1* needs layer *n*'s output: 23 sequential steps.
- So by default the model runs on **one core** while three sit idle (`cpu=100%` = one full core = 25%
  of the Pi).

## Two ways to split the work

| Approach | How | Gain | Cost |
|---|---|---|---|
| **Pipeline** (what we built) | Core A: layers 0–12 on block *n*; core B: layers 13–22 + head on block *n−1* | ~2× with 2 cores | **+1 block latency** (2.67 ms at 128); few sync points |
| **Data-parallel** | Split each layer's 8 channels across cores; sync after every layer | Up to ~3–4× on paper | No added latency, but 23 sync points per block; Lite (3 ch) splits poorly |

## How the pipeline works

- **Two buffer sets alternate.** During call *n*, the worker runs stage B of block *n−1* on set
  (n−1)%2 while the audio thread runs stage A of block *n* on set n%2. Then the audio thread waits
  and returns block *n−1*'s output.
- **Disjoint state:** stage A owns layers 0–12's histories; stage B owns 13–22 and the head. No overlap,
  so no locks.
- **Lock-free handoff:** two atomic counters (`_job`, `_done`) with acquire/release ordering. The
  worker **spin-waits** with `yield`: no syscalls, no sleeping (it burns a core, which is fine on a 4-core Pi).
- **Setup outside the audio path:** the worker thread is created at model load, pinned (core 2), and
  given SCHED_FIFO when `--rt` is used. Prewarm runs unpipelined.
- **Correctness:** the output is exactly one block late. After `--shift 128` it's **bit-identical** to
  the unpipelined output.

## Results

| Full, block 128 | p50 | Load | Cores |
|---|---|---|---|
| NEON kernel, 1 core | 3,511 µs | 132% | 1 |
| **Pipelined** | **1,785 µs** | **67%** | 2 (CPU 200%) |

That's 1.97× from 2 cores, nearly perfect scaling.

## Load balancing

- The cost model put layers 0–12 vs 13–22 + head at ≈ 91 vs 88 units (the two kernel-15 layers sit at
  14–15).
- **Measured** split 11/12/13/14/15 → p50 2,057 / 1,918 / **1,781** / 1,923 / 2,198 µs.
- Each layer off-balance costs ~140 µs per block: the slower stage sets the pace.

## When a pipeline doesn't help latency

- Lite pipelined at **16 samples**: 0 misses, p99 82 µs of a 333 µs deadline.
- But the delay is 16 + 16 = **32 samples**, the same as unpipelined Lite at 32.
- **For Lite, the pipeline buys headroom, not lower latency.** For Full, it's the difference between
  impossible and real time.

## Key takeaways

> - Pipelining trades a fixed amount of latency for throughput.
> - Keep stage state disjoint, and hand off with atomics; never locks in the audio path.
> - Balance stages by measured cost: the slowest stage sets the pace.
> - Check what the extra latency buys: Full needed it; Lite didn't.

## Check yourself

1. Why does a 2-stage pipeline add exactly one block of latency?
2. Why are no locks needed between the stages?
3. What happens to throughput if stage B is 20% heavier than stage A?
4. When would a data-parallel split beat a pipeline?
