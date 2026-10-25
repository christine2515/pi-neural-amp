# Low-latency plan: how close to "analog pedal" can the Pi 2 get?

Goal: cut total (round-trip) latency of the NAM A2 amp on the Raspberry Pi 2 toward what a player
feels as "no delay", by making the compute work at small blocks. Written 2026-10-25 by Claude for
Christine. Builds on [experiment-report.md](experiment-report.md) (iterations 1–15) and
[optimization-roadmap.md](optimization-roadmap.md). GPU/QPU, soft-float and int16 kernels are
another agent's work and are not covered.

**Labels.** **Measured (Pi)** = a CSV in `results-pi/` produced by the scripts named next to it.
**Checked (Mac)** = a Mac build/render, not a Pi timing. **Estimate** = arithmetic or a source's
number, with the reasoning next to it. Nothing labeled estimate should go in the README as a result.

Setup unless stated: Pi 2 (4× Cortex-A7, 900 MHz, `performance` governor), stock 6.18 kernel
(PREEMPT_VOLUNTARY, HZ=100, no isolcpus), clang 16 cross build, 48 kHz, `--rt` (SCHED_FIFO 80,
main thread on CPU 3, `mlockall`), `audio/di.wav`.

---

## 0. Answer first

**How low can total latency go on this Pi?**

| Path, model | Block and cores (offline, paced `--rt`, stock kernel) | Late blocks (measured) | Digital delay | Fixed I/O floor (estimate) | **Round trip (estimate)** |
|---|---|---|---|---|---|
| **I2S, Lite** | 16, 1 core | 0.002% | 0.67 ms | ~0.5–0.8 ms | **~1.2–1.5 ms** |
| **I2S, Full**, today's kernel | 48, 3 cores (frame split) | 0.018% | 2.0 ms | ~0.5–0.8 ms | **~2.5–2.8 ms** |
| **I2S, Full**, cores 1–3 isolated (predicted) | 24, 3 cores (frame split) | 0.08% today; needs isolation | 1.0 ms | ~0.5–0.8 ms | **~1.5–1.8 ms** |
| **Scarlett USB, Lite** | 16–32 if USB allows (unmeasured) | 0.002% | 0.67–1.33 ms | ~2–3 ms | **~2.7–4.3 ms** |
| **Scarlett USB, Full** | 48, 3 cores | 0.018% | 2.0 ms | ~2–3 ms | **~4–5 ms** |
| Before this round: Full, pipeline, 128 | 128 + 1 block | 0.001% | 8.0 ms | ~2–3 ms | ~10–11 ms |

- **"Analog-like" (≤ ~1–2 ms round trip, section 1.1) is reachable only with an I2S codec.** Lite
  gets there today (16 samples, one core, 3 late blocks in 416,400). Full gets to ~2.5 ms round trip
  today (48 samples on 3 cores) and, with isolated cores, plausibly ~1.5–1.8 ms (24 samples on 3
  cores: compute fits at 51% of the deadline; the remaining misses are measured OS events).
- **With the Scarlett, the USB path, not the model, sets the floor** (~2–3 ms, an estimate from
  published numbers; it must be measured with `latency_test.sh`). The model now fits in blocks far
  smaller than the USB floor, so after this work every further compute gain buys nothing on USB.
- Compute is no longer the limit for latency. The limits are (1) the I/O path, (2) OS jitter at
  small blocks: the 100 Hz tick (+40–150 µs per hit) and a measured **every-2-seconds 200–400 µs
  kernel event**, which matter only when the deadline is 167–500 µs and which a per-layer barrier
  amplifies, and (3) core count, if the OS keeps one core.

**What changed in this round (measured):** a **frame-split** patch (`patches/a2_frame_split.patch`)
runs each layer on 2–4 cores at once with a barrier per layer. Full at 128: 3,461 → 1,770 µs on 2
cores (same as the pipeline, **without its +1 block of latency**), 943 µs on 4 cores. Full at 24 on 3
cores: 244 µs (49% of the deadline). Output is **bit-identical** to `nhead`. At equal total delay it
has fewer late blocks than the pipeline; its weakness is that every barrier waits for the slowest
core, so it needs isolated cores to be clean below ~2 ms.

---

## 1. Latency budget

### 1.1 What "analog-like" means (sources)

- An analog pedal's delay is microseconds (analog filter group delay), i.e. effectively zero.
- Sound travels ~0.34 m per ms, so standing 1–3 m from a guitar amp is **~3–9 ms** of acoustic
  delay that players never call "latency" (physics, not a study). Each 1 ms of digital latency is
  like standing 34 cm further from the amp.
- Wessel & Wright (2002) set the commonly used target for digital instruments: **< 10 ms latency,
  < 1 ms jitter** (as cited by Jack et al. 2018 and McPherson et al. 2016).
- Jack, Mehrabi, Stockman & McPherson (Music Perception, 2018): percussionists rated 0 ms better than
  10 ± 3 ms or 20 ms on several quality measures, but **10 ms was not clearly worse than 0** in their
  data. McPherson, Jack & Moro (NIME 2016) found most common setups fail the 10 ms / 1 ms guideline and
  introduced Bela at sub-millisecond latency.
- Anecdote, not a study: a guitarist on LKML reported hearing differences between 1.3, 2.6 and 5 ms.

So: **< 10 ms is "playable", < ~3 ms is "like standing near the amp", < ~1–2 ms is
indistinguishable from an analog pedal in any study I found.** "Analog-like" below = < 2 ms round trip.

### 1.2 The model

Round trip = **ADC delay + capture transport + ALSA buffering + model compute slot + pipeline delay
+ playback transport + DAC delay**.

- **ALSA buffering (Measured design, `src/live.cpp`):** `live` pre-fills the playback buffer with
  `periods` periods of silence and starts capture and playback linked, so a captured sample leaves
  the DAC **periods × N** samples later. Compute must finish inside one period (with 2 periods). The
  model's own compute time is not added as long as it meets the deadline.
- **Pipeline delay:** +N with `a2_pipeline` (iteration 8). **0 with frame split** (this doc).
- **Model algorithmic delay:** 0. A2 is causal (no lookahead).
- **Converters (Estimate, datasheets / TI):** delta-sigma decimation and interpolation filters.
  PCM1808 ADC: **17.4/fs = 0.36 ms** at 48 kHz (TI E2E). PCM5102A DAC: **20/fs = 0.42 ms** normal
  filter, **3.5/fs = 0.07 ms** low-latency filter (datasheet; another table says 22/fs). TI's PCM1840
  ADC has a 7.7/fs (0.16 ms) low-latency filter. Analog buffer/preamp: µs.
- **I2S transport (Estimate):** DMA + FIFO of a few frames each way, ~0.02–0.1 ms total.
- **USB transport (Estimate, unmeasured on this Pi):** snd-usb-audio moves audio in URBs of 125 µs
  micro-frames (high speed) and queues URBs ahead on playback. The kernel's `lowlatency` playback mode
  (5.15+) cuts the queue, but **it is disabled for devices that use implicit feedback sync**
  (kernel patch notes); check `/proc/asound/card*/stream0` to see what the Scarlett uses. The Scarlett's
  internal converters add their own filter delay. Published end-to-end numbers: Focusrite quotes 2.74 ms
  round trip for 2nd-gen Scarletts at 96 kHz / 32 samples on macOS; the Reddit Pi 5 thread reports
  3.5 ms with a Scarlett vs 1.9 ms with I2S; an Ardour-forum user measured 16 ms for a 1st-gen 6i6
  at 128 × 2 (5.3 ms of it JACK buffering). I use **2–3 ms of fixed USB + converter overhead** at
  48 kHz as the estimate; the Pi 2's shared LAN951x hub/dwc_otg path could make it worse.

### 1.3 Budget table (2 periods, 48 kHz; ms)

Fixed floors are estimates (1.2). Digital buffering is exact arithmetic. "Model fits?" is measured
offline (sections 2–3) and still needs `live` confirmation on the real device.

| Block N | 2 × N | + pipeline (N) | I2S round trip (fixed ~0.5–0.8) | USB round trip (fixed ~2–3) | Lite fits? (measured) | Full fits? (measured) |
|---|---|---|---|---|---|---|
| 8 | 0.33 | 0.50 | **0.8–1.1** | 2.3–3.3 | 1 core 39% load, 0.015% late (paced) | no (2 cores: 77% p50, 2.4% missed burst) |
| 16 | 0.67 | 1.00 | **1.2–1.5** | 2.7–3.7 | **1 core, 0.002% late (paced)** | 4 cores: 42% p50, 0.3% late (paced) |
| 24 | 1.00 | 1.50 | 1.5–1.8 | 3.0–4.0 | yes | 3 cores: 51% p50, 0.08% late (needs isolation) |
| 32 | 1.33 | 2.00 | 1.8–2.1 | 3.3–4.3 | yes (0.002% late) | 3 cores 52% p50 (burst); 4 cores 0.15% late |
| 48 | 2.00 | 3.00 | 2.5–2.8 | 4.0–5.0 | yes | **3 cores: 49% p50, 0.018% late (paced)** |
| 64 | 2.67 | 4.00 | 3.2–3.5 | 4.7–5.7 | yes | 2 cores: 68% p50, 0.04% late; pipeline 0.004% (+64) |
| 128 | 5.33 | 8.00 | 5.8–6.1 | 7.3–8.3 | yes | 2 cores, 66% (frame split or pipeline) |

What a smaller block buys: halving the block from N to N/2 saves N/48 ms (2 periods of N/2 samples). From 128 to 32 that is 4 ms; from 32
to 16 it is 0.67 ms; from 16 to 8 it is 0.33 ms. **On USB the fixed 2–3 ms dominates below N=32;
on I2S the converters (0.5–0.8 ms) dominate below N=16.** A 3rd period (absorbs a late block) costs
+N: +0.33 ms at 16, +2.67 ms at 128.

---

## 2. Per-block fixed costs at small N (Q2)

### 2.1 Measured: cost vs block size is almost perfectly linear

`TAG=x_smallN_burst` (`nhead`, unpipelined, burst, `--rt`, 30 s per run, 2 runs per block;
`results-pi/summary_x_smallN_burst.csv`, fit with `scripts/lowlat_analysis.py fit`, rows in
`results-pi/lowlat_fit.csv`):

| | Fit t(N) (mean) | N=8 | N=16 | N=32 | N=64 | N=128 |
|---|---|---|---|---|---|---|
| **Lite** p50 (µs) | **10.7 µs + 6.56 µs × N** | 61.2 | 112.5 | 216.2 | 425.6 | 840.5 |
| Lite p50 / deadline | | 36.7% | 33.7% | 32.4% | 31.9% | 31.5% |
| Lite fixed share of mean | | 17% | 9% | 5% | 2.5% | 1.3% |
| **Full** p50 (µs) | **15.5 µs + 27.13 µs × N** | 226.5 | 443.3 | 867.3 | 1,742 | 3,461 |
| Full fixed share | | 6.7% | 3.4% | 1.8% | 0.9% | 0.4% |

- **The per-call fixed cost is only ~11 µs (Lite) / ~16 µs (Full)** ≈ 10–14 k cycles per
  `process()`: 23 layers + head, i.e. ~400–600 cycles per layer. At N=16 that is 9% (Lite) / 3.4%
  (Full) of the block. **Fixed cost is not what stops small blocks.**
- The earlier "8.5 µs/sample at 16 vs 7.8 at 128" (iteration 5) was the scalar Lite build plus paced
  wake-ups; with the NEON kernels the per-sample cost at 16 is 7.0 vs 6.6 µs at 128 (+7%).

### 2.2 Where the fixed cost goes (code inspection + `perf`)

Per layer per call (from `a2_fast.cpp` as patched):
1. `_ring_write`: a `memcpy` call (glibc dispatch ~50–100 cycles for tiny sizes) + rewind check. The
   amortized `memmove` rewind is proportional to N (every ~lookback/N blocks it moves lookback columns),
   so it is not fixed, but it is a **spike**: e.g. Lite at N=8, the D=239 layer rewinds every 150
   blocks and moves 14 KB.
2. `_layer_forward` switch + `_layer_forward_k` setup: K tap pointers (K multiplies), loop setup.
3. Per tile (every 4 frames, so it scales with N, not fixed): a `taps[K]` array built and passed, bias /
   mixin / 1×1 weights reloaded (~20 q-loads), function call.
4. Cache: Full's 46 KB of weights don't fit the 32 KB L1D with the histories, so each layer's weights
   come from L2 once per block (~700 lines/block). This is a per-block cost that grows in relative terms
   as N shrinks (Estimate: ≤ ~5–10 µs; partly hidden).
5. Head setup, rechannel loop, `memset` of `head_sum`, output copy, the container/slimmable wrapper,
   and one `clock_gettime` (Measured 223 ns) inside the timed region.

**No T=1 tail tiles at power-of-two blocks.** The Full kernel only falls back to 1-frame tiles when N
is not a multiple of 4; 8–128 are all multiples of 4, so "tile remainder" costs nothing here.

**Measured split (Pi):** `perfstat.sh` at N=8 and N=128 (`SET=perf_smalln`, tags `x_perf_n8`,
`x_perf_n128`, 10 s of audio, user-space counters, `results-pi/perf/x_perf_n*/`). Per-block extra =
(count at N=8 − count at N=128 scaled to the same samples) / blocks at N=8:

| Per extra `process()` call | Lite | Full |
|---|---|---|
| cycles | **9,300** (10.3 µs; fit says 10.7) | **18,100** (20 µs; fit says 15.5) |
| instructions | 4,900 (~200 per layer) | 4,750 |
| branch misses (~8–10 cycles each, assumed) | 106 (~4–5 per layer: loop exits) | 93 |
| L1D misses | 226 | **1,142** |
| L2 accesses | 434 | 2,289 |
| L2 misses | ~1 | ~6 |

| Hotspot share (perf record) | Lite N=128 | Lite N=8 | Full N=128 | Full N=8 |
|---|---|---|---|---|
| kernels (`a2_*_tile*`) | 80.6% | 70.6% | 92.7% | 89.4% |
| `_layer_forward` (setup, inlined dispatch) | 3.5% | **8.7%** | 0.9% | 2.1% |
| `memcpy` (ring writes) | 4.5% | **7.5%** | 3.0% | 4.3% |
| head dot product | 2.6% | 2.4% | 1.9% | 1.8% |

**Reading:** the fixed cost is ~200 instructions + ~4 branch misses per layer (setup, `memcpy` call,
loop exits), and, for Full, **~1,100 L1 misses per call that hit in L2**: the 46 KB of weights stream
from L2 once per block whatever the block size (L2 misses don't change). For Full that is ~60% of
the fixed cost (Estimate, at ~10 cycles per L2 hit). At N=8 the fixed cost is 17% of Lite's time and
~7% of Full's; at N=16 9% / 3.4%.

### 2.3 What actually hurts small blocks: OS jitter, a fixed number of µs

`scripts/lowlat_analysis.py ticks` on the per-block CSVs (`results-pi/lowlat_ticks.csv`):

| Lite, burst, `--rt` | median | blocks > 1.5× median | dominant spacing | mean excess | missed |
|---|---|---|---|---|---|
| N=8, run 1 | 61 µs | 1,376 (121/s) | **9–10 ms** | 42 µs | 17 / 180,000 |
| N=8, run 2 | 61 µs | 1,696 (148/s) | **9–10 ms** | 62 µs | 60 / 180,000 |
| N=16, run 1 | 112 µs | 98 (9/s) | — | 105 µs | 6 / 90,000 |
| N=32, run 1 | 216 µs | 31 (3/s) | — | 178 µs | 0 |

- The ~10 ms spacing is the **100 Hz kernel tick** (`CONFIG_HZ=100`, `NO_HZ_IDLE`, no `NO_HZ_FULL` in
  the stock `rpi-v7` config, Measured from `/boot/config-6.18.50+rpt-rpi-v7`). Each tick adds ~40–60
  µs to the block it lands in. At N=8 that is +25–35% of a 167 µs deadline.
- Several of the misses at N=8 sit at **exactly 2.000 s intervals** (1.313, 3.313, 5.313, … s in run
  1): periodic kernel housekeeping costing 200–400 µs.
- `dwc_otg` (USB + Ethernet) and the mailbox IRQ are delivered to **all four cores** (Measured,
  `/proc/interrupts`: equal counts per CPU), so no core is IRQ-free today.

These are fixed µs costs that a smaller kernel/block can't shrink. The fixes are the OS items in
section 5 (isolcpus + nohz_full + irqaffinity + RT kernel), not model code.

### 2.4 Fixes considered for the fixed cost (ranked, all small)

| Fix | Saves (Estimate) | Notes |
|---|---|---|
| Fuse ring writes into the kernel: the 1×1 output of layer *l* written straight into layer *l+1*'s history column; drop `layer_in` | `memcpy` is 4.5% (Lite) / 3.0% (Full) at 128 and 7.5% / 4.3% at 8 (measured share): ~3–6% | Kernel reads its residual from its own history (tap K−1 is the current input). Bit-identical. Medium effort (touches every path). |
| Hoist per-layer setup: precompute tap pointer offsets (k×D×C) at `SetMaxBufferSize` | ~100 cycles × 23 | Trivial. |
| Ring mode 1 with dirty-only mirror (`a2_ring_mirror_dirty.patch`, round 2) at small N | removes the rewind `memmove` spikes (up to ~76 KB for Lite if several layers coincide) | Constant per-block work; worth re-testing at N=8–16 where a 20–50 µs spike matters. |
| Larger `history_cols` in mode 0 (e.g. lookback + 64×N) | fewer rewinds, same spike size | Memory ×~2; doesn't fix the worst case. |

None of these moves the smallest block size by itself (fixed cost is 3–9% at N=16). Frame split
(section 3) does.

---

## 3. Parallelism without extra latency (Q3)

### 3.1 Cross-core handoff on the A7 (`src/sync_peak.cpp`)

Measured (Pi), `./build-x-tools/sync_peak --rt --cpu 3 --helpers 2,1,0`, 200,000 iterations per row,
`results-pi/x_sync_peak.csv`. Each per-iteration time includes one `clock_gettime` (223 ns, measured
in the same run); the "net" column subtracts it.

| Test | Threads (CPUs) | mean (raw) | **net** | p99 (raw) |
|---|---|---|---|---|
| Ping-pong, one-way handoff, `yield` spin | 2 (3↔2, 3↔1, 3↔0 all equal) | 247 ns | **~135 ns (~120 cycles)** | 338 ns |
| Ping-pong, `tight` spin (no hint) | 2 | 245 ns | ~135 ns | 338 ns |
| Ping-pong, `wfe`/`sev` | 2 | 297 ns | ~185 ns | 390 ns |
| Flag barrier (own counter per thread) | 2 / 3 / 4 | 417 / 514 / 617 ns | **~195 / 290 / 395 ns** | 573 / 729 / 833 ns |
| Central counter barrier (`ldrex/strex`) | 2 / 3 / 4 | 372 / 384 / 432 ns | ~150 / 160 / 210 ns | 573 / 625 / 625 ns |
| Payload 2 KiB written by each thread, read by all, 2 barriers | 2 vs 1 thread | 3,749 vs 2,707 ns | migration of 2 KiB ≈ 0.65 µs (~20 ns per 64 B line) | |

- **A cross-core handoff costs ~0.13 µs; a barrier 0.2–0.4 µs.** All four cores are equidistant
  (one cluster, shared L2). `wfe` saves power but adds ~50 ns per handoff.
- The central counter barrier scales better at 4 threads (~0.21 vs ~0.40 µs); the frame-split patch
  uses the flag barrier (simpler, no atomics RMW). Switching is a small follow-up (~5 µs/block at P=4).
- Max values of 20–250 µs are interrupts on one of the cores: **a barrier waits for the slowest core**,
  so every participating core's interruptions land on the block.

**Prediction from these numbers (made before prototyping):** per block, a P-way split costs about
t₁(N)/P + 26 barriers × (0.2–0.4 µs) + data migration (the previous layer's output for the other
threads' frames, ~N×32 B/P per layer) ≈ t₁(N)/P + 5–15 µs. The pipeline costs a whole block of
latency (N/48 ms) for the same 2× throughput. **The barrier scheme should win at every block size
≥ 8**, because its overhead is µs of compute, never latency.

### 3.2 The prototype: `patches/a2_frame_split.patch`

Applies on top of `nhead` (ring mode 0 + full_neon_tile + lite_neon_lanes + pipeline + head_neon).

- **Split frames, not channels.** Each of P threads (the calling thread + P−1 pinned workers) runs
  every layer on its own slice of the block's frames, in whole 4-frame tiles. A layer reads the
  previous layer's output at earlier frames (dilations 1–239), which may belong to another thread, so
  there is **one barrier per layer** (23) plus start, head and end: 26 per block.
- Why not split Full's 8 output channels? Every layer's 8×8 conv and 1×1 mix all channels, so a
  channel split needs **two** barriers per layer (after the conv and after the 1×1) and doesn't apply
  to Lite (3 channels). Frame split also reuses the existing kernels unchanged, so the output is
  bit-identical.
- The calling thread does all ring rewinds and advances every write position before the start
  barrier; during the block each thread only copies its own frames into the histories.
- Off by default. At model load: `NAM_A2_FSPLIT=8|3|all`, `NAM_A2_FSPLIT_THREADS=2..4`,
  `NAM_A2_FSPLIT_CPUS=2,1,0`, `NAM_A2_FSPLIT_PRIO=80`. Ignored if the pipeline is on, during prewarm,
  and for blocks that aren't a multiple of 4 (runs unsplit). Workers are created with a 256 KB stack
  and with their core and SCHED_FIFO priority set **in the pthread attributes** (a lesson from
  `sync_peak`: a thread that inherits a spinning SCHED_FIFO creator's core can never run to move
  itself). This also avoids the 8 MB-per-thread `mlockall` cost noted in `live-testing-todo.md`.

**Accuracy.** Measured (Pi): 10 s renders at block 32, 2/3/4 threads, Lite and Full vs `nhead`:
**bit-identical** (`null_test.py`, max difference 0; `results-pi/x_null_tests.csv`). Checked (Mac,
arm64): blocks 16/36/128 (split) and 30 (unsplit fallback), 2/3/4 threads: byte-identical WAVs.

### 3.3 Measured (Pi): burst, `--rt`, 30 s per run, 2 runs per row

Tags `x_fs_full_p2/p3/p4`, `x_fs_lite_p2/p4`, control `x_fs_off` (same binary, split off: Full 3,458
µs at 128 vs `nhead` 3,461; Lite 838 vs 840). `results-pi/summary_x_fs_*.csv`, fits in
`results-pi/lowlat_fit.csv`. p50 in µs (p50 / deadline); p99 = worse run; missed = both runs.

| Full | 1 core (`nhead`) | 2 cores | 3 cores | 4 cores | Deadline |
|---|---|---|---|---|---|
| N=8 | 226.5 (136%) | 127.8 (77%), p99 184, missed 2.4% | — | — | 166.7 |
| N=12 | — | — | 131.7 (53%), p99 224, 0.2% | — | 250 |
| N=16 | 443.3 (133%) | 239.7 (72%), p99 346, 1.4% | 235.1 (71%)¹ | **135.2 (41%), p99 238, 0.07%** | 333.3 |
| N=24 | — | — | **244.3 (49%), p99 351, 0.05%** | — | 500 |
| N=32 | 867.3 (130%) | 460.2 (69%), p99 584², 0.2% | **349.3 (52%), p99 465, 0.02%** | 251.0 (38%), p99 434, 0.02% | 666.7 |
| N=48 | — | — | 472.8 (47%), p99 619, 0.02% | — | 1,000 |
| N=64 | 1,742 (131%) | **888.2 (67%), p99 1,047, 0.03%** | — | 482.6 (36%), p99 689³ | 1,333 |
| N=96 | — | — | 916.6 (46%), 0 missed | — | 2,000 |
| N=128 | 3,461 (130%) | **1,770 (66%), p99 1,984, 2 / 22,500** | — | 942.8 (35%)³ | 2,667 |

¹ 16 frames = 4 tiles over 3 threads (2/1/1): no better than 2 threads. Use N = 12, 24, 48 with 3.
² One of the two N=32 runs had a ~7 s stretch of 2–3× slower blocks (p99 1,381 µs); the other was
clean (p99 584). See 3.5.
³ 4-core runs at 64/128 had single **41–51 ms** stalls (max column). Likely cause (inference, not
traced): with a SCHED_FIFO spinner on every core, kernel threads can't run anywhere until RT
throttling (`sched_rt_runtime_us=950000`, `NO_RT_RUNTIME_SHARE`, both read on the Pi) forces a
~50 ms break. **Don't use all 4 cores on the stock kernel.**

| Lite | 1 core | 2 cores | 4 cores | Deadline |
|---|---|---|---|---|
| N=8 | 61.2 (37%), missed 77/360k | **40.1 (24%), p99 62, missed 46/360k** | — | 166.7 |
| N=16 | 112.5 (34%) | 65.6 (20%), p99 104, 4/180k | 46.7 (14%), p99 85 | 333.3 |
| N=32 | 216.2 (32%) | 117.2 (18%) | 73.4 (11%) | 666.7 |
| N=64 | 425.6 (32%) | 222.8 (17%) | — | 1,333 |

**Speedups (p50):** Full 2 cores 1.77× (N=8) → 1.96× (N=128); 4 cores 3.28× (N=16) → 3.67× (N=128).
Lite 2 cores 1.53× (N=8) → 1.91× (N=64).

**Fit:** Full 2 cores t(N) = 33 µs + 13.7 µs×N (vs 15.5 + 27.1×N on one core): the split adds ~18 µs
per block (26 barriers + load imbalance per layer + data migration), within the 5–15 µs + imbalance
predicted. **It matches the pipeline's throughput (1,770 vs 1,767 µs at 128) without its block of
delay, so the pipeline is dominated at every block size measured.**

### 3.4 Paced (`--rt`), the realistic callback pattern

Measured (Pi), paced (sleep to each deadline), `--rt`, one pass of the 69 s file per run, 2 runs
per row. Tags `x_fsp_*` (frame split), `x_pipe_full_paced` and `x_lite_paced_small` (`nhead`).
"Missed" = compute over the deadline; "+wake" also counts late wake-ups. Total digital delay =
2 periods (+1 block for the pipeline).

| Config | Block | Digital delay | p50 / p99 (µs) | missed | missed + wake | rate (+wake) |
|---|---|---|---|---|---|---|
| Lite, 1 core | 8 | 0.33 ms | 64 / 89 | 30 / 832,800 | 128 | 0.015% |
| **Lite, 1 core** | **16** | **0.67 ms** | 116 / 154 | **3 / 416,400** | 7 | 0.002% |
| Lite, 1 core | 32 | 1.33 ms | 220 / 264 | 0 / 208,200 | 4 | 0.002% |
| Lite, 2 cores (split) | 8 | 0.33 ms | 43 / 65 | 143 / 832,800 | 332 | 0.04% |
| **Lite, 2 cores (split)** | **16** | **0.67 ms** | 68 / 106 | **0 / 416,400** | 10 | 0.002% |
| Full, 4 cores (split) | 16 | 0.67 ms | 141 / 246 | 380 / 416,400 | 1,312 | 0.3% |
| Full, 3 cores (split) | 24 | 1.00 ms | 254 / 385 | 157 / 277,600 | 232 | 0.08% |
| Full, 4 cores (split) | 32 | 1.33 ms | 258 / 377 | 87 / 208,200 | 311 | 0.15% |
| Full, 2 cores (split) | 32 | 1.33 ms | 470 / 1,277 | 7,466 / 208,200 | 26,348 | (bad episode, see 3.5) |
| Full, pipeline | 32 | 2.00 ms | 451 / 526 | 126 / 208,200 | 190 | 0.09% |
| **Full, 3 cores (split)** | **48** | **2.00 ms** | 485 / 629 | **13 / 138,800** | 25 | 0.018% |
| Full, 2 cores (split) | 64 | 2.67 ms | 906 / 1,077 | 25 / 104,100 | 41 | 0.04% |
| Full, pipeline | 64 | 4.00 ms | 891 / 987 | 2 / 104,100 | 4 | 0.004% |
| Full, pipeline (iteration 12, 30 min) | 128 | 8.00 ms | 1,786 / 1,922 | 7 / 675,000 | 7 | 0.001% |

**At equal total delay, frame split beats the pipeline** (2.0 ms: split-3 at 48 has 25 vs the
pipeline-at-32's 190 late blocks), and it reaches delays the pipeline can't (1.0 ms with 3 cores).
**But on the stock kernel nothing below ~2 ms is glitch-free for Full**, while Lite is essentially
clean at 16 (0.67 ms) on one core and at 16 with 2 cores.

**Why the split runs miss (Measured, `lowlat_analysis.py ticks` + miss timestamps):**
- Full split-3 at 24, run 1: of 74 misses, most sit at **x.69 s, exactly every 2.000 s** (1.873, 2.69,
  4.69, 6.69, 8.69, … s). The same 2 s event appears in the single-core Lite runs at N=8 (x.313 s).
  Something in the kernel or a driver runs every 2 s for 200–400 µs on some core; with a barrier
  per layer, **any** core's stall stalls the block. Identifying it needs `trace-cmd`/ftrace as root
  (Christine): candidates are a driver poll (Ethernet PHY, vc4/HDMI, firmware thermal) or a timer.
- Blocks > 1.5× median come every **25 ms** (split, 2–4 cores, ~48/s) instead of every 10 ms (one
  core): the cores' 100 Hz ticks are not aligned, so a split block gets hit by more ticks, each
  adding ~130–155 µs (vs ~40–60 µs on one core) because the other threads wait at the next barrier.
- 4 cores include CPU 0 (most housekeeping): max 5–10 ms stalls.

These are exactly what `isolcpus` + `nohz_full` + `irqaffinity=0` remove from cores 1–3. The
prediction (Estimate): with cores 1–3 isolated and tick-free, split-3 at 24 should approach the
single-core Lite miss rate, i.e. Full at **1.0 ms digital delay**. That is the test to run after
step 3 in section 8.

### 3.5 Caveats

- Each extra core multiplies exposure to interrupts: the block waits for the slowest participant
  (sync_peak max column; p99/p50 grows from 1.08 on one core at 128 to ~1.45 on two at 16).
  Isolating the participating cores is what makes this production-grade.
- **Unexplained: 2 threads at N=32.** In 2 of 4 runs (one burst, one paced) there were multi-second
  stretches where blocks took 2–3× longer (paced run 1: 9.3–16.9 s, ~60% of blocks over 1 ms). No
  other configuration showed it, `get_throttled` read 0x0 after the session, and no other benchmark
  was running. I can't rule out a background job on core 2 (my own SSH checks were running during
  one of them). Re-run `x_fsp_full_p2` at 32 on isolated cores before trusting or dismissing it.
- Leave **core 0** for the OS and IRQs: use 2–3 threads on cores 1–3 (`NAM_A2_FSPLIT_CPUS=2,1`).
- The workers spin between blocks (like the pipeline worker). That's 100% of 1–3 cores at SCHED_FIFO.

---

## 4. Kernel-level gains still on the table (Q4)

### 4.1 Measured: what the A7 does with loads (`src/kernel_peak.cpp`)

`./build-x-tools/kernel_peak --rt --cpu 3`, `results-pi/x_kernel_peak.csv` (kernels) and
`results-pi/x_kernel_peak2.csv` (kernels repeated + load forms). Each loop has 16 independent `vmla.f32
q` (never waiting on the loads) plus loads of one form from L1:

| Added to 16 VMLA q (65.1 cycles) | cycles per load instruction | Load alone (no VMLA) |
|---|---|---|
| `vld1.32 {d}` / `vldr d` / `{d[]}` / `{d:64}` | **+2.0** | 1.25 |
| `vld1.32 {q}`, with or without `:128` | **+3.0** | 2.25 |
| `vld1.32 {d0-d3}` (32 B) or `vldm`, with or without `:256` | **+6.0** (2 per iter) / +8.1 (1 per iter) | 4.5 |
| `ldrd` + `vmov d, r, r` | +2.8 | |
| integer `ldr`, `pld` | +1.0 | |

**NEON loads don't overlap with NEON arithmetic on the A7.** Every load costs about 1 cycle per 8
bytes plus ~1 cycle, on top of the VMLAs. Even an integer `ldr` or a `pld` costs a cycle. Alignment
hints change nothing.

### 4.2 Measured: the Full kernel in L1, three ways (K=6, T=4, same data)

| Variant | ns per tile | MAC/cycle |
|---|---|---|
| `intr` (verbatim `nhead` kernel) | 3,842 | 0.518 |
| `asm`: conv + 1×1 in inline asm, loads software-pipelined (weights 8 VMLAs ahead, inputs one pair ahead in a second d-bank, 4-register weight loads) | 3,805 | 0.523 (**+1.0%**) |
| `asm_a` (same + `:128`/`:64` hints) | 3,805 | 0.523 |
| conv only: `intr` / `asm` | 2,921 / 2,786 | 0.584 / 0.612 (**+4.8%**) |
| 64 VMLA per tap, no loads (ceiling) | 1,790 | 0.953 |

`asm` vs `intr` outputs: **bit-identical** (checked in the binary on the Pi).

What it says:
- **Software pipelining is worth ~5% on the conv and ~1% on the tile.** The loads were not stalling on
  latency; they cost issue cycles. The `objdump` of `a2_full_tile<6,4>` (`build-armhf-nhead`) shows
  clang already interleaves loads and VMLAs well; it spends 32 load instructions per tap (16 `vld1 q`
  weights + 16 `vld1 d` inputs) where the asm uses 24 (8 × 4-register + 16 d).
- **The epilogue is 24% of the tile** (3,842 − 2,921 ns ≈ 920 ns per 4 frames ≈ 830 cycles for the
  bias / mixin / LeakyReLU / head-sum / 1×1 / residual): every q-register add, mul or max also costs 4
  cycles on this core.
- In the model, Full runs at 0.48 MAC/cycle (27.1 µs/sample) vs 0.518 in L1: **cache effects cost
  ~7%** of Full's time; that is the upper bound for `pld`, which itself costs 1 cycle per line.

### 4.3 Proposals (Estimates unless noted)

| Change | Expected | Effort | Notes |
|---|---|---|---|
| Epilogue: `vmla` instead of `vmul`+`vadd` for the mixin (clang splits `vmlaq_n_f32` even with `no-slowfpvmlx`; see the `vmul.f32 q5, q2, q4` / `vadd` pairs after the tap loop) | −8 q-ops per tile ≈ −1% Full | 1 h | Bit-identical (VMLA is unfused). Use `vmlaq_lane_f32` or inline asm. |
| Epilogue: start the conv accumulators at the bias (Lite already does) | −8 q-ops per tile ≈ −1% | 1 h | Changes summation order: null-test (expect ~−130 dB). |
| 1×1 from registers: use the activations' lanes as VMLA scalars instead of storing `zs` and reloading 16 d | ~−1.5% | 1 day | Needs the activations in q0–q7; tight on registers. |
| Conv in asm (measured +4.8% on the conv in L1) | ~−3% Full | done in `kernel_peak.cpp`, needs a patch | Clang-only asm; GCC would need the same constraints. |
| `pld` for the D=101/239 layers, one per 64 B line ahead | ≤ 7% bound, likely 2–3% | 1 day | Measure with `perfstat` L1/L2 misses first. |
| Lite: the same epilogue work; Lite pays 4 lanes for 3 channels (25% waste) | ~5–10% Lite | 1–2 days | Lite is already fast enough at 8–16; low priority. |
| q vs d registers | 0 | — | Measured: equal per-lane throughput (iteration 7); d loads cost less per instruction but not per byte. |
| Weight-stationary loops at small N | 0 | — | Weights can't stay in registers (64 floats/tap > 16 q with accumulators), and per-tile reload already amortizes over 4 frames regardless of N. |

**Combined kernel-level estimate: ~5–8% for Full.** Real, but an order of magnitude less than the
frame split, and it doesn't change the smallest feasible block. Not prototyped as a model patch for
that reason (the asm conv is ready in `src/kernel_peak.cpp` if wanted).

---

## 5. System-level (Q5)

| Item | Effect on latency | Who | Notes |
|---|---|---|---|
| **3 cores for Full, core 0 for the OS** | Full at 48 today (0.018% late, paced), 24 after isolation (Estimate) | Claude (done: env vars) | Prefer 3 threads on CPUs 3,2,1. 4 threads only on an isolated/RT kernel (RT throttling stall, 3.3³). |
| **`isolcpus=1-3 nohz_full=1-3 rcu_nocbs=1-3 irqaffinity=0`** | Removes the 100 Hz tick (+40–60 µs per hit) and the 2 s housekeeping from the audio cores; frame-split tails depend on it | Christine (cmdline + reboot) | `nohz_full` needs **`CONFIG_NO_HZ_FULL=y`**, not set in the stock `rpi-v7` kernel (measured from `/boot/config-…-rpi-v7`): **add it to the RT kernel build.** |
| `sched_rt_runtime_us = -1` (or keep core 0 free) | Removes the 50 ms RT-throttling stall when every core spins at FIFO | Christine (sudo sysctl) | Only needed for 4 threads. |
| PREEMPT_RT kernel (`rpi-7.1.y`+) | Tail; wake-ups already 16–24 µs p99 with `--rt` | Christine | Unchanged from the roadmap; matters most at N ≤ 16. Watch the USB IRQ thread priority and the dwc_otg FIQ. |
| USB IRQ placement | `dwc_otg` lands on all four cores today (measured) | Christine | With the Scarlett at 8,000 micro-frames/s this becomes the biggest jitter source; `irqaffinity` / `/proc/irq/89/smp_affinity` = core 0. |
| Overclock 900 → 1,000 MHz | Compute −10% (Estimate, ≤ 1000/900; DRAM doesn't scale) | Christine (`arm_freq=1000`, maybe `over_voltage`) | Doesn't change which block sizes fit after frame split; only worth it for margin. |
| `mlockall` | Already done by `--rt`; frame-split workers now use 256 KB stacks | done | |
| Huge pages | ≈0 (Estimate): dTLB misses 152 k per 20 s run for Full (iteration 6 profile) | — | `CONFIG_ARM_LPAE` off; not worth it. |
| Cache-way partitioning | N/A: the A7's L2 has no partitioning; all four cores share it | — | |
| Lite on one core, I/O on another | Possible: Lite needs 1 core at N≥16 | — | With `live`, ALSA work happens in the audio thread (blocking read/write), so a separate I/O core only helps if IRQs are moved there. |

---

## 6. Model-side options (Q6)

Measured (Mac renders, `scripts/model_surgery.py` + `null_test.py`, 20 s, block 128,
`results-pi/x_model_surgery_null.csv`). Residual = error relative to the unmodified Full, without any
retraining:

| Variant (Full, no retraining) | Arithmetic saved | Residual vs Full |
|---|---|---|
| Lite (the file's own small model) vs Full | 85% | **−14.3 dB** (reference point) |
| every conv tap matrix rank 6 (SVD) | none (rank ≥ 4 costs ≥ 64 MACs) | −29.9 dB |
| rank 4 | none | −19.5 dB |
| rank 3 | 25% of conv | −15.9 dB |
| rank 2 | 50% of conv | −11.1 dB |
| 1×1 matrices rank 4 | none | −10.4 dB |
| prune the weakest channel (zeroed, 8 → 7) | ~23% | −6.0 dB |
| prune 2 channels | ~44% | −5.3 dB |

- **Post-hoc compression doesn't work for this model**: anything that saves arithmetic is as far from
  Full as Lite already is, or further. The −100 dB bar used for code changes is out of reach by 70+ dB.
- So "cheaper Full" means **retraining** (a pruned or low-rank architecture trained with the NAM
  trainer, or TONE3000's own smaller presets). That needs a training pipeline, which isn't measurable
  here. The model already ships its answer to "Nano": **Lite**, which now runs at 8 samples on 2
  cores or 16 on one.
- Recommendation: pick per use. Lite for the lowest latency on 1–2 cores; Full with frame split at
  16–24 samples when the extra 2 cores are free.

---

## 7. Everything measured this round

| Tag / file | What | Where |
|---|---|---|
| `x_smallN_burst` | `nhead`, Lite + Full unpipelined, N = 8–128, burst `--rt`, 30 s × 2 | `results-pi/summary_x_smallN_burst.csv`, `blocks/x_smallN_burst/` |
| `x_sync_peak.csv` | handoff, barriers, payload migration | `results-pi/x_sync_peak.csv` |
| `x_kernel_peak.csv`, `x_kernel_peak2.csv` | Full kernel variants in L1; load-form costs | `results-pi/` |
| `x_fs_full_p2/p3/p4`, `x_fs_lite_p2/p4`, `x_fs_off` | frame split, burst `--rt` | `results-pi/summary_x_fs_*.csv`, `blocks/x_fs_*` |
| `x_fsp_*`, `x_pipe_full_paced`, `x_lite_paced_small` | paced `--rt` | `results-pi/summary_x_fsp_*.csv` etc. |
| `x_null_tests.csv` | Pi null tests, frame split (2/3/4 threads) vs `nhead`, Lite + Full: all bit-identical | `results-pi/x_null_tests.csv` |
| `x_perf_n8`, `x_perf_n128` | perf counters at N=8 vs 128 | `results-pi/perf/x_perf_n*/` |
| `x_model_surgery_null.csv` | model-side residuals (Mac) | `results-pi/` |
| `lowlat_fit.csv`, `lowlat_ticks.csv` | analysis outputs | `results-pi/` |

New files: `src/sync_peak.cpp`, `src/kernel_peak.cpp` (ARMv7 CMake targets added at the end of
`CMakeLists.txt`), `patches/a2_frame_split.patch`, `scripts/lowlat_runs.sh` (all Pi runs above, by
`SET`), `scripts/lowlat_analysis.py`, `scripts/model_surgery.py` (needs numpy; on this Mac it runs
under `arch -x86_64 python3`). Builds: `build-armhf-x-tools` (microbenchmarks), `build-armhf-x-fsplit`
(`nhead` + frame split); on the Pi `build-x-tools/`, `build-x-fsplit/`.

**Exact diff proposed for `src/common.h` + `bench.cpp` + `live.cpp` (not applied):** add a
`ConfigureA2FrameSplit(spec, threads, cpus, prio, buildFlags)` next to `ConfigureA2Pipeline` that
refuses unless `buildFlags` contains `a2_frame_split`, and sets `NAM_A2_FSPLIT`,
`NAM_A2_FSPLIT_THREADS`, `NAM_A2_FSPLIT_CPUS`, `NAM_A2_FSPLIT_PRIO` (= `--rt-prio` under `--rt`, else 0).
In both `bench` and `live`: flags `--fsplit C` (8, 3, all), `--fsplit-threads P` (default 2),
`--fsplit-cpus 2,1` (default: the `P−1` cores below `--cpu`), called before `nam::get_dsp`; and
`modelName += "+fs" + P` when it applies (mirroring `PipelineApplies`). Until then the runs above set
the environment variables directly (`scripts/lowlat_runs.sh`).

---

## 8. Ranked next steps

| # | Step | Gain | Effort | Risk | Who |
|---|---|---|---|---|---|
| 1 | **Adopt frame split** for Full (3 threads on cores 3/2/1) at N=48 now, N=24 after step 3; Lite stays on 1 core at 16; add the `--fsplit` flags (section 7 diff); 30-min sustained paced run | Full digital delay 8 ms (pipeline at 128) → **2.0 ms** at 0.018% late blocks today (**Measured**), 1.0 ms once isolated (**Estimate**) | 0.5 day | Low (bit-identical; off by default) | Claude |
| 2 | **Measure the I/O floor**: `live --bypass` + `latency_test.sh` on the Scarlett at 32/64/128 × 2/3; read `stream0` (implicit feedback? HS?) | Turns the 2–3 ms estimate into a number; decides whether I2S is needed | 1 h with hardware | Low | Christine (cable) + Claude (runs) |
| 3 | **Isolate cores 1–3** (`isolcpus`, `irqaffinity=0`, `rcu_nocbs`), and in the RT kernel build enable **`CONFIG_NO_HZ_FULL`** + `nohz_full=1-3`; find the 2 s event with ftrace | Removes the 100 Hz tick and the 2 s stall from the barrier path: Full 48 → 24 (2.0 → 1.0 ms digital) (**Estimate**; rerun `SET=fsplit_paced`) | reboot / kernel build | Medium (USB IRQ behavior) | Christine, then Claude reruns |
| 4 | **I2S codec** (PCM1808/PCM5102A class, or a codec HAT with a low-latency filter) + guitar buffer | Round trip ~1.2–1.8 ms for Full, ~0.9–1.2 ms for Lite (**Estimate**), i.e. analog-like | hardware + 1 day | Medium (analog front end) | Christine |
| 5 | Kernel polish: mixin `vmla`, bias-first accumulators, asm conv, then `pld` if `perfstat` shows misses | ~5–8% Full (**Estimate**; asm conv +4.8% measured in L1) | 2–3 days | Low–medium (clang-specific asm) | Claude |
| 6 | Central (ldrex) barrier for 3–4 threads; fused ring writes | ~5 µs/block + ~2–4% (**Estimate**) | 1 day | Low | Claude |
| 7 | Overclock to 1 GHz | ≤ 10% compute (**Estimate**) | 30 min + 30-min stability run | Medium | Christine |
| — | Post-hoc model compression | Not viable: ≥ −16 dB error for any arithmetic saving (**Measured**) | — | — | — |

**Cost of each step toward the floor (estimates except where marked):** Full before this round: 8 ms
digital + I/O. Step 1: 2.0 ms digital (measured offline). Step 3: 1.0 ms digital for Full (Lite is
already at 0.67 ms, measured). Step 2 tells whether USB adds ~2–3 ms; if so, step 4 (I2S) is the only
way under ~2 ms round trip: Lite ~1.2–1.5 ms, Full ~1.5–1.8 ms. Steps 5–7 only add margin.

---

## Sources

- Wessel & Wright 2002 guideline (< 10 ms latency, < 1 ms jitter), as cited in Jack et al., "Action-sound
  Latency and the Perceived Quality of Digital Musical Instruments", Music Perception 2018:
  https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/44614/Jack%20Action-sound%20Latency%202018%20Accepted.pdf?sequence=1
- McPherson, Jack & Moro, "Action-Sound Latency: Are Our Tools Fast Enough?", NIME 2016:
  https://nime.org/proc/nime2016_mcpherson/
- Jack, Stockman & McPherson, Audio Mostly 2016 (latency and performer interaction):
  https://instrumentslab.org/data/andrew/jack_am2016.pdf
- LKML thread "Human timing perception (was: RT patch)", 2005 (guitarist anecdote):
  https://lkml.rescloud.iu.edu/0505.3/1919.html
- TI PCM5102A datasheet (SLAS859C), filter group delay 20/fs normal, 3.5/fs low latency:
  https://www.ti.com/document-viewer/PCM5102A/datasheet/specifications ; the 20 vs 22/fs discrepancy:
  https://e2e.ti.com/support/audio/f/audio-forum/694045/pcm5102a-group-delay
- PCM1808 latency 17.4/fs and PCM1840 low-latency 7.7/fs (TI E2E):
  https://e2e.ti.com/support/audio-group/audio/f/audio-forum/917072/pcm1808-conversion-delay
- snd-usb-audio low-latency playback, and its disabling for implicit-feedback devices (Linux 5.15/5.16):
  https://lkml.rescloud.iu.edu/2112.0/05834.html , https://lkml.iu.edu/hypermail/linux/kernel/2112.0/05646.html ,
  https://www.phoronix.com/news/Linux-5.16-USB-Audio-LL
- Focusrite 2nd-gen Scarlett round trip 2.74 ms (96 kHz, 32 samples, macOS):
  https://www.ravepubs.com/focusrite-scarlett-second-generation
- Ardour forum, 1st-gen Scarlett 6i6 measured round trip on Linux:
  https://discourse.ardour.org/t/typical-latency-with-usb-interfaces/104145
- r/NAM_NeuralAmpModeler, "1.9ms round-trip latency using a Raspberry Pi 5" (3.5 ms with a Scarlett,
  as summarized in `optimization-roadmap.md`; I could not re-fetch Reddit for this doc):
  https://www.reddit.com/r/NAM_NeuralAmpModeler/comments/1t7w4ho/19ms_roundtrip_latency_using_a_raspberry_pi_5/
