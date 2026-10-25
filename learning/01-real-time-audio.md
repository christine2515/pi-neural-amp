# 01: Real-time audio fundamentals

## Samples, blocks, deadlines

- **Sample rate:** 48,000 samples per second, so one sample lasts **20.8 µs**.
- **Block** (also called a buffer or period): the chunk of samples processed at once.
- **Deadline:** a block must finish before the next one arrives: deadline = block size ÷ sample rate.

| Block | Deadline (and the latency it adds) |
|---|---|
| 16 | 0.33 ms |
| 32 | 0.67 ms |
| 64 | 1.33 ms |
| 128 | 2.67 ms |
| 256 | 5.33 ms |
| 512 | 10.67 ms |

**The trade-off:** smaller blocks mean lower latency, which feels better to play, but less time per
block and more fixed per-call overhead. Guitarists start to feel latency at roughly 10 ms.

## Latency vs throughput: why "too slow" means dropouts

The project's first Pi number: the Full model at block 128 took **6.66 ms** against a **2.67 ms**
deadline, so **load = 250%**. The Pi needed ~3 minutes to process the 69 s test file.

- **Offline**, that's fine: the output is correct, just late.
- **Live**, latency is fixed by the buffer size, not by compute time. The audio device takes a block
  every 2.67 ms and needs one back every 2.67 ms. If the model is slower, the output buffer runs dry
  and the driver plays silence or garbage. That's an **xrun** (underrun), heard as clicks.
- A system that "just waited" would fall further behind forever (1.5 s behind after 1 s of playing). No
  real-time audio system does that.

**Per-sample view:** Full cost 6,664 µs ÷ 128 ≈ **52 µs per sample** vs a **20.8 µs** budget. Bigger
blocks raise both the deadline and the work by the same factor, so block size can't fix an average
overload.

## Do bigger buffers help?

- **At 250% load: no.** The model produces only 40% of the audio needed, so ~60% of the output is
  missing whatever the buffer size. A buffer of size B just delays the first dropout, to about B ÷ 0.6
  (e.g. 5.3 ms → first gap after ~9 ms).
- **Where buffers help:** average load **under** 100% with occasional spikes. With 2 periods, one
  spike can empty the buffer; a 3rd period gives a full block of slack to absorb it.
- **Rule:** average load > 100% → make the model faster. Average < 100% with spikes → buffers and
  scheduling (`--rt`, isolated cores).

## Periods and the ALSA buffer

- ALSA (Linux audio) uses a ring buffer of `periods × period_size` frames. `live` pre-fills playback
  with silence, so I/O latency = period × periods.
- 128 × 2 = 256 frames = 5.33 ms of buffering, plus converter and USB latency.
- **Round-trip latency** = input to output through the whole chain. `live --latency-test` measures it
  by playing a click into a loopback cable and counting samples until it comes back, which is exact to
  1/48000 s.

## Burst vs paced measurement

| Mode | What it does | What it hides or shows |
|---|---|---|
| **burst** | Blocks back to back | Warm caches, CPU always busy: the best case for compute |
| **paced** (`--paced`) | Sleep until each block's deadline, like an audio callback | Shows **wake-up latency** and cold-cache effects |

On the Mac, paced runs were several times slower per block than burst for the same model. On the Pi,
paced vs burst p50 for Lite at 128 was nearly identical (998 vs 991 µs). The difference shows up in the
tail.

**Wake-up latency** = how late the thread wakes after its sleep. A block "misses with wake" when
wake lateness + compute > deadline. That's what real-time scheduling targets: `--rt` cut wake p99 from
73–80 µs to 16–24 µs.

## Measured "smallest glitch-free block"

- **Lite:** 32 samples (0.67 ms), paced, 0 misses in 208,200 blocks. At 16 it failed (72 misses
  normal, 4 with `--rt`).
- **Full (two-core pipeline):** clean at 256; at 128, 7 marginal overruns (≤35 µs over) in 30 minutes.

## Key takeaways

> - Deadline = block ÷ 48 kHz; one sample = 20.8 µs.
> - Too slow means dropouts, not delay. Latency is set by buffers.
> - Buffers absorb spikes; they can't fix an average overload.
> - Measure paced (realistic) as well as burst (best case), and track wake-up latency.

## Check yourself

1. Full ran at 250% load. What fraction of output audio is missing in steady state, and why can't a 1 s buffer save it?
2. Why does a 3rd ALSA period help a model at 70% average load with rare 130% spikes?
3. What's the difference between the 2.67 ms "deadline" and the measured round-trip latency?
4. Why can burst-mode numbers be too optimistic?
