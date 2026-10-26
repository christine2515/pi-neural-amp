# State of the union: how the audio runs right now

> **2026-10-25.** The current live setup end to end, with the `nhead2` build. Every
> number comes from [experiment-report.md](../experiment-report.md) ("Live" sections),
> [low-latency-plan.md](../low-latency-plan.md) or a CSV in [`results-pi/`](../../results-pi/), with the label
> or file next to it. **Estimate** marks anything not measured. The full list of changes behind this
> build: [optimizations-summary.md](optimizations-summary.md).

## 1. Signal chain

```
Squier Strat (single coils)
  → Scarlett Solo 3rd gen, input 2, INST on          (ALSA capture channel 1; input 1 empty, −91 dBFS floor)
  → USB 2.0 high speed (480M), synchronous endpoints, 125 µs packets, S32_LE, 48 kHz, 2 in / 2 out
  → ALSA hw:USB, linked capture + playback, RW interleaved access, period × periods (e.g. 64 × 2)
  → live: S32 → float, take channel 1 → noise gate → model->process() → float → S32, same sample to both outputs
  → Scarlett outputs (RCA line outs / ¼" headphone jack)
```

- **ALSA.** Blocking `snd_pcm_readi` / `snd_pcm_writei` of one period each. Playback is pre-filled with
  `periods` periods of silence, so I/O delay is fixed at period × periods. If a read or write comes back
  short (an xrun), `live` counts it and restarts both streams (`src/live.cpp`).
- **Timed region.** Only `model->process()` is timed (`proc_*_us` in `live_summary.csv`). Format
  conversion and the gate run outside it, in the same thread.
- **Device facts** (experiment report, Live setup): guitar peaks −6.8 to −2.7 dBFS, 0 clipped samples.
  No headphones yet (needs a ¼" adapter), so the live runs measure timing; listening is still to do.

## 2. Threads and cores (`--rt --cpu 3 --pipeline 8`)

| Core | What runs there | Settings |
|---|---|---|
| **CPU 3** | Audio thread: ALSA read → gate → stage A of the model → wait for the worker → write | SCHED_FIFO 80, pinned, `mlockall(MCL_CURRENT \| MCL_FUTURE)`, 256 KB of stack pre-faulted (`src/common.h` `ApplyRealtime`) |
| **CPU 2** | Pipeline worker (Full only): stage B of the previous block | Same SCHED_FIFO 80, pinned at model load, **256 KB stack** (`a2_pipeline_stack`), spin-waits with `yield` between blocks (never sleeps) |
| CPU 0–1 | OS, SSH, the fork-free thermal logger when used | normal scheduling |
| all four | Interrupts: the USB controller (`dwc_otg`) lands on all four cores (measured, low-latency-plan §5). The Pi 2's Ethernet sits behind the same USB chip | **not isolated**: no `isolcpus`, no IRQ affinity, stock `PREEMPT_VOLUNTARY` kernel 6.18, 100 Hz tick |

Lite runs without the pipeline: CPU 3 only, CPU 2 idle.

## 3. Inside `process()` (build `nhead2`)

Build: clang 16, `-DBENCH_A2_RING_MODE=0`, patches `a2_full_neon_tile` + `a2_lite_neon_lanes` +
`a2_pipeline` + `a2_head_neon` + `a2_pipeline_stack` (`build_flags` in `live10_full_pipe_64x2`).

| Piece | Full (`--slim 1`, 8 channels, 11,776 MAC/sample) | Lite (`--slim 0.25`, 3 channels, 1,731 MAC/sample) |
|---|---|---|
| Layer kernel | Register-blocked NEON tile: 8 outputs × 4 frames, all taps + bias + LeakyReLU + skip sum + 1×1 fused (`a2_full_tile<6,4>`, `<15,4>`) | NEON lanes kernel: 3 channels in lanes, 4-frame tiles (`a2_lite_tile4`) |
| History buffers | Ring mode 0: linear buffer per layer, `memmove` rewind only when it fills | same |
| Output head | One NEON dot product of 16 taps × 8 channels per sample | 16 × 3 |
| Cores | Two-stage pipeline: layers 0–12 on CPU 3, layers 13–22 + head on CPU 2, split at 13 | one core |
| Extra delay | +1 block (the pipeline) | none |

One period through the pipeline (period n arrives; the worker is still finishing period n−1):

```
          CPU 3 (audio thread)                              CPU 2 (worker)
 t0  readi(period n) → gate
     stage A(n): rechannel + layers 0–12   ║   stage B(n−1): layers 13–22 + head
     signal "A(n) ready"; spin until B(n−1) done ◄──────── signal "B(n−1) done"
     writei(output of period n−1)          ║   picks up A(n) → stage B(n) …
 t1  readi(period n+1) …
```

Handoff is two atomics and two alternating buffer sets: no locks, no syscalls, no allocation. The
output is the unpipelined output shifted by exactly one block (bit-identical, iteration 8).

## 4. Recommended commands and what they measure

Both on the Pi from `~/pi-neural-amp`. `--duration` defaults to 30 s; pass a longer one to play.

**Lite (lowest latency):**

    ./build/live --device hw:USB --in-channel 1 --model models/a2_fender_super_reverb.nam --slim 0.25 \
      --rt --cpu 3 --period 32 --periods 2 --gate -50 --duration 600

**Full (best sound):**

    ./build/live --device hw:USB --in-channel 1 --model models/a2_fender_super_reverb.nam --slim 1 \
      --pipeline 8 --rt --cpu 3 --period 64 --periods 2 --gate -50 --duration 600

| Measured | Lite, 32 × 2 | Full pipelined, 64 × 2 |
|---|---|---|
| Deadline (one period) | 667 µs | 1,333 µs |
| Model time p50 / p99 / max | 236 / 303 / 435 µs (`small_live`) | 967 / 1,084 / 1,709 µs (`live10_full_pipe_64x2`) |
| p50 / p99 as % of deadline | 35% / 45% | 73% / 81% |
| Periods over the deadline | 0 | 43 of 450,000 (0.01%), mostly by < 100 µs |
| Underruns | **0 / 45,000** (30 s) and 0 / 30,000 (20 s, `lsw_lite`) | **1 / 450,000** (10 min) |
| In-device buffering | 2 × 32 = **1.33 ms** | 2 × 64 + 64 pipeline = **4.0 ms** |
| CPU (`cpu_pct`, 100% = one core) | 44% | 178% (audio thread + spinning worker) |
| Memory, whole process (PSS) | **5.6 MB** (5,583 kB, `lsw_lite` 32 × 2) | **5.8 MB** (5,842 kB, `memfix_build-nhead2`, 15 s) |
| Temperature | not logged for Lite | peak 46.5 °C over 10 min, no throttling during the run (`thermal_live10.csv`) |
| Period arrival jitter p99 | 169 µs (~25% of the period) | 349 µs (~26%) |

**Noise gate** (`--gate -50`: opens at −50 dBFS, closes at −54, 80 ms hold, 1 ms attack, 100 ms
release, no look-ahead, so **no added latency**). Measured on Full pipelined 64 × 2, 30 s takes
(`gate_compare.csv`): amp output in the 10 quietest windows **−58.3 → −88.5 dBFS (−30 dB)**; playing
passages 0.0 dB change (offline, same recording). The gated take had 2 underruns in 30 s
(`rec_full_gate50`). The gate was not run with Lite or in the 10-minute run.

**Memory caveat.** `live` preallocates per-period timing arrays (16 bytes per period), so PSS grows with
`--duration`: the 10-minute Full run showed 12.8 MB. An hour at 32-sample periods would add about
5.4 M × 16 B ≈ 86 MB (**estimate**, arithmetic). The model's footprint is the short-run number above.

## 5. Not in the current config, and why

| Option | Why not (measured unless marked) |
|---|---|
| ALSA mmap (`--mmap`) | No gain for Lite; pipelined Full at 2 periods fails (688 / 15,000 underruns at 64; every period at 128). 3 periods is clean (`lsw_*_mmap`) |
| Frame split (`a2_frame_split`) | **Now measured live** (build `nfs`, `live --fsplit 8` (3 threads on cores 3/2/1; same as env `NAM_A2_FSPLIT=8 …`)): Full at **48 × 2 = 2.0 ms**, **1 underrun in 600,000 periods over 10 min** (`live10_full_fsplit3_48x2`), p99 769 µs / 1,000 deadline, 3 cores (CPU 267%), 49 °C. Replaces the pipeline as the lowest-latency Full setting. Command: `./build-nfs/live --device hw:USB --in-channel 1 --model models/a2_fender_super_reverb.nam --slim 1 --fsplit 8 --period 48 --periods 2 --rt --cpu 3 --gate -50` |
| `isolcpus` / `nohz_full` / IRQ affinity | Needs sudo + reboot (🧑); `nohz_full` also needs a kernel rebuild (`CONFIG_NO_HZ_FULL` off in stock `rpi-v7`) |
| PREEMPT_RT | Needs a kernel build (🧑). `--rt` already gives 16–24 µs wake-up p99 (`rt_comparison.csv`) |
| GPU (VideoCore IV) | No compute path with the `vc4-kms-v3d` overlay; `/dev/vcio` is root-only (beyond-fpu §5) |
| int16 fixed point | −65 to −75 dB vs float (`quant_sim.csv`), fails the −100 dB bar |
| 16-sample periods | USB floor at 2 periods: bypass 16 × 2 underran on half the periods; Lite 16 × 4 = 1 / 60,000 is no shorter than 32 × 2 |

## 6. Known limitations

- **Round-trip latency is unmeasured.** 1.33 / 4.0 ms is buffering inside the Pi; the Solo's converters
  and USB add to it. Needs an RCA → ¼" loopback cable; `scripts/latency_test.sh` is ready. The USB floor
  is ~2–3 ms (**estimate** from published numbers, low-latency-plan §0).
- **Rare overruns at 64 for Full.** 43 periods in 10 minutes took longer than one period; the second
  period absorbed all but one. Earlier 30 s takes at 64 × 2 had 0, 1 and 2 underruns. Likely sources:
  interrupts and kernel housekeeping on cores 2–3 (an every-2 s 200–400 µs event and the 100 Hz tick
  were measured offline; not traced live).
- **Hum from single-coil pickups.** The chain is clean (−126 dBFS with guitar volume at 0); the pickups
  bring 60 Hz hum, up to −69 dBFS and ~20 dB position-dependent (`hum_check.csv`). The gate hides it in
  silence; it is still there under playing.
- **Under-voltage at plug-in.** `get_throttled` = `0x50000` (under-voltage and throttling occurred since
  boot) when the Solo was connected; no new events during the 10-minute run. Use a stronger supply or a
  powered hub.
- **The worker spins.** Full keeps CPU 2 at 100% even in silence.
- **Live model time is ~6% above `bench`** (Full 1,875 vs 1,767 µs at 128): likely sample conversion and
  USB interrupt work on the audio core (inference).

## 7. Two settings to pick from

| Goal | Lite | Full | Measured support |
|---|---|---|---|
| **Lower latency** | `--period 32 --periods 2` → 1.33 ms (or 24 × 3 → 1.5 ms, even 0.5 ms wake-ups) | **frame split, 3 cores: `--period 48 --periods 2` → 2.0 ms** (no pipeline block) | Lite 0 / 45,000; Full 1 / 600,000 in 10 min (`live10_full_fsplit3_48x2`). Going lower needs isolated cores (docs/sudo-session.md) |
| **Maximum robustness** | `--period 64 --periods 2` → 2.67 ms (or 32 × 3 → 2.0 ms) | `--period 64 --periods 3 --pipeline 8` → 5.33 ms (4.0 ms ALSA + 1.33 ms pipeline) | Lite 64 × 2: 0 / 22,500 and 0 / 15,000; 32 × 3: 0 / 30,000. Full 64 × 3: 0 / 15,000 (`lsw_full_pipe`) and 0 / 11,250 (`memfix_build-nhead2`). Full 128 × 2/3 also 0 (8 ms+), and 30 min paced offline at 128 had 7 marginal overruns, 44.4 °C peak (`sustained_full_pipe3`) |

The robust settings have only 15–30 s runs behind them; a 30-minute live run of either is still open
(live-testing-todo §7).
