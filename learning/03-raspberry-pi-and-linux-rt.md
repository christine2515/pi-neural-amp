# 03: The Raspberry Pi 2 and real-time Linux

## The hardware

| | Raspberry Pi 2 Model B |
|---|---|
| CPU | 4× ARM Cortex-A7 @ 900 MHz, ARMv7, **32-bit** ("armhf" = hard-float ABI) |
| Caches | 32 KB L1 data per core; **512 KB L2 shared** by all four cores |
| SIMD | NEON (single precision only; flush-to-zero, not fully IEEE-754) |
| RAM | 1 GB |
| Network/USB | USB ports **and** Ethernet sit behind one LAN951x hub chip, on one USB 2.0 link |

Our OS: Raspbian 13 (trixie) 32-bit, kernel 6.18, GCC 14.2. Check the core with
`grep -m1 "CPU part" /proc/cpuinfo` (0xc07 = A7, 0xd03 = A53 on the v1.2 board).
Context vs a Pi 5: [../docs/hardware-comparison.md](../docs/hardware-comparison.md). The Geekbench 5
numbers there are single public uploads; open the links to verify them.

## Headless setup lessons

- No Wi-Fi on a Pi 2: SSH over Ethernet (`ssh pi`, an alias that forces IPv4; IPv6 gave
  "Connection reset" once).
- Green LED = SD card activity; red = power. Flashing then dark means it booted fine.
- **The SD slot is push-to-eject.** The card popped out mid-session and SSH failed with
  `kex_exchange_identification`. Linux had kept running from RAM until it needed the card.

## CPU frequency governor

- `ondemand` lowers the clock when the CPU isn't fully busy. `performance` pins 900 MHz.
- **Measured:** the `ondemand` run had Lite deadline misses (max **3.4 ms**). After switching to
  `performance` (and with the sampler off), Lite had 0 misses with max 1.7 ms. Controls showed the
  governor caused the >2.67 ms excursions.
- Setting it needs sudo: `./scripts/set_governor.sh performance` (resets at reboot).

## Real-time scheduling toolkit

| Tool | What it does | Project result |
|---|---|---|
| **SCHED_FIFO** (priority 80) | The thread preempts normal tasks | Wake-up p99 73–80 → **16–24 µs** (~4×) |
| **CPU pinning** (`sched_setaffinity`) | Keep the thread on one core (core 3) | Fewer migrations, predictable caches |
| **`mlockall`** | Lock pages in RAM: no page faults in the loop | Part of `--rt` |
| **rtprio/memlock limits** | `@audio - rtprio 95` in limits.d, so `--rt` works without sudo | |
| **isolcpus + irqaffinity** | Keep the OS and interrupts off the audio cores | Not run yet (sudo + reboot); the next lever for Full's tail |
| **PREEMPT_RT** | Kernel that can preempt almost anything | Not run yet |

- **RT throttling:** by default `sched_rt_runtime_us = 950000`, so RT tasks get at most 95% of each
  second. A SCHED_FIFO loop that **never sleeps** (burst mode) would stall. That's why `--rt` runs use
  paced mode.
- **PREEMPT_RT on 32-bit ARM** only reached mainline in **Linux 7.1** (Phoronix), and the Pi kernel
  enables it from its `rpi-7.1.y` branch. On the Pi 5 (arm64) it's been mainline since 6.12. An older
  report found an RT kernel on the Pi 2's chip wouldn't boot until the USB driver's FIQ was turned off;
  whether that still applies is unknown.
- `--rt` fixes **wake-up** latency, not compute time (p50 within 1%).

## ALSA basics

- `hw:` devices talk to the hardware directly (`arecord -l` lists cards; Scarlett is usually `hw:USB`).
- **Period** = the chunk ALSA hands over; **buffer** = periods × period size. Linked capture and
  playback streams start together.
- **RW** (`snd_pcm_readi`/`writei`) copies between the ALSA buffer and yours. **mmap** writes into the
  buffer directly. The estimated saving is <1% of a 2.67 ms deadline for a direct `hw:` client
  (planned as a test).
- **xrun** = buffer ran empty (playback) or overflowed (capture). `live` counts them and restarts.

## USB audio vs I2S

- From the Reddit Pi 5 thread (claims, not our measurements): 3.5 ms round trip with a Scarlett vs 1.9 ms
  (later 1.1 ms) with I2S converters; USB packetization sets a floor around 16–32 samples.
- On the Pi 2, USB shares the hub with Ethernet, so SSH traffic can affect audio timing.
- I2S (DAC/ADC on the GPIO header) skips USB entirely, but a guitar needs a Hi-Z buffer and shielding.
- **Planned:** measure the Scarlett floor first (`live --bypass`, `latency_test.sh`), then decide on I2S.

## Key takeaways

> - The Pi 2 has 4 slow in-order cores sharing a 512 KB L2, and USB shares a chip with Ethernet.
> - Use the `performance` governor for benchmarks; `ondemand` caused deadline misses.
> - SCHED_FIFO + pinning + mlockall fix wake-ups; isolcpus/PREEMPT_RT target what's left.
> - RT throttling: a FIFO thread that never sleeps will stall. Use paced mode with `--rt`.

## Check yourself

1. Why did we run `--rt` only in paced mode?
2. What does `mlockall` prevent, and why does that matter in an audio loop?
3. Why could an Ethernet transfer cause audio glitches on a Pi 2 specifically?
4. What would `isolcpus=3 irqaffinity=0-2` change?
