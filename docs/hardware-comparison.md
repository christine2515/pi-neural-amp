# Hardware context: Raspberry Pi 2 vs Raspberry Pi 5

For framing in the README. These are **published reference numbers, not measurements from
this project**. Every number here has a source link; none came from `results/`.

## Specs

| | Raspberry Pi 2 Model B (v1.1) | Raspberry Pi 5 |
|---|---|---|
| Released | February 2015 | October 2023 |
| SoC | Broadcom BCM2836 | Broadcom BCM2712 |
| CPU | 4× Cortex-A7 @ 900 MHz (ARMv7, 32-bit) | 4× Cortex-A76 @ 2.4 GHz (ARMv8.2, 64-bit) |
| SIMD | NEON (single precision only) | NEON / ASIMD (single + double precision) |
| RAM | 1 GB LPDDR2 | 1–16 GB LPDDR4X |
| Networking | 100 Mb Ethernet (shares the USB chip) | Gigabit Ethernet, Wi-Fi |

## Geekbench

There is no same-version, well-sourced Geekbench result for both boards. What exists:

| Benchmark | Pi 2 Model B | Pi 5 | Source quality |
|---|---|---|---|
| Geekbench 6.2, single-core | no result found | **764 ± 6** | Raspberry Pi official, avg of 100+ runs (4 KB pages) [1] |
| Geekbench 6.2, multi-core | no result found | **1,604 ± 22** | Raspberry Pi official, avg of 100+ runs (4 KB pages) [1] |
| Geekbench 5, single-core | 52 ⚠️ | 585 ⚠️ | one user upload each [2][3] |
| Geekbench 5, multi-core | 99 ⚠️ | not found | one user upload [2] |

⚠️ = taken from search-engine summaries of Geekbench browser pages, which block automated
reads. **Open links [2] and [3] yourself and confirm the numbers before using them.**

How to use this:
- The defensible headline: on Geekbench 5 single-core, **a Pi 5 scores roughly 11× a Pi 2**
  (585 vs 52). Say it comes from single public uploads, not controlled runs.
- Don't compare Geekbench 5 numbers with Geekbench 6 numbers; the scales differ.
- Stronger option: run Geekbench 5 on *your* Pi 2 (Linux ARMv7 build) and cite your own
  result URL. That makes the Pi 2 number a measurement instead of someone else's upload.

## Why it matters for this project

The r/NAM_NeuralAmpModeler post that got 1.9 ms (later 1.1 ms) round trip on a Pi 5 reports
CPU staying under 20% [4]. This project runs the same kind of model on hardware roughly an
order of magnitude slower per core, so where it stops keeping up, and what moves that point,
is the result.

## Sources
1. Raspberry Pi, "Benchmarking Raspberry Pi 5": https://www.raspberrypi.com/news/benchmarking-raspberry-pi-5/
2. Geekbench 5, Raspberry Pi 2 Model B Rev 1.1 (Linux ARMv7): https://browser.geekbench.com/v5/cpu/23548532
3. Geekbench 5, Raspberry Pi 5 user upload (Oct 2023, Debian 12): https://browse.geekbench.ca/v5/cpu/23155634
4. r/NAM_NeuralAmpModeler, "1.9ms round-trip latency using a Raspberry Pi 5": https://www.reddit.com/r/NAM_NeuralAmpModeler/comments/1t7w4ho/19ms_roundtrip_latency_using_a_raspberry_pi_5/
