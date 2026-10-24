# pi-neural-amp: neural guitar amp on a Raspberry Pi 2, profiled like a product

## Goal
Run NAM models in real time on a Raspberry Pi 2 (32-bit ARM, 1 GB RAM) and measure
per-block time against the deadline at 64-512 sample blocks (48 kHz), the smallest
block size with zero missed deadlines, CPU, memory (PSS), boot to ready, and
sustained performance over 30 minutes.

## Rules
- The audio path (process loop) has no allocation, locks, logging or syscalls.
- All measurements write CSV to results/; plots come from CSV only.
- Never report a number that a script didn't produce.

## Division of labor
- I (Christine) choose what to measure, design experiments, and write the
  analysis and conclusions in README.md.
- You write build files, the benchmark harness, metric scripts, the systemd
  unit, plotting, and first drafts I edit.

## Layout
src/bench.cpp   offline real-time benchmark (file in, timings out)
src/live.cpp    optional live ALSA/JACK client
scripts/        thermal logger, memory sampler, boot timing, plots
results/        CSVs and charts
