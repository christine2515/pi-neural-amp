# Tonight's runbook

Everything below writes CSV to `results/`. README numbers come from `results/kpi_tables.md`
and `results/*.csv`, nothing else.

**Build used for all results from here on: clang 16, cross-compiled on the Mac** (section 1b), deployed
to the Pi's `build/` so every script's default `./build/bench` / `./build/live` is the clang build.
After any code change: `./scripts/cross_build.sh && PI_USER=<you> ./scripts/deploy.sh`.
The one GCC 14.2 vs clang comparison (block 128, one run each) is kept in `results-pi/summary_gcc.csv`
and `summary_clang.csv`: GCC was 7% faster at Lite p50 (10% at p99), and the two were within ~1% on Full.

## 0. Flash + boot (15 min, laptop)
Raspberry Pi Imager → Raspberry Pi OS Lite (32-bit) → settings: hostname `pi-amp`, your user,
SSH with public key. Ethernet to the router, power on, then:

    ssh <you>@pi-amp.local

## 1. Copy the repo and build (start before anything else, it's the slow part)
From the laptop, in the folder that contains `pi-neural-amp/`:

    rsync -a --exclude 'build*' --exclude .sysroot --exclude .venv --exclude audio/raw --exclude results pi-neural-amp <you>@pi-amp.local:~/

**Always `--exclude results`**: rsync overwrites files with the same name, so Mac test runs in
`results/summary.csv` would replace the Pi's. The `host` column in every CSV shows which machine wrote a row.

On the Pi:

    cd ~/pi-neural-amp && ./scripts/pi_setup.sh 2>&1 | tee results/build_log.txt

`pi_setup.sh` installs packages, sets 2 GB swap, adds rtprio/memlock limits for the `audio` group,
detects A7 vs A53 and builds with `-j2`. When it finishes, log out and back in (for the group).
If the build is still crawling after ~45 min, check `free -m`: if swap is pinned, `-j1`.

### 1b. Cross-compiling on the Mac (instead of the slow Pi build)
One-time: `brew install lld` (Apple clang compiles; LLVM lld links ELF). Then:

    ./scripts/cross_build.sh                       # fetches .sysroot/ (Debian Bookworm armhf) once, builds build-armhf/
    PI_USER=<you> ./scripts/deploy.sh              # rsyncs bench + live to ~/pi-neural-amp/build/ on the Pi

A2-generic copy for 4c: `BUILD_DIR=build-armhf-generic ./scripts/cross_build.sh -DBENCH_A2_FAST=OFF`, then
`PI_USER=<you> BUILD_DIR=build-armhf-generic DEST=build-generic ./scripts/deploy.sh`. Binaries link the Pi's own
glibc/libstdc++ (needs Bookworm or newer). They are built with clang, not the Pi's GCC: `build_flags` in the
summary CSVs says `AppleClang-...`, so don't mix cross-built and Pi-built numbers in one comparison.
For an A53 board add `-DPI_CPU=cortex-a53`.

## 2. Inputs (while it builds)
- `models/`: `a2_fender_super_reverb.nam` is already there. It's an A2 file, so every sweep runs it
  twice, as `@lite` (3 channels) and `@full` (8 channels): same amp, two sizes.
- `audio/di.wav`: 30-60 s mono 48 kHz guitar DI.
- `rsync -a --exclude raw models audio <you>@pi-amp.local:~/pi-neural-amp/`

Sanity check that it sounds like an amp (laptop build is fine for this):

    ./build/bench --model models/<x>.nam --input audio/di.wav --block 512 --render results/check.wav

## 3. Decide the conditions (your call; write it in the README)
- Governor: `./scripts/set_governor.sh performance` (bench logs the governor in every row either way).
- Mode: sweeps default to `MODE=paced` (sleep to each deadline, like an audio callback). `MODE=burst`
  runs back to back. On the Mac, paced p99 was several times burst p99 for the same model, which is
  why both are worth reporting. Pick one as the headline.

## 4. KPI 1-3: sweep, smallest block, --rt before/after (~50 min with a 30 s DI, 3 repeats)

    ./scripts/compare_rt.sh

Writes `summary_baseline.csv`, `summary_rt.csv`, `kpi_by_block.csv`, `kpi_smallest_block.csv`,
`rt_comparison.csv`, `kpi_memory.csv` (KPI 4 comes free: PSS + CPU% are sampled on every run).
Quick look: `cat results/kpi_tables.md`.

Paced runs also record **wake-up latency** (`wake_p99_us`, `missed_with_wake`): how late the thread
woke after each sleep. That's the scheduler's share of the deadline, separate from model compute.

### 4b. Third arm: isolated core (~25 min + 2 reboots)

    ./scripts/isolcpus.sh enable 3 && sudo reboot
    ./scripts/isolcpus.sh status           # expect "isolated: 3"
    ./scripts/compare_rt.sh                # skips arms already run, adds rt_isol, rebuilds the table
    ./scripts/isolcpus.sh disable && sudo reboot

`rt_comparison.csv` then has baseline / rt / rt_isol columns side by side, plus % change vs baseline
for p99 block time and p99 wake-up latency. Charts: `rt_comparison.png`, `wake_latency.png`.

Optional burst-mode baseline (no `--rt`, see note in `compare_rt.sh`):

    TAG=burst MODE=burst ./scripts/sweep.sh

### 4c. Optional optimization: NAM's A2 fast path on vs off

`bench` is built with NAM's hand-optimized A2 WaveNet path. Build a second copy with the generic
path (another long Pi build, so start it before something slow like the 30-minute run):

    cmake -S . -B build-generic -DCMAKE_BUILD_TYPE=Release -DBENCH_A2_FAST=OFF && cmake --build build-generic -j2
    BENCH=./build-generic/bench TAG=a2_generic ./scripts/sweep.sh
    python3 scripts/kpis.py --compare a2_generic baseline

`build_flags` in each summary row records `a2_fast=ON/OFF`.

## 5. Profile the hot loop with perf (10 min; needed for "profiled with perf")

    sudo perf record -g -o results/perf.data ./build/bench --model models/<std>.nam \
      --input audio/di.wav --block 128 --summary-out results/summary_perf.csv
    sudo perf report -i results/perf.data --stdio --no-children | head -60 > results/perf_top.txt

## 6. KPI 5: sustained 30 min + boot (start it, then go write)

    MODEL=models/<smallest that passes at 128>.nam BLOCK=128 ./scripts/sustained.sh

Boot to first block (each reboot adds a row; do 3-5):

    ./scripts/install_boot_unit.sh models/<x>.nam && sudo reboot

Disable afterwards: `sudo systemctl disable pi-neural-amp`.

## 7. Tables + charts

    python3 scripts/kpis.py --compare baseline rt
    python3 scripts/plot.py --blocks results/blocks/baseline/<model>__128__r1.csv

Copy back to the laptop into its own folder, so Pi data never mixes with Mac test runs:
`rsync -a <you>@pi-amp.local:~/pi-neural-amp/results/ results-pi/` Then on the Mac: `RESULTS_DIR=results-pi python3 scripts/plot.py`.

## 8. Live tuning with the Scarlett (after the offline runs)

Wiring: guitar → Scarlett instrument input (input 2 on a Solo, INST on; ALSA capture channel 1) → USB → Pi → Scarlett headphones. **Direct Monitor off.**
If the Scarlett drops out or crackles, power it from a powered USB hub instead of the Pi.

    arecord -l                     # find the card name; usually "USB" -> DEVICE=hw:USB
    ./build/live --device hw:USB --in-channel 1 --model models/a2_fender_super_reverb.nam --slim 0.25 --period 32 --periods 2 --rt --cpu 3 --gate -50 --duration 60   # Lite
    ./build-nfs/live --device hw:USB --in-channel 1 --model models/a2_fender_super_reverb.nam --slim 1 --fsplit 8 --period 48 --periods 2 --rt --cpu 3 --gate -50 --duration 60   # Full, frame split

Current recommended settings and their measured numbers: docs/step2/state-of-the-union.md.

That's the playable amp. Each run appends a row to `results/live_summary.csv` (xruns, process
time, period jitter, CPU). Then tune:

    ./scripts/live_sweep.sh                                       # baseline: 32-256 frames x 2/3 periods
    EXTRA="--rt --cpu 3" TAG=live_rt ./scripts/live_sweep.sh      # same with real-time scheduling
    python3 scripts/kpis.py                                       # live_by_config / live_smallest_buffer

The offline `kpi_smallest_block.csv` predicts where live should start working; the gap between
the two is driver + USB overhead (`--bypass` rows show it with no model at all).

**Round-trip latency.** Unplug the guitar and connect a cable from a Scarlett output to the instrument input (input 2 on a Solo; RCA → ¼" from its RCA line outs):

    ./scripts/latency_test.sh                                     # results/live_latency.csv

`live_latency_summary.csv` shows measured round trip next to the buffer size (period × periods).
The difference is converter + USB latency. The Reddit Pi 5 post reports 3.5 ms with a Scarlett.
