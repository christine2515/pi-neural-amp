# Tonight's runbook

Everything below writes CSV to `results/`. README numbers come from `results/kpi_tables.md`
and `results/*.csv`, nothing else.

## 0. Flash + boot (15 min, laptop)
Raspberry Pi Imager → Raspberry Pi OS Lite (32-bit) → settings: hostname `pi-amp`, your user,
SSH with public key. Ethernet to the router, power on, then:

    ssh <you>@pi-amp.local

## 1. Copy the repo and build (start before anything else, it's the slow part)
From the laptop, in the folder that contains `pi-neural-amp/`:

    rsync -a --exclude build --exclude .venv pi-neural-amp <you>@pi-amp.local:~/

On the Pi:

    cd ~/pi-neural-amp && ./scripts/pi_setup.sh 2>&1 | tee results/build_log.txt

`pi_setup.sh` installs packages, sets 2 GB swap, adds rtprio/memlock limits for the `audio` group,
detects A7 vs A53 and builds with `-j2`. When it finishes, log out and back in (for the group).
If the build is still crawling after ~45 min, check `free -m`: if swap is pinned, `-j1`.

## 2. Inputs (while it builds)
- `models/`: one TONE3000 capture in several sizes (standard / lite / feather / nano), named to sort by size.
- `audio/di.wav`: 30-60 s mono 48 kHz guitar DI.
- `rsync -a models audio <you>@pi-amp.local:~/pi-neural-amp/`

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

Optional burst-mode baseline (no `--rt`, see note in `compare_rt.sh`):

    TAG=burst MODE=burst ./scripts/sweep.sh

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

Copy back to the laptop: `rsync -a <you>@pi-amp.local:~/pi-neural-amp/results/ results/`

## Resume line: where each number comes from
| Placeholder | Source |
|---|---|
| X ms smallest glitch-free buffer | `kpi_smallest_block.csv` → `latency_ms` (say which model and tag) |
| Y ms p99 block time | `kpi_by_block.csv` → `p99_us` at that model/block |
| Z MB memory | `kpi_memory.csv` → `peak_pss_mb` |
| W% p99 cut by --rt | `rt_comparison.csv` → `p99_change_pct` (pick the row you'll defend; report the range) |
