#!/usr/bin/env bash
# Run the sweep without and with --rt, then build the comparison table.
# Same env vars as sweep.sh (MODE, BLOCKS, REPEATS, ...). RT_CPU picks the pinned core (default 3).
# Output: results/summary_baseline.csv, results/summary_rt.csv, results/rt_comparison.{csv,md}
#
# Note: with MODE=burst, a SCHED_FIFO loop that never sleeps hits the kernel's RT throttle
# (sched_rt_runtime_us, 95% per second by default) and stalls. Use MODE=paced (the default) for --rt.
set -euo pipefail
cd "$(dirname "$0")/.."
RT_CPU=${RT_CPU:-3}

TAG=baseline EXTRA="" ./scripts/sweep.sh
TAG=rt EXTRA="--rt --rt-prio 80 --cpu $RT_CPU" ./scripts/sweep.sh
python3 scripts/kpis.py --compare baseline rt
