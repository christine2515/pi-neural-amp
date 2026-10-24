#!/usr/bin/env bash
# Run the sweep under each scheduling arm, then build the comparison table.
#   baseline  normal scheduling
#   rt        SCHED_FIFO 80 + pinned to RT_CPU + mlockall
#   rt_isol   same as rt, with RT_CPU isolated by isolcpus (needs ./scripts/isolcpus.sh enable + reboot)
#
# Env: ARMS (default: "baseline rt", plus "rt_isol" automatically if RT_CPU is isolated),
#      RT_CPU (default 3), and everything sweep.sh takes (MODE, BLOCKS, REPEATS, ...).
# Arms whose summary file already exists are skipped, so the usual flow is:
#   ./scripts/compare_rt.sh                     # baseline + rt
#   ./scripts/isolcpus.sh enable && sudo reboot
#   ./scripts/compare_rt.sh                     # adds rt_isol, rebuilds the table
#   ./scripts/isolcpus.sh disable && sudo reboot
# Output: results/summary_<arm>.csv, results/rt_comparison.{csv,md}
#
# Note: with MODE=burst, a SCHED_FIFO loop that never sleeps hits the kernel's RT throttle
# (sched_rt_runtime_us, 95% per second by default) and stalls. Use MODE=paced (the default) for --rt.
set -euo pipefail
cd "$(dirname "$0")/.."
RT_CPU=${RT_CPU:-3}

isolated=$(cat /sys/devices/system/cpu/isolated 2>/dev/null || true)
if [ -z "${ARMS:-}" ]; then
  ARMS="baseline rt"
  [ "$isolated" = "$RT_CPU" ] && ARMS="$ARMS rt_isol"
fi

for arm in $ARMS; do
  if [ -s "results/summary_${arm}.csv" ]; then
    echo "== $arm: results/summary_${arm}.csv exists, skipping (delete it to rerun)"
    continue
  fi
  case "$arm" in
    baseline)
      [ -n "$isolated" ] && { echo "baseline must run without isolcpus (isolated=$isolated)"; exit 1; }
      TAG=baseline EXTRA="" ./scripts/sweep.sh ;;
    rt)
      [ -n "$isolated" ] && { echo "rt must run without isolcpus (isolated=$isolated)"; exit 1; }
      TAG=rt EXTRA="--rt --rt-prio 80 --cpu $RT_CPU" ./scripts/sweep.sh ;;
    rt_isol)
      [ "$isolated" = "$RT_CPU" ] || { echo "rt_isol needs CPU $RT_CPU isolated: ./scripts/isolcpus.sh enable $RT_CPU && sudo reboot"; exit 1; }
      TAG=rt_isol EXTRA="--rt --rt-prio 80 --cpu $RT_CPU" ./scripts/sweep.sh ;;
    *) echo "unknown arm $arm"; exit 2 ;;
  esac
done

compare=()
for arm in baseline rt rt_isol; do
  [ -s "results/summary_${arm}.csv" ] && compare+=("$arm")
done
python3 scripts/kpis.py --compare "${compare[@]}"
