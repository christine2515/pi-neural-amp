#!/usr/bin/env bash
# Run one experiment on the Pi from the Mac: deploy a cross-built variant, run sweep.sh there
# (detached, so a dropped SSH connection doesn't kill it), wait, and pull results into results-pi/.
#
# Usage (from the Mac):
#   VARIANT=fm TAG=a2fm SLIM=lite ./scripts/remote_sweep.sh
# Env:
#   VARIANT   cross build to use: build-armhf-$VARIANT (default: base)
#   TAG       results tag (required); refuses to overwrite results/summary_$TAG.csv on the Pi
#   PI        ssh host (default: pi, the ~/.ssh/config alias)
#   anything sweep.sh reads (SLIM, BLOCKS, REPEATS, MODE, EXTRA, MEM_SAMPLE, KEEP_BLOCKS) is passed on
set -euo pipefail
cd "$(dirname "$0")/.."

VARIANT=${VARIANT:-base}
TAG=${TAG:?set TAG}
PI=${PI:-pi}
BUILD_DIR=build-armhf-$VARIANT
DEST=build-$VARIANT
[ -x "$BUILD_DIR/bench" ] || { echo "missing $BUILD_DIR/bench: BUILD_DIR=$BUILD_DIR ./scripts/cross_build.sh <flags>"; exit 1; }

if ssh "$PI" "pgrep -f '[s]cripts/sweep.sh|[b]uild[^ ]*/bench ' >/dev/null"; then
  echo "something is already benchmarking on the Pi; not starting"; exit 1
fi
if ssh "$PI" "test -s ~/pi-neural-amp/results/summary_$TAG.csv"; then
  echo "results/summary_$TAG.csv already exists on the Pi; pick a new TAG"; exit 1
fi

rsync -a --rsync-path="mkdir -p ~/pi-neural-amp/$DEST && rsync" "$BUILD_DIR/bench" "$BUILD_DIR/live" "$PI:~/pi-neural-amp/$DEST/"
# Keep the Pi's scripts in step with the Mac (never results/).
rsync -a scripts "$PI:~/pi-neural-amp/"

envs="BENCH=./$DEST/bench TAG=$TAG"
for v in SLIM BLOCKS REPEATS MODE EXTRA MEM_SAMPLE KEEP_BLOCKS; do
  [ -n "${!v:-}" ] && envs="$envs $v='${!v}'"
done
echo "starting on $PI: $envs ./scripts/sweep.sh"
ssh "$PI" "cd ~/pi-neural-amp && $envs setsid nohup ./scripts/sweep.sh > results/log_$TAG.txt 2>&1 < /dev/null &"

sleep 5
# One long-lived SSH session that checks every 30 s (a new SSH handshake each time costs the Pi 2 CPU).
ssh "$PI" "while pgrep -f [s]cripts/sweep.sh >/dev/null; do sleep 30; done"
ssh "$PI" "tail -n 3 ~/pi-neural-amp/results/log_$TAG.txt"

mkdir -p results-pi
rsync -a --exclude '*.wav' --exclude 'perf*.data*' "$PI:~/pi-neural-amp/results/" results-pi/
echo "pulled into results-pi/ (summary_$TAG.csv)"
