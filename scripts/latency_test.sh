#!/usr/bin/env bash
# Round-trip latency: `live --latency-test` plays a click every 0.5 s and times it coming back.
#
# Setup: cable from one Scarlett OUTPUT (line out / monitor out) to the INPUT given by IN_CHANNEL.
# Turn Direct Monitor off, set the input gain low and output volume moderate. Unplug the guitar.
#
# Env: DEVICE=hw:USB, IN_CHANNEL=0, PERIOD_SIZES="32 64 128 256", PERIOD_COUNTS="2 3",
#      DURATION=10, EXTRA="" (e.g. "--rt --cpu 3"), TAG=latency
# Output: results/live_latency.csv (one row per click), results/live_summary.csv
set -euo pipefail
cd "$(dirname "$0")/.."

TAG=${TAG:-latency}
DEVICE=${DEVICE:-hw:USB}
IN_CHANNEL=${IN_CHANNEL:-0}
PERIOD_SIZES=${PERIOD_SIZES:-"32 64 128 256"}
PERIOD_COUNTS=${PERIOD_COUNTS:-"2 3"}
DURATION=${DURATION:-10}
EXTRA=${EXTRA:-}

[ -x ./build/live ] || { echo "missing ./build/live"; exit 1; }
for n in $PERIOD_COUNTS; do
  for p in $PERIOD_SIZES; do
    # shellcheck disable=SC2086
    ./build/live --latency-test --device "$DEVICE" --in-channel "$IN_CHANNEL" --period "$p" --periods "$n" \
      --duration "$DURATION" $EXTRA --label "${TAG}__${p}x${n}" || echo "failed: ${p}x${n}"
    sleep 1
  done
done
python3 scripts/kpis.py >/dev/null && grep -A30 "Round-trip latency" results/kpi_tables.md || true
