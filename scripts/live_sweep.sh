#!/usr/bin/env bash
# Live tuning: run `live` through the interface at each period size / period count and count xruns.
# Play guitar while it runs if you want to listen; the timing doesn't depend on the input.
#
# Env vars (all optional):
#   TAG=live                 label prefix; rows go to results/live_summary.csv
#   DEVICE=hw:USB            ALSA device (`arecord -l` lists cards)
#   IN_CHANNEL=0             capture channel the guitar is on
#   MODELS_DIR=models        every .nam (A2 files run as @lite and @full)
#   PERIOD_SIZES="32 64 128 256"
#   PERIOD_COUNTS="2 3"
#   DURATION=20              seconds per configuration
#   EXTRA=""                 e.g. "--rt --cpu 3", "--mmap", "--pipeline 8" (Full) — applied to every run
#   SLIM="lite"              only these A2 submodels (lite/full); default all
#   BYPASS=1                 also run each configuration with no model (interface baseline)
#
# Example: EXTRA="--rt --cpu 3" TAG=live_rt ./scripts/live_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."

TAG=${TAG:-live}
DEVICE=${DEVICE:-hw:USB}
IN_CHANNEL=${IN_CHANNEL:-0}
MODELS_DIR=${MODELS_DIR:-models}
PERIOD_SIZES=${PERIOD_SIZES:-"32 64 128 256"}
PERIOD_COUNTS=${PERIOD_COUNTS:-"2 3"}
DURATION=${DURATION:-20}
EXTRA=${EXTRA:-}
BYPASS=${BYPASS:-1}

LIVE=${LIVE:-./build/live}
[ -x "$LIVE" ] || { echo "missing $LIVE"; exit 1; }
shopt -s nullglob

run() {  # run <label> <live args...>
  local label=$1; shift
  # shellcheck disable=SC2086
  "$LIVE" --device "$DEVICE" --in-channel "$IN_CHANNEL" --duration "$DURATION" $EXTRA \
    --label "$label" "$@" || echo "live failed: $label"
  sleep 1
}

for n in $PERIOD_COUNTS; do
  for p in $PERIOD_SIZES; do
    [ "$BYPASS" = 1 ] && run "${TAG}__bypass__${p}x${n}" --bypass --period "$p" --periods "$n"
    for model in "$MODELS_DIR"/*.nam; do
      variants=$("${BENCH:-./build/bench}" --model "$model" --print-slim-sizes)
      [ -n "$variants" ] || variants="- -"
      while read -r slim vname <&3; do
        if [ -n "${SLIM:-}" ] && [[ " $SLIM " != *" $vname "* ]]; then continue; fi
        mname=$(basename "$model" .nam)
        slim_args=()
        if [ "$slim" != "-" ]; then
          mname="${mname}@${vname}"
          slim_args=(--slim "$slim")
        fi
        run "${TAG}__${mname}__${p}x${n}" --model "$model" ${slim_args[@]+"${slim_args[@]}"} --period "$p" --periods "$n"
      done 3<<< "$variants"
    done
  done
done
echo "wrote results/live_summary.csv"
