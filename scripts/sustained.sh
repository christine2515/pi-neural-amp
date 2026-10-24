#!/usr/bin/env bash
# 30-minute sustained run with thermal logging.
# Usage: MODEL=models/x.nam BLOCK=128 ./scripts/sustained.sh
# Env: DURATION=1800, IDLE=30, TAIL=60, INPUT=audio/di.wav, EXTRA="--rt" (optional), TAG=sustained
# Output: results/thermal_$TAG.csv, results/summary_$TAG.csv, results/blocks/$TAG/sustained.csv,
#         results/mem/$TAG/sustained.csv
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:?set MODEL=models/<file>.nam}
BLOCK=${BLOCK:-128}
DURATION=${DURATION:-1800}
INPUT=${INPUT:-audio/di.wav}
EXTRA=${EXTRA:-}
TAG=${TAG:-sustained}
mkdir -p results/blocks/$TAG results/mem/$TAG

./scripts/thermal_logger.sh "results/thermal_${TAG}.csv" 1 &
LOGGER=$!
trap 'kill $LOGGER 2>/dev/null || true' EXIT
sleep "${IDLE:-30}"   # idle baseline before load

# shellcheck disable=SC2086
./build/bench --model "$MODEL" --input "$INPUT" --block "$BLOCK" --paced --duration "$DURATION" $EXTRA \
  --label "${TAG}__$(basename "$MODEL" .nam)__${BLOCK}" --summary-out "results/summary_${TAG}.csv" \
  --blocks-out "results/blocks/$TAG/sustained.csv" &
pid=$!
./scripts/mem_sampler.sh "$pid" "results/mem/$TAG/sustained.csv" 1 &
wait "$pid"
sleep "${TAIL:-60}"   # cool-down tail
echo "done: results/thermal_${TAG}.csv"
