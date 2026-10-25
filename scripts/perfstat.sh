#!/usr/bin/env bash
# Hardware-counter profile of bench on the Pi, written as CSV, plus a perf-record hotspot list.
# Counts user space only (works with the default perf_event_paranoid=2, no sudo).
# The Cortex-A7 PMU has 4 counters, so events run in groups of 4 (separate runs) to avoid multiplexing.
#
# Usage (on the Pi): TAG=perf_base BENCH=./build-base/bench ./scripts/perfstat.sh
# Env: TAG (required), BENCH=./build/bench, SLIMS="0.25 1", BLOCK=128, DURATION=20 (s of audio),
#      INPUT=audio/di.wav, RECORD=1 (also perf record)
# Output: results/perf/<TAG>/stat_<variant>_<group>.csv, results/perf/<TAG>/top_<variant>.txt
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${TAG:?set TAG}
BENCH=${BENCH:-./build/bench}
SLIMS=${SLIMS:-"0.25 1"}
BLOCK=${BLOCK:-128}
DURATION=${DURATION:-20}
INPUT=${INPUT:-audio/di.wav}
RECORD=${RECORD:-1}
MODEL=${MODEL:-models/a2_fender_super_reverb.nam}
OUT=results/perf/$TAG
mkdir -p "$OUT"

GROUPS_=(
  "core:cycles:u,instructions:u,branches:u,branch-misses:u"
  "l1d:cycles:u,L1-dcache-loads:u,L1-dcache-load-misses:u,dTLB-load-misses:u"
  "l2:cycles:u,LLC-loads:u,LLC-load-misses:u,L1-icache-load-misses:u"
)

for slim in $SLIMS; do
  name=$("$BENCH" --model "$MODEL" --print-slim-sizes | awk -v s="$slim" '$1==s {print $2}')
  name=${name:-slim$slim}
  run=("$BENCH" --model "$MODEL" --slim "$slim" --input "$INPUT" --block "$BLOCK" --duration "$DURATION"
       --label "${TAG}__${name}" --summary-out "results/summary_${TAG}.csv")
  for g in "${GROUPS_[@]}"; do
    gname=${g%%:*}
    events=${g#*:}
    perf stat -x, -e "$events" -o "$OUT/stat_${name}_${gname}.csv" "${run[@]}"
  done
  if [ "$RECORD" = 1 ]; then
    perf record -q --all-user -F 2000 -o "$OUT/perf_${name}.data" "${run[@]}" >/dev/null
    # (|| true: head closing the pipe early is not an error)
    { perf report -q -i "$OUT/perf_${name}.data" --stdio --no-children --sort symbol 2>/dev/null \
        | sed -E 's/ +$//; s/  +/ /g' | grep -v '^$' | head -40 > "$OUT/top_${name}.txt"; } || true
  fi
done
echo "wrote $OUT"
