#!/usr/bin/env bash
# Run every model in models/ at every block size, with memory/CPU sampling.
#
# Env vars (all optional):
#   TAG=baseline          names the output files: results/summary_$TAG.csv, results/mem/$TAG/...
#   MODELS_DIR=models     directory of .nam files
#   INPUT=audio/di.wav    48 kHz mono guitar DI
#   BLOCKS="64 128 256 512"
#   REPEATS=3
#   MODE=paced            paced (sleep to each deadline, like an audio callback) or burst (back to back)
#   EXTRA=""              extra bench flags, e.g. "--rt --cpu 3"
#   KEEP_BLOCKS=1         write per-block CSVs to results/blocks/$TAG/
#
# Example: TAG=baseline ./scripts/sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."

TAG=${TAG:-baseline}
MODELS_DIR=${MODELS_DIR:-models}
INPUT=${INPUT:-audio/di.wav}
BLOCKS=${BLOCKS:-"64 128 256 512"}
REPEATS=${REPEATS:-3}
MODE=${MODE:-paced}
EXTRA=${EXTRA:-}
KEEP_BLOCKS=${KEEP_BLOCKS:-1}
BENCH=${BENCH:-./build/bench}

SUMMARY=results/summary_${TAG}.csv
MEMDIR=results/mem/${TAG}
BLKDIR=results/blocks/${TAG}
mkdir -p "$MEMDIR" "$BLKDIR"

[ -x "$BENCH" ] || { echo "missing $BENCH: build first"; exit 1; }
[ -f "$INPUT" ] || { echo "missing input $INPUT"; exit 1; }
shopt -s nullglob
MODELS=("$MODELS_DIR"/*.nam)
[ ${#MODELS[@]} -gt 0 ] || { echo "no .nam files in $MODELS_DIR"; exit 1; }

MODE_FLAG=""
[ "$MODE" = "paced" ] && MODE_FLAG="--paced"

echo "sweep tag=$TAG mode=$MODE extra='$EXTRA' models=${#MODELS[@]} blocks='$BLOCKS' repeats=$REPEATS"
for model in "${MODELS[@]}"; do
  # Slimmable (A2) files hold several submodels; run each one as its own model ("<name>@lite", "@full").
  variants=$("$BENCH" --model "$model" --print-slim-sizes)
  [ -n "$variants" ] || variants="- -"
  while read -r slim vname <&3; do
    mname=$(basename "$model" .nam)
    slim_args=()
    if [ "$slim" != "-" ]; then
      mname="${mname}@${vname}"
      slim_args=(--slim "$slim")
    fi
    for block in $BLOCKS; do
      for rep in $(seq 1 "$REPEATS"); do
        label="${TAG}__${mname}__${block}__r${rep}"
        blk_args=()
        [ "$KEEP_BLOCKS" = 1 ] && blk_args=(--blocks-out "$BLKDIR/${mname}__${block}__r${rep}.csv")
        # shellcheck disable=SC2086
        "$BENCH" --model "$model" ${slim_args[@]+"${slim_args[@]}"} --input "$INPUT" --block "$block" \
          $MODE_FLAG $EXTRA --label "$label" --summary-out "$SUMMARY" ${blk_args[@]+"${blk_args[@]}"} &
        pid=$!
        if [ -r /proc/$pid/smaps_rollup ]; then
          ./scripts/mem_sampler.sh "$pid" "$MEMDIR/${mname}__${block}__r${rep}.csv" &
          spid=$!
        else
          spid=""
        fi
        wait "$pid" || echo "bench failed for $label (exit $?)"
        [ -n "$spid" ] && wait "$spid" 2>/dev/null || true
        sleep 1   # let the CPU settle between runs
      done
    done
  done 3<<< "$variants"
done
echo "wrote $SUMMARY"
