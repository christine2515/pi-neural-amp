#!/usr/bin/env bash
# Low-latency experiments (docs/low-latency-plan.md), run ON THE PI, one after another, never in parallel.
# Each experiment is one sweep.sh call with its own TAG (results/summary_<TAG>.csv) and environment.
#
# Usage (on the Pi, detached):
#   SET=fsplit_burst setsid nohup ./scripts/lowlat_runs.sh > results/log_x_lowlat.txt 2>&1 < /dev/null &
# Sets:
#   smalln_burst   nhead build, Lite + Full unpipelined, blocks 8-128, burst, --rt (per-block overhead)
#   fsplit_burst   frame-split build (patches/a2_frame_split.patch), Full 2/3/4 threads, Lite 2/4, burst
#   fsplit_paced   frame-split build, paced --rt at the smallest blocks that fit
#   pipe_paced     nhead build, pipelined Full at 32/64 and Lite at 8/16/32, paced --rt (comparison)
#   nulltest       10 s renders, nhead vs frame split (2/3/4 threads), null_test.py on the Pi
#   perf_smalln    perfstat.sh (counters + perf record) at block 8 and 128, nhead
# Env: NHEAD=./build/bench, FSPLIT=./build-x-fsplit/bench, REPEATS (default 2), DUR (burst seconds, 30).
set -euo pipefail
cd "$(dirname "$0")/.."

NHEAD=${NHEAD:-./build/bench}
FSPLIT=${FSPLIT:-./build-x-fsplit/bench}
REPEATS=${REPEATS:-2}
DUR=${DUR:-30}
SET=${SET:?set SET}

run() { # run TAG BENCH MODE SLIM BLOCKS EXTRA [VAR=value ...]
  local tag=$1 bench=$2 mode=$3 slim=$4 blocks=$5 extra=$6
  shift 6
  if [ -s "results/summary_$tag.csv" ]; then echo "skip $tag (exists)"; return; fi
  echo "=== $tag: $bench $mode slim=$slim blocks='$blocks' extra='$extra' $*"
  env "$@" BENCH="$bench" TAG="$tag" MODE="$mode" SLIM="$slim" BLOCKS="$blocks" REPEATS="$REPEATS" \
    EXTRA="$extra" KEEP_BLOCKS=1 ./scripts/sweep.sh
}

FS2=(NAM_A2_FSPLIT_THREADS=2 NAM_A2_FSPLIT_CPUS=2 NAM_A2_FSPLIT_PRIO=80)
FS3=(NAM_A2_FSPLIT_THREADS=3 NAM_A2_FSPLIT_CPUS=2,1 NAM_A2_FSPLIT_PRIO=80)
FS4=(NAM_A2_FSPLIT_THREADS=4 NAM_A2_FSPLIT_CPUS=2,1,0 NAM_A2_FSPLIT_PRIO=80)
BURST="--duration $DUR --rt --cpu 3"
PACED="--rt --cpu 3"

case "$SET" in
  smalln_burst)
    run x_smallN_burst "$NHEAD" burst "lite full" "8 16 32 64 128" "$BURST"
    ;;
  fsplit_burst)
    run x_fs_full_p2 "$FSPLIT" burst full "8 16 32 64 128" "$BURST" NAM_A2_FSPLIT=8 "${FS2[@]}"
    run x_fs_full_p3 "$FSPLIT" burst full "12 16 24 32 48 96" "$BURST" NAM_A2_FSPLIT=8 "${FS3[@]}"
    run x_fs_full_p4 "$FSPLIT" burst full "16 32 64 128" "$BURST" NAM_A2_FSPLIT=8 "${FS4[@]}"
    run x_fs_lite_p2 "$FSPLIT" burst lite "8 16 32 64" "$BURST" NAM_A2_FSPLIT=3 "${FS2[@]}"
    run x_fs_lite_p4 "$FSPLIT" burst lite "16 32" "$BURST" NAM_A2_FSPLIT=3 "${FS4[@]}"
    # Control: the frame-split build with the split off must match nhead.
    run x_fs_off "$FSPLIT" burst "lite full" "16 128" "$BURST"
    ;;
  fsplit_paced)
    run x_fsp_full_p2 "$FSPLIT" paced full "32 64" "$PACED" NAM_A2_FSPLIT=8 "${FS2[@]}"
    run x_fsp_full_p3 "$FSPLIT" paced full "24 48" "$PACED" NAM_A2_FSPLIT=8 "${FS3[@]}"
    run x_fsp_full_p4 "$FSPLIT" paced full "16 32" "$PACED" NAM_A2_FSPLIT=8 "${FS4[@]}"
    run x_fsp_lite_p2 "$FSPLIT" paced lite "8 16" "$PACED" NAM_A2_FSPLIT=3 "${FS2[@]}"
    ;;
  pipe_paced)
    run x_pipe_full_paced "$NHEAD" paced full "32 64" "$PACED --pipeline 8"
    run x_lite_paced_small "$NHEAD" paced lite "8 16 32" "$PACED"
    ;;
  nulltest)
    # 10 s renders on the Pi: nhead (reference) vs frame split, then scripts/null_test.py.
    M=models/a2_fender_super_reverb.nam
    R=results/x_render
    mkdir -p "$R"
    for slim in 0.25 1; do
      "$NHEAD" --model $M --slim $slim --input audio/di.wav --block 32 --duration 10 \
        --render "$R/nhead_$slim.wav" --summary-out results/summary_x_render.csv
      for th in 2 3 4; do
        env NAM_A2_FSPLIT=all NAM_A2_FSPLIT_THREADS=$th "$FSPLIT" --model $M --slim $slim --input audio/di.wav \
          --block 32 --duration 10 --render "$R/fsplit${th}_$slim.wav" --summary-out results/summary_x_render.csv
        python3 scripts/null_test.py "$R/nhead_$slim.wav" "$R/fsplit${th}_$slim.wav" --label "x_fsplit${th}_slim$slim" --out results/x_null_tests.csv
      done
    done
    ;;
  perf_smalln)
    # Hardware counters + hotspots at block 8 vs 128 (nhead, unpipelined), 10 s of audio each.
    for b in 8 128; do
      TAG=x_perf_n$b BENCH="$NHEAD" BLOCK=$b DURATION=10 ./scripts/perfstat.sh
    done
    ;;
  *)
    echo "unknown SET=$SET"
    exit 2
    ;;
esac
echo "done SET=$SET"
