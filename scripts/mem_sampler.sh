#!/usr/bin/env bash
# Sample PSS/RSS (from /proc/PID/smaps_rollup) and CPU% (from /proc/PID/stat) until PID exits.
# Usage: ./scripts/mem_sampler.sh PID results/mem/run.csv [interval_s]
set -uo pipefail
PID=${1:?usage: mem_sampler.sh PID OUT.csv [interval_s]}
OUT=${2:?usage: mem_sampler.sh PID OUT.csv [interval_s]}
INTERVAL=${3:-0.2}
mkdir -p "$(dirname "$OUT")"
HZ=$(getconf CLK_TCK)

echo "unix_time,pid,pss_kb,rss_kb,private_dirty_kb,swap_kb,cpu_pct" > "$OUT"
prev_ticks=""
prev_t=""
while [ -r "/proc/$PID/smaps_rollup" ]; do
  t=$(date +%s.%N)
  read -r pss rss pdirty swap < <(awk '
    /^Pss:/ {p=$2} /^Rss:/ {r=$2} /^Private_Dirty:/ {d=$2} /^Swap:/ {s=$2}
    END {print p+0, r+0, d+0, s+0}' "/proc/$PID/smaps_rollup" 2>/dev/null) || break
  # utime + stime are fields 14 and 15; strip "pid (comm)" first in case comm has spaces.
  ticks=$(sed 's/^.*) //' "/proc/$PID/stat" 2>/dev/null | awk '{print $12 + $13}') || break
  cpu=na
  if [ -n "$prev_ticks" ] && [ -n "$ticks" ]; then
    cpu=$(awk "BEGIN{dt=$t-$prev_t; if (dt>0) printf \"%.1f\", 100*($ticks-$prev_ticks)/$HZ/dt; else print \"na\"}")
  fi
  [ -n "$pss" ] && echo "$t,$PID,$pss,$rss,$pdirty,$swap,$cpu" >> "$OUT"
  prev_ticks=$ticks
  prev_t=$t
  sleep "$INTERVAL"
done
