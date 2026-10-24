#!/usr/bin/env bash
# 1 Hz logger: SoC temperature, CPU frequency, vcgencmd get_throttled flags.
# Usage: ./scripts/thermal_logger.sh results/thermal.csv [interval_s]
# Stop with Ctrl-C or kill. Works without vcgencmd (those columns become "na").
set -uo pipefail
OUT=${1:-results/thermal.csv}
INTERVAL=${2:-1}
mkdir -p "$(dirname "$OUT")"

HAVE_VC=0
command -v vcgencmd >/dev/null 2>&1 && HAVE_VC=1

if [ ! -s "$OUT" ]; then
  echo "unix_time,uptime_s,temp_c,scaling_cur_freq_khz,arm_clock_hz,throttled_hex,under_voltage_now,freq_capped_now,throttled_now,soft_temp_limit_now,under_voltage_ever,freq_capped_ever,throttled_ever,soft_temp_limit_ever" > "$OUT"
fi

while true; do
  now=$(date +%s.%N)
  case $now in *N) now=$(date +%s) ;; esac   # BSD date has no %N
  up=$(cut -d' ' -f1 /proc/uptime 2>/dev/null || echo na)
  t_milli=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo "")
  if [ -n "$t_milli" ]; then temp=$(awk "BEGIN{printf \"%.2f\", $t_milli/1000}"); else temp=na; fi
  freq=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo na)
  if [ "$HAVE_VC" = 1 ]; then
    arm=$(vcgencmd measure_clock arm | cut -d= -f2)
    thr=$(vcgencmd get_throttled | cut -d= -f2)
    v=$((thr))
    bits="$((v & 1)),$(((v >> 1) & 1)),$(((v >> 2) & 1)),$(((v >> 3) & 1)),$(((v >> 16) & 1)),$(((v >> 17) & 1)),$(((v >> 18) & 1)),$(((v >> 19) & 1))"
  else
    arm=na; thr=na; bits="na,na,na,na,na,na,na,na"
  fi
  echo "$now,$up,$temp,$freq,$arm,$thr,$bits" >> "$OUT"
  sleep "$INTERVAL"
done
