#!/usr/bin/env bash
# 1 Hz logger: SoC temperature, CPU frequency and under-voltage alarm, from sysfs only.
# Usage: ./scripts/thermal_logger.sh results/thermal.csv [interval_s]
#
# It does NOT call vcgencmd in the loop: on the Pi 2, vcgencmd's firmware mailbox calls stalled the
# audio cores for several ms (experiment-report.md, iteration 12). `vcgencmd get_throttled` is read once
# at start and once at exit instead; its "has occurred" bits (16-19) are sticky since boot, so the exit
# reading still shows whether throttling, capping or under-voltage happened at any point during the run.
# VCGENCMD_EVERY=N (seconds) re-enables periodic vcgencmd calls if you accept the interference.
set -uo pipefail
OUT=${1:-results/thermal.csv}
INTERVAL=${2:-1}
VCGENCMD_EVERY=${VCGENCMD_EVERY:-0}
mkdir -p "$(dirname "$OUT")"

UV_ALARM=$(ls /sys/devices/platform/soc/soc:firmware/raspberrypi-hwmon/hwmon/hwmon*/in0_lcrit_alarm 2>/dev/null | head -1)
HAVE_VC=0
command -v vcgencmd >/dev/null 2>&1 && HAVE_VC=1

throttled_row() {  # throttled_row <phase>: one row of decoded vcgencmd get_throttled bits
  local thr v
  if [ "$HAVE_VC" = 1 ]; then
    thr=$(vcgencmd get_throttled | cut -d= -f2)
    v=$((thr))
    echo "$1,$thr,$((v & 1)),$(((v >> 1) & 1)),$(((v >> 2) & 1)),$(((v >> 3) & 1)),$(((v >> 16) & 1)),$(((v >> 17) & 1)),$(((v >> 18) & 1)),$(((v >> 19) & 1))"
  else
    echo "$1,na,na,na,na,na,na,na,na,na"
  fi
}

THR_OUT="${OUT%.csv}_throttled.csv"
echo "phase,throttled_hex,under_voltage_now,freq_capped_now,throttled_now,soft_temp_limit_now,under_voltage_ever,freq_capped_ever,throttled_ever,soft_temp_limit_ever" > "$THR_OUT"
throttled_row start >> "$THR_OUT"
trap 'throttled_row end >> "$THR_OUT"; exit 0' INT TERM EXIT

if [ ! -s "$OUT" ]; then
  echo "unix_time,uptime_s,temp_c,scaling_cur_freq_khz,under_voltage_alarm,throttled_hex" > "$OUT"
fi

i=0
while true; do
  now=$(date +%s.%N)
  case $now in *N) now=$(date +%s) ;; esac   # BSD date has no %N
  up=$(cut -d' ' -f1 /proc/uptime 2>/dev/null || echo na)
  t_milli=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo "")
  if [ -n "$t_milli" ]; then temp=$((t_milli / 1000)).$(printf '%03d' $((t_milli % 1000))); else temp=na; fi
  freq=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo na)
  uv=na
  [ -n "$UV_ALARM" ] && uv=$(cat "$UV_ALARM" 2>/dev/null || echo na)
  thr=na
  if [ "$HAVE_VC" = 1 ] && [ "$VCGENCMD_EVERY" -gt 0 ] && [ $((i % VCGENCMD_EVERY)) = 0 ]; then
    thr=$(vcgencmd get_throttled | cut -d= -f2)
  fi
  echo "$now,$up,$temp,$freq,$uv,$thr" >> "$OUT"
  i=$((i + 1))
  sleep "$INTERVAL"
done
