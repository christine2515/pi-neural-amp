#!/usr/bin/env bash
# Set the cpufreq governor on all CPUs: ./scripts/set_governor.sh performance|ondemand
# bench records the active governor in every summary row.
set -euo pipefail
GOV=${1:?usage: set_governor.sh performance|ondemand|powersave}
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  echo "$GOV" | sudo tee "$g" >/dev/null
done
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq -c
