#!/usr/bin/env bash
# Isolate one CPU core from the scheduler and IRQs so only the pinned audio thread runs there.
# Usage: ./scripts/isolcpus.sh enable [cpu]   (default cpu 3; then reboot)
#        ./scripts/isolcpus.sh disable         (then reboot)
#        ./scripts/isolcpus.sh status
# Edits the kernel command line (single line). A backup is kept at cmdline.txt.bak.
set -euo pipefail

CMDLINE=/boot/firmware/cmdline.txt
[ -f "$CMDLINE" ] || CMDLINE=/boot/cmdline.txt
ACTION=${1:-status}
CPU=${2:-3}

strip() { sed -E 's/ ?(isolcpus|irqaffinity)=[^ ]*//g' "$CMDLINE"; }

case "$ACTION" in
  enable)
    OTHERS=$(seq 0 $(( $(nproc --all) - 1 )) | grep -vx "$CPU" | paste -sd, -)   # IRQs go to these cores
    sudo cp "$CMDLINE" "$CMDLINE.bak"
    new="$(strip) isolcpus=$CPU irqaffinity=$OTHERS"
    echo "$new" | sudo tee "$CMDLINE" >/dev/null
    echo "cmdline now: $(cat "$CMDLINE")"
    echo "Reboot, then check: ./scripts/isolcpus.sh status"
    ;;
  disable)
    sudo cp "$CMDLINE" "$CMDLINE.bak"
    strip | sudo tee "$CMDLINE" >/dev/null
    echo "cmdline now: $(cat "$CMDLINE")"
    echo "Reboot to apply."
    ;;
  status)
    echo "cmdline:  $(grep -oE '(isolcpus|irqaffinity)=[^ ]*' "$CMDLINE" | tr '\n' ' ')"
    echo "isolated: $(cat /sys/devices/system/cpu/isolated 2>/dev/null || echo na)"
    ;;
  *) echo "usage: isolcpus.sh enable [cpu] | disable | status"; exit 2 ;;
esac
