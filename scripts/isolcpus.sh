#!/usr/bin/env bash
# Isolate CPU cores from the scheduler and IRQs so only the pinned audio threads run there.
# Usage: ./scripts/isolcpus.sh enable [cpus]   cpus: "3" (default) or a list/range like "1-3" or "2,3";
#                                              IRQs go to the remaining cores. Then reboot.
#        ./scripts/isolcpus.sh disable         (then reboot)
#        ./scripts/isolcpus.sh status
# Sets isolcpus=, irqaffinity= and rcu_nocbs= on the kernel command line (single line); a backup is
# kept at cmdline.txt.bak. nohz_full= is NOT set: the stock rpi-v7 kernel lacks CONFIG_NO_HZ_FULL.
set -euo pipefail

CMDLINE=/boot/firmware/cmdline.txt
[ -f "$CMDLINE" ] || CMDLINE=/boot/cmdline.txt
ACTION=${1:-status}
CPUS=${2:-3}

strip() { sed -E 's/ ?(isolcpus|irqaffinity|rcu_nocbs|nohz_full)=[^ ]*//g' "$CMDLINE"; }

expand() {  # "1-3" or "2,3" -> one number per line
  echo "$1" | tr ',' '\n' | while read -r part; do
    if [[ "$part" == *-* ]]; then seq "${part%-*}" "${part#*-}"; else echo "$part"; fi
  done
}

case "$ACTION" in
  enable)
    ALL=$(seq 0 $(( $(nproc --all) - 1 )))
    ISO=$(expand "$CPUS")
    OTHERS=$(echo "$ALL" | grep -vxF "$ISO" | paste -sd, -)   # IRQs and housekeeping go to these cores
    [ -n "$OTHERS" ] || { echo "leave at least one core for the OS"; exit 2; }
    sudo cp "$CMDLINE" "$CMDLINE.bak"
    new="$(strip) isolcpus=$CPUS irqaffinity=$OTHERS rcu_nocbs=$CPUS"
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
    echo "cmdline:  $(grep -oE '(isolcpus|irqaffinity|rcu_nocbs|nohz_full)=[^ ]*' "$CMDLINE" | tr '\n' ' ')"
    echo "isolated: $(cat /sys/devices/system/cpu/isolated 2>/dev/null || echo na)"
    echo "USB IRQ (dwc_otg) affinity: $(cat /proc/irq/$(awk -F: '/dwc_otg/{gsub(/ /,"",$1); print $1; exit}' /proc/interrupts)/smp_affinity_list 2>/dev/null || echo na)"
    ;;
  *) echo "usage: isolcpus.sh enable [cpus] | disable | status"; exit 2 ;;
esac
