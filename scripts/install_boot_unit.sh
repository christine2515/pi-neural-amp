#!/usr/bin/env bash
# Install and enable the boot-timing unit. Each reboot appends a row to results/boot_first_block.csv.
# Usage: ./scripts/install_boot_unit.sh models/<file>.nam      (disable: sudo systemctl disable pi-neural-amp)
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=$(realpath "${1:?usage: install_boot_unit.sh models/<file>.nam}")
REPO=$(pwd)
sed -e "s|@USER@|$USER|g" -e "s|@REPO@|$REPO|g" -e "s|@MODEL@|$MODEL|g" systemd/pi-neural-amp.service \
  | sudo tee /etc/systemd/system/pi-neural-amp.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable pi-neural-amp.service
echo "Enabled. Reboot a few times (sudo reboot), then: python3 scripts/kpis.py"
