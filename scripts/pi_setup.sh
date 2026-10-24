#!/usr/bin/env bash
# One-time Pi setup: packages, swap for the build, real-time limits, build.
# Run from the repo root on the Pi:  ./scripts/pi_setup.sh
# With cross-compiled binaries (scripts/cross_build.sh + deploy.sh): SKIP_BUILD=1 ./scripts/pi_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== CPU =="
grep -m1 "CPU part" /proc/cpuinfo || true   # 0xc07 = Cortex-A7, 0xd03 = Cortex-A53
uname -a

echo "== packages =="
sudo apt-get update
PKGS="git cmake g++ python3-matplotlib python3-numpy libasound2-dev alsa-utils"
# shellcheck disable=SC2086
sudo apt-get install -y $PKGS linux-perf || sudo apt-get install -y $PKGS

echo "== swap (Eigen builds need it on 1 GB) =="
if [ -f /etc/dphys-swapfile ]; then
  sudo sed -i 's/^#\?CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
  sudo sed -i 's/^#\?CONF_MAXSWAP=.*/CONF_MAXSWAP=2048/' /etc/dphys-swapfile
  sudo systemctl restart dphys-swapfile
fi
free -m

echo "== real-time limits for the audio group (lets --rt run without sudo) =="
sudo tee /etc/security/limits.d/99-pi-neural-amp.conf >/dev/null <<'EOF'
@audio - rtprio 95
@audio - memlock unlimited
EOF
sudo usermod -aG audio "$USER"
echo "Log out and back in (or reboot) for the audio group / limits to apply."

if [ "${SKIP_BUILD:-0}" = 1 ]; then
  echo "== SKIP_BUILD=1: not building; deploy cross-built binaries from the Mac =="
  cat /etc/os-release | grep -E "^(PRETTY_NAME|VERSION_CODENAME)="
  ldd --version | head -1
  exit 0
fi

echo "== submodules =="
git submodule update --init --recursive

echo "== build =="
CPU_PART=$(grep -m1 "CPU part" /proc/cpuinfo | awk '{print $4}')
PI_CPU=cortex-a7
[ "$CPU_PART" = "0xd03" ] && PI_CPU=cortex-a53
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPI_CPU="$PI_CPU"
# -j2: -j4 runs out of memory compiling Eigen-heavy files on 1 GB.
time cmake --build build -j2
./build/bench --help || true
