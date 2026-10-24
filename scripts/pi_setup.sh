#!/usr/bin/env bash
# One-time Pi setup: packages, swap for the build, real-time limits, build.
# Run from the repo root on the Pi:  ./scripts/pi_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== CPU =="
grep -m1 "CPU part" /proc/cpuinfo || true   # 0xc07 = Cortex-A7, 0xd03 = Cortex-A53
uname -a

echo "== packages =="
sudo apt-get update
sudo apt-get install -y git cmake g++ python3-matplotlib python3-numpy linux-perf || \
  sudo apt-get install -y git cmake g++ python3-matplotlib python3-numpy

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
