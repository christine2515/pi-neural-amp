#!/usr/bin/env bash
# Copy cross-built binaries to the Pi: <BUILD_DIR>/{bench,live} -> ~/pi-neural-amp/<DEST>/ on the Pi.
#   PI_USER=you ./scripts/deploy.sh
#   PI_USER=you BUILD_DIR=build-armhf-generic DEST=build-generic ./scripts/deploy.sh
# PI defaults to $PI_USER@pi-amp.local; set PI=user@host to override.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${PI_USER:?set PI_USER to your login on the Pi}"
PI="${PI:-$PI_USER@pi-amp.local}"
BUILD_DIR="${BUILD_DIR:-build-armhf}"
case "$BUILD_DIR" in /*) ;; *) BUILD_DIR="$ROOT/$BUILD_DIR" ;; esac
DEST="${DEST:-build}"

bins=()
for b in bench live; do [[ -f "$BUILD_DIR/$b" ]] && bins+=("$BUILD_DIR/$b"); done
[[ ${#bins[@]} -gt 0 ]] || { echo "nothing in $BUILD_DIR: run scripts/cross_build.sh" >&2; exit 1; }

rsync -av --rsync-path="mkdir -p ~/pi-neural-amp/$DEST && rsync" "${bins[@]}" "$PI:~/pi-neural-amp/$DEST/"
echo "on the Pi: cd ~/pi-neural-amp && ./$DEST/bench --help"
