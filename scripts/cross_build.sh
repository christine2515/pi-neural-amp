#!/usr/bin/env bash
# Cross-build bench + live for the Pi 2 (armhf) on a Mac.
#   ./scripts/cross_build.sh                                   -> build-armhf/
#   BUILD_DIR=build-armhf-generic ./scripts/cross_build.sh -DBENCH_A2_FAST=OFF
# Extra args are passed to the CMake configure step. Needs: cmake, Homebrew lld.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-build-armhf}"
case "$BUILD_DIR" in /*) ;; *) BUILD_DIR="$ROOT/$BUILD_DIR" ;; esac
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 8)}"

"$ROOT/scripts/fetch_sysroot.sh"

cmake -S "$ROOT" -B "$BUILD_DIR" \
  -DCMAKE_TOOLCHAIN_FILE="$ROOT/cmake/armhf-toolchain.cmake" \
  -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$BUILD_DIR" -j "$JOBS"

for b in bench live; do
  if [[ -f "$BUILD_DIR/$b" ]]; then file "$BUILD_DIR/$b"; else echo "$b: NOT BUILT" >&2; exit 1; fi
done
