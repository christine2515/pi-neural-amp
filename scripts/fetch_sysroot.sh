#!/usr/bin/env bash
# Build a Debian 12 (Bookworm) armhf sysroot in .sysroot/ for cross-compiling on a Mac.
# Idempotent: re-running with the same pins is a no-op. Downloads are SHA256-checked and only
# unpacked (ar + tar) as data; nothing from the packages is ever executed.
#
# Bookworm = glibc 2.36 / libstdc++ 12, so binaries linked here also run on Trixie (newer glibc
# and libstdc++ are backward compatible), but not on anything older than Bookworm.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${SYSROOT_DIR:-$ROOT/.sysroot}"
SYSROOT="$OUT/bookworm-armhf"
DEBS="$OUT/debs"
MIRROR="${DEBIAN_MIRROR:-https://deb.debian.org/debian}"
# The pool only keeps the current point release; older pins fall back to snapshot.debian.org.
SNAPSHOT="https://snapshot.debian.org/archive/debian/20261024T000000Z"

# name  pool path  sha256  (from dists/bookworm/main/binary-armhf/Packages, 2026-10-24)
PINS='
libc6            pool/main/g/glibc/libc6_2.36-9+deb12u14_armhf.deb                      758c68b92654747025b48476a45c315bf16df1321224abdd9be672bfd120be45
libc6-dev        pool/main/g/glibc/libc6-dev_2.36-9+deb12u14_armhf.deb                  ca9486dc7f2dfb25c61c0c93d561c01f8abdf9866bd4c157f0ccd518295bbbb7
linux-libc-dev   pool/main/l/linux/linux-libc-dev_6.1.176-1_armhf.deb                   5641c2f8c252bacc210abb9456deb4344357ccf32ca5821cb8ffe3a9a193a1fb
libstdc++-12-dev pool/main/g/gcc-12/libstdc++-12-dev_12.2.0-14+deb12u1_armhf.deb        77dc865277bf0486ec2fbdebb51dc701cfd90e7702d214f2356270e722e3b6ec
libstdc++6       pool/main/g/gcc-12/libstdc++6_12.2.0-14+deb12u1_armhf.deb              66d53ddeb10a37f22e455611c2c515f9d444e523b55d873ff48bde47267963db
libgcc-12-dev    pool/main/g/gcc-12/libgcc-12-dev_12.2.0-14+deb12u1_armhf.deb           f4af3b766ae50185d8c94933bfb369266aeb4b9288916d2c9789bf0283f3df5a
libgcc-s1        pool/main/g/gcc-12/libgcc-s1_12.2.0-14+deb12u1_armhf.deb               f58562bf01efd6112b914182147ee149e0d4c90869046e31844da8803669eeb3
libasound2       pool/main/a/alsa-lib/libasound2_1.2.8-1+b1_armhf.deb                   97718ff5e552c9a7b9e77e8debbe219b0068a9ab5dcabedd0800c69dd8f0eb96
libasound2-dev   pool/main/a/alsa-lib/libasound2-dev_1.2.8-1+b1_armhf.deb               acbe234a3a0298bf07dcb30034eb17ee92392c894a5f445f18de48ea8bad8c13
'

STAMP="$SYSROOT/.pins"
if [[ -f "$STAMP" ]] && [[ "$(cat "$STAMP")" == "$PINS" ]]; then
  echo "sysroot up to date: $SYSROOT"
  exit 0
fi

mkdir -p "$DEBS"
rm -rf "$SYSROOT" "$OUT/unpack"
mkdir -p "$SYSROOT"

while read -r name path sha; do
  [[ -z "${name:-}" ]] && continue
  deb="$DEBS/$(basename "$path")"
  if [[ ! -f "$deb" ]] || [[ "$(shasum -a 256 "$deb" | cut -d' ' -f1)" != "$sha" ]]; then
    echo "fetch $name"
    curl -fsSL --retry 3 -o "$deb.part" "$MIRROR/$path" \
      || curl -fsSL --retry 3 -o "$deb.part" "$SNAPSHOT/$path"
    got="$(shasum -a 256 "$deb.part" | cut -d' ' -f1)"
    if [[ "$got" != "$sha" ]]; then
      echo "SHA256 mismatch for $name: got $got" >&2; rm -f "$deb.part"; exit 1
    fi
    mv "$deb.part" "$deb"
  fi
  # .deb = ar archive; only data.tar.* is unpacked (control scripts are ignored).
  work="$OUT/unpack/$name"; mkdir -p "$work"
  (cd "$work" && ar x "$deb")
  data="$(ls "$work"/data.tar.* | head -1)"
  tar -xf "$data" -C "$SYSROOT"
done <<< "$PINS"
rm -rf "$OUT/unpack"

# Packages ship absolute symlinks (e.g. libfoo.so -> /lib/arm-linux-gnueabihf/libfoo.so.1);
# rewrite them relative so they resolve inside the sysroot, not on the Mac.
find "$SYSROOT" -type l | while read -r link; do
  target="$(readlink "$link")"
  if [[ "$target" == /* ]]; then
    dir="$(dirname "${link#"$SYSROOT"}")"
    rel="$(python3 -I -c 'import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$target" "$dir")"
    ln -sfn "$rel" "$link"
  fi
done

printf '%s' "$PINS" > "$STAMP"
echo "sysroot ready: $SYSROOT"
