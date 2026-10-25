# 08: Build and tooling

## Cross-compiling

- A Mac build makes a **macOS Mach-O** program for Apple Silicon. The Pi needs a **Linux ELF** for
  32-bit ARM. Different instruction set and OS, so it can't run there.
- **Three pieces** (see `../cmake/armhf-toolchain.cmake`, `../scripts/fetch_sysroot.sh`):
  1. **Compiler:** Apple clang with `--target=armv7a-linux-gnueabihf` (arch-OS-ABI triple; `hf` =
     hard-float) plus `-mcpu=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=hard`.
  2. **Sysroot:** the Pi's headers and libraries (glibc 2.36, libstdc++ 12, ALSA) from 9 pinned,
     SHA256-checked Debian Bookworm `.deb` files, unpacked only as data.
  3. **Linker:** LLVM `lld` (Apple's linker only makes Mach-O).
- The binary records the minimum library versions it needs (glibc 2.34, GLIBCXX 3.4.29), so it runs on
  Bookworm or newer (we have trixie).
- **Payoff:** ~12 s per build on the Mac vs 12.4 min natively on the Pi (the earlier "an hour" guess was
  wrong). Variants (fast-math, patches) take seconds.
- **Caveat:** compilers matter. Clang and GCC builds of the same code differed by up to 26%, so compare
  like with like. `build_flags` records which compiler built each row.

## CMake lessons

- **OBJECT vs STATIC libraries.** NAM registers its model parsers from static initializers. A STATIC
  library let the linker drop those object files, so `bench` crashed with "No config parser registered
  for architecture: SlimmableContainer". An OBJECT library keeps every file.
- **Experiment knobs** recorded in `build_flags`: `BENCH_A2_FASTMATH`, `BENCH_A2_RING_MODE`,
  `BENCH_NAM_PATCHES`, `CMAKE_CXX_FLAGS`.
- **Never edit vendored code.** Patches (`../patches/*.patch`) are applied to a **build-dir copy** of
  NAM/, so the submodule stays pristine and every variant is reproducible.

## The remote experiment loop

`../scripts/remote_sweep.sh` (run from the Mac):
1. Deploy `build-armhf-<variant>` to the Pi.
2. Start `sweep.sh` detached (`setsid nohup`), so a dropped SSH session doesn't kill it.
3. Wait with **one** long-lived SSH session. A new handshake every poll costs the Pi CPU.
4. Pull results into `results-pi/`; analyze on the Mac with `RESULTS_DIR=results-pi`.

**Pitfall:** `pgrep -f scripts/sweep.sh` matched its **own** shell's command line, so the wait loop
never ended. Fix: `pgrep -f "[s]cripts/sweep.sh"` (the regex still matches "scripts", but the literal
text "[s]cripts" doesn't).

## rsync and SSH pitfalls

- **rsync overwrites same-named files.** Mac test rows in `results/summary.csv` replaced the Pi's own
  run. The `host` column exposed it. Fix: always `--exclude results`, and pull Pi data into a separate
  `results-pi/`.
- **CSV schema changes:** adding columns under an old header silently misaligns data, so `bench` refuses
  to append to an old-format file.
- **SSH:** a `~/.ssh/config` alias (`Host pi`, `AddressFamily inet`) after IPv6 "connection reset"
  errors. Key auth with `id_ed25519`. A loose SD card looked like an SSH failure.
- `sudo` on the Pi needs a password, so Claude can't change system settings. Those steps stay with the user.

## Reproducibility checklist

- Every result row records: host, kernel, governor, compiler + flags + patches, isolated CPUs, PSS.
- Numbers only come from CSVs produced by scripts (project rule).
- Keep negative results and controls in the report.

## Key takeaways

> - Cross-compile = compiler target + sysroot + ELF linker; compare like with like.
> - OBJECT libraries keep self-registering code; patches keep vendored code pristine.
> - Detach long runs, poll gently, never rsync over results, and record provenance in every row.

## Check yourself

1. Why can't the Mac's `build/bench` run on the Pi?
2. What went wrong with the STATIC library, and why did OBJECT fix it?
3. Why does `pgrep -f "[s]cripts/sweep.sh"` avoid matching itself?
4. How did the `host` column save the dataset?
