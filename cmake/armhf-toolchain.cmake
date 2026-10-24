# Cross-compile for the Pi 2 (Raspberry Pi OS 32-bit, armhf) from macOS.
# Compiler: Apple clang (or any clang; override with -DCROSS_CLANG=...). Linker: LLVM lld
# (brew install lld). Libraries/headers: Debian Bookworm sysroot from scripts/fetch_sysroot.sh.
# Usage: ./scripts/cross_build.sh  (or cmake -DCMAKE_TOOLCHAIN_FILE=cmake/armhf-toolchain.cmake)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR armv7)   # matches the ARM flag block in CMakeLists.txt

set(_triple armv7a-linux-gnueabihf)
get_filename_component(_repo "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

set(PI_SYSROOT "${_repo}/.sysroot/bookworm-armhf" CACHE PATH "armhf sysroot (scripts/fetch_sysroot.sh)")
if(NOT EXISTS "${PI_SYSROOT}/usr/lib/arm-linux-gnueabihf/libc.so")
  message(FATAL_ERROR "No sysroot at ${PI_SYSROOT}: run scripts/fetch_sysroot.sh")
endif()

find_program(CROSS_CLANG NAMES clang PATHS /usr/bin NO_DEFAULT_PATH)
find_program(CROSS_CLANGXX NAMES clang++ PATHS /usr/bin NO_DEFAULT_PATH)
find_program(CROSS_LLD NAMES ld.lld
  PATHS /opt/homebrew/opt/lld/bin /opt/homebrew/opt/llvm/bin /usr/local/opt/lld/bin /usr/local/opt/llvm/bin)
if(NOT CROSS_LLD)
  message(FATAL_ERROR "ld.lld not found: brew install lld")
endif()

set(CMAKE_C_COMPILER   "${CROSS_CLANG}")
set(CMAKE_CXX_COMPILER "${CROSS_CLANGXX}")
set(CMAKE_C_COMPILER_TARGET   ${_triple})
set(CMAKE_CXX_COMPILER_TARGET ${_triple})
set(CMAKE_SYSROOT "${PI_SYSROOT}")

# GCC 12's crt*.o, libgcc and libstdc++ headers live under usr/lib/gcc/arm-linux-gnueabihf/12;
# clang's Linux driver finds them via the sysroot. Link with lld; libstdc++ (the Pi's), not libc++.
set(CMAKE_EXE_LINKER_FLAGS_INIT    "--ld-path=${CROSS_LLD} -Wl,--build-id")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "--ld-path=${CROSS_LLD}")

# Look for libraries/headers (e.g. ALSA) only in the sysroot, programs only on the host.
set(CMAKE_FIND_ROOT_PATH "${PI_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
set(CMAKE_LIBRARY_ARCHITECTURE arm-linux-gnueabihf)
# No pkg-config for the target on the Mac; keep the host's from leaking in.
set(PKG_CONFIG_EXECUTABLE "" CACHE FILEPATH "" FORCE)
