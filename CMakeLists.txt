cmake_minimum_required(VERSION 3.16)
project(pi_neural_amp CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Target CPU for armhf builds. Original Pi 2 (BCM2836) is cortex-a7; Pi 2 v1.2 (BCM2837) is cortex-a53.
# Check `grep -m1 "CPU part" /proc/cpuinfo`: 0xc07 = A7, 0xd03 = A53.
set(PI_CPU "cortex-a7" CACHE STRING "ARM -mcpu for armhf builds")
option(BENCH_FAST_MATH "Compile with -ffast-math" OFF)
option(BENCH_A2_FAST "Use NAM's hand-optimized A2 WaveNet path (OFF = generic WaveNet)" ON)

set(NAM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/NeuralAmpModelerCore")
if(NOT EXISTS "${NAM_DIR}/NAM/dsp.h")
  message(FATAL_ERROR "NeuralAmpModelerCore missing: run `git submodule update --init --recursive`")
endif()

# Experiment knobs that only touch NAM's A2 fast path (NAM/wavenet/a2_fast.cpp).
option(BENCH_A2_FASTMATH "Compile only a2_fast.cpp with -ffast-math (lets ARMv7 NEON vectorize Lite)" OFF)
set(BENCH_A2_RING_MODE "" CACHE STRING "NAM_A2_RING_MODE for a2_fast.cpp (0 or 1; empty = NAM default)")
# Patches for experiments on vendored NAM code, applied to a copy in the build dir (the submodule
# itself is never modified). Semicolon-separated paths, e.g. -DBENCH_NAM_PATCHES=patches/x.patch
set(BENCH_NAM_PATCHES "" CACHE STRING "Patches to apply to a build-dir copy of NAM/")

set(NAM_SRC_ROOT "${NAM_DIR}")
if(BENCH_NAM_PATCHES)
  set(NAM_SRC_ROOT "${CMAKE_BINARY_DIR}/nam_patched")
  file(REMOVE_RECURSE "${NAM_SRC_ROOT}")
  file(COPY "${NAM_DIR}/NAM" DESTINATION "${NAM_SRC_ROOT}")
  foreach(_patch IN LISTS BENCH_NAM_PATCHES)
    get_filename_component(_patch_abs "${_patch}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    execute_process(COMMAND patch -p1 --forward --silent -d "${NAM_SRC_ROOT}" -i "${_patch_abs}"
                    RESULT_VARIABLE _patch_rc)
    if(NOT _patch_rc EQUAL 0)
      message(FATAL_ERROR "Patch failed: ${_patch}")
    endif()
    # Re-run CMake when a patch changes.
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_patch_abs}")
  endforeach()
endif()

file(GLOB NAM_SOURCES "${NAM_SRC_ROOT}/NAM/*.cpp" "${NAM_SRC_ROOT}/NAM/*/*.cpp")
set(A2_SOURCE "${NAM_SRC_ROOT}/NAM/wavenet/a2_fast.cpp")
set(A2_FLAGS "")
if(BENCH_A2_FASTMATH)
  list(APPEND A2_FLAGS -ffast-math)
endif()
if(NOT BENCH_A2_RING_MODE STREQUAL "")
  list(APPEND A2_FLAGS -DNAM_A2_RING_MODE=${BENCH_A2_RING_MODE})
endif()
if(A2_FLAGS)
  set_source_files_properties("${A2_SOURCE}" PROPERTIES COMPILE_OPTIONS "${A2_FLAGS}")
endif()
set(WAV_SOURCES "${NAM_DIR}/Dependencies/AudioDSPTools/dsp/wav.cpp")

set(ARCH_FLAGS "")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(armv7|armhf|arm$)")
  set(ARCH_FLAGS -mcpu=${PI_CPU} -mfpu=neon-vfpv4 -mfloat-abi=hard)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)" AND NOT APPLE)
  set(ARCH_FLAGS -mcpu=native)
endif()
set(OPT_FLAGS -O3)
if(BENCH_FAST_MATH)
  list(APPEND OPT_FLAGS -ffast-math)
endif()

# NAM Core compiled once (the slow part on a Pi) and shared by bench and live. OBJECT, not
# STATIC: NAM registers model parsers from static initializers the linker would otherwise drop.
add_library(namcore OBJECT ${NAM_SOURCES} ${WAV_SOURCES})
target_include_directories(namcore SYSTEM PUBLIC
  "${NAM_SRC_ROOT}"
  "${NAM_DIR}/Dependencies/eigen"
  "${NAM_DIR}/Dependencies/nlohmann"
  "${NAM_DIR}/Dependencies/AudioDSPTools/dsp")
# float I/O buffers: ARMv7 NEON has no double-precision SIMD.
target_compile_definitions(namcore PUBLIC NAM_SAMPLE_FLOAT)
if(BENCH_A2_FAST)
  target_compile_definitions(namcore PUBLIC NAM_ENABLE_A2_FAST)
endif()
target_compile_options(namcore PUBLIC ${ARCH_FLAGS} ${OPT_FLAGS} -Wno-deprecated-declarations)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  find_package(Threads REQUIRED)
  target_link_libraries(namcore PUBLIC Threads::Threads)
endif()

add_executable(bench src/bench.cpp)
target_link_libraries(bench PRIVATE namcore)

# Record the exact build config in every summary row.
set(_patch_names "")
foreach(_patch IN LISTS BENCH_NAM_PATCHES)
  get_filename_component(_n "${_patch}" NAME_WE)
  list(APPEND _patch_names "${_n}")
endforeach()
string(REPLACE ";" "+" _patch_str "${_patch_names}")
string(REPLACE ";" " " _a2_str "${A2_FLAGS}")
string(REPLACE ";" " " BUILD_FLAGS_STR
  "${ARCH_FLAGS} ${OPT_FLAGS} a2_fast=${BENCH_A2_FAST} a2_flags=[${_a2_str}] patches=[${_patch_str}] cxxflags=[${CMAKE_CXX_FLAGS}]")
target_compile_definitions(bench PRIVATE
  BENCH_BUILD_FLAGS="${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION} ${BUILD_FLAGS_STR}")

# live: ALSA client (Linux only; needs libasound2-dev).
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  find_package(ALSA)
  if(ALSA_FOUND)
    add_executable(live src/live.cpp)
    target_link_libraries(live PRIVATE namcore ALSA::ALSA)
  else()
    message(WARNING "ALSA not found: skipping live (sudo apt install libasound2-dev)")
  endif()
endif()

# neon_peak: measured NEON/VFP throughput and latency (ARMv7 builds only).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(armv7|armhf|arm$)")
  add_executable(neon_peak src/neon_peak.cpp)
  target_compile_options(neon_peak PRIVATE ${ARCH_FLAGS} -O2)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(neon_peak PRIVATE Threads::Threads)
  endif()
endif()
