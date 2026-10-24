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

set(NAM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/NeuralAmpModelerCore")
if(NOT EXISTS "${NAM_DIR}/NAM/dsp.h")
  message(FATAL_ERROR "NeuralAmpModelerCore missing: run `git submodule update --init --recursive`")
endif()

file(GLOB NAM_SOURCES "${NAM_DIR}/NAM/*.cpp" "${NAM_DIR}/NAM/*/*.cpp")
set(WAV_SOURCES "${NAM_DIR}/Dependencies/AudioDSPTools/dsp/wav.cpp")

add_executable(bench src/bench.cpp ${NAM_SOURCES} ${WAV_SOURCES})
target_include_directories(bench SYSTEM PRIVATE
  "${NAM_DIR}"
  "${NAM_DIR}/Dependencies/eigen"
  "${NAM_DIR}/Dependencies/nlohmann"
  "${NAM_DIR}/Dependencies/AudioDSPTools/dsp")
# float I/O buffers: ARMv7 NEON has no double-precision SIMD.
target_compile_definitions(bench PRIVATE NAM_SAMPLE_FLOAT NAM_ENABLE_A2_FAST)

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
target_compile_options(bench PRIVATE ${ARCH_FLAGS} ${OPT_FLAGS} -Wno-deprecated-declarations)

# Record the exact build config in every summary row.
string(REPLACE ";" " " BUILD_FLAGS_STR "${ARCH_FLAGS} ${OPT_FLAGS}")
target_compile_definitions(bench PRIVATE
  BENCH_BUILD_FLAGS="${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION} ${BUILD_FLAGS_STR}")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  find_package(Threads REQUIRED)
  target_link_libraries(bench PRIVATE Threads::Threads)
endif()
