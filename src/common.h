// Helpers shared by bench and live: real-time setup, clocks, system info, stats.
#pragma once

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <sys/resource.h>
#include <sys/utsname.h>
#include <unistd.h>

#ifdef __linux__
  #include <pthread.h>
  #include <sched.h>
  #include <sys/mman.h>
#endif

namespace pna
{
using Clock = std::chrono::steady_clock;

// Apply real-time settings. Returns false (and says why) if any step fails, so a run labelled
// rt=1 is never silently a normal run.
inline bool ApplyRealtime(int rtPrio, int cpu)
{
#ifdef __linux__
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0)
  {
    std::perror("mlockall");
    return false;
  }
  // Pre-fault some stack so the first deep call in the loop doesn't page fault.
  volatile unsigned char stack[256 * 1024];
  for (size_t i = 0; i < sizeof(stack); i += 4096)
    stack[i] = 0;

  if (cpu >= 0)
  {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0)
    {
      std::perror("sched_setaffinity");
      return false;
    }
  }
  sched_param sp{};
  sp.sched_priority = rtPrio;
  if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0)
  {
    std::cerr << "pthread_setschedparam(SCHED_FIFO) failed: run with sudo or set an rtprio limit\n";
    return false;
  }
  return true;
#else
  (void)rtPrio;
  (void)cpu;
  std::cerr << "--rt is only supported on Linux\n";
  return false;
#endif
}

inline bool PinOnly(int cpu)
{
#ifdef __linux__
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  if (sched_setaffinity(0, sizeof(set), &set) != 0)
  {
    std::perror("sched_setaffinity");
    return false;
  }
  return true;
#else
  (void)cpu;
  std::cerr << "--cpu is only supported on Linux\n";
  return false;
#endif
}

inline double ProcessCpuSeconds()
{
  timespec ts{};
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

inline double BootSeconds()
{
#ifdef __linux__
  timespec ts{};
  clock_gettime(CLOCK_BOOTTIME, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
#else
  return -1.0;
#endif
}

inline std::string ReadFirstLine(const char* path)
{
  std::ifstream f(path);
  std::string line;
  if (f && std::getline(f, line))
    return line;
  return "na";
}

// Absolute sleep on the monotonic clock. On Linux, steady_clock is CLOCK_MONOTONIC, so
// clock_nanosleep(TIMER_ABSTIME) avoids the relative-sleep error of sleep_until.
inline void SleepUntil(Clock::time_point target)
{
#ifdef __linux__
  const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(target.time_since_epoch()).count();
  timespec ts{};
  ts.tv_sec = static_cast<time_t>(ns / 1000000000LL);
  ts.tv_nsec = static_cast<long>(ns % 1000000000LL);
  while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) == EINTR)
  {
  }
#else
  std::this_thread::sleep_until(target);
#endif
}

// Kernel release plus preemption model ("PREEMPT_RT", "PREEMPT", or "none"), from uname.
inline std::string KernelInfo()
{
  utsname u{};
  if (uname(&u) != 0)
    return "unknown";
  const std::string v = u.version;
  const char* preempt = v.find("PREEMPT_RT") != std::string::npos ? "PREEMPT_RT"
                        : v.find("PREEMPT") != std::string::npos  ? "PREEMPT"
                                                                  : "none";
  return std::string(u.release) + " " + preempt;
}

// Name of the submodel a slimmable size selects. NAM A2 files hold two: Lite then Full.
inline std::string SlimVariantName(const std::vector<double>& breakpoints, double v)
{
  size_t idx = 0;
  while (idx < breakpoints.size() && v >= breakpoints[idx])
    idx++;
  if (breakpoints.size() == 1)
    return idx == 0 ? "lite" : "full";
  return "s" + std::to_string(idx);
}

inline std::string Hostname()
{
  char buf[256] = {0};
  if (gethostname(buf, sizeof(buf) - 1) != 0)
    return "unknown";
  return buf;
}

inline double Percentile(const std::vector<int64_t>& sorted, double p)
{
  if (sorted.empty())
    return 0.0;
  // Nearest-rank percentile.
  size_t rank = static_cast<size_t>(std::ceil(p / 100.0 * sorted.size()));
  rank = std::clamp<size_t>(rank, 1, sorted.size());
  return static_cast<double>(sorted[rank - 1]);
}

// A field from /proc/self/smaps_rollup in kB (e.g. "Pss"), or -1 where unavailable (macOS).
// Called after the audio loop: it reads a file.
inline long SelfMemKb(const char* field)
{
  std::ifstream f("/proc/self/smaps_rollup");
  const std::string key = std::string(field) + ":";
  std::string line;
  while (f && std::getline(f, line))
    if (line.rfind(key, 0) == 0)
      return std::strtol(line.c_str() + key.size(), nullptr, 10);
  return -1;
}

// Peak resident set size of this process in kB (ru_maxrss is kB on Linux, bytes on macOS).
inline long PeakRssKb()
{
  rusage ru{};
  if (getrusage(RUSAGE_SELF, &ru) != 0)
    return -1;
#ifdef __APPLE__
  return static_cast<long>(ru.ru_maxrss / 1024);
#else
  return static_cast<long>(ru.ru_maxrss);
#endif
}

// True if `path` doesn't exist yet or starts with exactly `header`. Appending rows with new
// columns under an old header would silently misalign every column, so callers refuse instead.
inline bool CsvHeaderMatches(const std::string& path, const std::string& header)
{
  std::ifstream f(path);
  if (!f)
    return true;
  std::string first;
  std::getline(f, first);
  if (first + "\n" == header)
    return true;
  std::cerr << path << " was written by an older version with different columns.\n"
            << "Use a new --summary-out / TAG, or move the old file aside.\n";
  return false;
}

// Configure the A2 two-core pipeline (patches/a2_pipeline.patch) through the environment it reads at
// model construction. Call before nam::get_dsp. spec: "8" (Full), "3" (Lite), "all". Returns false
// (and says why) if this binary wasn't built with the patch, so a run is never mislabeled as pipelined.
inline bool ConfigureA2Pipeline(const std::string& spec, int cpu, int split, int prio, const char* buildFlags)
{
  if (std::string(buildFlags).find("a2_pipeline") == std::string::npos)
  {
    std::cerr << "--pipeline needs a build with patches/a2_pipeline.patch\n";
    return false;
  }
  setenv("NAM_A2_PIPELINE", spec.c_str(), 1);
  setenv("NAM_A2_PIPELINE_CPU", std::to_string(cpu).c_str(), 1);
  setenv("NAM_A2_PIPELINE_PRIO", std::to_string(prio).c_str(), 1);
  if (split > 0)
    setenv("NAM_A2_PIPELINE_SPLIT", std::to_string(split).c_str(), 1);
  return true;
}

// Whether a pipeline spec applies to an A2 submodel ("lite" = 3 channels, "full" = 8). Used for labels.
inline bool PipelineApplies(const std::string& spec, const std::string& variant)
{
  if (spec.empty())
    return false;
  if (spec == "all" || spec == "1")
    return true;
  return (variant == "full" && spec.find('8') != std::string::npos)
         || (variant == "lite" && spec.find('3') != std::string::npos);
}

inline void EnsureParentDir(const std::string& path)
{
  const auto parent = std::filesystem::path(path).parent_path();
  if (!parent.empty())
    std::filesystem::create_directories(parent);
}
} // namespace pna
