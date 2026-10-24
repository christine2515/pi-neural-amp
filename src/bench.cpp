// bench: offline real-time benchmark for NAM models.
//
// Loads a .nam model and a mono WAV, processes the file in fixed-size blocks, and times
// every model->process() call with a monotonic clock. Writes per-block times to CSV and
// appends one summary row (p50/p99/max, missed deadlines, DSP load, CPU%) to a summary CSV.
//
// Modes:
//   default    blocks back to back (max throughput, warm caches)
//   --paced    sleep until each block's deadline, like an audio callback would
//   --rt       SCHED_FIFO + CPU pinning + mlockall (Linux only)
//
// The timed loop does no allocation, locking, logging or I/O. In --paced mode the only
// syscall is the sleep between blocks, which stands in for waiting on the audio interrupt
// and sits outside the timed region.
//
// In --paced mode bench also records wake-up latency: how late the thread wakes after each
// sleep (actual wake - scheduled deadline). That is the scheduler's contribution, which
// isolcpus / PREEMPT_RT target. A block "misses with wake" when wake + process > deadline.

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <sys/utsname.h>
#include <unistd.h>

#ifdef __linux__
  #include <pthread.h>
  #include <sched.h>
  #include <sys/mman.h>
#endif

#include "NAM/dsp.h"
#include "NAM/get_dsp.h"
#include "wav.h"

#ifndef BENCH_BUILD_FLAGS
  #define BENCH_BUILD_FLAGS "unknown"
#endif

namespace
{
using Clock = std::chrono::steady_clock;

struct Args
{
  std::string model;
  std::string input;
  int block = 128;
  int warmupBlocks = 100;
  double durationSec = 0.0; // 0 = one pass over the file
  bool paced = false;
  bool rt = false;
  int rtPrio = 80;
  int cpu = -1;
  std::string label = "run";
  std::string blocksOut;     // per-block CSV; empty = skip
  std::string summaryOut = "results/summary.csv";
  std::string firstBlockOut; // boot-time CSV; empty = skip
  std::string renderOut;     // processed audio WAV; empty = skip
};

void Usage()
{
  std::cerr << "Usage: bench --model M.nam --input in.wav [options]\n"
               "  --block N            block size in samples (default 128)\n"
               "  --warmup N           blocks to discard before measuring (default 100)\n"
               "  --duration S         loop the input to S seconds of audio (default: one pass)\n"
               "  --paced              sleep to each block deadline (simulates audio callback)\n"
               "  --rt                 SCHED_FIFO, mlockall, pin to --cpu (Linux)\n"
               "  --rt-prio P          SCHED_FIFO priority (default 80)\n"
               "  --cpu C              pin to CPU C (default: 3 with --rt, none otherwise)\n"
               "  --label L            free-text label stored in the summary\n"
               "  --blocks-out F.csv   per-block timings\n"
               "  --summary-out F.csv  summary row is appended here (default results/summary.csv)\n"
               "  --first-block-out F  write CLOCK_BOOTTIME at end of first block (boot timing)\n"
               "  --render F.wav       write processed audio\n";
}

bool ParseArgs(int argc, char** argv, Args& a)
{
  for (int i = 1; i < argc; i++)
  {
    std::string s = argv[i];
    auto next = [&](const char* name) -> const char* {
      if (i + 1 >= argc)
      {
        std::cerr << name << " needs a value\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (s == "--model")
      a.model = next("--model");
    else if (s == "--input")
      a.input = next("--input");
    else if (s == "--block")
      a.block = std::atoi(next("--block"));
    else if (s == "--warmup")
      a.warmupBlocks = std::atoi(next("--warmup"));
    else if (s == "--duration")
      a.durationSec = std::atof(next("--duration"));
    else if (s == "--paced")
      a.paced = true;
    else if (s == "--rt")
      a.rt = true;
    else if (s == "--rt-prio")
      a.rtPrio = std::atoi(next("--rt-prio"));
    else if (s == "--cpu")
      a.cpu = std::atoi(next("--cpu"));
    else if (s == "--label")
      a.label = next("--label");
    else if (s == "--blocks-out")
      a.blocksOut = next("--blocks-out");
    else if (s == "--summary-out")
      a.summaryOut = next("--summary-out");
    else if (s == "--first-block-out")
      a.firstBlockOut = next("--first-block-out");
    else if (s == "--render")
      a.renderOut = next("--render");
    else if (s == "-h" || s == "--help")
      return false;
    else
    {
      std::cerr << "Unknown argument: " << s << "\n";
      return false;
    }
  }
  if (a.model.empty() || a.input.empty() || a.block <= 0 || a.block > 8192 || a.warmupBlocks < 0)
    return false;
  if (a.rt && a.cpu < 0)
    a.cpu = 3;
  return true;
}

// Apply real-time settings. Returns false (and says why) if any step fails, so a run labelled
// rt=1 is never silently a normal run.
bool ApplyRealtime(const Args& a)
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

  if (a.cpu >= 0)
  {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(a.cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0)
    {
      std::perror("sched_setaffinity");
      return false;
    }
  }
  sched_param sp{};
  sp.sched_priority = a.rtPrio;
  if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0)
  {
    std::cerr << "pthread_setschedparam(SCHED_FIFO) failed: run with sudo or set an rtprio limit\n";
    return false;
  }
  return true;
#else
  (void)a;
  std::cerr << "--rt is only supported on Linux\n";
  return false;
#endif
}

bool PinOnly(int cpu)
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

double ProcessCpuSeconds()
{
  timespec ts{};
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

double BootSeconds()
{
#ifdef __linux__
  timespec ts{};
  clock_gettime(CLOCK_BOOTTIME, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
#else
  return -1.0;
#endif
}

std::string ReadFirstLine(const char* path)
{
  std::ifstream f(path);
  std::string line;
  if (f && std::getline(f, line))
    return line;
  return "na";
}

// Absolute sleep on the monotonic clock. On Linux, steady_clock is CLOCK_MONOTONIC, so
// clock_nanosleep(TIMER_ABSTIME) avoids the relative-sleep error of sleep_until.
void SleepUntil(Clock::time_point target)
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
std::string KernelInfo()
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

std::string Hostname()
{
  char buf[256] = {0};
  if (gethostname(buf, sizeof(buf) - 1) != 0)
    return "unknown";
  return buf;
}

double Percentile(const std::vector<int64_t>& sorted, double p)
{
  if (sorted.empty())
    return 0.0;
  // Nearest-rank percentile.
  size_t rank = static_cast<size_t>(std::ceil(p / 100.0 * sorted.size()));
  rank = std::clamp<size_t>(rank, 1, sorted.size());
  return static_cast<double>(sorted[rank - 1]);
}

bool WriteWavFloat32(const std::string& path, const std::vector<float>& samples, uint32_t sr)
{
  std::ofstream out(path, std::ios::binary);
  if (!out)
    return false;
  const uint32_t dataSize = static_cast<uint32_t>(samples.size() * sizeof(float));
  const uint32_t riffSize = 36 + dataSize, fmtSize = 16, byteRate = sr * 4;
  const uint16_t fmt = 3, ch = 1, align = 4, bits = 32;
  out.write("RIFF", 4);
  out.write(reinterpret_cast<const char*>(&riffSize), 4);
  out.write("WAVEfmt ", 8);
  out.write(reinterpret_cast<const char*>(&fmtSize), 4);
  out.write(reinterpret_cast<const char*>(&fmt), 2);
  out.write(reinterpret_cast<const char*>(&ch), 2);
  out.write(reinterpret_cast<const char*>(&sr), 4);
  out.write(reinterpret_cast<const char*>(&byteRate), 4);
  out.write(reinterpret_cast<const char*>(&align), 2);
  out.write(reinterpret_cast<const char*>(&bits), 2);
  out.write("data", 4);
  out.write(reinterpret_cast<const char*>(&dataSize), 4);
  out.write(reinterpret_cast<const char*>(samples.data()), dataSize);
  return out.good();
}

void EnsureParentDir(const std::string& path)
{
  const auto parent = std::filesystem::path(path).parent_path();
  if (!parent.empty())
    std::filesystem::create_directories(parent);
}
} // namespace

int main(int argc, char** argv)
{
  Args a;
  if (!ParseArgs(argc, argv, a))
  {
    Usage();
    return 2;
  }

  // ---- Load model (timed: part of "boot to ready") ----
  const auto tLoad0 = Clock::now();
  std::unique_ptr<nam::DSP> model = nam::get_dsp(std::filesystem::path(a.model));
  const auto tLoad1 = Clock::now();
  if (!model)
  {
    std::cerr << "Failed to load model " << a.model << "\n";
    return 1;
  }
  if (model->NumInputChannels() != 1 || model->NumOutputChannels() != 1)
  {
    std::cerr << "Only mono-in/mono-out models are supported\n";
    return 1;
  }

  // ---- Load input ----
  std::vector<float> audio;
  double fileRate = 0.0;
  const auto rc = dsp::wav::Load(a.input.c_str(), audio, fileRate);
  if (rc != dsp::wav::LoadReturnCode::SUCCESS)
  {
    std::cerr << "Failed to load " << a.input << ": " << dsp::wav::GetMsgForLoadReturnCode(rc) << "\n";
    return 1;
  }
  const double modelRate = model->GetExpectedSampleRate();
  if (modelRate > 0 && std::abs(modelRate - fileRate) > 0.5)
  {
    std::cerr << "Sample rate mismatch: file " << fileRate << " Hz, model expects " << modelRate << " Hz\n";
    return 1;
  }
  const double sr = fileRate;
  const int N = a.block;
  if (audio.size() < static_cast<size_t>(N))
  {
    std::cerr << "Input shorter than one block\n";
    return 1;
  }

  // ---- Reset (allocates internal buffers; prewarms by default) ----
  const auto tReset0 = Clock::now();
  model->Reset(sr, N);
  const auto tReset1 = Clock::now();

  // ---- Preallocate everything the loop touches ----
  const size_t measuredBlocks = a.durationSec > 0
                                  ? static_cast<size_t>(std::ceil(a.durationSec * sr / N))
                                  : audio.size() / N;
  const size_t totalBlocks = measuredBlocks + a.warmupBlocks;
  const size_t fileLen = audio.size();

  std::vector<NAM_SAMPLE> inBuf(N, 0), outBuf(N, 0);
  NAM_SAMPLE* inPtrs[1] = {inBuf.data()};
  NAM_SAMPLE* outPtrs[1] = {outBuf.data()};
  std::vector<int64_t> blockNs(totalBlocks, 0);
  std::vector<int64_t> startNs(totalBlocks, 0);
  std::vector<int64_t> wakeNs(totalBlocks, 0); // lateness of the wake-up before block b (paced only)
  std::vector<float> rendered;
  if (!a.renderOut.empty())
    rendered.assign(totalBlocks * N, 0.0f);

  const std::string governor = ReadFirstLine("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
  std::string isolated = ReadFirstLine("/sys/devices/system/cpu/isolated");
  if (isolated.empty())
    isolated = "none";
  const std::string kernel = KernelInfo();

  if (a.rt)
  {
    if (!ApplyRealtime(a))
      return 3;
  }
  else if (a.cpu >= 0)
  {
    if (!PinOnly(a.cpu))
      return 3;
  }

  const int64_t periodNs = static_cast<int64_t>(std::llround(1e9 * N / sr));
  const double bootAtLoopStart = BootSeconds();
  const double cpu0 = ProcessCpuSeconds();
  const auto loopStart = Clock::now();

  // ---- Timed loop: no allocation, locks, logging or I/O ----
  size_t readPos = 0;
  for (size_t b = 0; b < totalBlocks; b++)
  {
    for (int i = 0; i < N; i++)
    {
      inBuf[i] = static_cast<NAM_SAMPLE>(audio[readPos]);
      if (++readPos == fileLen)
        readPos = 0;
    }

    const auto t0 = Clock::now();
    model->process(inPtrs, outPtrs, N);
    const auto t1 = Clock::now();

    blockNs[b] = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    startNs[b] = std::chrono::duration_cast<std::chrono::nanoseconds>(t0 - loopStart).count();
    if (!rendered.empty())
      for (int i = 0; i < N; i++)
        rendered[b * N + i] = static_cast<float>(outBuf[i]);

    if (a.paced)
    {
      const auto target = loopStart + std::chrono::nanoseconds(periodNs * static_cast<int64_t>(b + 1));
      SleepUntil(target);
      // If the block overran, no sleep happened and this is how far behind schedule we are.
      if (b + 1 < totalBlocks)
        wakeNs[b + 1] = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - target).count();
    }
  }

  const auto loopEnd = Clock::now();
  const double cpu1 = ProcessCpuSeconds();

  // ---- Stats over measured (post-warmup) blocks ----
  std::vector<int64_t> measured(blockNs.begin() + a.warmupBlocks, blockNs.end());
  size_t missed = 0;
  long double sumNs = 0;
  for (int64_t t : measured)
  {
    sumNs += t;
    if (t > periodNs)
      missed++;
  }
  std::vector<int64_t> sorted = measured;
  std::sort(sorted.begin(), sorted.end());

  std::vector<int64_t> wakeSorted(wakeNs.begin() + a.warmupBlocks, wakeNs.end());
  std::sort(wakeSorted.begin(), wakeSorted.end());
  size_t missedWithWake = 0;
  for (size_t b = a.warmupBlocks; b < totalBlocks; b++)
    if (wakeNs[b] + blockNs[b] > periodNs)
      missedWithWake++;
  const double wakeP50 = a.paced ? Percentile(wakeSorted, 50) / 1e3 : 0.0;
  const double wakeP99 = a.paced ? Percentile(wakeSorted, 99) / 1e3 : 0.0;
  const double wakeMax = a.paced && !wakeSorted.empty() ? wakeSorted.back() / 1e3 : 0.0;

  const double deadlineUs = periodNs / 1e3;
  const double p50 = Percentile(sorted, 50) / 1e3;
  const double p90 = Percentile(sorted, 90) / 1e3;
  const double p99 = Percentile(sorted, 99) / 1e3;
  const double p999 = Percentile(sorted, 99.9) / 1e3;
  const double maxUs = sorted.empty() ? 0.0 : sorted.back() / 1e3;
  const double meanUs = measured.empty() ? 0.0 : static_cast<double>(sumNs / measured.size()) / 1e3;
  const double missedPct = measured.empty() ? 0.0 : 100.0 * missed / measured.size();
  // DSP load: compute time as a fraction of the audio time it produced.
  const double dspLoadPct = 100.0 * meanUs / deadlineUs;
  const double wallS = std::chrono::duration<double>(loopEnd - loopStart).count();
  const double cpuPct = wallS > 0 ? 100.0 * (cpu1 - cpu0) / wallS : 0.0;
  const double loadMs = std::chrono::duration<double, std::milli>(tLoad1 - tLoad0).count();
  const double resetMs = std::chrono::duration<double, std::milli>(tReset1 - tReset0).count();
  const std::string modelName = std::filesystem::path(a.model).filename().string();

  // ---- Outputs ----
  if (!a.blocksOut.empty())
  {
    EnsureParentDir(a.blocksOut);
    std::ofstream f(a.blocksOut);
    f << "block_index,warmup,start_ns,elapsed_ns,deadline_ns,wake_ns\n";
    for (size_t b = 0; b < totalBlocks; b++)
      f << b << "," << (b < static_cast<size_t>(a.warmupBlocks) ? 1 : 0) << "," << startNs[b] << "," << blockNs[b]
        << "," << periodNs << "," << wakeNs[b] << "\n";
  }

  {
    EnsureParentDir(a.summaryOut);
    const bool exists = std::filesystem::exists(a.summaryOut);
    std::ofstream f(a.summaryOut, std::ios::app);
    if (!exists)
      f << "unix_time,host,label,model,block,sample_rate,deadline_us,measured_blocks,warmup_blocks,"
           "p50_us,p90_us,p99_us,p999_us,max_us,mean_us,missed,missed_pct,"
           "wake_p50_us,wake_p99_us,wake_max_us,missed_with_wake,dsp_load_pct,cpu_pct,"
           "paced,rt,rt_prio,cpu,isolated_cpus,governor,kernel,load_ms,reset_ms,build_flags\n";
    f << std::time(nullptr) << "," << Hostname() << "," << a.label << "," << modelName << "," << N << "," << sr
      << "," << deadlineUs << "," << measured.size() << "," << a.warmupBlocks << "," << p50 << "," << p90 << ","
      << p99 << "," << p999 << "," << maxUs << "," << meanUs << "," << missed << "," << missedPct << ","
      << wakeP50 << "," << wakeP99 << "," << wakeMax << "," << missedWithWake << "," << dspLoadPct << ","
      << cpuPct << "," << a.paced << "," << a.rt << "," << (a.rt ? a.rtPrio : 0) << "," << a.cpu << ",\""
      << isolated << "\"," << governor << ",\"" << kernel << "\"," << loadMs << "," << resetMs << ",\""
      << BENCH_BUILD_FLAGS << "\"\n";
  }

  if (!a.firstBlockOut.empty() && bootAtLoopStart >= 0 && totalBlocks > 0)
  {
    // CLOCK_BOOTTIME counts from kernel start (firmware/bootloader time is not included).
    const double firstBlockBootS = bootAtLoopStart + (startNs[0] + blockNs[0]) / 1e9;
    EnsureParentDir(a.firstBlockOut);
    const bool exists = std::filesystem::exists(a.firstBlockOut);
    std::ofstream f(a.firstBlockOut, std::ios::app);
    if (!exists)
      f << "unix_time,model,block,first_block_boottime_s,model_load_ms,reset_ms\n";
    f << std::time(nullptr) << "," << modelName << "," << N << "," << firstBlockBootS << "," << loadMs << ","
      << resetMs << "\n";
  }

  if (!a.renderOut.empty())
  {
    EnsureParentDir(a.renderOut);
    if (!WriteWavFloat32(a.renderOut, rendered, static_cast<uint32_t>(sr)))
      std::cerr << "Failed to write " << a.renderOut << "\n";
  }

  std::printf("%s block=%d deadline=%.1fus p50=%.1fus p99=%.1fus max=%.1fus missed=%zu/%zu (%.3f%%) load=%.1f%% "
              "cpu=%.1f%%%s%s\n",
              modelName.c_str(), N, deadlineUs, p50, p99, maxUs, missed, measured.size(), missedPct, dspLoadPct,
              cpuPct, a.paced ? " paced" : "", a.rt ? " rt" : "");
  if (a.paced)
    std::printf("  wake p50=%.1fus p99=%.1fus max=%.1fus missed_with_wake=%zu\n", wakeP50, wakeP99, wakeMax,
                missedWithWake);
  return 0;
}
