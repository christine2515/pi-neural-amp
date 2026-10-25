// live: real-time NAM amp through an ALSA interface (e.g. Focusrite Scarlett), plus a
// round-trip latency test. Linux only.
//
// Full duplex on linked capture/playback PCMs: read one period from the guitar input,
// run the model, write one period out. Playback is pre-filled with `periods` periods of
// silence, so the I/O latency is fixed by period size x periods; the model has to finish
// inside one period or the stream underruns (an audible click, counted as an xrun).
//
// Modes:
//   default          guitar in -> model -> all output channels
//   --bypass         guitar in -> output, no model (interface + driver baseline)
//   --latency-test   play a click every 0.5 s and detect it on the input; needs a cable from
//                    an output to the input channel. Reports round trip in frames and ms.
//
// The timed region (model->process) does no allocation, locking, logging or I/O. The
// blocking snd_pcm_readi/writei calls are the audio callback's wait and sit outside it.

#include <alsa/asoundlib.h>
#include <csignal>
#include <cstring>

#include "NAM/dsp.h"
#include "NAM/get_dsp.h"
#include "NAM/slimmable.h"
#include "common.h"

#ifndef BENCH_BUILD_FLAGS
  #define BENCH_BUILD_FLAGS "unknown"
#endif

using namespace pna;

namespace
{
volatile std::sig_atomic_t gStop = 0;
void OnSignal(int) { gStop = 1; }

// Read/write entry points: plain RW access or mmap access (same signatures).
snd_pcm_sframes_t (*gReadi)(snd_pcm_t*, void*, snd_pcm_uframes_t) = snd_pcm_readi;
snd_pcm_sframes_t (*gWritei)(snd_pcm_t*, const void*, snd_pcm_uframes_t) = snd_pcm_writei;

// pss_kb / peak_rss_kb: the whole live process (model, ALSA buffers, libraries), read after the loop.
const std::string kSummaryHeader =
  "unix_time,host,label,device,mode,model,rate,period,periods,buffer_frames,io_buffer_ms,format,"
  "in_channels,out_channels,seconds,periods_processed,xruns,proc_p50_us,proc_p99_us,proc_max_us,"
  "deadline_us,cycle_jitter_p99_us,cycle_jitter_max_us,cpu_pct,rt,rt_prio,cpu,isolated_cpus,kernel,"
  "pss_kb,peak_rss_kb,access,pipeline,build_flags\n";

struct Args
{
  std::string device = "hw:USB";
  std::string model;
  double slim = -1.0;
  int rate = 48000;
  int period = 128;
  int periods = 2;
  int inChannel = 0;
  double durationSec = 30.0;
  bool bypass = false;
  bool latencyTest = false;
  double clickThreshold = 0.05;
  bool rt = false;
  int rtPrio = 80;
  int cpu = -1;
  std::string label = "live";
  std::string summaryOut = "results/live_summary.csv";
  std::string latencyOut = "results/live_latency.csv";
  std::string periodsOut; // per-period CSV; empty = skip
  std::string recordOut;  // stereo WAV: left = dry input, right = model output; empty = skip
  double gateDb = 0.0;    // noise gate open threshold in dBFS (e.g. -50); 0 = off
  double gateHoldMs = 80.0, gateReleaseMs = 100.0;
  bool mmap = false;       // MMAP_INTERLEAVED access instead of RW_INTERLEAVED
  std::string pipeline;    // A2 two-core pipeline spec ("8", "3", "all"); needs the a2_pipeline patch
  int pipelineCpu = -1;    // worker core; default 2 when --cpu 3
  int pipelineSplit = 0;
};

void Usage()
{
  std::cerr << "Usage: live [--model M.nam | --bypass | --latency-test] [options]\n"
               "  --device D           ALSA device (default hw:USB; see `arecord -l`)\n"
               "  --rate R             sample rate (default 48000)\n"
               "  --period N           frames per period (default 128)\n"
               "  --periods K          periods in the buffer (default 2)\n"
               "  --in-channel C       capture channel with the guitar (default 0)\n"
               "  --duration S         seconds to run (default 30; Ctrl-C stops early)\n"
               "  --slim V             slimmable (A2) submodel size 0..1\n"
               "  --bypass             pass input to output with no model\n"
               "  --latency-test       click out, detect on input (loopback cable needed)\n"
               "  --click-threshold T  detection level 0..1 (default 0.05)\n"
               "  --rt / --rt-prio P / --cpu C   same as bench\n"
               "  --label L\n"
               "  --summary-out F.csv  (default results/live_summary.csv)\n"
               "  --latency-out F.csv  (default results/live_latency.csv)\n"
               "  --periods-out F.csv  per-period timings\n"
               "  --record F.wav       save dry input (left, before the gate) and output (right) as a float WAV\n"
               "  --gate DB            noise gate before the model, opens above DB dBFS (e.g. -50); 4 dB hysteresis\n"
               "  --gate-hold MS / --gate-release MS   (defaults 80 / 100)\n"
               "  --mmap               ALSA mmap access (snd_pcm_mmap_readi/writei) instead of read/write\n"
               "  --pipeline C         two-core A2 pipeline (8 = Full, 3 = Lite); adds one period of latency\n"
               "  --pipeline-cpu N     worker core (default 2 with --cpu 3)\n"
               "  --pipeline-split L   first layer run by the worker\n";
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
    if (s == "--device")
      a.device = next("--device");
    else if (s == "--model")
      a.model = next("--model");
    else if (s == "--slim")
      a.slim = std::atof(next("--slim"));
    else if (s == "--rate")
      a.rate = std::atoi(next("--rate"));
    else if (s == "--period")
      a.period = std::atoi(next("--period"));
    else if (s == "--periods")
      a.periods = std::atoi(next("--periods"));
    else if (s == "--in-channel")
      a.inChannel = std::atoi(next("--in-channel"));
    else if (s == "--duration")
      a.durationSec = std::atof(next("--duration"));
    else if (s == "--bypass")
      a.bypass = true;
    else if (s == "--latency-test")
      a.latencyTest = true;
    else if (s == "--click-threshold")
      a.clickThreshold = std::atof(next("--click-threshold"));
    else if (s == "--rt")
      a.rt = true;
    else if (s == "--rt-prio")
      a.rtPrio = std::atoi(next("--rt-prio"));
    else if (s == "--cpu")
      a.cpu = std::atoi(next("--cpu"));
    else if (s == "--label")
      a.label = next("--label");
    else if (s == "--summary-out")
      a.summaryOut = next("--summary-out");
    else if (s == "--latency-out")
      a.latencyOut = next("--latency-out");
    else if (s == "--gate")
      a.gateDb = std::atof(next("--gate"));
    else if (s == "--gate-hold")
      a.gateHoldMs = std::atof(next("--gate-hold"));
    else if (s == "--gate-release")
      a.gateReleaseMs = std::atof(next("--gate-release"));
    else if (s == "--record")
      a.recordOut = next("--record");
    else if (s == "--mmap")
      a.mmap = true;
    else if (s == "--pipeline")
      a.pipeline = next("--pipeline");
    else if (s == "--pipeline-cpu")
      a.pipelineCpu = std::atoi(next("--pipeline-cpu"));
    else if (s == "--pipeline-split")
      a.pipelineSplit = std::atoi(next("--pipeline-split"));
    else if (s == "--periods-out")
      a.periodsOut = next("--periods-out");
    else if (s == "-h" || s == "--help")
      return false;
    else
    {
      std::cerr << "Unknown argument: " << s << "\n";
      return false;
    }
  }
  if (a.model.empty() && !a.bypass && !a.latencyTest)
    return false;
  if (a.period <= 0 || a.periods < 2 || a.durationSec <= 0 || a.slim > 1.0)
    return false;
  if (a.rt && a.cpu < 0)
    a.cpu = 3;
  return true;
}

// One opened PCM with the configuration ALSA actually granted.
struct Pcm
{
  snd_pcm_t* h = nullptr;
  snd_pcm_format_t format = SND_PCM_FORMAT_UNKNOWN;
  unsigned channels = 0;
  snd_pcm_uframes_t period = 0;
  snd_pcm_uframes_t buffer = 0;
  int bytesPerSample = 0;
};

bool Check(int err, const char* what)
{
  if (err < 0)
  {
    std::cerr << what << ": " << snd_strerror(err) << "\n";
    return false;
  }
  return true;
}

bool OpenPcm(Pcm& p, const Args& a, snd_pcm_stream_t stream)
{
  const char* name = stream == SND_PCM_STREAM_CAPTURE ? "capture" : "playback";
  if (!Check(snd_pcm_open(&p.h, a.device.c_str(), stream, 0), name))
    return false;

  snd_pcm_hw_params_t* hw;
  snd_pcm_hw_params_alloca(&hw);
  snd_pcm_hw_params_any(p.h, hw);
  if (!Check(snd_pcm_hw_params_set_rate_resample(p.h, hw, 0), "disable resampling") ||
      !Check(snd_pcm_hw_params_set_access(
               p.h, hw, a.mmap ? SND_PCM_ACCESS_MMAP_INTERLEAVED : SND_PCM_ACCESS_RW_INTERLEAVED),
             "set access"))
    return false;

  // Scarlett 3rd/4th gen use S32_LE; 2nd gen uses S24_3LE.
  for (snd_pcm_format_t f : {SND_PCM_FORMAT_S32_LE, SND_PCM_FORMAT_S24_3LE, SND_PCM_FORMAT_S16_LE})
    if (snd_pcm_hw_params_test_format(p.h, hw, f) == 0)
    {
      p.format = f;
      break;
    }
  if (p.format == SND_PCM_FORMAT_UNKNOWN)
  {
    std::cerr << name << ": no supported sample format (S32_LE, S24_3LE, S16_LE)\n";
    return false;
  }
  snd_pcm_hw_params_set_format(p.h, hw, p.format);
  p.bytesPerSample = snd_pcm_format_physical_width(p.format) / 8;

  // USB interfaces usually only allow their full channel count, so take the minimum allowed.
  snd_pcm_hw_params_get_channels_min(hw, &p.channels);
  if (!Check(snd_pcm_hw_params_set_channels(p.h, hw, p.channels), "set channels"))
    return false;

  unsigned rate = a.rate;
  if (!Check(snd_pcm_hw_params_set_rate(p.h, hw, rate, 0), "set rate"))
    return false;

  p.period = a.period;
  int dir = 0;
  if (!Check(snd_pcm_hw_params_set_period_size_near(p.h, hw, &p.period, &dir), "set period"))
    return false;
  p.buffer = p.period * a.periods;
  if (!Check(snd_pcm_hw_params_set_buffer_size_near(p.h, hw, &p.buffer), "set buffer") ||
      !Check(snd_pcm_hw_params(p.h, hw), "apply hw params"))
    return false;

  snd_pcm_sw_params_t* sw;
  snd_pcm_sw_params_alloca(&sw);
  snd_pcm_sw_params_current(p.h, sw);
  // Start explicitly (both streams together through the link), never on a threshold.
  snd_pcm_sw_params_set_start_threshold(p.h, sw, 0x7fffffff);
  snd_pcm_sw_params_set_avail_min(p.h, sw, p.period);
  return Check(snd_pcm_sw_params(p.h, sw), "apply sw params");
}

inline float ReadSample(const uint8_t* buf, const Pcm& p, size_t frame, unsigned ch)
{
  const uint8_t* s = buf + (frame * p.channels + ch) * p.bytesPerSample;
  switch (p.format)
  {
    case SND_PCM_FORMAT_S32_LE:
    {
      int32_t v;
      std::memcpy(&v, s, 4);
      return v / 2147483648.0f;
    }
    case SND_PCM_FORMAT_S24_3LE:
    {
      int32_t v = s[0] | (s[1] << 8) | (s[2] << 16);
      if (v & 0x800000)
        v -= 0x1000000;
      return v / 8388608.0f;
    }
    default:
    {
      int16_t v;
      std::memcpy(&v, s, 2);
      return v / 32768.0f;
    }
  }
}

inline void WriteSample(uint8_t* buf, const Pcm& p, size_t frame, unsigned ch, float x)
{
  x = std::clamp(x, -1.0f, 0.99999994f);
  uint8_t* d = buf + (frame * p.channels + ch) * p.bytesPerSample;
  switch (p.format)
  {
    case SND_PCM_FORMAT_S32_LE:
    {
      const int32_t v = static_cast<int32_t>(x * 2147483648.0f);
      std::memcpy(d, &v, 4);
      break;
    }
    case SND_PCM_FORMAT_S24_3LE:
    {
      const int32_t v = static_cast<int32_t>(x * 8388608.0f);
      d[0] = v & 0xff;
      d[1] = (v >> 8) & 0xff;
      d[2] = (v >> 16) & 0xff;
      break;
    }
    default:
    {
      const int16_t v = static_cast<int16_t>(x * 32768.0f);
      std::memcpy(d, &v, 2);
      break;
    }
  }
}

// (Re)start the linked streams: prepare, pre-fill playback with silence, start.
// Linked streams start together; if the device can't link them (e.g. ALSA's "null" PCM), start both.
bool gLinked = true;

bool StartStreams(Pcm& cap, Pcm& pb, std::vector<uint8_t>& silence)
{
  snd_pcm_drop(cap.h);
  snd_pcm_drop(pb.h);
  if (!Check(snd_pcm_prepare(cap.h), "prepare capture") || !Check(snd_pcm_prepare(pb.h), "prepare playback"))
    return false;
  for (snd_pcm_uframes_t done = 0; done < pb.buffer; done += pb.period)
    if (!Check(static_cast<int>(gWritei(pb.h, silence.data(), pb.period)), "pre-fill playback"))
      return false;
  if (!gLinked && !Check(snd_pcm_start(pb.h), "start playback"))
    return false;
  return Check(snd_pcm_start(cap.h), "start");
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
  if (!CsvHeaderMatches(a.summaryOut, kSummaryHeader))
    return 2;
  if (a.mmap)
  {
    gReadi = snd_pcm_mmap_readi;
    gWritei = snd_pcm_mmap_writei;
  }
  // The pipeline worker starts when the model is constructed, so configure it first.
  if (!a.pipeline.empty())
  {
    if (a.pipelineCpu < 0 && a.cpu == 3)
      a.pipelineCpu = 2;
    if (!ConfigureA2Pipeline(a.pipeline, a.pipelineCpu, a.pipelineSplit, a.rt ? a.rtPrio : 0, BENCH_BUILD_FLAGS))
      return 2;
  }
  std::signal(SIGINT, OnSignal);
  std::signal(SIGTERM, OnSignal);

  // ---- Model ----
  std::unique_ptr<nam::DSP> model;
  std::string modelName = a.latencyTest ? "latency-test" : a.bypass ? "bypass" : "";
  if (!a.bypass && !a.latencyTest)
  {
    model = nam::get_dsp(std::filesystem::path(a.model));
    if (!model || model->NumInputChannels() != 1 || model->NumOutputChannels() != 1)
    {
      std::cerr << "Failed to load a mono model from " << a.model << "\n";
      return 1;
    }
    if (model->GetExpectedSampleRate() > 0 && std::abs(model->GetExpectedSampleRate() - a.rate) > 0.5)
    {
      std::cerr << "Model expects " << model->GetExpectedSampleRate() << " Hz, --rate is " << a.rate << "\n";
      return 1;
    }
    modelName = std::filesystem::path(a.model).filename().string();
    model->Reset(a.rate, a.period);
    if (a.slim >= 0.0)
    {
      auto* slimmable = dynamic_cast<nam::SlimmableModel*>(model.get());
      if (!slimmable)
      {
        std::cerr << "--slim given but the model is not slimmable\n";
        return 1;
      }
      slimmable->SetSlimmableSize(a.slim);
      model->Reset(a.rate, a.period);
      modelName += "@" + SlimVariantName(slimmable->GetSlimmableSizeBreakpoints(), a.slim);
    }
    const auto at = modelName.rfind('@');
    if (PipelineApplies(a.pipeline, at == std::string::npos ? std::string() : modelName.substr(at + 1)))
      modelName += "+pipe" + (a.pipelineSplit > 0 ? std::to_string(a.pipelineSplit) : std::string());
  }

  // ---- ALSA ----
  Pcm cap, pb;
  if (!OpenPcm(cap, a, SND_PCM_STREAM_CAPTURE) || !OpenPcm(pb, a, SND_PCM_STREAM_PLAYBACK))
    return 1;
  if (cap.period != pb.period || cap.period != static_cast<snd_pcm_uframes_t>(a.period))
  {
    std::cerr << "Device granted period " << cap.period << "/" << pb.period << " instead of " << a.period
              << "; pick a size it supports\n";
    return 1;
  }
  if (a.inChannel < 0 || static_cast<unsigned>(a.inChannel) >= cap.channels)
  {
    std::cerr << "--in-channel " << a.inChannel << " out of range (device has " << cap.channels << ")\n";
    return 1;
  }
  if (const int err = snd_pcm_link(cap.h, pb.h); err < 0)
  {
    std::cerr << "note: can't link capture+playback (" << snd_strerror(err) << "); starting them separately\n";
    gLinked = false;
  }
  const int N = a.period;
  std::fprintf(stderr, "device %s: %d Hz, period %d x %d (%lu frames buffer), %s, in %u ch / out %u ch\n",
               a.device.c_str(), a.rate, N, a.periods, static_cast<unsigned long>(pb.buffer),
               snd_pcm_format_name(cap.format), cap.channels, pb.channels);

  // ---- Preallocate everything the loop touches ----
  std::vector<uint8_t> inRaw(N * cap.channels * cap.bytesPerSample, 0);
  std::vector<uint8_t> outRaw(N * pb.channels * pb.bytesPerSample, 0);
  std::vector<uint8_t> silence(outRaw.size(), 0);
  std::vector<NAM_SAMPLE> inBuf(N, 0), outBuf(N, 0);
  NAM_SAMPLE* inPtrs[1] = {inBuf.data()};
  NAM_SAMPLE* outPtrs[1] = {outBuf.data()};
  const size_t maxPeriods = static_cast<size_t>(std::ceil(a.durationSec * a.rate / N)) + 1;
  std::vector<int64_t> procNs(maxPeriods, 0), cycleNs(maxPeriods, 0);
  // Recording buffer, preallocated for the whole run (interleaved dry/out), filled per good period.
  std::vector<float> rec(a.recordOut.empty() ? 0 : maxPeriods * N * 2, 0.0f);
  std::vector<NAM_SAMPLE> rawIn(N, 0);
  std::vector<uint32_t> xrunAt(1024, 0);
  // Latency test: emission frame of the current click, and measured round trips.
  const int64_t clickEvery = a.rate / 2, clickLen = 16;
  std::vector<int64_t> roundTrips(static_cast<size_t>(a.durationSec * 2) + 4, 0);
  size_t nRoundTrips = 0;

  NoiseGate gate;
  if (a.gateDb < 0.0)
    gate.Configure(static_cast<float>(a.gateDb), 4.0f, static_cast<float>(a.gateHoldMs),
                   static_cast<float>(a.gateReleaseMs), static_cast<float>(a.rate));

  if (a.rt)
  {
    if (!ApplyRealtime(a.rtPrio, a.cpu))
      return 3;
  }
  else if (a.cpu >= 0 && !PinOnly(a.cpu))
    return 3;

  if (!StartStreams(cap, pb, silence))
    return 1;

  // Frame counters since the last (re)start. Playback is ahead by the pre-filled buffer.
  int64_t inFrames = 0, outFrames = static_cast<int64_t>(pb.buffer);
  int64_t lastClickOut = -1;
  bool armed = false;
  size_t nPeriods = 0, nXruns = 0;
  auto prevRead = Clock::now();
  const double cpu0 = ProcessCpuSeconds();
  const auto runStart = Clock::now();

  // ---- Audio loop ----
  while (!gStop && nPeriods < maxPeriods - 1)
  {
    snd_pcm_sframes_t r = gReadi(cap.h, inRaw.data(), N);
    const auto tRead = Clock::now();
    if (r == N)
    {
      for (int i = 0; i < N; i++)
        inBuf[i] = static_cast<NAM_SAMPLE>(ReadSample(inRaw.data(), cap, i, a.inChannel));
      if (!rec.empty())
        std::copy(inBuf.begin(), inBuf.end(), rawIn.begin()); // record the ungated input
      if (!a.latencyTest)
        gate.Process(inBuf.data(), N);

      if (a.latencyTest)
      {
        for (int i = 0; i < N && armed; i++)
          if (std::fabs(inBuf[i]) > a.clickThreshold)
          {
            if (nRoundTrips < roundTrips.size())
              roundTrips[nRoundTrips++] = inFrames + i - lastClickOut;
            armed = false;
          }
        for (int i = 0; i < N; i++)
        {
          const int64_t f = outFrames + i;
          const int64_t phase = f % clickEvery;
          outBuf[i] = phase < clickLen ? 0.5f : 0.0f;
          if (phase == 0)
          {
            lastClickOut = f;
            armed = true;
          }
        }
        procNs[nPeriods] = 0;
      }
      else
      {
        const auto t0 = Clock::now();
        if (model)
          model->process(inPtrs, outPtrs, N);
        else
          std::copy(inBuf.begin(), inBuf.end(), outBuf.begin());
        const auto t1 = Clock::now();
        procNs[nPeriods] = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
      }

      for (int i = 0; i < N; i++)
        for (unsigned ch = 0; ch < pb.channels; ch++)
          WriteSample(outRaw.data(), pb, i, ch, static_cast<float>(outBuf[i]));
      r = gWritei(pb.h, outRaw.data(), N);
    }

    if (r != N)
    {
      // xrun (or short read/write): note it, restart both streams, reset the frame clocks.
      if (nXruns < xrunAt.size())
        xrunAt[nXruns] = static_cast<uint32_t>(nPeriods);
      nXruns++;
      if (!StartStreams(cap, pb, silence))
        break;
      inFrames = 0;
      outFrames = static_cast<int64_t>(pb.buffer);
      armed = false;
      prevRead = Clock::now();
      continue;
    }
    cycleNs[nPeriods] = std::chrono::duration_cast<std::chrono::nanoseconds>(tRead - prevRead).count();
    prevRead = tRead;
    inFrames += N;
    outFrames += N;
    if (!rec.empty())
    {
      float* r = rec.data() + nPeriods * static_cast<size_t>(N) * 2;
      for (int i = 0; i < N; i++)
      {
        r[2 * i] = static_cast<float>(rawIn[i]);
        r[2 * i + 1] = static_cast<float>(outBuf[i]);
      }
    }
    nPeriods++;
  }

  const double wallS = std::chrono::duration<double>(Clock::now() - runStart).count();
  const long pssKb = SelfMemKb("Pss"); // before closing the PCMs, so their buffers still count
  const long peakRssKb = PeakRssKb();
  const double cpuPct = wallS > 0 ? 100.0 * (ProcessCpuSeconds() - cpu0) / wallS : 0.0;
  snd_pcm_drop(cap.h);
  snd_pcm_drop(pb.h);
  if (gLinked)
    snd_pcm_unlink(cap.h);
  snd_pcm_close(cap.h);
  snd_pcm_close(pb.h);

  // ---- Stats (skip the first periods while the streams settle) ----
  const size_t skip = std::min<size_t>(nPeriods, 50);
  std::vector<int64_t> proc(procNs.begin() + skip, procNs.begin() + nPeriods);
  std::vector<int64_t> jitter;
  const int64_t periodNs = static_cast<int64_t>(std::llround(1e9 * N / a.rate));
  for (size_t i = skip; i < nPeriods; i++)
    jitter.push_back(std::llabs(cycleNs[i] - periodNs));
  std::sort(proc.begin(), proc.end());
  std::sort(jitter.begin(), jitter.end());
  const double deadlineUs = periodNs / 1e3;
  const double ioLatencyMs = 1e3 * static_cast<double>(pb.buffer) / a.rate;
  std::string isolated = ReadFirstLine("/sys/devices/system/cpu/isolated");
  if (isolated.empty())
    isolated = "none";

  EnsureParentDir(a.summaryOut);
  {
    const bool exists = std::filesystem::exists(a.summaryOut);
    std::ofstream f(a.summaryOut, std::ios::app);
    if (!exists)
      f << kSummaryHeader;
    const char* mode = a.latencyTest ? "latency_test" : a.bypass ? "bypass" : "model";
    f << std::time(nullptr) << "," << Hostname() << "," << a.label << "," << a.device << "," << mode << ","
      << modelName << "," << a.rate << "," << N << "," << a.periods << "," << pb.buffer << "," << ioLatencyMs
      << "," << snd_pcm_format_name(cap.format) << "," << cap.channels << "," << pb.channels << "," << wallS
      << "," << nPeriods << "," << nXruns << "," << Percentile(proc, 50) / 1e3 << ","
      << Percentile(proc, 99) / 1e3 << "," << (proc.empty() ? 0.0 : proc.back() / 1e3) << "," << deadlineUs
      << "," << Percentile(jitter, 99) / 1e3 << "," << (jitter.empty() ? 0.0 : jitter.back() / 1e3) << ","
      << cpuPct << "," << a.rt << "," << (a.rt ? a.rtPrio : 0) << "," << a.cpu << ",\"" << isolated << "\",\""
      << KernelInfo() << "\"," << pssKb << "," << peakRssKb << "," << (a.mmap ? "mmap" : "rw") << ","
      << (a.pipeline.empty() ? "none" : a.pipeline) << ",\"" << BENCH_BUILD_FLAGS << "\"\n";
  }

  if (!a.recordOut.empty())
  {
    EnsureParentDir(a.recordOut);
    if (!WriteWavFloat(a.recordOut, rec.data(), nPeriods * static_cast<size_t>(N), 2, a.rate))
      std::cerr << "Failed to write " << a.recordOut << "\n";
    else
      std::fprintf(stderr, "recorded %.1f s to %s (left = dry input, right = output)\n",
                   static_cast<double>(nPeriods) * N / a.rate, a.recordOut.c_str());
  }

  if (!a.periodsOut.empty())
  {
    EnsureParentDir(a.periodsOut);
    std::ofstream f(a.periodsOut);
    f << "period_index,proc_ns,cycle_ns,deadline_ns\n";
    for (size_t i = 0; i < nPeriods; i++)
      f << i << "," << procNs[i] << "," << cycleNs[i] << "," << periodNs << "\n";
  }

  if (a.latencyTest)
  {
    EnsureParentDir(a.latencyOut);
    const bool exists = std::filesystem::exists(a.latencyOut);
    std::ofstream f(a.latencyOut, std::ios::app);
    if (!exists)
      f << "unix_time,label,device,rate,period,periods,click_index,roundtrip_frames,roundtrip_ms\n";
    for (size_t i = 0; i < nRoundTrips; i++)
      f << std::time(nullptr) << "," << a.label << "," << a.device << "," << a.rate << "," << N << ","
        << a.periods << "," << i << "," << roundTrips[i] << "," << 1e3 * roundTrips[i] / a.rate << "\n";
    std::vector<int64_t> rts(roundTrips.begin(), roundTrips.begin() + nRoundTrips);
    std::sort(rts.begin(), rts.end());
    if (rts.empty())
      std::printf("latency test: no clicks detected. Check the loopback cable, --in-channel and levels "
                  "(or lower --click-threshold).\n");
    else
      std::printf("latency test: %zu clicks, round trip median %.0f frames (%.2f ms), min %.2f ms, max %.2f ms\n",
                  rts.size(), Percentile(rts, 50), 1e3 * Percentile(rts, 50) / a.rate,
                  1e3 * rts.front() / a.rate, 1e3 * rts.back() / a.rate);
  }

  std::printf("%s period=%d x%d (buffer %.2f ms) periods=%zu xruns=%zu proc p50=%.1fus p99=%.1fus max=%.1fus "
              "deadline=%.1fus jitter p99=%.1fus cpu=%.1f%%%s\n",
              modelName.c_str(), N, a.periods, ioLatencyMs, nPeriods, nXruns, Percentile(proc, 50) / 1e3,
              Percentile(proc, 99) / 1e3, proc.empty() ? 0.0 : proc.back() / 1e3, deadlineUs,
              Percentile(jitter, 99) / 1e3, cpuPct, a.rt ? " rt" : "");
  return 0;
}
