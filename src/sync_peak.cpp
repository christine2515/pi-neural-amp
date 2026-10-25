// sync_peak: what does it cost to hand work between cores on the Pi 2's Cortex-A7?
//
// Question (docs/low-latency-plan.md, Q3): a data-parallel split of one block across cores needs a
// barrier per layer (23+ per block). Is a barrier cheap enough to do that at small blocks, where the
// two-stage pipeline (one handoff per block, +1 block of latency) is the alternative?
//
// Tests (all threads pinned, SCHED_FIFO 80 with --rt; main thread on --cpu, helpers on --helpers):
//   pingpong   two threads bounce a counter: A stores i, B waits for i and stores i, A waits.
//              one-way handoff = round trip / 2. Spin styles: "yield" (the pipeline patch's loop),
//              "tight" (no hint), "wfe" (wait-for-event; writer issues dsb + sev).
//   barrier    P = 2/3/4 threads, each with its own counter on its own cache line ("flag barrier",
//              what patches/a2_frame_split.patch uses): store own step (release), wait until every
//              other counter has reached it (acquire). Time per barrier with no work in between.
//   barrier+payload  per iteration: every thread writes B bytes of its own slice of a shared buffer,
//              barrier, reads the whole buffer (all P slices, one float per 16 B), barrier. Like a
//              layer whose next layer needs the other cores' frames. P = 1 is the local-only
//              baseline; (P row) - (P=1 row) - 2 x (no-payload barrier) = cache-to-cache migration.
//   central    P threads, one shared counter with atomic fetch_add (ldrex/strex) + generation flag,
//              for comparison with the flag barrier.
//
// Timing: steady_clock around a batch of iterations (mean), plus per-iteration timestamps on the
// main thread for p50/p99/max (the Pi 2's arch timer ticks at 19.2 MHz: 52 ns resolution).
//
// Usage: sync_peak [--rt] [--cpu 3] [--helpers 2,1,0] [--iters N] [--out results/sync_peak.csv]

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <pthread.h>

#include "common.h"

using namespace pna;

namespace
{
struct alignas(64) Padded
{
  std::atomic<uint32_t> v{0};
  char pad[64 - sizeof(std::atomic<uint32_t>)];
};

enum class Spin
{
  Yield,
  Tight,
  Wfe
};
const char* SpinName(Spin s)
{
  return s == Spin::Yield ? "yield" : s == Spin::Tight ? "tight" : "wfe";
}

inline void Relax(Spin s)
{
#if defined(__arm__) || defined(__aarch64__)
  if (s == Spin::Yield)
    asm volatile("yield" ::: "memory");
  else if (s == Spin::Wfe)
    asm volatile("wfe" ::: "memory");
#else
  (void)s;
#endif
}
inline void Notify(Spin s)
{
#if defined(__arm__) || defined(__aarch64__)
  if (s == Spin::Wfe)
    asm volatile("dsb ish\n sev" ::: "memory");
#else
  (void)s;
#endif
}

struct Opts
{
  bool rt = false;
  int cpu = 3;
  std::vector<int> helpers{2, 1, 0};
  long iters = 200000;
  std::string out = "results/sync_peak.csv";
};

bool SetupThread(const Opts& o, int cpu)
{
#ifdef __linux__
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0)
    return false;
  if (o.rt)
  {
    sched_param sp{};
    sp.sched_priority = 80;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0)
      return false;
  }
#else
  (void)o;
  (void)cpu;
#endif
  return true;
}

// Start fn(i) on helper threads 1..n-1 (pinned), run fn(0) here, join. Small explicit stacks so that
// mlockall doesn't pin 8 MB per thread.
template <class F>
void RunThreads(const Opts& o, int n, F fn)
{
  struct Arg
  {
    const Opts* o;
    int idx;
    F* fn;
  };
  std::vector<pthread_t> th(n);
  std::vector<Arg> args(n);
  for (int i = 1; i < n; i++)
  {
    args[i] = {&o, i, &fn};
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 256 * 1024);
#ifdef __linux__
    // Set core and policy at creation: a thread that inherits the creator's core and SCHED_FIFO
    // priority can never run to move itself while the creator spins on that core.
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(o.helpers[i - 1], &set);
    pthread_attr_setaffinity_np(&attr, sizeof(set), &set);
    if (o.rt)
    {
      sched_param sp{};
      sp.sched_priority = 80;
      pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);
      pthread_attr_setschedpolicy(&attr, SCHED_FIFO);
      pthread_attr_setschedparam(&attr, &sp);
    }
#endif
    pthread_create(
      &th[i], &attr,
      [](void* p) -> void* {
        Arg* a = static_cast<Arg*>(p);
        if (!SetupThread(*a->o, a->o->helpers[a->idx - 1]))
          std::fprintf(stderr, "helper %d: affinity/priority failed\n", a->idx);
        (*a->fn)(a->idx);
        return nullptr;
      },
      &args[i]);
    pthread_attr_destroy(&attr);
  }
  fn(0);
  for (int i = 1; i < n; i++)
    pthread_join(th[i], nullptr);
  std::fflush(stdout);
}

struct Result
{
  double mean_ns = 0, p50 = 0, p99 = 0, max = 0;
};

Result Stats(std::vector<int64_t>& per, double total_ns, long iters, double scale)
{
  Result r;
  r.mean_ns = total_ns / iters * scale;
  std::sort(per.begin(), per.end());
  r.p50 = Percentile(per, 50) * scale;
  r.p99 = Percentile(per, 99) * scale;
  r.max = per.empty() ? 0 : per.back() * scale;
  return r;
}

int64_t NowNs()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}

// ---- ping-pong ----
Result PingPong(const Opts& o, Spin spin)
{
  Padded a, b;
  const long n = o.iters;
  std::vector<int64_t> per(n);
  double total = 0;
  RunThreads(o, 2, [&](int idx) {
    if (idx == 1)
    {
      for (uint32_t i = 1; i <= static_cast<uint32_t>(n); i++)
      {
        while (a.v.load(std::memory_order_acquire) != i)
          Relax(spin);
        b.v.store(i, std::memory_order_release);
        Notify(spin);
      }
      return;
    }
    // Let the helper reach its spin loop.
    for (int k = 0; k < 2000000; k++)
      asm volatile("" ::: "memory");
    const int64_t t0 = NowNs();
    int64_t prev = t0;
    for (uint32_t i = 1; i <= static_cast<uint32_t>(n); i++)
    {
      a.v.store(i, std::memory_order_release);
      Notify(spin);
      while (b.v.load(std::memory_order_acquire) != i)
        Relax(spin);
      const int64_t t = NowNs();
      per[i - 1] = t - prev;
      prev = t;
    }
    total = static_cast<double>(prev - t0);
  });
  // Report one-way handoff (= round trip / 2). The per-iteration numbers include one clock read.
  return Stats(per, total, n, 0.5);
}

// ---- flag barrier, optional payload ----
// Each thread owns a step counter; barrier = publish own step, wait for all others to reach it.
Result FlagBarrier(const Opts& o, int P, Spin spin, int payload_bytes)
{
  std::vector<Padded> step(P);
  const long n = o.iters;
  std::vector<int64_t> per(n);
  double total = 0;
  const int slice = std::max(payload_bytes, 64);
  // Shared buffer: P slices, each starting on its own cache line.
  std::vector<float> buf(static_cast<size_t>(slice / 4) * P + 16, 0.0f);
  float* base = reinterpret_cast<float*>((reinterpret_cast<uintptr_t>(buf.data()) + 63) & ~uintptr_t(63));
  std::atomic<float> sink{0};
  RunThreads(o, P, [&](int idx) {
    float acc = 0;
    float* mine = base + static_cast<size_t>(idx) * (slice / 4);
    const int nfl = payload_bytes / 4;
    uint32_t s = 0;
    auto barrier = [&] {
      ++s;
      step[idx].v.store(s, std::memory_order_release);
      Notify(spin);
      for (int u = 0; u < P; u++)
        if (u != idx)
          while (static_cast<int32_t>(step[u].v.load(std::memory_order_acquire) - s) < 0)
            Relax(spin);
    };
    // Warm-up / start line.
    barrier();
    const int64_t t0 = NowNs();
    int64_t prev = t0;
    for (uint32_t i = 0; i < static_cast<uint32_t>(n); i++)
    {
      if (nfl > 0)
      {
        for (int k = 0; k < nfl; k++)
          mine[k] = static_cast<float>(i + k);
      }
      barrier();
      if (nfl > 0)
      {
        // Read every slice (own and others'), as the next layer would.
        for (int u = 0; u < P; u++)
        {
          const float* s = base + static_cast<size_t>(u) * (slice / 4);
          for (int k = 0; k < nfl; k += 4)
            acc += s[k];
        }
        // Second barrier: nobody overwrites its slice until everyone has read it (a real layer
        // sequence gets this ordering from the next layer's barrier).
        barrier();
      }
      if (idx == 0)
      {
        const int64_t t = NowNs();
        per[i] = t - prev;
        prev = t;
      }
    }
    if (idx == 0)
      total = static_cast<double>(prev - t0);
    sink.store(acc, std::memory_order_relaxed);
  });
  (void)sink.load();
  return Stats(per, total, n, 1.0);
}

// ---- centralized counter barrier (sense reversal) ----
Result CentralBarrier(const Opts& o, int P, Spin spin)
{
  alignas(64) static std::atomic<int> count{0};
  alignas(64) static std::atomic<uint32_t> gen{0};
  count.store(0);
  gen.store(0);
  const long n = o.iters;
  std::vector<int64_t> per(n);
  double total = 0;
  RunThreads(o, P, [&](int idx) {
    auto barrier = [&] {
      const uint32_t g = gen.load(std::memory_order_acquire);
      if (count.fetch_add(1, std::memory_order_acq_rel) == P - 1)
      {
        count.store(0, std::memory_order_relaxed);
        gen.store(g + 1, std::memory_order_release);
        Notify(spin);
      }
      else
        while (gen.load(std::memory_order_acquire) == g)
          Relax(spin);
    };
    barrier();
    const int64_t t0 = NowNs();
    int64_t prev = t0;
    for (long i = 0; i < n; i++)
    {
      barrier();
      if (idx == 0)
      {
        const int64_t t = NowNs();
        per[i] = t - prev;
        prev = t;
      }
    }
    if (idx == 0)
      total = static_cast<double>(prev - t0);
  });
  return Stats(per, total, n, 1.0);
}

// Cost of one clock read (subtract mentally from per-iteration percentiles).
double ClockCostNs()
{
  const int n = 200000;
  const int64_t t0 = NowNs();
  int64_t x = 0;
  for (int i = 0; i < n; i++)
    x += NowNs();
  const int64_t t1 = NowNs();
  if (x == 42)
    std::puts("");
  return static_cast<double>(t1 - t0) / n;
}

std::vector<int> ParseList(const char* s)
{
  std::vector<int> v;
  while (*s)
  {
    v.push_back(std::atoi(s));
    while (*s && *s != ',')
      s++;
    if (*s == ',')
      s++;
  }
  return v;
}
} // namespace

int main(int argc, char** argv)
{
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  Opts o;
  for (int i = 1; i < argc; i++)
  {
    const std::string s = argv[i];
    auto next = [&] { return i + 1 < argc ? argv[++i] : (std::fprintf(stderr, "%s needs a value\n", s.c_str()), std::exit(2), ""); };
    if (s == "--rt")
      o.rt = true;
    else if (s == "--cpu")
      o.cpu = std::atoi(next());
    else if (s == "--helpers")
      o.helpers = ParseList(next());
    else if (s == "--iters")
      o.iters = std::atol(next());
    else if (s == "--out")
      o.out = next();
    else
    {
      std::fprintf(stderr, "usage: sync_peak [--rt] [--cpu 3] [--helpers 2,1,0] [--iters N] [--out F.csv]\n");
      return 2;
    }
  }
  if (o.rt)
  {
    if (!ApplyRealtime(80, o.cpu))
      return 3;
  }
  else
    PinOnly(o.cpu);

  EnsureParentDir(o.out);
  const bool exists = std::filesystem::exists(o.out);
  std::ofstream f(o.out, std::ios::app);
  if (!exists)
    f << "unix_time,host,test,threads,cpus,spin,payload_bytes,iterations,mean_ns,p50_ns,p99_ns,max_ns,clock_read_ns,rt\n";
  const double clk = ClockCostNs();
  std::printf("clock read: %.0f ns\n", clk);

  auto cpus = [&](int P) {
    std::string c = std::to_string(o.cpu);
    for (int i = 0; i < P - 1 && i < static_cast<int>(o.helpers.size()); i++)
      c += "+" + std::to_string(o.helpers[i]);
    return c;
  };
  auto emit = [&](const char* test, int P, Spin s, int payload, const Result& r) {
    std::printf("%-16s P=%d cpus=%-8s spin=%-5s payload=%5d  mean %7.0f ns  p50 %7.0f  p99 %7.0f  max %8.0f\n", test,
                P, cpus(P).c_str(), SpinName(s), payload, r.mean_ns, r.p50, r.p99, r.max);
    f << std::time(nullptr) << "," << Hostname() << "," << test << "," << P << "," << cpus(P) << "," << SpinName(s)
      << "," << payload << "," << o.iters << "," << r.mean_ns << "," << r.p50 << "," << r.p99 << "," << r.max << ","
      << clk << "," << o.rt << "\n";
    f.flush();
  };

  const int maxP = std::min(4, 1 + static_cast<int>(o.helpers.size()));
  // Ping-pong against each helper core (all four A7 cores share one L2; distances should be equal).
  for (size_t h = 0; h < o.helpers.size(); h++)
  {
    Opts oh = o;
    oh.helpers = {o.helpers[h]};
    for (Spin s : {Spin::Yield, Spin::Tight, Spin::Wfe})
    {
      const Result r = PingPong(oh, s);
      std::printf("pingpong 3<->%d ", o.helpers[h]);
      f << std::time(nullptr) << "," << Hostname() << ",pingpong_oneway,2," << o.cpu << "+" << o.helpers[h] << ","
        << SpinName(s) << ",0," << o.iters << "," << r.mean_ns << "," << r.p50 << "," << r.p99 << "," << r.max << ","
        << clk << "," << o.rt << "\n";
      std::printf("spin=%-5s one-way mean %6.0f ns  p50 %6.0f  p99 %6.0f  max %8.0f\n", SpinName(s), r.mean_ns, r.p50,
                  r.p99, r.max);
    }
  }
  for (int P = 2; P <= maxP; P++)
    for (Spin s : {Spin::Yield, Spin::Wfe})
      emit("barrier", P, s, 0, FlagBarrier(o, P, s, 0));
  for (int P = 2; P <= maxP; P++)
    emit("central", P, Spin::Yield, 0, CentralBarrier(o, P, Spin::Yield));
  // Payloads: one thread's share of a layer's output. Full (8 ch x 4 B) at 128 frames split 2 ways =
  // 2 KiB; at 16 frames split 2 ways = 256 B. Lite (3 ch) is 3/8 of that.
  for (int P = 1; P <= maxP; P++)
    for (int bytes : {128, 256, 512, 1024, 2048})
      emit("barrier_payload", P, Spin::Yield, bytes, FlagBarrier(o, P, Spin::Yield, bytes));
  return 0;
}
