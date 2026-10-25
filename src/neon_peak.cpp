// neon_peak: measured FP throughput/latency of the Pi 2's Cortex-A7 for the instruction forms the
// NAM kernels can use. Each variant runs a loop of 8 instructions written in inline asm (so the
// compiler can't rewrite them, e.g. clang splitting VMLA into VMUL+VADD on cortex-a7) and reports
// cycles per instruction at the measured clock. "indep" = 8 independent accumulators (throughput);
// "chain" = 8 instructions on one accumulator (latency). ARMv7 + NEON only.
//
// Usage: neon_peak [--mhz 900] [--out results/neon_peak.csv]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "common.h"

using namespace pna;

#if defined(__ARM_NEON) && !defined(__aarch64__)

namespace
{
constexpr long kIters = 20'000'000;

#define REP8(x) x x x x x x x x
#define LOOP(body)                                                                                                     \
  [] {                                                                                                                 \
    long n = kIters;                                                                                                   \
    asm volatile("vmov.f32 q0, #1.0\n vmov.f32 q1, #0.5\n vmov.i32 q8, #0\n vmov.i32 q9, #0\n"                        \
                 "vmov.i32 q10, #0\n vmov.i32 q11, #0\n vmov.i32 q12, #0\n vmov.i32 q13, #0\n"                         \
                 "vmov.i32 q14, #0\n vmov.i32 q15, #0\n vmov.i32 q4, #0\n vmov.i32 q5, #0\n vmov.f32 q7, #0.5\n"                                                               \
                 "1:\n" body "subs %0, %0, #1\n bne 1b\n"                                                              \
                 : "+r"(n)                                                                                             \
                 :                                                                                                     \
                 : "q0", "q1", "q4", "q5", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "cc");                            \
  }

struct Variant
{
  const char* name;
  int lanes; // MACs (or flops) per instruction
  void (*fn)();
};

const Variant kVariants[] = {
  {"vmla.f32 q by-scalar indep", 4,
   LOOP("vmla.f32 q8, q0, d2[0]\n vmla.f32 q9, q0, d2[0]\n vmla.f32 q10, q0, d2[0]\n vmla.f32 q11, q0, d2[0]\n"
        "vmla.f32 q12, q0, d2[0]\n vmla.f32 q13, q0, d2[0]\n vmla.f32 q14, q0, d2[0]\n vmla.f32 q15, q0, d2[0]\n")},
  {"vmla.f32 q indep", 4,
   LOOP("vmla.f32 q8, q0, q1\n vmla.f32 q9, q0, q1\n vmla.f32 q10, q0, q1\n vmla.f32 q11, q0, q1\n"
        "vmla.f32 q12, q0, q1\n vmla.f32 q13, q0, q1\n vmla.f32 q14, q0, q1\n vmla.f32 q15, q0, q1\n")},
  {"vmla.f32 d indep", 2,
   LOOP("vmla.f32 d16, d0, d2\n vmla.f32 d18, d0, d2\n vmla.f32 d20, d0, d2\n vmla.f32 d22, d0, d2\n"
        "vmla.f32 d24, d0, d2\n vmla.f32 d26, d0, d2\n vmla.f32 d28, d0, d2\n vmla.f32 d30, d0, d2\n")},
  {"vfma.f32 q indep", 4,
   LOOP("vfma.f32 q8, q0, q1\n vfma.f32 q9, q0, q1\n vfma.f32 q10, q0, q1\n vfma.f32 q11, q0, q1\n"
        "vfma.f32 q12, q0, q1\n vfma.f32 q13, q0, q1\n vfma.f32 q14, q0, q1\n vfma.f32 q15, q0, q1\n")},
  {"vfma.f32 d indep", 2,
   LOOP("vfma.f32 d16, d0, d2\n vfma.f32 d18, d0, d2\n vfma.f32 d20, d0, d2\n vfma.f32 d22, d0, d2\n"
        "vfma.f32 d24, d0, d2\n vfma.f32 d26, d0, d2\n vfma.f32 d28, d0, d2\n vfma.f32 d30, d0, d2\n")},
  {"vmul.f32 q indep", 4,
   LOOP("vmul.f32 q8, q0, q1\n vmul.f32 q9, q0, q1\n vmul.f32 q10, q0, q1\n vmul.f32 q11, q0, q1\n"
        "vmul.f32 q12, q0, q1\n vmul.f32 q13, q0, q1\n vmul.f32 q14, q0, q1\n vmul.f32 q15, q0, q1\n")},
  {"vadd.f32 q indep", 4,
   LOOP("vadd.f32 q8, q0, q1\n vadd.f32 q9, q0, q1\n vadd.f32 q10, q0, q1\n vadd.f32 q11, q0, q1\n"
        "vadd.f32 q12, q0, q1\n vadd.f32 q13, q0, q1\n vadd.f32 q14, q0, q1\n vadd.f32 q15, q0, q1\n")},
  {"vmla.f32 s (VFP scalar) indep", 1,
   LOOP("vmla.f32 s16, s28, s29\n vmla.f32 s17, s28, s29\n vmla.f32 s18, s28, s29\n vmla.f32 s19, s28, s29\n"
        "vmla.f32 s20, s28, s29\n vmla.f32 s21, s28, s29\n vmla.f32 s22, s28, s29\n vmla.f32 s23, s28, s29\n")},
  {"vfma.f32 s (VFP scalar) indep", 1,
   LOOP("vfma.f32 s16, s28, s29\n vfma.f32 s17, s28, s29\n vfma.f32 s18, s28, s29\n vfma.f32 s19, s28, s29\n"
        "vfma.f32 s20, s28, s29\n vfma.f32 s21, s28, s29\n vfma.f32 s22, s28, s29\n vfma.f32 s23, s28, s29\n")},
  {"vmla.f32 q chain (latency)", 4, LOOP(REP8("vmla.f32 q8, q0, q1\n"))},
  {"vmla.f32 d chain (latency)", 2, LOOP(REP8("vmla.f32 d16, d0, d2\n"))},
  {"vfma.f32 q chain (latency)", 4, LOOP(REP8("vfma.f32 q8, q0, q1\n"))},
  {"vadd.f32 q chain (latency)", 4, LOOP(REP8("vadd.f32 q8, q8, q1\n"))},
};
} // namespace

int main(int argc, char** argv)
{
  double mhz = 900.0;
  std::string out = "results/neon_peak.csv";
  for (int i = 1; i + 1 < argc; i += 2)
  {
    if (!std::strcmp(argv[i], "--mhz"))
      mhz = std::atof(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--out"))
      out = argv[i + 1];
  }
  if (!ApplyRealtime(80, 3))
    std::fprintf(stderr, "(no --rt: running at normal priority)\n");

  EnsureParentDir(out);
  const bool exists = std::filesystem::exists(out);
  std::ofstream f(out, std::ios::app);
  if (!exists)
    f << "unix_time,host,variant,lanes,instructions,seconds,cycles_per_instr,ops_per_cycle,mhz_assumed\n";
  for (const Variant& v : kVariants)
  {
    v.fn(); // warm up
    const auto t0 = Clock::now();
    v.fn();
    const double s = std::chrono::duration<double>(Clock::now() - t0).count();
    const double instr = 8.0 * kIters;
    const double cpi = s * mhz * 1e6 / instr;
    std::printf("%-32s %6.2f cycles/instr  %5.2f ops/cycle\n", v.name, cpi, v.lanes / cpi);
    f << std::time(nullptr) << "," << Hostname() << ",\"" << v.name << "\"," << v.lanes << "," << instr << "," << s
      << "," << cpi << "," << v.lanes / cpi << "," << mhz << "\n";
  }
  return 0;
}

#else
int main()
{
  std::fprintf(stderr, "neon_peak only runs on 32-bit ARM with NEON\n");
  return 1;
}
#endif
