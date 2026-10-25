// int_peak: measured integer / fixed-point throughput and latency of the Pi 2's Cortex-A7, the
// integer counterpart of neon_peak. Question: can int16/int32 arithmetic (NEON or the scalar
// integer pipeline) beat the ~1 float lane-op per cycle that every FP form reaches (neon_peak.csv)?
// Also times software float emulation (libgcc's __aeabi_fmul/__aeabi_fadd, which is what
// -mfloat-abi=soft calls, plus a minimal hand-written emulation as a best case).
//
// Same method as neon_peak: each variant is a loop body of 8 instructions in inline asm (the
// compiler can't substitute), 20 M iterations, cycles = time x --mhz. "indep" = 8 different
// accumulators (throughput); "chain" = all 8 on one accumulator (latency). Mixed bodies (e.g. NEON
// float + scalar integer) report MACs per cycle for the whole body.
//
// Usage: int_peak [--mhz 900] [--out results/int_peak.csv] [--rt] [--cpu 3]

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "common.h"

using namespace pna;

#if defined(__ARM_NEON) && !defined(__aarch64__)

extern "C" unsigned __aeabi_fmul(unsigned, unsigned); // libgcc soft-float (base AAPCS: core regs)
extern "C" unsigned __aeabi_fadd(unsigned, unsigned);

namespace
{
constexpr long kIters = 20'000'000;
alignas(16) int16_t g_buf[64];

#define REP8(x) x x x x x x x x
#define NEON_INIT                                                                                                      \
  "vmov.i16 q0, #3\n vmov.i16 q1, #5\n vmov.f32 q2, #1.0\n vmov.f32 q3, #0.5\n"                                        \
  "vmov.i32 q8, #0\n vmov.i32 q9, #0\n vmov.i32 q10, #0\n vmov.i32 q11, #0\n"                                          \
  "vmov.i32 q12, #0\n vmov.i32 q13, #0\n vmov.i32 q14, #0\n vmov.i32 q15, #0\n"
#define NEON_CLOBBERS "q0", "q1", "q2", "q3", "q4", "q5", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "cc", "memory"

// NEON-only body. %[p] = pointer to a 16-byte-aligned buffer (for load variants).
#define LOOP(body)                                                                                                     \
  [] {                                                                                                                 \
    long n = kIters;                                                                                                   \
    const int16_t* p = g_buf;                                                                                          \
    asm volatile(NEON_INIT "1:\n" body "subs %[n], %[n], #1\n bne 1b\n"                                                \
                 : [n] "+r"(n)                                                                                         \
                 : [p] "r"(p)                                                                                          \
                 : NEON_CLOBBERS);                                                                                     \
  }

// Scalar-integer body (optionally mixed with NEON): accumulators %[a0]..%[a7], inputs %[x], %[y].
#define SLOOP(body)                                                                                                    \
  [] {                                                                                                                 \
    long n = kIters;                                                                                                   \
    int a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;                                                \
    int x = 0x00030005, y = 0x00070002;                                                                                \
    asm volatile(NEON_INIT "1:\n" body "subs %[n], %[n], #1\n bne 1b\n"                                                \
                 : [n] "+r"(n), [a0] "+r"(a0), [a1] "+r"(a1), [a2] "+r"(a2), [a3] "+r"(a3), [a4] "+r"(a4),             \
                   [a5] "+r"(a5), [a6] "+r"(a6), [a7] "+r"(a7)                                                         \
                 : [x] "r"(x), [y] "r"(y)                                                                              \
                 : NEON_CLOBBERS);                                                                                     \
  }

#define Q8(op, srcs)                                                                                                   \
  op " q8, " srcs "\n " op " q9, " srcs "\n " op " q10, " srcs "\n " op " q11, " srcs "\n " op " q12, " srcs "\n " op  \
     " q13, " srcs "\n " op " q14, " srcs "\n " op " q15, " srcs "\n"
#define D8(op, srcs)                                                                                                   \
  op " d16, " srcs "\n " op " d18, " srcs "\n " op " d20, " srcs "\n " op " d22, " srcs "\n " op " d24, " srcs "\n " op \
     " d26, " srcs "\n " op " d28, " srcs "\n " op " d30, " srcs "\n"
#define S8(op, srcs)                                                                                                   \
  op " %[a0], " srcs ", %[a0]\n " op " %[a1], " srcs ", %[a1]\n " op " %[a2], " srcs ", %[a2]\n " op " %[a3], " srcs   \
     ", %[a3]\n " op " %[a4], " srcs ", %[a4]\n " op " %[a5], " srcs ", %[a5]\n " op " %[a6], " srcs ", %[a6]\n " op   \
     " %[a7], " srcs ", %[a7]\n"

struct Variant
{
  const char* name;
  const char* kind; // mac | mul | add | other | latency | mixed
  double macs_per_body; // useful MACs (or ops, for mul/add/other) per loop body
  void (*fn)();
  int instrs = 8; // instructions per loop body
};

const Variant kVariants[] = {
  // --- NEON integer multiply-accumulate ---
  {"vmla.i16 q indep", "mac", 8 * 8, LOOP(Q8("vmla.i16", "q0, q1"))},
  {"vmla.i16 q by-scalar indep", "mac", 8 * 8, LOOP(Q8("vmla.i16", "q0, d2[0]"))},
  {"vmla.i16 d indep", "mac", 8 * 4, LOOP(D8("vmla.i16", "d0, d2"))},
  {"vmla.i32 q indep", "mac", 8 * 4, LOOP(Q8("vmla.i32", "q0, q1"))},
  {"vmlal.s16 (16x16->32) indep", "mac", 8 * 4, LOOP(Q8("vmlal.s16", "d0, d2"))},
  {"vmlal.s16 by-scalar indep", "mac", 8 * 4, LOOP(Q8("vmlal.s16", "d0, d2[0]"))},
  {"vmlal.s8 (8x8->16) indep", "mac", 8 * 8, LOOP(Q8("vmlal.s8", "d0, d2"))},
  {"vqdmlal.s16 (sat. doubling) indep", "mac", 8 * 4, LOOP(Q8("vqdmlal.s16", "d0, d2"))},
  // --- NEON integer multiply / rescale (not accumulating) ---
  {"vmull.s16 indep", "mul", 8 * 4, LOOP(Q8("vmull.s16", "d0, d2"))},
  {"vqdmulh.s16 q indep", "mul", 8 * 8, LOOP(Q8("vqdmulh.s16", "q0, q1"))},
  {"vqrdmulh.s16 q indep", "mul", 8 * 8, LOOP(Q8("vqrdmulh.s16", "q0, q1"))},
  {"vqrdmulh.s16 q by-scalar indep", "mul", 8 * 8, LOOP(Q8("vqrdmulh.s16", "q0, d2[0]"))},
  {"vqrdmulh.s32 q indep", "mul", 8 * 4, LOOP(Q8("vqrdmulh.s32", "q0, q1"))},
  {"vmul.i16 q indep", "mul", 8 * 8, LOOP(Q8("vmul.i16", "q0, q1"))},
  // --- NEON integer add / housekeeping ---
  {"vadd.i16 q indep", "add", 8 * 8, LOOP(Q8("vadd.i16", "q0, q1"))},
  {"vadd.i32 q indep", "add", 8 * 4, LOOP(Q8("vadd.i32", "q0, q1"))},
  {"vqadd.s16 q indep", "add", 8 * 8, LOOP(Q8("vqadd.s16", "q0, q1"))},
  {"vmax.s16 q indep", "other", 8 * 8, LOOP(Q8("vmax.s16", "q0, q1"))},
  {"vqrshrn.s32 #15 (narrow 32->16) indep", "other", 8 * 4, LOOP(D8("vqrshrn.s32", "q0, #15"))},
  // --- loads mixed with MACs (1 load : 1 MAC instruction) ---
  {"vmlal.s16 by-scalar + vld1.16 d (1:1)", "mixed", 4 * 4,
   LOOP("vmlal.s16 q8, d0, d2[0]\n vld1.16 {d4}, [%[p]:64]\n vmlal.s16 q9, d0, d2[0]\n vld1.16 {d5}, [%[p]:64]\n"
        "vmlal.s16 q10, d0, d2[0]\n vld1.16 {d6}, [%[p]:64]\n vmlal.s16 q11, d0, d2[0]\n vld1.16 {d7}, [%[p]:64]\n")},
  {"vmla.i16 q by-scalar + vld1.16 q (1:1)", "mixed", 4 * 8,
   LOOP("vmla.i16 q8, q0, d2[0]\n vld1.16 {d4-d5}, [%[p]:128]\n vmla.i16 q9, q0, d2[0]\n vld1.16 {d6-d7}, [%[p]:128]\n"
        "vmla.i16 q10, q0, d2[0]\n vld1.16 {d4-d5}, [%[p]:128]\n vmla.i16 q11, q0, d2[0]\n vld1.16 {d6-d7}, [%[p]:128]\n")},
  {"vmla.f32 q by-scalar + vld1.32 q (1:1)", "mixed", 4 * 4,
   LOOP("vmla.f32 q8, q2, d6[0]\n vld1.32 {d8-d9}, [%[p]:128]\n vmla.f32 q9, q2, d6[0]\n vld1.32 {d10-d11}, [%[p]:128]\n"
        "vmla.f32 q10, q2, d6[0]\n vld1.32 {d8-d9}, [%[p]:128]\n vmla.f32 q11, q2, d6[0]\n vld1.32 {d10-d11}, [%[p]:128]\n")},
  // --- conv-tile instruction mixes (64 MACs each), loads grouped before the MACs ---
  // f32, as a2_full_neon_tile per input pair: 4 q weight loads + 4 d input loads + 16 vmla.f32 q by-scalar.
  {"conv-tile mix f32: 4 vld1 q + 4 vld1 d + 16 vmla.f32 q by-scalar", "mixed", 64,
   LOOP("vld1.32 {d8-d9}, [%[p]:128]\n vld1.32 {d10-d11}, [%[p]:128]\n vld1.32 {d8-d9}, [%[p]:128]\n"
        "vld1.32 {d10-d11}, [%[p]:128]\n vld1.32 {d6}, [%[p]:64]\n vld1.32 {d7}, [%[p]:64]\n"
        "vld1.32 {d6}, [%[p]:64]\n vld1.32 {d7}, [%[p]:64]\n" Q8("vmla.f32", "q2, d6[0]") Q8("vmla.f32", "q2, d6[1]")),
   24},
  // Same mix with each load placed between two MACs, as clang 16 schedules a2_full_tile<6,4> today.
  {"conv-tile mix f32, loads interleaved (clang order)", "mixed", 64,
   LOOP("vld1.32 {d8-d9}, [%[p]:128]\n vmla.f32 q8, q2, d6[0]\n vld1.32 {d10-d11}, [%[p]:128]\n vmla.f32 q9, q2, d6[0]\n"
        "vld1.32 {d8-d9}, [%[p]:128]\n vmla.f32 q10, q2, d6[0]\n vld1.32 {d10-d11}, [%[p]:128]\n vmla.f32 q11, q2, d6[0]\n"
        "vld1.32 {d6}, [%[p]:64]\n vmla.f32 q12, q2, d7[0]\n vld1.32 {d7}, [%[p]:64]\n vmla.f32 q13, q2, d6[0]\n"
        "vld1.32 {d6}, [%[p]:64]\n vmla.f32 q14, q2, d7[0]\n vld1.32 {d7}, [%[p]:64]\n vmla.f32 q15, q2, d6[0]\n"
        Q8("vmla.f32", "q2, d6[1]")),
   24},
  // int16 equivalent (8 out x 4 frames x 2 inputs): 2 q weight loads + 1 q input load + 16 vmlal.s16 by-scalar.
  {"conv-tile mix s16: 2 vld1 q + 1 vld1 q + 16 vmlal.s16 by-scalar", "mixed", 64,
   LOOP("vld1.16 {d4-d5}, [%[p]:128]\n vld1.16 {d6-d7}, [%[p]:128]\n vld1.16 {d2-d3}, [%[p]:128]\n"
        Q8("vmlal.s16", "d4, d2[0]") Q8("vmlal.s16", "d5, d2[1]")),
   19},
  // --- NEON latency (dependent chains) ---
  {"vmla.i16 q chain (latency)", "latency", 8 * 8, LOOP(REP8("vmla.i16 q8, q0, q1\n"))},
  {"vmlal.s16 chain (latency)", "latency", 8 * 4, LOOP(REP8("vmlal.s16 q8, d0, d2\n"))},
  {"vqrdmulh.s16 q chain (latency)", "latency", 8 * 8, LOOP(REP8("vqrdmulh.s16 q8, q8, q1\n"))},
  {"vadd.i16 q chain (latency)", "latency", 8 * 8, LOOP(REP8("vadd.i16 q8, q8, q1\n"))},
  // --- scalar integer pipeline (ARM core registers) ---
  {"mla (32x32+32) indep", "mac", 8, SLOOP(S8("mla", "%[x], %[y]"))},
  {"smlabb (16x16+32) indep", "mac", 8, SLOOP(S8("smlabb", "%[x], %[y]"))},
  {"smlawb (32x16+32, Q15 style) indep", "mac", 8, SLOOP(S8("smlawb", "%[x], %[y]"))},
  {"smlad (dual 16x16+32) indep", "mac", 16, SLOOP(S8("smlad", "%[x], %[y]"))},
  {"mla chain (latency)", "latency", 8, SLOOP(REP8("mla %[a0], %[x], %[y], %[a0]\n"))},
  {"smlad chain (latency)", "latency", 16, SLOOP(REP8("smlad %[a0], %[x], %[y], %[a0]\n"))},
  // --- NEON float and scalar integer side by side: do the two pipelines overlap? ---
  {"vmla.f32 q x4 + smlad x4 interleaved", "mixed", 4 * 4 + 4 * 2,
   SLOOP("vmla.f32 q8, q2, q3\n smlad %[a0], %[x], %[y], %[a0]\n vmla.f32 q9, q2, q3\n smlad %[a1], %[x], %[y], %[a1]\n"
         "vmla.f32 q10, q2, q3\n smlad %[a2], %[x], %[y], %[a2]\n vmla.f32 q11, q2, q3\n smlad %[a3], %[x], %[y], "
         "%[a3]\n")},
  {"vmla.i16 q x4 + smlad x4 interleaved", "mixed", 4 * 8 + 4 * 2,
   SLOOP("vmla.i16 q8, q0, q1\n smlad %[a0], %[x], %[y], %[a0]\n vmla.i16 q9, q0, q1\n smlad %[a1], %[x], %[y], %[a1]\n"
         "vmla.i16 q10, q0, q1\n smlad %[a2], %[x], %[y], %[a2]\n vmla.i16 q11, q0, q1\n smlad %[a3], %[x], %[y], "
         "%[a3]\n")},
  {"vmla.f32 q x4 + vmla.i16 q x4 interleaved", "mixed", 4 * 4 + 4 * 8,
   LOOP("vmla.f32 q8, q2, q3\n vmla.i16 q12, q0, q1\n vmla.f32 q9, q2, q3\n vmla.i16 q13, q0, q1\n"
        "vmla.f32 q10, q2, q3\n vmla.i16 q14, q0, q1\n vmla.f32 q11, q2, q3\n vmla.i16 q15, q0, q1\n")},
  // --- diagnostics: what interleaving costs (pairing rules of the in-order pipeline) ---
  {"vmla.f32 q x4 + vadd.f32 q x4 interleaved", "mixed", 4 * 4 + 4 * 4,
   LOOP("vmla.f32 q8, q2, q3\n vadd.f32 q12, q2, q3\n vmla.f32 q9, q2, q3\n vadd.f32 q13, q2, q3\n"
        "vmla.f32 q10, q2, q3\n vadd.f32 q14, q2, q3\n vmla.f32 q11, q2, q3\n vadd.f32 q15, q2, q3\n")},
  {"vmla.f32 d + vld1.32 d (1:1)", "mixed", 4 * 2,
   LOOP("vmla.f32 d16, d4, d6[0]\n vld1.32 {d8}, [%[p]:64]\n vmla.f32 d18, d4, d6[0]\n vld1.32 {d9}, [%[p]:64]\n"
        "vmla.f32 d20, d4, d6[0]\n vld1.32 {d10}, [%[p]:64]\n vmla.f32 d22, d4, d6[0]\n vld1.32 {d11}, [%[p]:64]\n")},
  {"vmla.f32 q by-scalar x6 + vld1.32 q x2 (3:1)", "mixed", 6 * 4,
   LOOP("vmla.f32 q8, q2, d6[0]\n vmla.f32 q9, q2, d6[0]\n vmla.f32 q10, q2, d6[0]\n vld1.32 {d8-d9}, [%[p]:128]\n"
        "vmla.f32 q11, q2, d6[0]\n vmla.f32 q12, q2, d6[0]\n vmla.f32 q13, q2, d6[0]\n vld1.32 {d10-d11}, [%[p]:128]\n")},
  {"vmla.f32 q by-scalar x6 + 2 consecutive vld1.32 q", "mixed", 6 * 4,
   LOOP("vmla.f32 q8, q2, d6[0]\n vmla.f32 q9, q2, d6[0]\n vmla.f32 q10, q2, d6[0]\n"
        "vmla.f32 q11, q2, d6[0]\n vmla.f32 q12, q2, d6[0]\n vmla.f32 q13, q2, d6[0]\n"
        "vld1.32 {d8-d9}, [%[p]:128]\n vld1.32 {d10-d11}, [%[p]:128]\n")},
  {"vmla.f32 q x4 + vmul.f32 q x4 interleaved", "mixed", 4 * 4 + 4 * 4,
   LOOP("vmla.f32 q8, q2, q3\n vmul.f32 q12, q2, q3\n vmla.f32 q9, q2, q3\n vmul.f32 q13, q2, q3\n"
        "vmla.f32 q10, q2, q3\n vmul.f32 q14, q2, q3\n vmla.f32 q11, q2, q3\n vmul.f32 q15, q2, q3\n")},
  {"vmla.f32 q by-scalar x4 + ARM add x4 (1:1)", "mixed", 4 * 4,
   SLOOP("vmla.f32 q8, q2, d6[0]\n add %[a0], %[a0], %[x]\n vmla.f32 q9, q2, d6[0]\n add %[a1], %[a1], %[x]\n"
         "vmla.f32 q10, q2, d6[0]\n add %[a2], %[a2], %[x]\n vmla.f32 q11, q2, d6[0]\n add %[a3], %[a3], %[x]\n")},
  {"vmlal.s16 by-scalar x4 + ARM add x4 (1:1)", "mixed", 4 * 4,
   SLOOP("vmlal.s16 q8, d0, d2[0]\n add %[a0], %[a0], %[x]\n vmlal.s16 q9, d0, d2[0]\n add %[a1], %[a1], %[x]\n"
         "vmlal.s16 q10, d0, d2[0]\n add %[a2], %[a2], %[x]\n vmlal.s16 q11, d0, d2[0]\n add %[a3], %[a3], %[x]\n")},
  {"vmlal.s16 by-scalar x6 + vld1.16 d x2 (3:1)", "mixed", 6 * 4,
   LOOP("vmlal.s16 q8, d0, d2[0]\n vmlal.s16 q9, d0, d2[0]\n vmlal.s16 q10, d0, d2[0]\n vld1.16 {d4}, [%[p]:64]\n"
        "vmlal.s16 q11, d0, d2[0]\n vmlal.s16 q12, d0, d2[0]\n vmlal.s16 q13, d0, d2[0]\n vld1.16 {d5}, [%[p]:64]\n")},
  {"vmla.i16 q x4 + vadd.i16 q x4 interleaved", "mixed", 4 * 8 + 4 * 8,
   LOOP("vmla.i16 q8, q0, q1\n vadd.i16 q12, q0, q1\n vmla.i16 q9, q0, q1\n vadd.i16 q13, q0, q1\n"
        "vmla.i16 q10, q0, q1\n vadd.i16 q14, q0, q1\n vmla.i16 q11, q0, q1\n vadd.i16 q15, q0, q1\n")},
};

// ---- software float emulation ------------------------------------------------------------------
constexpr long kSoftIters = 1'000'000; // x 8 independent multiply-adds

inline unsigned AsBits(float f)
{
  unsigned u;
  std::memcpy(&u, &f, 4);
  return u;
}
inline float AsFloat(unsigned u)
{
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// Minimal float multiply/add on integers: normal numbers only, truncating, no NaN/Inf/denormal/zero
// handling, no rounding. Not correct IEEE; a lower bound on what any emulation must spend.
inline unsigned EmuMul(unsigned a, unsigned b)
{
  const unsigned s = (a ^ b) & 0x80000000u;
  const int e = static_cast<int>((a >> 23) & 0xff) + static_cast<int>((b >> 23) & 0xff) - 127;
  const uint64_t p = static_cast<uint64_t>((a & 0x7fffffu) | 0x800000u) * ((b & 0x7fffffu) | 0x800000u);
  const unsigned top = static_cast<unsigned>(p >> 47); // product in [2^46, 2^48)
  const unsigned m = static_cast<unsigned>(p >> (23 + top));
  return s | (static_cast<unsigned>(e + static_cast<int>(top)) << 23) | (m & 0x7fffffu);
}
inline unsigned EmuAdd(unsigned a, unsigned b)
{
  if ((a & 0x7fffffffu) < (b & 0x7fffffffu))
  {
    const unsigned t = a;
    a = b;
    b = t;
  }
  const int ea = (a >> 23) & 0xff, eb = (b >> 23) & 0xff;
  const unsigned ma = ((a & 0x7fffffu) | 0x800000u) << 7; // leading bit at 30
  int d = ea - eb;
  if (d > 31)
    d = 31;
  const unsigned mb = (((b & 0x7fffffu) | 0x800000u) << 7) >> d;
  unsigned m = ((a ^ b) & 0x80000000u) ? ma - mb : ma + mb;
  if (m == 0)
    return 0;
  const int sh = __builtin_clz(m) - 1; // move leading bit to 30
  int e = ea - sh;
  m = sh >= 0 ? m << sh : m >> 1;
  return (a & 0x80000000u) | (static_cast<unsigned>(e) << 23) | ((m >> 7) & 0x7fffffu);
}

template <class Mul, class Add>
double TimeSoft(Mul mul, Add add, float* check)
{
  unsigned acc[8], x[8];
  for (int i = 0; i < 8; i++)
  {
    acc[i] = AsBits(1.0f + 0.1f * i);
    x[i] = AsBits(0.5f + 0.05f * i);
  }
  const unsigned w = AsBits(0.999f);
  const auto t0 = Clock::now();
  for (long it = 0; it < kSoftIters; it++)
  {
    for (int i = 0; i < 8; i++)
    {
      unsigned xi = x[i];
      asm volatile("" : "+r"(xi)); // keep the product inside the loop
      acc[i] = add(acc[i], mul(xi, w));
    }
  }
  const double s = std::chrono::duration<double>(Clock::now() - t0).count();
  for (int i = 0; i < 8; i++)
    check[i] = AsFloat(acc[i]);
  return s;
}
} // namespace

int main(int argc, char** argv)
{
  double mhz = 900.0;
  std::string out = "results/int_peak.csv";
  bool rt = false;
  int cpu = 3;
  for (int i = 1; i < argc; i++)
  {
    if (!std::strcmp(argv[i], "--mhz") && i + 1 < argc)
      mhz = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--out") && i + 1 < argc)
      out = argv[++i];
    else if (!std::strcmp(argv[i], "--cpu") && i + 1 < argc)
      cpu = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--rt"))
      rt = true;
  }
  if (rt && !ApplyRealtime(80, cpu))
    std::fprintf(stderr, "(--rt failed: running at normal priority)\n");

  EnsureParentDir(out);
  const bool exists = std::filesystem::exists(out);
  std::ofstream f(out, std::ios::app);
  if (!exists)
    f << "unix_time,host,variant,kind,macs_per_body,instructions,seconds,cycles_per_instr,macs_per_cycle,"
         "mhz_assumed,rt\n";
  for (const Variant& v : kVariants)
  {
    v.fn(); // warm up
    const auto t0 = Clock::now();
    v.fn();
    const double s = std::chrono::duration<double>(Clock::now() - t0).count();
    const double instr = static_cast<double>(v.instrs) * kIters;
    const double cpi = s * mhz * 1e6 / instr;
    const double mpc = v.macs_per_body * kIters / (s * mhz * 1e6);
    std::printf("%-42s %6.2f cycles/instr  %5.2f ops/cycle\n", v.name, cpi, mpc);
    f << std::time(nullptr) << "," << Hostname() << ",\"" << v.name << "\"," << v.kind << "," << v.macs_per_body << ","
      << instr << "," << s << "," << cpi << "," << mpc << "," << mhz << "," << rt << "\n";
  }

  // Software float: one "instruction" = one float multiply + one float add (one emulated MAC).
  struct Soft
  {
    const char* name;
    double s;
  };
  float chk_lib[8], chk_emu[8], chk_hw[8];
  const double s_lib = TimeSoft([](unsigned a, unsigned b) { return __aeabi_fmul(a, b); },
                                [](unsigned a, unsigned b) { return __aeabi_fadd(a, b); }, chk_lib);
  const double s_emu = TimeSoft(EmuMul, EmuAdd, chk_emu);
  // Same C loop with hardware VFP floats, as a reference for loop overhead.
  const double s_hw = TimeSoft([](unsigned a, unsigned b) { return AsBits(AsFloat(a) * AsFloat(b)); },
                               [](unsigned a, unsigned b) { return AsBits(AsFloat(a) + AsFloat(b)); }, chk_hw);
  for (const Soft& v : {Soft{"soft-float libgcc __aeabi_fmul+__aeabi_fadd", s_lib},
                        Soft{"soft-float minimal emulation (no IEEE edge cases)", s_emu},
                        Soft{"hardware VFP mul+add, same C loop", s_hw}})
  {
    const double instr = 8.0 * kSoftIters;
    const double cpi = v.s * mhz * 1e6 / instr;
    std::printf("%-42s %6.2f cycles/MAC     %5.3f MACs/cycle\n", v.name, cpi, 1.0 / cpi);
    f << std::time(nullptr) << "," << Hostname() << ",\"" << v.name << "\",softfloat,1," << instr << "," << v.s << ","
      << cpi << "," << 1.0 / cpi << "," << mhz << "," << rt << "\n";
  }
  std::printf("check (lib / emu / hw, acc[0]): %.6g %.6g %.6g\n", chk_lib[0], chk_emu[0], chk_hw[0]);
  return 0;
}

#else
int main()
{
  std::fprintf(stderr, "int_peak only runs on 32-bit ARM with NEON\n");
  return 1;
}
#endif
