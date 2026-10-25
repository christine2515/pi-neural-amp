// kernel_peak: how fast can Full's layer kernel (8 output channels x 4 frames, K taps) run on one
// Cortex-A7 core when every operand is in L1, and how much of the gap to the VMLA ceiling is the
// instruction schedule? (docs/low-latency-plan.md, Q4.)
//
// Variants, each run on the same L1-resident data (K=6 taps, 4-frame tiles, Full's 8x8 weights):
//   intr       a verbatim copy of a2_full_tile<K,4> from patches/a2_full_neon_tile.patch (clang
//              intrinsics + scheduling fences): what the nhead build runs.
//   asm        same arithmetic, conv taps + 1x1 written in inline asm with the loads software-
//              pipelined: each pair's weights are loaded 8 VMLAs before first use, each frame's
//              inputs one pair ahead into a second bank of d-registers, weights as 4-register
//              vld1 (2 instructions per input pair instead of 4). Same per-accumulator operation
//              order as intr, so outputs are bit-identical (checked here).
//   asm_a      asm with alignment hints (:128 on weights, :64 on inputs); needs 16-byte-aligned weights.
//   conv_*     the K-tap convolution part alone (no bias/activation/1x1 epilogue).
//   vmla_only  64 VMLA per tap with no loads at all: the ceiling for this instruction mix.
// Reports ns per tile and MACs per cycle at --mhz. One tile = 4 frames x (K x 64 + 64) MACs.
//
// Usage: kernel_peak [--rt] [--cpu 3] [--mhz 900] [--out results/kernel_peak.csv]

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common.h"

using namespace pna;

#if defined(__ARM_NEON) && !defined(__aarch64__)
  #include <arm_neon.h>

  #define KP_KERNEL __attribute__((noinline, target("no-slowfpvmlx")))

namespace
{
// ---------------------------------------------------------------------------------------------
// Verbatim copy of the nhead kernel (a2_full_neon_tile.patch).
template <int T>
__attribute__((always_inline)) inline void a2_neon_fence(float32x4_t (&lo)[T], float32x4_t (&hi)[T])
{
  if constexpr (T == 4)
    asm volatile("" : "+w"(lo[0]), "+w"(lo[1]), "+w"(lo[2]), "+w"(lo[3]), "+w"(hi[0]), "+w"(hi[1]), "+w"(hi[2]),
                 "+w"(hi[3]));
  else
    for (int t = 0; t < T; t++)
      asm volatile("" : "+w"(lo[t]), "+w"(hi[t]));
}

template <int T>
__attribute__((always_inline)) inline void a2_full_mac8(float32x4_t (&lo)[T], float32x4_t (&hi)[T],
                                                        const float* __restrict w, const float* __restrict x)
{
  #pragma clang loop unroll(full)
  for (int jp = 0; jp < 4; jp++)
  {
    const float* wj = w + jp * 16;
    const float32x4_t w0l = vld1q_f32(wj + 0);
    const float32x4_t w0h = vld1q_f32(wj + 4);
    const float32x4_t w1l = vld1q_f32(wj + 8);
    const float32x4_t w1h = vld1q_f32(wj + 12);
    float32x2_t xv[T];
    for (int t = 0; t < T; t++)
      xv[t] = vld1_f32(x + t * 8 + jp * 2);
    for (int t = 0; t < T; t++)
    {
      lo[t] = vmlaq_lane_f32(lo[t], w0l, xv[t], 0);
      hi[t] = vmlaq_lane_f32(hi[t], w0h, xv[t], 0);
    }
    for (int t = 0; t < T; t++)
    {
      lo[t] = vmlaq_lane_f32(lo[t], w1l, xv[t], 1);
      hi[t] = vmlaq_lane_f32(hi[t], w1h, xv[t], 1);
    }
    a2_neon_fence<T>(lo, hi);
  }
}

template <int K, int T>
KP_KERNEL void a2_full_tile(const float* const* __restrict taps, const float* __restrict conv_w,
                            const float* __restrict conv_b, const float* __restrict mixin_w,
                            const float* __restrict l1x1_w, const float* __restrict l1x1_b,
                            const float* __restrict cond, float* __restrict hsum, float* __restrict lin,
                            float leaky_slope)
{
  float32x4_t lo[T], hi[T];
  for (int t = 0; t < T; t++)
  {
    lo[t] = vdupq_n_f32(0.0f);
    hi[t] = vdupq_n_f32(0.0f);
  }
  #pragma clang loop unroll(disable)
  for (int k = 0; k < K; k++)
    a2_full_mac8<T>(lo, hi, conv_w + k * 64, taps[k]);

  const float32x4_t bl = vld1q_f32(conv_b), bh = vld1q_f32(conv_b + 4);
  const float32x4_t ml = vld1q_f32(mixin_w), mh = vld1q_f32(mixin_w + 4);
  alignas(16) float zs[T * 8];
  for (int t = 0; t < T; t++)
  {
    float32x4_t a = vaddq_f32(lo[t], bl);
    float32x4_t b = vaddq_f32(hi[t], bh);
    a = vmlaq_n_f32(a, ml, cond[t]);
    b = vmlaq_n_f32(b, mh, cond[t]);
    a = vmaxq_f32(a, vmulq_n_f32(a, leaky_slope));
    b = vmaxq_f32(b, vmulq_n_f32(b, leaky_slope));
    float* h = hsum + t * 8;
    vst1q_f32(h, vaddq_f32(vld1q_f32(h), a));
    vst1q_f32(h + 4, vaddq_f32(vld1q_f32(h + 4), b));
    vst1q_f32(zs + t * 8, a);
    vst1q_f32(zs + t * 8 + 4, b);
  }
  for (int t = 0; t < T; t++)
  {
    lo[t] = vdupq_n_f32(0.0f);
    hi[t] = vdupq_n_f32(0.0f);
  }
  a2_full_mac8<T>(lo, hi, l1x1_w, zs);
  const float32x4_t ll = vld1q_f32(l1x1_b), lh = vld1q_f32(l1x1_b + 4);
  for (int t = 0; t < T; t++)
  {
    float* p = lin + t * 8;
    vst1q_f32(p, vaddq_f32(vaddq_f32(vld1q_f32(p), lo[t]), ll));
    vst1q_f32(p + 4, vaddq_f32(vaddq_f32(vld1q_f32(p + 4), hi[t]), lh));
  }
}

template <int K>
KP_KERNEL void conv_intr(const float* const* __restrict taps, const float* __restrict conv_w, float* __restrict out)
{
  float32x4_t lo[4], hi[4];
  for (int t = 0; t < 4; t++)
    lo[t] = hi[t] = vdupq_n_f32(0.0f);
  #pragma clang loop unroll(disable)
  for (int k = 0; k < K; k++)
    a2_full_mac8<4>(lo, hi, conv_w + k * 64, taps[k]);
  for (int t = 0; t < 4; t++)
  {
    vst1q_f32(out + t * 8, lo[t]);
    vst1q_f32(out + t * 8 + 4, hi[t]);
  }
}

// ---------------------------------------------------------------------------------------------
// Software-pipelined asm. Accumulators q8..q15 = lo0 hi0 lo1 hi1 lo2 hi2 lo3 hi3 (stored to out in
// that order, i.e. out[t*8 + 0..3] = lo[t], out[t*8 + 4..7] = hi[t]). Weights for the current input
// pair in q0..q3 (w0l w0h w1l w1h); inputs in bank A (d8..d11, frames 0..3) or bank B (d12..d15).
// PAIR(X bank, Y bank, P): 16 VMLAs on bank X while loading the next pair's inputs into bank Y from
// pointer P (post-increment 32 B = one frame, then back 120 B = next input pair) and the next pair's
// weights into q0/q1 (after their last use) and q2/q3.
  #define KP_VMLA(acc, w, x, lane) "vmla.f32 " acc ", " w ", " x "[" lane "]\n"
  #define KP_PAIR_BODY(X0, X1, X2, X3, LOADY0, LOADY1, LOADY2, LOADY3, LOADW01, LOADW23)                              \
    KP_VMLA("q8", "q0", X0, "0") KP_VMLA("q10", "q0", X1, "0") LOADY0 KP_VMLA("q12", "q0", X2, "0")                  \
    KP_VMLA("q14", "q0", X3, "0") KP_VMLA("q9", "q1", X0, "0") LOADY1 KP_VMLA("q11", "q1", X1, "0")                  \
    KP_VMLA("q13", "q1", X2, "0") KP_VMLA("q15", "q1", X3, "0") LOADW01 KP_VMLA("q8", "q2", X0, "1")                 \
    LOADY2 KP_VMLA("q10", "q2", X1, "1") KP_VMLA("q12", "q2", X2, "1") KP_VMLA("q14", "q2", X3, "1")                 \
    LOADY3 KP_VMLA("q9", "q3", X0, "1") KP_VMLA("q11", "q3", X1, "1") KP_VMLA("q13", "q3", X2, "1")                  \
    KP_VMLA("q15", "q3", X3, "1") LOADW23
  #define KP_LDX(d, P, AL) "vld1.32 {" d "}, [%[" P "]" AL "], %[rs]\n"
  #define KP_PAIR(X0, X1, X2, X3, Y0, Y1, Y2, Y3, P, AW, AX)                                                            \
    KP_PAIR_BODY(X0, X1, X2, X3, KP_LDX(Y0, P, AX), KP_LDX(Y1, P, AX), KP_LDX(Y2, P, AX), KP_LDX(Y3, P, AX),          \
                 "vld1.32 {d0-d3}, [%[w]" AW "]!\n", "vld1.32 {d4-d7}, [%[w]" AW "]!\n")                               \
    "sub %[" P "], %[" P "], #120\n"
  #define KP_PAIR_W(X0, X1, X2, X3, AW) /* last pair of the last tap: no loads */                                     \
    KP_PAIR_BODY(X0, X1, X2, X3, "", "", "", "", "", "")
  #define KP_EXPAND(m, ...) m(__VA_ARGS__) /* expands KP_A/KP_B into 4 arguments first */
  #define KP_A "d8", "d9", "d10", "d11"
  #define KP_B "d12", "d13", "d14", "d15"

  // One tap = 4 input pairs, banks A B A B. In the loop, pair 3 loads the next tap's pair 0 (via rt).
  #define KP_CONV_ASM(AW, AX)                                                                                          \
    "vmov.i32 q8, #0\n vmov.i32 q9, #0\n vmov.i32 q10, #0\n vmov.i32 q11, #0\n"                                       \
    "vmov.i32 q12, #0\n vmov.i32 q13, #0\n vmov.i32 q14, #0\n vmov.i32 q15, #0\n"                                     \
    "ldr %[nx], [%[tp]], #4\n"                                                                                         \
    "vld1.32 {d0-d3}, [%[w]" AW "]!\n vld1.32 {d4-d7}, [%[w]" AW "]!\n"                                               \
    KP_LDX("d8", "nx", AX) KP_LDX("d9", "nx", AX) KP_LDX("d10", "nx", AX) KP_LDX("d11", "nx", AX)                     \
    "sub %[nx], %[nx], #120\n"                                                                                         \
    "cmp %[k], #0\n beq 2f\n"                                                                                          \
    "1:\n"                                                                                                             \
    "ldr %[rt], [%[tp]], #4\n"                                                                                         \
    KP_EXPAND(KP_PAIR, KP_A, KP_B, "nx", AW, AX)                                                                                  \
    KP_EXPAND(KP_PAIR, KP_B, KP_A, "nx", AW, AX)                                                                                  \
    KP_EXPAND(KP_PAIR, KP_A, KP_B, "nx", AW, AX)                                                                                  \
    KP_EXPAND(KP_PAIR, KP_B, KP_A, "rt", AW, AX)                                                                                  \
    "mov %[nx], %[rt]\n"                                                                                               \
    "subs %[k], %[k], #1\n bne 1b\n"                                                                                   \
    "2:\n"                                                                                                             \
    KP_EXPAND(KP_PAIR, KP_A, KP_B, "nx", AW, AX)                                                                                  \
    KP_EXPAND(KP_PAIR, KP_B, KP_A, "nx", AW, AX)                                                                                  \
    /* pair 2 loads pair 3's inputs and weights; pair 3 loads nothing */                                               \
    KP_EXPAND(KP_PAIR_BODY, KP_A, KP_LDX("d12", "nx", AX), KP_LDX("d13", "nx", AX), KP_LDX("d14", "nx", AX),                   \
                 KP_LDX("d15", "nx", AX), "vld1.32 {d0-d3}, [%[w]" AW "]!\n", "vld1.32 {d4-d7}, [%[w]" AW "]!\n")      \
    KP_EXPAND(KP_PAIR_W, KP_B, AW)                                                                                                \
    "vst1.32 {d16-d19}, [%[out]]!\n vst1.32 {d20-d23}, [%[out]]!\n"                                                   \
    "vst1.32 {d24-d27}, [%[out]]!\n vst1.32 {d28-d31}, [%[out]]\n"

  #define KP_CLOBBERS                                                                                                  \
    "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "cc", "memory"

// out[32] = conv over k taps (taps[k] = frame-0 column of tap k, frames 8 floats apart).
__attribute__((always_inline)) inline void conv_asm_u(const float* const* taps, const float* w, int K, float* out)
{
  const float* nx;
  const float* rt;
  int km1 = K - 1;
  asm volatile(KP_CONV_ASM("", "")
               : [w] "+r"(w), [tp] "+r"(taps), [k] "+r"(km1), [out] "+r"(out), [nx] "=&r"(nx), [rt] "=&r"(rt)
               : [rs] "r"(32)
               : KP_CLOBBERS);
}
__attribute__((always_inline)) inline void conv_asm_a(const float* const* taps, const float* w, int K, float* out)
{
  const float* nx;
  const float* rt;
  int km1 = K - 1;
  asm volatile(KP_CONV_ASM(":128", ":64")
               : [w] "+r"(w), [tp] "+r"(taps), [k] "+r"(km1), [out] "+r"(out), [nx] "=&r"(nx), [rt] "=&r"(rt)
               : [rs] "r"(32)
               : KP_CLOBBERS);
}

template <int K, bool Aligned>
KP_KERNEL void a2_full_tile_asm(const float* const* __restrict taps, const float* __restrict conv_w,
                                const float* __restrict conv_b, const float* __restrict mixin_w,
                                const float* __restrict l1x1_w, const float* __restrict l1x1_b,
                                const float* __restrict cond, float* __restrict hsum, float* __restrict lin,
                                float leaky_slope)
{
  alignas(16) float acc[32];
  if constexpr (Aligned)
    conv_asm_a(taps, conv_w, K, acc);
  else
    conv_asm_u(taps, conv_w, K, acc);
  const float32x4_t bl = vld1q_f32(conv_b), bh = vld1q_f32(conv_b + 4);
  const float32x4_t ml = vld1q_f32(mixin_w), mh = vld1q_f32(mixin_w + 4);
  alignas(16) float zs[32];
  for (int t = 0; t < 4; t++)
  {
    float32x4_t a = vaddq_f32(vld1q_f32(acc + t * 8), bl);
    float32x4_t b = vaddq_f32(vld1q_f32(acc + t * 8 + 4), bh);
    a = vmlaq_n_f32(a, ml, cond[t]);
    b = vmlaq_n_f32(b, mh, cond[t]);
    a = vmaxq_f32(a, vmulq_n_f32(a, leaky_slope));
    b = vmaxq_f32(b, vmulq_n_f32(b, leaky_slope));
    float* h = hsum + t * 8;
    vst1q_f32(h, vaddq_f32(vld1q_f32(h), a));
    vst1q_f32(h + 4, vaddq_f32(vld1q_f32(h + 4), b));
    vst1q_f32(zs + t * 8, a);
    vst1q_f32(zs + t * 8 + 4, b);
  }
  const float* zp[1] = {zs};
  if constexpr (Aligned)
    conv_asm_a(zp, l1x1_w, 1, acc);
  else
    conv_asm_u(zp, l1x1_w, 1, acc);
  const float32x4_t ll = vld1q_f32(l1x1_b), lh = vld1q_f32(l1x1_b + 4);
  for (int t = 0; t < 4; t++)
  {
    float* p = lin + t * 8;
    vst1q_f32(p, vaddq_f32(vaddq_f32(vld1q_f32(p), vld1q_f32(acc + t * 8)), ll));
    vst1q_f32(p + 4, vaddq_f32(vaddq_f32(vld1q_f32(p + 4), vld1q_f32(acc + t * 8 + 4)), lh));
  }
}

template <int K>
KP_KERNEL void conv_asm(const float* const* taps, const float* w, float* out)
{
  conv_asm_u(taps, w, K, out);
}

// Ceiling: the same 64 VMLAs per tap, no loads.
template <int K>
KP_KERNEL void vmla_only(float* out)
{
  int k = K;
  asm volatile("vmov.i32 q8, #0\n vmov.i32 q9, #0\n vmov.i32 q10, #0\n vmov.i32 q11, #0\n"
               "vmov.i32 q12, #0\n vmov.i32 q13, #0\n vmov.i32 q14, #0\n vmov.i32 q15, #0\n"
               "1:\n" KP_EXPAND(KP_PAIR_BODY, KP_A, "", "", "", "", "", "") KP_EXPAND(KP_PAIR_BODY, KP_B, "", "", "", "", "", "")
                 KP_EXPAND(KP_PAIR_BODY, KP_A, "", "", "", "", "", "") KP_EXPAND(KP_PAIR_BODY, KP_B, "", "", "", "", "", "")
               "subs %[k], %[k], #1\n bne 1b\n"
               "vst1.32 {d16-d19}, [%[out]]!\n vst1.32 {d20-d23}, [%[out]]!\n"
               "vst1.32 {d24-d27}, [%[out]]!\n vst1.32 {d28-d31}, [%[out]]\n"
               : [k] "+r"(k), [out] "+r"(out)
               :
               : KP_CLOBBERS);
}

// ---------------------------------------------------------------------------------------------
// Load cost when mixed with VMLA: each loop iteration = 16 independent VMLA.F32 q (by scalar, operands
// never reloaded, so no dependency on the loads) + a few loads of one form from an L1-resident buffer.
// cycles/iter - (16 VMLA alone) = what those loads add.
alignas(64) float g_lbuf[256];
constexpr long kLIters = 2'000'000;
  #define KP_V16                                                                                                     \
    "vmla.f32 q8, q0, d2[0]\n vmla.f32 q9, q0, d2[1]\n vmla.f32 q10, q0, d3[0]\n vmla.f32 q11, q0, d3[1]\n"         \
    "vmla.f32 q12, q1, d2[0]\n vmla.f32 q13, q1, d2[1]\n vmla.f32 q14, q1, d3[0]\n vmla.f32 q15, q1, d3[1]\n"       \
    "vmla.f32 q8, q0, d2[0]\n vmla.f32 q9, q0, d2[1]\n vmla.f32 q10, q0, d3[0]\n vmla.f32 q11, q0, d3[1]\n"         \
    "vmla.f32 q12, q1, d2[0]\n vmla.f32 q13, q1, d2[1]\n vmla.f32 q14, q1, d3[0]\n vmla.f32 q15, q1, d3[1]\n"
  #define KP_LLOOP(loads)                                                                                            \
    [] {                                                                                                             \
      long n = kLIters;                                                                                              \
      const float* p = g_lbuf;                                                                                       \
      asm volatile("vmov.f32 q0, #1.0\n vmov.f32 q1, #0.5\n vmov.i32 q8, #0\n vmov.i32 q9, #0\n"                    \
                   "vmov.i32 q10, #0\n vmov.i32 q11, #0\n vmov.i32 q12, #0\n vmov.i32 q13, #0\n"                     \
                   "vmov.i32 q14, #0\n vmov.i32 q15, #0\n"                                                           \
                   "1:\n" loads "subs %[n], %[n], #1\n bne 1b\n"                                                     \
                   : [n] "+r"(n)                                                                                     \
                   : [p] "r"(p)                                                                                      \
                   : "r4", "r5", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12",    \
                     "q13", "q14", "q15", "cc", "memory");                                                           \
    }
struct LoadVariant
{
  const char* name;
  int loads; // load instructions per iteration
  void (*fn)();
};
const LoadVariant kLoadVariants[] = {
  {"16vmla", 0, KP_LLOOP(KP_V16)},
  {"16vmla+4 vld1.32 {d}", 4,
   KP_LLOOP(KP_V16 "vld1.32 {d8}, [%[p]]\n vld1.32 {d9}, [%[p]]\n vld1.32 {d10}, [%[p]]\n vld1.32 {d11}, [%[p]]\n")},
  {"16vmla+4 vldr d", 4,
   KP_LLOOP(KP_V16 "vldr d8, [%[p]]\n vldr d9, [%[p], #8]\n vldr d10, [%[p], #16]\n vldr d11, [%[p], #24]\n")},
  {"16vmla+4 vld1.32 {d:64}", 4,
   KP_LLOOP(KP_V16 "vld1.32 {d8}, [%[p]:64]\n vld1.32 {d9}, [%[p]:64]\n vld1.32 {d10}, [%[p]:64]\n vld1.32 {d11}, "
                   "[%[p]:64]\n")},
  {"16vmla+4 vld1.32 {q}", 4,
   KP_LLOOP(KP_V16 "vld1.32 {d8,d9}, [%[p]]\n vld1.32 {d10,d11}, [%[p]]\n vld1.32 {d12,d13}, [%[p]]\n vld1.32 "
                   "{d14,d15}, [%[p]]\n")},
  {"16vmla+4 vld1.32 {q:128}", 4,
   KP_LLOOP(KP_V16 "vld1.32 {d8,d9}, [%[p]:128]\n vld1.32 {d10,d11}, [%[p]:128]\n vld1.32 {d12,d13}, [%[p]:128]\n "
                   "vld1.32 {d14,d15}, [%[p]:128]\n")},
  {"16vmla+2 vld1.32 {4d}", 2, KP_LLOOP(KP_V16 "vld1.32 {d8-d11}, [%[p]]\n vld1.32 {d12-d15}, [%[p]]\n")},
  {"16vmla+2 vld1.32 {4d:256}", 2,
   KP_LLOOP(KP_V16 "vld1.32 {d8-d11}, [%[p]:256]\n vld1.32 {d12-d15}, [%[p]:256]\n")},
  {"16vmla+1 vld1.32 {4d}", 1, KP_LLOOP(KP_V16 "vld1.32 {d8-d11}, [%[p]]\n")},
  {"16vmla+4 vld1.32 {d[]}", 4,
   KP_LLOOP(KP_V16 "vld1.32 {d8[]}, [%[p]]\n vld1.32 {d9[]}, [%[p]]\n vld1.32 {d10[]}, [%[p]]\n vld1.32 {d11[]}, "
                   "[%[p]]\n")},
  {"16vmla+2 vldm 4d", 2, KP_LLOOP(KP_V16 "vldmia %[p], {d8-d11}\n vldmia %[p], {d12-d15}\n")},
  {"16vmla+4 ldrd+vmov", 4,
   KP_LLOOP(KP_V16 "ldrd r4, r5, [%[p]]\n vmov d8, r4, r5\n ldrd r4, r5, [%[p], #8]\n vmov d9, r4, r5\n"
                   "ldrd r4, r5, [%[p], #16]\n vmov d10, r4, r5\n ldrd r4, r5, [%[p], #24]\n vmov d11, r4, r5\n")},
  {"16vmla+4 ldr (int only)", 4,
   KP_LLOOP(KP_V16 "ldr r4, [%[p]]\n ldr r5, [%[p], #8]\n ldr r4, [%[p], #16]\n ldr r5, [%[p], #24]\n")},
  {"16vmla+4 pld", 4,
   KP_LLOOP(KP_V16 "pld [%[p]]\n pld [%[p], #64]\n pld [%[p], #128]\n pld [%[p], #192]\n")},
  {"0vmla+4 vld1.32 {d}", 4,
   KP_LLOOP("vld1.32 {d8}, [%[p]]\n vld1.32 {d9}, [%[p]]\n vld1.32 {d10}, [%[p]]\n vld1.32 {d11}, [%[p]]\n")},
  {"0vmla+4 vld1.32 {q}", 4,
   KP_LLOOP("vld1.32 {d8,d9}, [%[p]]\n vld1.32 {d10,d11}, [%[p]]\n vld1.32 {d12,d13}, [%[p]]\n vld1.32 {d14,d15}, "
            "[%[p]]\n")},
  {"0vmla+2 vld1.32 {4d}", 2, KP_LLOOP("vld1.32 {d8-d11}, [%[p]]\n vld1.32 {d12-d15}, [%[p]]\n")},
  {"0vmla+4 vst1.32 {q}", 4,
   KP_LLOOP("vst1.32 {d8,d9}, [%[p]]\n vst1.32 {d10,d11}, [%[p]]\n vst1.32 {d12,d13}, [%[p]]\n vst1.32 {d14,d15}, "
            "[%[p]]\n")},
};

constexpr int kK = 6;
constexpr int kTiles = 8; // 32 frames per pass, all in L1
constexpr long kPasses = 40000;

struct Data
{
  // Weights 16-byte aligned (heap + manual alignment) so the :128 variant is legal.
  std::vector<float> wbuf, hist, hsum, lin, cond;
  float* conv_w;
  float* l1x1_w;
  float conv_b[8], mixin_w[8], l1x1_b[8];
  Data()
  {
    wbuf.assign((kK + 1) * 64 + 8, 0.0f);
    conv_w = reinterpret_cast<float*>((reinterpret_cast<uintptr_t>(wbuf.data()) + 15) & ~uintptr_t(15));
    l1x1_w = conv_w + kK * 64;
    uint32_t s = 12345;
    auto rnd = [&] {
      s = s * 1664525u + 1013904223u;
      return static_cast<float>((s >> 8) & 0xffff) / 65536.0f - 0.5f;
    };
    for (int i = 0; i < (kK + 1) * 64; i++)
      conv_w[i] = rnd() * 0.3f;
    for (int i = 0; i < 8; i++)
    {
      conv_b[i] = rnd() * 0.1f;
      mixin_w[i] = rnd();
      l1x1_b[i] = rnd() * 0.1f;
    }
    // History: dilation 1, (kK - 1) lookback + 32 frames, 8 floats per frame (8-byte aligned).
    hist.assign((kK - 1 + kTiles * 4) * 8, 0.0f);
    for (auto& v : hist)
      v = rnd();
    hsum.assign(kTiles * 32, 0.0f);
    lin.assign(kTiles * 32, 0.0f);
    cond.assign(kTiles * 4, 0.0f);
    for (auto& v : cond)
      v = rnd();
  }
  void Taps(int tile, const float* (&taps)[kK]) const
  {
    for (int k = 0; k < kK; k++)
      taps[k] = hist.data() + (tile * 4 + k) * 8; // tap k = frame f + k (k = 0 oldest)
  }
};

using TileFn = void (*)(const float* const*, const float*, const float*, const float*, const float*, const float*,
                        const float*, float*, float*, float);

double RunTile(Data& d, TileFn fn)
{
  const float* taps[kK];
  const auto t0 = Clock::now();
  for (long p = 0; p < kPasses; p++)
    for (int t = 0; t < kTiles; t++)
    {
      d.Taps(t, taps);
      fn(taps, d.conv_w, d.conv_b, d.mixin_w, d.l1x1_w, d.l1x1_b, d.cond.data() + t * 4, d.hsum.data() + t * 32,
         d.lin.data() + t * 32, 0.01f);
    }
  return std::chrono::duration<double, std::nano>(Clock::now() - t0).count() / (kPasses * kTiles);
}

template <class F>
double RunConv(Data& d, F fn)
{
  const float* taps[kK];
  alignas(16) float out[32];
  float sink = 0;
  const auto t0 = Clock::now();
  for (long p = 0; p < kPasses; p++)
    for (int t = 0; t < kTiles; t++)
    {
      d.Taps(t, taps);
      fn(taps, out);
      sink += out[t];
    }
  const double ns = std::chrono::duration<double, std::nano>(Clock::now() - t0).count() / (kPasses * kTiles);
  if (sink == 12345.678f)
    std::puts("");
  return ns;
}
} // namespace

int main(int argc, char** argv)
{
  bool rt = false;
  int cpu = 3;
  double mhz = 900.0;
  std::string out = "results/kernel_peak.csv";
  for (int i = 1; i < argc; i++)
  {
    const std::string s = argv[i];
    if (s == "--rt")
      rt = true;
    else if (s == "--cpu" && i + 1 < argc)
      cpu = std::atoi(argv[++i]);
    else if (s == "--mhz" && i + 1 < argc)
      mhz = std::atof(argv[++i]);
    else if (s == "--out" && i + 1 < argc)
      out = argv[++i];
    else
    {
      std::fprintf(stderr, "usage: kernel_peak [--rt] [--cpu 3] [--mhz 900] [--out F.csv]\n");
      return 2;
    }
  }
  if (rt ? !ApplyRealtime(80, cpu) : !PinOnly(cpu))
    return 3;

  // Correctness: intr vs asm on one pass, bit-for-bit.
  {
    Data a, b, c;
    const float* taps[kK];
    for (int t = 0; t < kTiles; t++)
    {
      a.Taps(t, taps);
      a2_full_tile<kK, 4>(taps, a.conv_w, a.conv_b, a.mixin_w, a.l1x1_w, a.l1x1_b, a.cond.data() + t * 4,
                          a.hsum.data() + t * 32, a.lin.data() + t * 32, 0.01f);
      b.Taps(t, taps);
      a2_full_tile_asm<kK, false>(taps, b.conv_w, b.conv_b, b.mixin_w, b.l1x1_w, b.l1x1_b, b.cond.data() + t * 4,
                                  b.hsum.data() + t * 32, b.lin.data() + t * 32, 0.01f);
      c.Taps(t, taps);
      a2_full_tile_asm<kK, true>(taps, c.conv_w, c.conv_b, c.mixin_w, c.l1x1_w, c.l1x1_b, c.cond.data() + t * 4,
                                 c.hsum.data() + t * 32, c.lin.data() + t * 32, 0.01f);
    }
    const bool same = !std::memcmp(a.lin.data(), b.lin.data(), a.lin.size() * 4)
                      && !std::memcmp(a.hsum.data(), b.hsum.data(), a.hsum.size() * 4)
                      && !std::memcmp(a.lin.data(), c.lin.data(), a.lin.size() * 4);
    std::printf("asm vs intr outputs: %s\n", same ? "bit-identical" : "DIFFERENT");
    if (!same)
      return 1;
  }

  EnsureParentDir(out);
  const bool exists = std::filesystem::exists(out);
  std::ofstream f(out, std::ios::app);
  if (!exists)
    f << "unix_time,host,variant,K,macs_per_call,ns_per_call,macs_per_cycle,mhz_assumed,rt\n";
  auto emit = [&](const char* name, double macs, double ns) {
    const double mpc = macs / (ns * mhz / 1000.0);
    std::printf("%-12s %8.1f ns/call  %5.3f MAC/cycle\n", name, ns, mpc);
    f << std::time(nullptr) << "," << Hostname() << "," << name << "," << kK << "," << macs << "," << ns << "," << mpc
      << "," << mhz << "," << rt << "\n";
  };
  Data d;
  const double tileMacs = 4.0 * (kK * 64 + 64), convMacs = 4.0 * kK * 64;
  for (int rep = 0; rep < 2; rep++)
  {
    emit("intr", tileMacs, RunTile(d, a2_full_tile<kK, 4>));
    emit("asm", tileMacs, RunTile(d, a2_full_tile_asm<kK, false>));
    emit("asm_a", tileMacs, RunTile(d, a2_full_tile_asm<kK, true>));
    emit("conv_intr", convMacs, RunConv(d, [&](const float* const* t, float* o) { conv_intr<kK>(t, d.conv_w, o); }));
    emit("conv_asm", convMacs, RunConv(d, [&](const float* const* t, float* o) { conv_asm<kK>(t, d.conv_w, o); }));
    emit("vmla_only", convMacs, RunConv(d, [&](const float* const*, float* o) { vmla_only<kK>(o); }));
  }

  // Load forms mixed with VMLA (separate CSV columns reused: macs_per_call = loads per iteration,
  // ns_per_call = cycles per iteration, macs_per_cycle = extra cycles per load vs 16 VMLA alone).
  double base = 0;
  for (const LoadVariant& v : kLoadVariants)
  {
    v.fn();
    const auto t0 = Clock::now();
    v.fn();
    const double s = std::chrono::duration<double>(Clock::now() - t0).count();
    const double cyc = s * mhz * 1e6 / kLIters;
    if (v.loads == 0)
      base = cyc;
    const bool mixed = v.name[0] == '1';
    const double extra = v.loads ? (cyc - (mixed ? base : 0.0)) / v.loads : 0.0;
    std::printf("%-28s %6.2f cycles/iter  %5.2f cycles per load%s\n", v.name, cyc, extra,
                mixed ? " (above 16 VMLA)" : "");
    f << std::time(nullptr) << "," << Hostname() << ",\"load: " << v.name << "\",0," << v.loads << "," << cyc << ","
      << extra << "," << mhz << "," << rt << "\n";
  }
  return 0;
}

#else
int main()
{
  std::fprintf(stderr, "kernel_peak only runs on 32-bit ARM with NEON\n");
  return 1;
}
#endif
