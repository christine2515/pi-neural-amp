# 09: Open questions, next steps and self-test

## What's left

| Item | Needs | Why |
|---|---|---|
| `isolcpus` + IRQ affinity on cores 2–3 | sudo + reboot | Remove Full's last marginal overruns at 128 |
| PREEMPT_RT kernel (`rpi-7.1.y`+) | Building your own kernel | Tail latency, especially at ≤32 samples; watch the USB FIQ issue |
| Scarlett: `live`, `--mmap`, round-trip latency | Hardware connected | The real product number; find the USB floor |
| I2S ADC/DAC | Hardware | Only if USB sets the latency floor |
| `live --pipeline` | Code | `live` reads the env var only today: set `NAM_A2_PIPELINE=8` |
| Int16 fixed point / GPU (VideoCore IV) | Research (in progress) | Ways past the ~1 FP op/cycle limit |
| 30-min rerun with the `nhead` build | Pi time | The sustained run used the build before the head patch (~1% different) |

## Self-test: questions and short answers

**1. Why didn't a bigger buffer fix Full at 250% load?**
The model produced only 40% of the audio needed. Buffers redistribute a deficit; they can't erase it.
They only delay the first dropout.

**2. What does IPC 0.27 tell you on an in-order core?**
On its own, that the pipeline is often waiting. For the NEON Full kernel, most instructions are
4-cycle q ops, so low IPC is expected. Look at MACs per cycle (0.47) and the hotspots (44% on loads).

**3. Why did fast-math make Lite slower?**
It vectorized across frames with `vld3` de-interleaves, ran out of q registers (spills), and clang's
A7 model split VMLA into vmul + vadd. Fewer instructions, but more cycles.

**4. Why did Eigen's `lazyProduct` lose?**
It removed packing, but it reloaded weights for every column. Loads +18%, so the A7 became load-bound.

**5. Why is NEON not faster for float math on the A7?**
Measured: every FP form peaks at ~1 lane-op per cycle (q = 4 lanes in ~4 cycles). Vectorizing only cuts
non-math instructions.

**6. What did the pipeline cost, and was it worth it?**
+1 block of latency (2.67 ms at 128) and a second busy core. Yes for Full (it became real time); not for
Lite (same total delay as unpipelined 32).

**7. How do you know an optimization didn't change the sound?**
A null test: subtract two renders and require ≲ −100 dB residual (ours: bit-identical to −130 dB).

**8. What caused the 0.254 s spikes?**
The memory sampler starting `awk`/`sed`/`date` every 0.2 s. Shared L2 and memory-bus contention.
Confirmed with a sampler-on control.

**9. Why does `--rt` only run in paced mode?**
RT throttling (95% per second by default) would stall a SCHED_FIFO loop that never sleeps.

**10. Why do Lite and Full use the same memory in `bench`?**
12.7 MiB of the 16.9 MiB is the decoded test file, and both submodels are always loaded.

**11. Why compare clang to clang?**
Same code, different compilers, up to 26% apart (GCC spilled the Full kernel).

**12. Why split the pipeline at layer 13?**
It balances the measured cost of the two stages. Each layer off-balance costs ~140 µs per block.

## Key takeaways

> - The remaining gains are in the OS (isolation, RT kernel) and the audio path (USB vs I2S).
> - Research directions: integer math and the GPU, to get past the ~1 FP op/cycle ceiling.

## Check yourself

Cover the answers above and explain each in two sentences.
