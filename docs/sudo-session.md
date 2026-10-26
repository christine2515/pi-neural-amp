# Christine's sudo / hardware session

The remaining performance items need root, a reboot, or hardware. Each step says what to run, what it
answers, and what Claude does afterwards. Steps are independent; do them in this order if doing all.
Facts from the Pi (2026-10-25): kernel 6.18.50+rpt-rpi-v7, `CONFIG_PREEMPT_VOLUNTARY=y`, `CONFIG_HZ=100`,
`CONFIG_NO_HZ_FULL` not set, `kernel.sched_rt_runtime_us = 950000`, USB IRQ (`dwc_otg`, IRQ 89) on all 4
cores (~2,000 interrupts/s per core with the Scarlett attached), `perf` installed, `trace-cmd` not installed.

## 0. Power (no sudo)
- Use a 5 V / 2.5 A supply, or put the Scarlett on a powered USB hub. `vcgencmd get_throttled` showed
  `0x50000` (under-voltage and throttling occurred) after the Scarlett was plugged in.
- Claude afterwards: reboot-free check with `vcgencmd get_throttled` after a reboot → should be `0x0`.

## 1. Move USB interrupts off the audio cores (no reboot; reverts on reboot)
```bash
echo 1 | sudo tee /proc/irq/89/smp_affinity
```
Routes `dwc_otg` (USB + Ethernet) to core 0 only. **Answers:** do the rare live overruns at 64 × 2 / 48 × 2
and the jitter tails come from USB interrupts landing on cores 1–3? Claude reruns the live 10-minute runs
(`live10_full_pipe_64x2`, `live10_full_fsplit3_48x2`) and compares underruns.
Risk: low; if USB audio misbehaves, reboot to undo.

## 2. Kernel trace of what interrupts the audio cores (no reboot)
While Claude runs a paced benchmark on cores 2–3, run:
```bash
sudo perf record -a -C 2,3 -e sched:sched_switch -e irq:irq_handler_entry -e irq:softirq_entry \
  -e timer:hrtimer_expire_entry -o /tmp/audio_cores.data -- sleep 30
sudo perf script -i /tmp/audio_cores.data > ~/pi-neural-amp/results/audio_cores_trace.txt
sudo chown ctu ~/pi-neural-amp/results/audio_cores_trace.txt
```
**Answers:** what runs on the audio cores besides our threads, and identifies the 2.000 s event seen in
the low-latency runs (the `raspberrypi_hwmon` 2.08 s mailbox poll was already ruled out by correlation:
2 of 22 polls near a spike = chance). Claude parses the trace.

## 3. Real-time throttling test for the 4-core frame split (no reboot; temporary)
```bash
sudo sysctl kernel.sched_rt_runtime_us=-1     # disable RT throttling
# Claude runs the 4-thread frame-split test, then:
sudo sysctl kernel.sched_rt_runtime_us=950000 # restore
```
**Answers:** are the 41–51 ms stalls of 4-core frame split the kernel's RT throttling (95% per second)?
Risk: with throttling off, a runaway SCHED_FIFO thread can hang the Pi until reboot. Only during the test.

## 4. Core isolation (cmdline + reboot)
```bash
cd ~/pi-neural-amp && ./scripts/isolcpus.sh enable 1-3 && sudo reboot
# after reboot:
./scripts/isolcpus.sh status
```
Isolates cores 1–3 (`isolcpus=1-3 irqaffinity=0 rcu_nocbs=1-3`), leaving core 0 for Linux and IRQs.
**Answers:** do the remaining overruns disappear (offline 30-min run had 7 marginal ones; live 64 × 2 had
1 in 10 min), and can Full run at 32 or 24 frames with frame split? Claude reruns `compare_rt.sh`
(rt_isol arm), the live sweeps and a 30-minute run. Undo: `./scripts/isolcpus.sh disable && sudo reboot`.
Note: `nohz_full` (removing the 100 Hz tick, +40–60 µs per hit) needs `CONFIG_NO_HZ_FULL`, which the
stock kernel doesn't have: that belongs in the RT kernel build (step 7).

## 5. Graphics userspace for the GPU latency measurement (no reboot)
```bash
sudo apt install libegl1 libgles2 libgbm1 libegl-mesa0 libgl1-mesa-dri
```
~24 packages incl. ~30 MB Mesa. **Answers:** per-pass and readback latency of OpenGL ES fragment shaders
on the VideoCore IV vs the 25 µs/pass bar. Claude's GLES agent continues from its step 2.

## 6. Boot to first processed block (reboots)
```bash
cd ~/pi-neural-amp && ./scripts/install_boot_unit.sh models/a2_fender_super_reverb.nam
sudo reboot     # 3–5 times; each boot appends to results/boot_first_block.csv
sudo systemctl disable pi-neural-amp
```
**Answers:** the original plan's "boot to ready" KPI.

## 7. PREEMPT_RT kernel (later; a build project)
`rpi-7.1.y` or newer (32-bit ARM RT is mainline from Linux 7.1), with `CONFIG_PREEMPT_RT=y` and
`CONFIG_NO_HZ_FULL=y`; cross-compile on the Mac or a Linux VM. Then `nohz_full=1-3` on the cmdline.
Watch the USB driver's FIQ and the USB IRQ thread's priority under RT (see docs/optimization-roadmap.md).

## 8. Hardware
- **RCA → ¼" cable** (Solo line out → input 2) for round-trip latency: `./scripts/latency_test.sh`.
- Optional: I2S ADC/DAC + guitar buffer for sub-2 ms round trip (estimate; docs/low-latency-plan.md).
