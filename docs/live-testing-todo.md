# Live testing todo (Scarlett on the Pi 2)

Goal: get the measured offline results (Lite clean at 32 samples, Full clean at 128 with the two-core
pipeline) through real audio, and measure what offline can't: xruns, the USB/driver floor, true
round-trip latency, and the Reddit OS/audio ideas (mmap, isolcpus, periods).

**Who:** 🧑 = Christine (hardware, sudo, reboot, ears) · 🤖 = Claude (code, deploys, runs, analysis).
Everything writes CSV to `results/`; analysis goes into `docs/experiment-report.md` (new "Live" section).

## 0. Code prep (before the hardware session) ✅ done 2026-10-25
- [x] 🤖 `live --pipeline 8|3 [--pipeline-cpu N] [--pipeline-split L]` (shared `ConfigureA2Pipeline` with `bench`;
      worker gets the same SCHED_FIFO priority under `--rt`; label `+pipe` only when it applies to the submodel).
- [x] 🤖 `live --mmap` (`SND_PCM_ACCESS_MMAP_INTERLEAVED`, `snd_pcm_mmap_readi/writei`).
- [x] 🤖 `live` summary: new `access`, `pipeline`, `build_flags` columns (new header, so a new `live_summary.csv`).
- [x] 🤖 `live_sweep.sh`: `SLIM=` filter, `LIVE=`/`BENCH=` overrides; `latency_test.sh`: `LIVE=`; KPI table shows access + pipeline.
- [x] 🤖 `live` no longer fails on devices that can't link capture/playback (starts both streams instead).
- [x] 🤖 `nhead` (`build-armhf-nhead`) deployed to the Pi's `build/` (`bench` + `live`).
- [x] 🤖 Smoke test on ALSA's `null` device (no real timing): bypass, `--mmap`, Full `--pipeline 8 --rt`,
      Lite `--mmap --rt`, `--latency-test`. All run with 0 xruns; process times match `bench`
      (Full pipelined p50 1,763 µs, Lite 844 µs).
- [x] 🤖 **Fixed (patches/a2_pipeline_stack.patch, 13.8 → 5.8 MB):** with `--rt`, pipelined Full's PSS is 13.8 MB vs 5.6 MB for Lite. Likely cause: `mlockall`
      locks the worker thread's whole default stack (`ulimit -s` = 8 MB). Fix: create the worker with a
      small explicit stack (pthread attr in `a2_pipeline.patch`), then re-measure. Fix before quoting memory.

## 1. Hardware setup 🧑 (Scarlett Solo 3rd gen connected 2026-10-25; guitar = capture channel 1; ⚠️ under-voltage seen at plug-in; need a ¼" headphone adapter and an RCA → ¼" loopback cable)
- [x] Plug the Scarlett into the Pi's USB. If it crackles or drops out, use a powered USB hub.
- [x] **Direct Monitor OFF**, input 2 (Solo instrument jack) set to **INST**, gain low to start, headphones on the headphone out.
- [ ] Keep Ethernet plugged in (we need SSH), but don't stream anything during runs.
- [x] Tape the microSD card in (it popped out once).

## 2. Device checks 🤖 (read-only) ✅ hw:USB, 480M, S32_LE 2/2 ch, sync endpoints, 125 µs packets
- [x] `arecord -l` / `aplay -l`: card name (expected `USB` → `DEVICE=hw:USB`). ✅ card 2 "USB"
- [x] `lsusb -t`: the Scarlett should show **480M** (high speed). 12M would mean split transactions ✅ 480M
      through the Pi 2's hub, which is the worst case for the USB driver.
- [x] `cat /proc/asound/card*/stream0`: formats, rates, channels, endpoint interval. ✅ S32_LE 2/2 ch, sync, 125 µs
- [x] `vcgencmd get_throttled` once, before starting: `0x0`. Power is OK with the Scarlett attached. ⚠️ 0x50000 (under-voltage at plug-in; see docs/sudo-session.md)

## 3. First sound 🧑 + 🤖 (timing ✅: 0 underruns for bypass/Lite/Full at 128×2, Lite 32×2, Full pipe 64×2; listening pending headphones)
- [x] `live --bypass --period 128 --periods 2 --duration 20`: you should hear your dry guitar. Checks ✅ 0 underruns (timing; listened via --record)
      the wiring, channel and format.
- [x] `live --model … --slim 0.25 --period 128 --duration 60`: **Lite. Play it.** Does it sound like the ✅ recorded + listened
      Fender (`audio/raw/check_full_amp.wav` is the offline reference)? Any clicks?
- [x] `live --model … --slim 1 --pipeline 8 --rt --cpu 3 --period 128 --duration 60`: **Full, pipelined. Play it.** ✅ recorded + listened (now also frame split, see state of the union)

## 4. Round-trip latency 🧑 (cable) + 🤖 (runs)
- [ ] 🧑 Unplug the guitar; connect a cable from a Scarlett **output** to **input 2** (Solo: RCA line out → ¼"; line level, low gain).
- [ ] 🤖 `latency_test.sh` at periods 32/64/128/256 × 2/3 (bypass path): the **interface + driver floor**.
      Compare against buffer size (period × periods) to get the USB/converter overhead.
- [ ] 🤖 Same with `--rt`, and with `--mmap`.
- [ ] 🤖 Report: smallest stable configuration and its measured round trip.
      Reference: the Reddit Pi 5 post reports 3.5 ms with a Scarlett.

## 5. Live sweeps: xruns by configuration 🤖 (play or stay silent; timing doesn't depend on the input)
Each arm runs the period sizes (32/64/128/256) × periods (2/3) for 20–60 s; output is `live_by_config.csv`
and `live_smallest_buffer.csv`.
- [ ] Lite, plain
- [x] Lite, `--rt --cpu 3` ✅ lsw_lite
- [x] Lite, `--rt` + `--mmap` (**Reddit: ALSA mmap**): `SLIM=lite EXTRA="--rt --cpu 3 --mmap" TAG=live_lite_mmap ./scripts/live_sweep.sh` ✅ lsw_lite_mmap: no benefit
- [x] Full, `--pipeline 8 --rt --cpu 3`: `SLIM=full EXTRA="--pipeline 8 --rt --cpu 3" TAG=live_full_pipe ./scripts/live_sweep.sh` ✅ lsw_full_pipe
- [x] Full, pipelined + `--mmap` ✅ lsw_full_pipe_mmap: fails at 2 periods
- [x] Compare: offline smallest clean block vs live smallest xrun-free buffer. The gap is driver + USB cost. ✅ experiment-report Live sections

## 6. Core isolation 🧑 (sudo + reboot) then 🤖
- [ ] 🧑 `./scripts/isolcpus.sh enable 3` (Full pipeline: discuss `isolcpus=2,3`) → `sudo reboot`.
- [ ] 🤖 `isolcpus.sh status` → re-run the Lite `--rt` and Full pipelined arms (**Reddit: isolcpus**).
- [ ] 🤖 Also offline: `compare_rt.sh` adds the `rt_isol` arm, and a 30-min `sustained.sh` to see whether
      the 7 marginal overruns go away.
- [ ] 🧑 `./scripts/isolcpus.sh disable` + reboot when done (or keep it if it helps).

## 7. Robustness 🤖
- [ ] Ethernet interference: one Full live config while pushing traffic over Ethernet (`rsync` a big file).
      The Pi 2's USB and Ethernet share one chip.
- [ ] 30-min live run, Full pipelined at the smallest clean config, with `thermal_logger.py`: xruns per
      minute and temperature.
- [ ] Memory: `pss_kb` from `live_summary.csv`, the real memory number (no test file in memory).

## 8. Later / needs more from Christine
- [ ] 🧑 PREEMPT_RT kernel (`rpi-7.1.y`+, cross-compiled), then repeat sections 4–6. Watch the USB FIQ
      and IRQ-thread priority.
- [ ] 🧑 I2S ADC/DAC + guitar buffer, if section 4 shows the USB floor dominates.
- [ ] 🤖 Write up a "Live" section in the experiment report and update `learning/`.

## What "done" looks like
A table with **measured round-trip latency** and **xruns per minute** for Lite and Full at their smallest
stable buffer, each Reddit idea (mmap, isolcpus, periods, `--rt`) marked as measured help or no help,
and a sound check by ear.
