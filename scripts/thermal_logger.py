#!/usr/bin/env python3
"""Low-interference thermal logger for timed runs. Standard library only.

One long-lived process: every INTERVAL seconds it reads only the SoC temperature from sysfs and appends a
CSV row. It never forks, and it doesn't poll anything that goes through the VideoCore firmware during the
run. The shell logger's `vcgencmd` calls stalled the audio cores by up to ~5 ms, and even its sysfs version
(forking cat/cut/date and reading cpufreq/hwmon every second) caused periodic late wake-ups. See
experiment-report.md, iteration 12. CPU frequency, the under-voltage alarm and `vcgencmd get_throttled`
(whose "has occurred" bits are sticky since boot) are read once at start and once at exit, into
<out>_throttled.csv.

Usage: python3 scripts/thermal_logger.py results/thermal_<tag>.csv [interval_s]   (stop with SIGTERM/SIGINT)
"""
import glob
import os
import signal
import subprocess
import sys
import time

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/thermal.csv"
INTERVAL = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
TEMP = "/sys/class/thermal/thermal_zone0/temp"
FREQ = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
UV = (glob.glob("/sys/devices/platform/soc/soc:firmware/raspberrypi-hwmon/hwmon/hwmon*/in0_lcrit_alarm") or [None])[0]


def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except (OSError, TypeError):
        return "na"


def snapshot(phase):
    """Start/exit row: frequency, under-voltage alarm and decoded get_throttled bits."""
    try:
        thr = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=5).stdout
        hexv = thr.strip().split("=")[1]
        v = int(hexv, 16)
        bits = [v & 1, v >> 1 & 1, v >> 2 & 1, v >> 3 & 1, v >> 16 & 1, v >> 17 & 1, v >> 18 & 1, v >> 19 & 1]
    except (OSError, IndexError, ValueError, subprocess.SubprocessError):
        hexv, bits = "na", ["na"] * 8
    return [phase, f"{time.time():.3f}", read(FREQ), read(UV), hexv] + [str(b) for b in bits]


def main():
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    thr_out = OUT[:-4] + "_throttled.csv" if OUT.endswith(".csv") else OUT + "_throttled.csv"
    with open(thr_out, "w") as f:
        f.write("phase,unix_time,scaling_cur_freq_khz,under_voltage_alarm,throttled_hex,under_voltage_now,"
                "freq_capped_now,throttled_now,soft_temp_limit_now,under_voltage_ever,freq_capped_ever,"
                "throttled_ever,soft_temp_limit_ever\n")
        f.write(",".join(snapshot("start")) + "\n")

    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))

    new = not os.path.exists(OUT) or os.path.getsize(OUT) == 0
    with open(OUT, "a", buffering=1) as f:
        if new:
            f.write("unix_time,temp_c\n")
        next_t = time.monotonic()
        while not stop["now"]:
            t = read(TEMP)
            temp = f"{int(t) / 1000:.3f}" if t.isdigit() else "na"
            f.write(f"{time.time():.3f},{temp}\n")
            next_t += INTERVAL
            time.sleep(max(0.0, next_t - time.monotonic()))

    with open(thr_out, "a") as f:
        f.write(",".join(snapshot("end")) + "\n")


if __name__ == "__main__":
    main()
