# Electric guitars, pickups, hum, and noise gates

How the signal this project processes is made, why a Strat hums, how we proved where the hum came
from, and what a noise gate does about it. Numbers are from this project's measurements
([../docs/experiment-report.md](../docs/experiment-report.md), "Live: background hum diagnosis";
`results-pi/hum_check.csv`). General physics is labeled as background.

## 1. How an electric guitar makes a signal (background)

- **Steel strings, magnets, coils.** A pickup is a permanent magnet (or magnetic pole pieces) wrapped
  in thousands of turns of very thin copper wire. The magnet magnetizes the steel string just above
  it. When the string vibrates, it changes the magnetic field through the coil, and by **Faraday's
  law** a changing magnetic field induces a voltage in the coil. That voltage *is* the guitar signal.
  It's a few hundred millivolts, at the same frequency as the vibrating string.
- **No string, no signal; nylon strings don't work.** The pickup senses steel moving in a magnetic
  field, not sound in the air. That's why an electric guitar is quiet unplugged and its tone depends on
  pickup position: near the bridge sounds brighter, near the neck warmer.
- **The output is high-impedance and weak.** A passive pickup is a big inductor with high output
  impedance, which is why audio interfaces have a dedicated **instrument (Hi-Z) input**: the Solo's
  input 2 with **INST** on. Plug a guitar into a low-impedance input and you lose level and treble.
- **Volume and tone knobs** are just a variable resistor (a voltage divider: volume) and a
  resistor + capacitor (a low-pass filter: tone) after the pickups. At volume 0, the divider shorts
  the pickups' output to ground. We used that in the hum diagnosis.
- **The rest of the chain in this project:** guitar → cable → Solo preamp (gain knob) → 24-bit ADC at
  48 kHz → USB → ALSA → `live` → NAM model → back out. "dBFS" means decibels relative to the
  converter's full scale: 0 dBFS is the largest value the ADC can represent; everything below is negative.

## 2. Single coils vs humbuckers (background)

- A **single-coil pickup** (Strat, Tele) is one coil. Besides the strings, the coil is also an
  **antenna for any changing magnetic field nearby**, including the 60 Hz field (50 Hz in Europe)
  around mains wiring, power bricks, transformers, chargers, monitors and dimmers. That becomes **hum**
  at 60 Hz and its harmonics (120, 180, 240 Hz…), plus buzzier high harmonics when the source is a
  switching supply or rectifier.
- A **humbucker** (Les Paul style) is two coils wound in opposite directions with opposite magnet
  polarity. The string signal adds, while hum from an outside field (which hits both coils the same
  way) is **cancelled**. Hence "hum-bucking."
- **Strat trick:** on many Strats the middle pickup is reverse-wound / reverse-polarity (RWRP), so
  selector positions 2 and 4 (two pickups together) cancel hum like a humbucker. Not verified for
  this Squier.
- **Single-coil hum is directional and position-dependent.** It depends on how the coil is oriented
  relative to the field lines. Turning your body 90° or moving away from a power brick can change it
  by 10–20 dB.

## 3. Why the amp model makes hum louder

Like a real high-gain amp, the NAM capture **boosts quiet signals the most** (it compresses: in
[10-wavenet-explained.md](10-wavenet-explained.md), the output saturates near −10 dBFS RMS once the
input passes about −24 dBFS RMS). In silence, the only input is hum, so the amp turned −72 dBFS of dry
noise into −58 dBFS (+14 dB). It also emphasized the 2.4–2.9 kHz hum harmonics, which is what made it
sound like a **buzz** rather than a low hum. While you play, the notes are much louder and mask it.

## 4. How we found the source: change one thing at a time

The method (same as every experiment in this project): hold everything constant, change one
variable, measure. Each step was a 10 s silent recording of the guitar input, analyzed for the
60 Hz series and the 2.4–2.9 kHz cluster (`scripts/hum_check.py`).

| Step | What changed | Hum (60 Hz series) | What it tells us |
|---|---|---|---|
| A | guitar plugged in, **volume 0** | **−126 dBFS** | With the pickups shorted out, the system is silent → Solo, cable, Pi, USB and power are clean |
| B | guitar **unplugged** | −108 dBFS | An open Hi-Z input picks up a little; fine |
| E | unplugged, **gain at minimum** | −119 dBFS | The floor scales with gain → nothing is injected after the preamp (rules out USB/power) |
| C | volume up, **hand on strings** | −69.5 dBFS | Loud hum with the pickups live |
| C2 | same, **not touching** | −68.7 dBFS | Touching doesn't matter → not a grounding/shielding fault |
| D | same, **guitar turned/moved** | −75.8 dBFS | 7 dB better from position alone → magnetic pickup hum |

**Verdict: single-coil pickup hum from the room's 60 Hz fields.** It isn't the Pi, the Solo or the
power supply.

**Lessons from how the test went:**
- **Controls need to be real.** The first round used countdowns, and the changes weren't made in time,
  so four recordings were identical (−91 ± 0.3 dBFS). A *self-checking* test (sweeping the gain knob
  while recording) exposed that: a flat level proved nothing had changed. The fix was to let the
  person trigger each recording ("ready B").
- **A wrong hypothesis is fine if the next test can kill it.** Step C suggested a floating ground
  (touching made hum worse); C2 showed touching made no difference, so that idea was dropped.
- **Confounds:** step C changed two things at once (lower volume *and* touching). C2 separated them.

## 5. Fixes, cheapest first

1. **Orientation and distance:** face away from power bricks, chargers and screens (7 dB measured).
2. **Selector positions 2 or 4** on a Strat, if the middle pickup is RWRP.
3. **Noise gate** (below). Software, no added latency.
4. **Hardware:** noiseless single coils or humbuckers; copper-foil shielding of the pickup cavity
   (helps electric-field noise more than magnetic hum).

## 6. What a noise gate is, and how `live --gate` works

A **noise gate** is an automatic mute: when the input is quieter than a threshold, it turns the
signal down. When you play, it opens. Nearly every guitar amp simulator has one, placed **before**
the amp model so the amp never amplifies the hum in the first place.

`live --gate -50` (in [../src/common.h](../src/common.h), `NoiseGate`):
- **Envelope follower:** tracks the input's recent peak, rising instantly and decaying over ~50 ms,
  so a note's level is followed smoothly.
- **Threshold with hysteresis:** opens when the envelope exceeds **−50 dBFS**, closes only when it
  drops below **−54 dBFS**. The 4 dB gap stops it chattering on and off around the threshold.
- **Hold (80 ms):** stays open briefly after the level drops, so it doesn't chop between notes.
- **Ramps:** gain rises in 1 ms (fast enough to keep the pick attack) and falls with a 100 ms
  release, so closing never clicks.
- **Cost and latency:** a few multiplies and compares per sample, with **zero added latency**: it
  doesn't look ahead.

**Choosing the threshold from data:** hum peaks measured −56 to −53 dBFS (worst positions), while
playing peaks had a median of −13 dBFS. So −50 sits above the hum and well below normal playing.
Very soft playing (the Full take dipped into the −50s) can get cut, which is the classic gate
trade-off; lower the threshold if it chokes soft notes. Offline test on the Full take: quiet
passages −63 → −76 dBFS (−13 dB), playing unchanged (0.0 dB).

**What a gate can't do:** remove hum *while* you play. Hum and notes are mixed then; only
removing the source (pickups, position) or filtering would help.

## Key takeaways
- A pickup is a coil + magnet: Faraday's law turns string motion into voltage, and also turns stray
  60 Hz magnetic fields into hum.
- Single coils hum, humbuckers cancel it; Strat positions 2/4 can cancel it too.
- Amp models (like real amps) boost quiet signals, so hum that's barely there dry becomes an audible buzz.
- Find noise sources by changing one thing at a time, with controls that prove the change happened.
- A noise gate mutes silence before the amp, with hysteresis, hold and ramps, and adds no latency.

## Check yourself
1. Why does an electric guitar need a steel string and an instrument (Hi-Z) input?
2. Why does a single-coil pickup hum and a humbucker not?
3. In step A, why did the hum almost disappear with the guitar volume at 0, and what did that rule out?
4. What did step E (gain to minimum, unplugged) prove about the Pi's power and USB?
5. Why did four early recordings agree to within 0.3 dB, and how did we notice?
6. Why put the gate *before* the amp model rather than after?
7. What problem does hysteresis solve? What does the hold time solve?
8. Why does the gate add no latency, and what would a "look-ahead" gate cost?

<details><summary>Answers</summary>

1. The pickup senses steel moving in its magnetic field; the pickup is a weak, high-impedance source
   that a low-impedance input would load down.
2. One coil picks up stray magnetic fields; two opposite coils cancel fields that hit both equally.
3. Volume 0 shorts the pickups' output, so −126 dBFS is the rest of the chain's noise. That ruled
   out the Solo, cable, Pi, USB and power as hum sources.
4. The floor dropped with the gain, so the noise enters at or before the preamp; anything injected
   after it (power, USB) would have stayed constant.
5. The changes weren't made before the countdown ended; a gain sweep with a flat level (±0.1 dB) proved it.
6. The amp boosts quiet input; gating after it would also cut the amp's own decay and tails, and
   hum would already be amplified and distorted.
7. Hysteresis stops rapid open/close chatter near the threshold; hold stops it closing between notes.
8. It only uses past samples. A look-ahead gate delays the audio by its look-ahead window to open
   before transients, which adds latency.
</details>
