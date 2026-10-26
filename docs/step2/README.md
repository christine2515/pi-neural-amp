# Step 2 docs

Written after the first live tests (2026-10-25), for the next phase of the project.

| Doc | What |
|---|---|
| [state-of-the-union.md](state-of-the-union.md) | How the audio runs right now: signal chain, threads/cores, recommended `live` commands and measured numbers |
| [optimizations-summary.md](optimizations-summary.md) | Every optimization with its measured effect and status (adopted, rejected, pending, needs Christine) |
| [natural-language-tone-selection.md](natural-language-tone-selection.md) | Long-term design: describe a tone in words, TypeSafe's Jev picks a NAM model, `live` hot-swaps it |
| [connected-amp-design.md](connected-amp-design.md) | Hardware and integration: USB WiFi + phone hotspot, private model store (within TONE3000's terms), I2C OLED display, how `tonectl` calls Jev and drives `live`, interference tests, phases |

Step 1 material (the experiment record and research that led here) stays in [`docs/`](../).
