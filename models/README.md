# models/

Put .nam files here (not committed).

- **A2 files** (`"architecture": "SlimmableContainer"`) hold two submodels, Lite and Full. The sweep
  runs both automatically as `<name>@lite` and `<name>@full`, so one A2 file gives you a same-amp
  size comparison.
- Current: `a2_fender_super_reverb.nam` (Fender Super Reverb 1965 amp + cab, A2, 48 kHz,
  modeled by amalgamaudio, from TONE3000). Lite = 3 channels, Full = 8 channels, same 23 layers.
