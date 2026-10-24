# audio/

`di.wav`: dry guitar DI used by every benchmark. Not committed.

Made from `raw/Dry Guitar.wav` (44.1 kHz stereo 16-bit, 84.5 s): kept 12.8-82.2 s (dropped the quiet
intro and the silent tail), averaged L+R to mono, resampled to 48 kHz, saved as 32-bit float
(NAM's WAV reader rejects ffmpeg's 24-bit "extensible" WAVs):

    ffmpeg -ss 12.8 -to 82.2 -i "raw/Dry Guitar.wav" \
      -af "pan=mono|c0=0.5*c0+0.5*c1,aresample=48000:filter_size=64:cutoff=0.97" \
      -c:a pcm_f32le -flags +bitexact di.wav

Result: 69.4 s, mono, 48 kHz, Float32.
`raw/check_full_amp.wav` is the A2 Full model's output, for a listen.
