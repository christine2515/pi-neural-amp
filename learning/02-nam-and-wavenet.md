# 02: NAM and the WaveNet model

## What NAM is

- **Neural Amp Modeler (NAM)** is an open-source project that trains a small neural network to imitate a
  specific amp or pedal.
- **How a capture is made:** play a fixed test signal through the real amp, record the output, and train
  the network to map clean input → recorded output sample by sample.
- **`.nam` file:** JSON with the architecture settings and trained weights. Ours,
  `a2_fender_super_reverb.nam`, captures a 1965 Fender Super Reverb amp plus cab (by amalgamaudio, from
  TONE3000). It's a `SlimmableContainer` holding two submodels.
- **NeuralAmpModelerCore (NAM Core)** is the C++ **library** (no `main()`, no audio I/O) that loads and
  runs models. Our `bench` and `live` programs compile it in. It bundles **Eigen** (matrix math),
  **nlohmann/json**, and **AudioDSPTools** (WAV loading).
- **Inputs it expects:** a **DI** (direct input) signal: dry guitar, no amp. Ours: 69.4 s, mono, 48 kHz.
  The original download was 44.1 kHz stereo and had to be converted.

## WaveNet, one sample's path

The A2 model is a WaveNet run **causally** (output depends only on current and past input).

1. **Rechannel:** input sample → a C-wide vector.
2. **23 layers**, each:
   - **Dilated convolution:** combine the current vector with ones from d, 2d, … 5d samples ago
     (kernel 6; two layers use kernel 15), through C×C weight matrices. Dilations are
     1, 3, 7, 17, 41, 101, 239 (×3 groups, plus 1 and 13).
   - **Input mixin:** add a weighted copy of the raw guitar sample (conditioning).
   - **LeakyReLU(0.01):** keep positives, scale negatives by 0.01. The nonlinearity, where the
     "distortion" lives.
   - **Skip connection:** add into a running **head sum** shared by all layers.
   - **1×1 conv + residual:** mix channels (C×C) and add back to the layer input, which becomes the next
     layer's input.
3. **Head:** kernel-16 convolution, C → 1, times `head_scale`.

**Receptive field:** each output depends on the last **6,347 samples ≈ 132 ms**. That's how the model
captures slow behavior like compression and sag. Dilation is the trick: a few layers cover a long time
cheaply.

## A2 Lite vs Full

| | Lite | Full |
|---|---|---|
| Channels C | 3 | 8 |
| Weights | 1,871 | 12,146 |
| MAC per sample | **1,731** | **11,776** |
| Layers, dilations, receptive field | same | same |

Per TONE3000, both come from one **slimmable** training run: one file can run either size, and
`--slim` picks which. NAM's container keeps **both** submodels loaded, which is one reason Lite and
Full showed the same memory.

## Counting the MACs (how to derive them)

| Step | Formula | Lite | Full |
|---|---|---|---|
| Dilated convs | Σ K × C × C over 23 layers (21×6 + 2×15 = 156 taps) | 1,404 | 9,984 |
| 1×1 residual | 23 × C × C | 207 | 1,472 |
| Input mixin | 23 × C | 69 | 184 |
| Head | 16 × C | 48 | 128 |
| Rechannel | C | 3 | 8 |
| **Total** | | **1,731** | **11,776** |

**Sanity check:** counting parameters the same way gives exactly the file's 1,871 / 12,146 weights.
The predicted Full/Lite ratio is 6.8×; the Pi measured 6.9× (6,664 vs 964 µs). The cost is the
arithmetic, and it scales with **C²**.

## Reading list

- WaveNet paper, sections 2.1 (causal dilated convs) and 2.4 (conditioning): https://arxiv.org/abs/1609.03499
- Wright, Damskägg, Juvela, Välimäki, "Real-Time Guitar Amplifier Emulation with Deep Learning" (2020):
  [PDF](https://acris.aalto.fi/ws/portalfiles/portal/41964332/Real_time_guitar_amplifier_emulation.pdf)
- Same group, RNN alternative (DAFx 2019): [PDF](https://acris.aalto.fi/ws/portalfiles/portal/36768418/ELEC_Wright_Real_time_black_box_DAFx2019.pdf)
- Atkinson, "Slimmable NAM": https://arxiv.org/html/2511.07470v1
- TONE3000 A2 posts (vendor claims, not independently verified):
  [intro](https://www.tone3000.com/blog/introducing-neural-amp-modeler-nam-architecture-2-a2),
  [slimmable](https://www.tone3000.com/blog/a2-next-generation-neural-amp-modeler),
  [team](https://www.tone3000.com/blog/nam-a2-team)
- The code: `../external/NeuralAmpModelerCore/NAM/wavenet/a2_fast.cpp`: read the top comment, `process()`,
  `_layer_forward_k`, then `SetMaxBufferSize`.

## Why FFT doesn't fit here

FFT convolution beats direct convolution for **long** kernels (hundreds to thousands of taps, e.g.
cabinet impulse responses). A2's kernels are 6–16 taps, so FFT overhead would cost more than it saves.
NAM Core's FFT path is for *Linear* models.

## Key takeaways

> - NAM Core is a library; the WaveNet is causal and dilated, with a 132 ms receptive field.
> - Lite and Full differ only in width (3 vs 8); cost ∝ C² (1,731 vs 11,776 MAC/sample).
> - Derive costs from structure and check them against the parameter count.

## Check yourself

1. Why does a dilated conv with dilations up to 239 give a long receptive field with few weights?
2. Derive Full's dilated-conv MACs from K, C and the layer list.
3. Why does Lite use ~6.8× less compute than Full even though both have 23 layers?
4. Why is FFT convolution a poor fit for this model?
