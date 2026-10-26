# Natural-language tone selection: "warm bluesy clean" → the right NAM model, switched live

> **Draft (Claude), 2026-10-25.** Design doc for a long-term feature, not a plan of record.
> Christine decides scope and writes the conclusions. Project numbers come from the docs and CSVs named
> next to them. External facts link to their source. **Estimate** marks anything not measured; vendor
> numbers are labeled as vendor claims.

## 0. Summary

- **What "jev" is.** Most likely **Jev, TypeSafe AI's "decision model"**, released in early access in
  September 2026 ([InfoQ](https://www.infoq.com/news/2026/10/typesafe-ai-jev-released/),
  [typesafe.ai](https://typesafe.ai/)). It does not generate text. It answers typed questions (pick one
  of up to 255 options, score on a scale of up to 10 levels, true/false) with probabilities and a
  confidence value. It takes text only, is served only as a hosted API, and is billed per input token.
  That fits "pick one NAM model from a catalog" closely, and its output can only be a catalog entry.
  It cannot run on the Pi. Details and caveats are in section 2. The doc keeps the decider
  **pluggable** in case Christine meant something else.
- **v1 recommendation:** build an offline catalog on the Mac by rendering the same DI through every
  model, then measuring loudness, brightness and gain, and computing a **LAION-CLAP music** audio
  embedding. At query time, the request text is embedded (in the browser or on the Mac), and the
  nearest model is scored with simple descriptor rules on top. Switching on the Pi uses a model
  manager thread, a lock-free pointer swap at a block boundary, a short fade, and gain matching.
  Nothing goes to the cloud.
- **v2 recommendation:** hybrid. CLAP retrieval narrows the catalog to the top-K, then **Jev** makes
  the choice among those candidates with confidence gating, using the same API to classify the
  request as a new tone or a relative edit ("brighter", "less gain"). Claude (Haiku-class) is the
  fallback when Jev's confidence is low or Christine wants an explanation. Relative edits are
  computed in code from the measured descriptors.
- **The hard engineering is on the Pi, not in the ML.** As written, every pipelined Full model
  starts a spinning SCHED_FIFO worker when it is constructed (`patches/a2_pipeline.patch`), so
  "preload N models" would start N spinning threads. In addition, two Full models do not fit in the
  CPU budget at the same time, so a true crossfade is not possible for Full (section 5).

---

## 1. Problem statement and constraints

Christine types or says *"warm bluesy clean with a bit of breakup"*. Within about a second, the amp
changes to a capture that sounds like that, with no click, dropout or loudness jump, and the audio
never misses a deadline while this happens.

### 1.1 Hard constraints (from the project's measurements)

| Constraint | Value | Source |
|---|---|---|
| CPU | 4× Cortex-A7 @ 900 MHz, ~1 float op/cycle/core | beyond-fpu.md, "Correction first" |
| Audio cores in use | Lite: CPU 3 only, 35% / 45% of a 667 µs period (p50 / p99, 32 × 2). Full pipelined: CPU 3 + CPU 2, 73% / 81% of a 1,333 µs period (64 × 2), CPU 2 spinning at 100% | state-of-the-union.md §2, §4 |
| Free for everything else | **CPU 0–1**, which also run the OS, SSH and USB/Ethernet interrupts (not isolated) | state-of-the-union.md §2 |
| RAM | 1 GB total. `live` PSS 5.6 MB (Lite) / 5.8 MB (Full) whole process | state-of-the-union.md §4 |
| Network | 100 Mb Ethernet **behind the same USB chip as the Scarlett** | hardware-comparison.md; state-of-the-union.md §2 |
| GPU | None usable: vc4 KMS has no compute path, and no Mesa is installed | beyond-fpu.md §5 |
| Model load on the Pi (one A2 file, both submodels) | `load_ms` ≈ **100–120 ms** (JSON parse), `reset_ms` ≈ **175–470 ms** (allocate + prewarm) | `load_ms`, `reset_ms` columns of `results-pi/summary_*.csv` (e.g. `summary_pipe_full.csv`: 102.6 / 180.0 ms; `summary_baseline.csv`: 100.3 / 384.9 ms) |
| Receptive field | 6,347 samples = **132 ms**. A freshly loaded model needs 132 ms of input before its output is in steady state | learning/10 §4 |
| Audio-path rules | No allocation, locks, logging or syscalls in the process loop | CLAUDE.md |

### 1.2 What runs where

| Piece | When | Where | Why |
|---|---|---|---|
| Download captures, render DI, compute descriptors and audio embeddings | Offline, once per new model | **Mac** | Heavy (PyTorch, CLAP). Native arm64 renders are fast and match the Pi (null tests in optimization-roadmap.md) |
| Catalog JSON + `.nam` files | Copied to the Pi | Pi SD card | Small: one A2 file is 295 KB |
| Text → decision | Per request | Browser / Mac / cloud (v2), with a rules fallback on the Pi | The Pi can't run a useful text encoder or LLM next to the audio (section 3) |
| Load, prewarm and switch the model | Per request | **Pi, CPU 0–1 (manager thread)**, with the swap in the audio thread | Must never block the audio thread |

---

## 2. What "jev" is

### 2.1 Best match: TypeSafe AI's Jev (credible, with caveats)

- **What it is.** TypeSafe calls Jev a "System One" model: software sends a *state* (text or JSON)
  and a set of typed questions, and Jev returns typed answers with calibrated probabilities and a
  confidence value, not generated text ([TypeSafe docs: introduction](https://docs.typesafe.ai/introduction),
  [typesafe.ai](https://typesafe.ai/)).
- **Question types.** **Choice** (pick one; options given as `criteria` = name → description; **up to
  255 options**; returns `choice`, `probabilities`, `confidence`), **Score** (2–10 ordered levels;
  returns a probability-weighted score), **Noul** (0–1 truth of a statement)
  ([Choice](https://docs.typesafe.ai/primitives/choice.md), [Score](https://docs.typesafe.ai/primitives/score.md),
  [Pydantic AI integration](https://pydantic.dev/docs/ai/models/typesafe/)).
- **Models and limits.** `jev-1.13.0` and `jev-1.14.0` (Oct 14, 2026); `jev-latest` is an alias. The
  defaults are 64k tokens per request and 32k for the state plus any one question
  ([Models](https://docs.typesafe.ai/models.md)).
- **Modality.** **Text only.** It "does not accept audio or video" ([Models](https://docs.typesafe.ai/models.md)).
  For this project, that means the audio grounding has to be turned into text first (section 4's
  auto-generated descriptions and measured descriptors).
- **Size, weights, licensing.** No parameter count is published, the weights cannot be downloaded,
  and the docs describe access only through the hosted endpoint `POST /v1/systemone`
  ([Models](https://docs.typesafe.ai/models.md)). It is proprietary and usage-billed. **It cannot run
  on a Pi 2, or anywhere locally.** For an independent comparison against a fine-tuned classifier,
  see [Anaconda](https://www.anaconda.com/blog/decision-time).
- **Price and latency (vendor claims).** $0.042 per million input tokens, output free; 70–500 ms
  end to end ([InfoQ](https://www.infoq.com/news/2026/10/typesafe-ai-jev-released/),
  [Models](https://docs.typesafe.ai/models.md)). InfoQ also cites a third-party median of 76 ms.
- **Caveats.** "Can't hallucinate" means it can't return an out-of-schema answer. It can still pick
  the wrong valid option. TypeSafe's own notes say Jev is unreliable at counting, arithmetic and date
  comparison and should leave math to code
  ([InfoQ](https://www.infoq.com/news/2026/10/typesafe-ai-jev-released/)). It is in early access
  ([typesafe.ai](https://typesafe.ai/)), so availability and pricing can change.
- **Music precedent.** [Jevthoven](https://github.com/cocktailpeanut/jevthoven) uses Jev `Choice`
  calls to make each compositional decision for a MIDI generator, and it accepts plain-English
  revision requests. That is the same pattern proposed here: Jev chooses, code acts.

Search results also surfaced audio-flavored derivatives ("Qwen-Audio-Jev" on
[arXiv](https://arxiv.org/html/2610.21614v1), "AudioJev" on
[Hugging Face](https://huggingface.co/mocomoco-inc/AudioJev-AudioDecisionModel), and a "Jev-Omni"
mode on an unofficial playground). **I did not verify these.** TypeSafe's own docs list no audio
model.

### 2.2 Why this is probably what Christine meant

Jev was released the month before this doc and is described as a model that *decides*. Christine's
note says jev should "decide which nam model to use". Jev's Choice question with ≤255 described
options maps one-to-one onto "pick a NAM model from the catalog". I found no audio or music model
named Jev, and no common acronym that fits better.

### 2.3 If it's something else: keep the decider pluggable

Everything below talks to an interface rather than to Jev directly:

```python
class Decider(Protocol):
    def decide(self, request: str, candidates: list[CatalogEntry], current: ToneState) -> Decision: ...

@dataclass
class Decision:
    intent: Literal["new_tone", "relative_edit", "undo", "question"]
    model_id: str | None            # must be in candidates; validated in code
    edits: dict[str, float]         # e.g. {"brightness": +1, "gain": -1} for relative edits
    trim_db: float | None
    confidence: float               # 0..1; below threshold → fall back / ask
    explanation: str | None
```

The planned implementations are `RulesDecider` (Pi), `EmbeddingDecider` (text or CLAP), `JevDecider`
and `ClaudeDecider`. If "jev" turns out to be another model, it becomes one more class.

---

## 3. Architecture options

### 3a. Metadata + keyword rules on the Pi (baseline)

- **Inputs.** Per-model fields from the `.nam` file. Our file has `name` ("FNDR BFSR VB Edge WRM2 CAB
  FREE"), `modeled_by`, `gear_type` (`amp_cab`), `gear_make`, `gear_model`, `tone_type` ("T3K-Null"),
  `loudness` (−16.39), `gain` (0.628), `input_level_dbu`, `output_level_dbu`, `trainer`, `date`
  (`models/a2_fender_super_reverb.nam`). TONE3000 adds `title`, `description`, `gear`, `makes`,
  `tags`, `sizes`, `license` and the architecture per model
  ([TONE3000 API](https://www.tone3000.com/api)). The measured descriptors from section 4 can be added.
- **Method.** A hand-written lexicon: "clean" → gain descriptor low; "metal", "djent", "high gain"
  → gain high and `tags` ∋ metal; "fender", "blackface" → make; "bright", "dark", "warm" → spectral
  centroid; plus a BM25 or keyword score over names, tags and descriptions. Pure Python, under 10 ms.
- **Good:** offline, instant, private, explainable. **Bad:** misses paraphrase ("glassy", "spanky",
  "brown sound") unless it's in the lexicon. Capture names are inconsistent ("FNDR BFSR VB Edge
  WRM2").

### 3b. Text embeddings (request ↔ model description)

- **Model.** A MiniLM-class sentence embedder such as `all-MiniLM-L6-v2`: 22.7M parameters,
  384-dimensional output, inputs truncated at 256 word pieces, Apache 2.0
  ([model card mirror](https://huggingface.co/mmine/all-MiniLM-L6-v2),
  [Superlinked](https://superlinked.com/glossary/what-is-all-minilm-l6-v2)).
- **Catalog side.** Embed each model's text card (metadata, tags, and the auto-generated
  descriptor sentence from §4.4) offline. The catalog matrix is N × 384 floats: 200 models = 300 KB.
- **On the Pi 2? Possible but poor (estimate).**
  - Compute: about 2 × 22.7M ≈ 45 MFLOP per token, so a 16-token request is ≈ 0.7 GFLOP. At about
    1 flop/cycle on one free 900 MHz core, that is **≥ 0.8 s best case, likely 1.5–3 s**, on cores
    0–1, which already carry the OS and interrupts.
  - Memory: about 90 MB fp32 or about 23 MB int8.
  - Runtime: ONNX Runtime ships **no prebuilt 32-bit ARM packages**. ARM32v7 is a source-build target,
    and there are unofficial community wheels
    ([nknytk/built-onnxruntime-for-raspberrypi-linux](https://github.com/nknytk/built-onnxruntime-for-raspberrypi-linux),
    [ORT BUILD.md](https://github.com/lantiga/onnxruntime/blob/master/BUILD.md)). llama.cpp can embed
    BERT-class models, but recent fixes were still needed for 32-bit Arm fp16 builds
    ([llama.cpp commit cd644c39](https://app.semanticdiff.com/gh/ggml-org/llama.cpp/commit/cd644c39545aac3dca63261f99a9bfc35956cb25)).
  - It also competes with the audio for the shared memory bus.
- **Better placement.** Run it on the client. The page the Pi serves can run the embedder **in the
  browser** with transformers.js, then send the 384-float vector to the Pi, which computes N dot
  products (200 × 384 ≈ 77k MACs, well under 1 ms). Or run it on the Mac.
- **Limit:** it matches *words about the model*, not *how it sounds*.

### 3c. Audio–text joint embeddings (CLAP family): ground the choice in the sound

**Idea.** Offline, render a fixed DI riff through every model with `bench --render` and embed the
**audio** with a CLAP audio encoder. At query time, embed the **text** with the matching CLAP text
encoder and pick the nearest audio embedding. A capture called "FNDR BFSR VB Edge WRM2" then gets
matched on how it actually sounds.

| Checkpoint | Music? | Notes | Source |
|---|---|---|---|
| LAION-CLAP `music_audioset_epoch_15_esc_90.14.pt` | **Yes**, trained on music + AudioSet + LAION-Audio-630k | HTSAT-base audio encoder, **48 kHz input** (same rate as our renders), GTZAN 71%. Repo is CC0-1.0. API: `get_audio_embedding_from_filelist`, `get_text_embedding` | [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP) |
| LAION-CLAP `music_speech_audioset_…89.98.pt` | Mixed | GTZAN 51%, so prefer the music-only checkpoint | [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP) |
| `laion/larger_clap_music` (HF Transformers) | Yes | Swin-style audio encoder on log-Mel + RoBERTa text encoder | [model card mirror](https://huggingface.co/another-blue/larger_clap_music) |
| `Xenova/clap-htsat-unfused` (transformers.js) | General | `ClapTextModelWithProjection` runs **in the browser** and returns **512-d** `text_embeds` | [transformers.js modeling_clap](https://app.unpkg.com/@huggingface/transformers@4.2.0/files/src/models/clap/modeling_clap.js) |
| MS-CLAP 2023 (`msclap`) | General + music | pip package; versions 2022/2023/clapcap. Embedding size not confirmed in my sources (check the output shape) | [PyPI msclap](https://pypi.org/project/msclap) |
| MuQ-MuLan-large | **Music-specialized** | about 700M params, **24 kHz** input, code MIT but **weights CC-BY-NC 4.0** (personal use is fine; no commercial use) | [tencent-ailab/MuQ](https://github.com/tencent-ailab/MuQ) |

- **Embedding size.** The LAION-CLAP family maps both modalities into a 512-d space (transformers.js
  example above). 200 models × 512 floats = 400 KB.
- **Heavy part offline.** Audio embedding needs PyTorch and an HTSAT-base-sized model, which runs on
  the Mac, once per model.
- **Query-time text encoder.** RoBERTa-base size, about 125M parameters (**estimate** from the
  architecture name). On the Pi that is about 4 GFLOP per 16-token request, so **≥ 5 s** and
  about 500 MB fp32 (estimate), which is not viable. In the browser via transformers.js or on the
  Mac, it is fine.
- **Risk (open question).** CLAP was trained on general music and audio captions. It is unproven at
  telling *fine* amp differences apart ("blackface clean" vs "tweed clean"). It should separate
  coarse classes (clean / crunch / high gain, bright / dark). Section 7 tests this. That is why the
  descriptor rules (measured gain and brightness) stay in the scoring.

### 3d. LLM reading the catalog and choosing with tool calling

**Local on the Pi 2: not useful (estimate, with sources for the inputs).**
- The smallest instruction-tuned models are about 0.5B parameters. Qwen2.5-0.5B-Instruct Q4_K_M is a
  **398 MB** file ([HF upload](https://huggingface.co/ThomasBaruzier/Qwen2.5-0.5B-Instruct-GGUF/commit/d50bff78d3e58e25bf8568d9481f94538a12cb6a)).
- Compute: about 2 × 0.5B = 1 GFLOP per token. Just *reading* a 300-token prompt is 3 × 10¹¹ FLOP;
  at about 0.9–1.8 GFLOP/s on cores 0–1, that is **3–6 minutes** before the first output token. A
  12k-token catalog would take hours. No published Pi 2/3 numbers exist. Pi 5 (Cortex-A76) gets 15+
  tok/s on similar models ([Arm learning path](https://learn.arm.com/learning-paths/embedded-and-microcontrollers/raspberry-pi-smart-home/1-overview/)),
  and a Pi 1 got about 0.35 tok/s on a 90M model ([Better Stack](https://betterstack.com/community/guides/ai/llm-raspberry-pi.md)).
- It would also stream 398 MB of weights through the memory bus per token, next to the audio.
  **Ruled out.**

**Where it can run:** on the Mac (a small local model with llama.cpp or Ollama), on a phone, or in the
cloud. Claude via the API is the low-effort option. Indicative list prices for the current tiers are
Haiku 5.5 $0.10 / $0.50 and Sonnet 5.5 $2 / $10 per million input / output tokens
([Claude pricing](https://docs.claude.com/en/docs/about-claude/pricing); figures as of Oct 2026).
A 200-model catalog at about 60 tokens per entry is about 12k tokens: **about $0.001 per request on
Haiku** uncached, and less with prompt caching of the catalog prefix (estimate). Latency is about
0.5–2 s (estimate, not measured).

**Constrain the output to the catalog.** Use one tool with `strict: true` and an `enum` of the
candidate IDs, so a hallucinated ID cannot validate. The Pi still re-checks every ID
([tool use docs](https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview)).

```json
{
  "name": "select_tone",
  "description": "Pick the NAM capture that best matches the player's request, or a relative edit of the current tone.",
  "strict": true,
  "input_schema": {
    "type": "object",
    "additionalProperties": false,
    "required": ["intent", "model_id", "size", "trim_db", "gate_db", "confidence", "reason"],
    "properties": {
      "intent":     { "type": "string", "enum": ["new_tone", "relative_edit", "undo", "question"] },
      "model_id":   { "type": "string", "enum": ["t3k_29175_a2", "t3k_28241_a2", "fender_super_reverb_a2"] },
      "size":       { "type": "string", "enum": ["lite", "full"] },
      "trim_db":    { "type": "number", "description": "Output trim, -12..+6; the Pi clamps it" },
      "gate_db":    { "type": "number", "description": "Gate threshold dBFS, -80..-40; the Pi clamps it" },
      "confidence": { "type": "number", "description": "0..1 self-rated" },
      "reason":     { "type": "string", "description": "One sentence, shown in the UI" }
    }
  }
}
```

The enum is generated per request from the retrieved candidates. The catalog text (cards with
measured descriptors) goes in a cached system-prompt prefix.

**Jev as the decider (the same idea, different shape).** There is no generated text and no tool
loop. One `POST /v1/systemone` with several questions answered in parallel
([docs](https://docs.typesafe.ai/primitives/choice.md)):

```json
{
  "model": "jev-1.14.0",
  "state": "Request: \"warm bluesy clean with a bit of breakup\". Current tone: fender_super_reverb_a2 (clean, centroid 1.9 kHz, compression 2/5).",
  "questions": {
    "intent": { "type": "choice", "instructions": "What does the player want?",
      "criteria": { "new_tone": "A different amp or character", "relative_edit": "Adjust the current tone (brighter, less gain, ...)",
                    "undo": "Go back to the previous tone", "question": "Asking about the rig, not changing it" } },
    "model": { "type": "choice", "instructions": "Which capture best matches the requested sound?",
      "criteria": { "fender_super_reverb_a2": "1965 Fender Super Reverb amp+cab. Clean with slight edge; warm; low gain (2/5); centroid 1.9 kHz; tags: clean, blues",
                    "t3k_29175_a2": "...", "t3k_28241_a2": "..." } },
    "gain": { "type": "score", "instructions": "How much distortion is requested?",
      "criteria": ["Pristine clean", "Clean with edge", "Crunch", "High-gain rhythm", "Saturated lead"] },
    "brightness_edit": { "type": "score", "instructions": "If this is an edit, how should brightness change?",
      "criteria": ["Much darker", "Darker", "Same", "Brighter", "Much brighter"] }
  }
}
```

`model` can only be one of the listed IDs, and `confidence` decides whether to act or fall back.
Score answers become numeric targets that **code** turns into a model choice (Jev's docs advise
keeping math in code).

### 3e. Hybrid: retrieve with embeddings, rerank/decide with Jev or an LLM

1. CLAP (and/or MiniLM) retrieval → top-K candidates (K = 10–30, well under Jev's 255 cap and a
   short Claude prompt).
2. Jev Choice over the K candidates, plus intent and score questions. If `confidence ≥ τ`, act.
3. If not, fall back to Claude for a reasoned pick and an explanation, or to CLAP top-1 with
   "did you mean …?" in the UI.
4. Relative edits skip retrieval: code moves from the current model along the requested descriptor
   axis (section 6.3).

### 3.6 Comparison

| | a. Rules on Pi | b. Text embeddings | c. CLAP audio–text | d. LLM (Claude/Jev) over catalog | e. Hybrid c + Jev, Claude fallback |
|---|---|---|---|---|---|
| **Decision latency** | < 10 ms | 50–300 ms in browser/Mac (est.); 1.5–3 s on Pi (est.) | 100–500 ms text encode in browser/Mac (est.); ranking < 1 ms on Pi | Jev 70–500 ms (vendor); Claude about 0.5–2 s (est.) | about 0.2–1 s (est.) |
| **+ switch** (load + prewarm, if not preloaded) | +0.3–0.5 s measured (`load_ms` + `reset_ms`); about 0 if preloaded | same | same | same | same |
| **Matching intent** | Low–medium (lexicon-bound) | Medium (words, not sound) | Medium–high on coarse traits; unproven on fine ones | High on language, only as good as the catalog text | Highest expected |
| **Compute location** | Pi CPU 0–1 | Browser/Mac (Pi possible, poor) | Mac offline; browser/Mac online | Cloud (Jev only); cloud or Mac (Claude/local LLM) | Mac offline; browser + cloud |
| **Cost** | 0 | 0 | 0 | Jev ≈ $0.0005, Haiku ≈ $0.001 per request (est. at 12k tokens) | ≈ Jev cost; Claude only on low confidence |
| **Privacy** | Local | Local | Local | Request text + catalog leave the house | Same as d, only for the request |
| **Offline** | Yes | Yes (if browser model cached) | Yes | No | Degrades to c, then a |
| **Effort** | S | S–M | M | M | M–L |

**Recommendation.** **v1 = c with a's rules as tie-breakers and fallback.** It works offline,
it's free, it's grounded in the sound, and it's a good on-device-inference story. **v2 = e**, with
Jev as the default decider, Claude as the low-confidence fallback and explainer, and relative edits.

---

## 4. Catalog building (offline, Mac)

### 4.1 Getting models, within TONE3000's terms

TONE3000's API terms (effective July 7, 2026) say **"Do not bulk download, scrape, mirror, or cache
the TONE3000 catalog"** and that integrators should only download models when a user requests them.
They also forbid preloading catalog content onto devices without permission
([API terms](https://www.tone3000.com/api/terms)). The `GET /tones/{id}/download` endpoint is
**partner-only**, search is heavily rate-limited, and the free tier is for non-commercial products
([API docs](https://www.tone3000.com/api)). Per-tone licenses vary (`t3k`, `cc-by`, `cc-by-nc`, …,
[API docs](https://www.tone3000.com/api)). The default T3K license lets you use the file and publish
the *audio you make with it*, but not redistribute the file
([example tone page](https://www.tone3000.com/tones/ocd-29175)).

So the plan is:
- **Christine picks and downloads** each capture herself on the website (user-initiated, like any
  player). No crawler.
- For metadata, the catalog builder may use the free-tier, user-scoped endpoint **`GET
  /tones/downloaded`** (OAuth device flow, 100 req/min). It returns `title`, `description`, `gear`,
  `makes`, `tags`, `sizes`, `license` and `a2_models_count` for tones *she* downloaded
  ([API docs](https://www.tone3000.com/api)). `GET /models?tone_id=…&architecture=2` gives the model
  `size` and `architecture_version`.
- `.nam` files stay out of git (already the case: `.gitignore` has `models/*`). The catalog JSON
  stores IDs, licenses and derived descriptors, not weights. Renders stay local.
- If this ever becomes a product, it needs TONE3000's commercial terms and written permission to
  preload.

### 4.2 Filter to what the Pi can run

- **Prefer A2 files** (`architecture: SlimmableContainer`). They carry Lite + Full, and our NEON
  kernels are A2-specific. Costs are measured: **Lite 1,731 MAC/sample**, 32–40% of one core; **Full
  11,776 MAC/sample**, 66% of the deadline on 2 cores (README Results, experiment-report Summary).
- For other files, compute MAC/sample from the config (the derivation is in learning/02 "Counting
  the MACs") and admit them only if ≤ 1,731 × (a measured margin). A1 / custom files run on the
  generic NAM Core path, which was **251% of the deadline** for an 11,776-MAC model before our
  kernels (README Results), so A1 "standard" is out until it's measured. TONE3000 exposes `sizes`
  (`standard`, `lite`, `feather`, `nano`, `custom`) and an `architecture` filter
  ([API docs](https://www.tone3000.com/api)).
- Sample rate must be 48 kHz (`live` refuses a mismatch, `src/live.cpp`).

### 4.3 Render a standard DI through each model

```
for f in models/catalog/*.nam; do
  for s in 0.25 1; do
    ./build/bench --model "$f" --slim $s --input audio/di.wav --duration 20 --block 128 \
       --render renders/$(basename "$f" .nam)@$s.wav --summary-out results/summary_catalog.csv
  done
done
```

- Use the repo's own `bench --render` (`src/bench.cpp`), native on the Mac. It is bit-exact or
  within −130 dB of the Pi for A2 (experiment-report null tests), so the catalog describes exactly
  what the Pi will play.
- **DI set.** `audio/di.wav` (69.4 s, already used everywhere), plus three short standard clips
  (single notes, open chords, palm-muted chugs) cut from it or recorded on the Strat. Fix the input
  level, since gain-staging changes the character. Our captures report `input_level_dbu` (15 in our
  file); NAM Core exposes `GetInputLevel()` (`external/NeuralAmpModelerCore/NAM/dsp.h`).

### 4.4 Descriptors (`scripts/catalog_features.py`, NumPy only; Christine's from-scratch FFT preference applies)

| Descriptor | How | Why |
|---|---|---|
| `loudness_lufs` | K-weighted integrated loudness of the render (BS.1770; or RMS dBFS first) | Gain matching (section 5.5), "louder/quieter" |
| `centroid_hz`, `rolloff85_hz` | STFT on the render, energy-weighted mean / 85% roll-off | "bright / dark / warm" |
| `band_db[low, lowmid, mid, high]` | Energy in 80–250, 250–800, 800–3k, 3k–8k Hz relative to total | "scooped", "mid-forward", "boomy" |
| `small_signal_gain_db`, `knee_dbfs`, `ceiling_dbfs`, `compression_slope` | **Level-transfer curve**: 110 Hz sines at 3 dB steps, output RMS vs input RMS, the same method as figure `wn10_level_transfer.png` (learning/10 "Level in vs level out": our Fender is linear with about +14 dB below about −24 dBFS, flat near −10 dBFS above) | "clean vs breakup vs high gain", "touch sensitive" |
| `thd_pct@{-30,-12} dBFS` | Harmonic energy / fundamental for the same sines | Distortion amount at soft and hard picking |
| `crest_db` on the DI render | Peak/RMS of output | Compression feel |
| `gain_class` (0–4) | Thresholds on knee + THD, calibrated by Christine's labels | Coarse bucket used by rules and by Jev's Score answer |
| `clap_audio` (512 floats) | LAION-CLAP music checkpoint, mean over 10 s windows of the DI render | Audio-grounded retrieval |
| `text_card` | Template sentence from metadata + descriptors, e.g. "1965 Fender Super Reverb amp+cab; clean with slight edge (gain 1/4); warm (centroid 1.9 kHz); compression low" | Input to MiniLM, Jev and Claude |
| `text_emb` (384 floats) | MiniLM over `text_card` + tags + description | Text retrieval fallback |

### 4.5 Catalog JSON schema (proposed)

```json
{
  "schema": "pi-neural-amp/tone-catalog@1",
  "built_at": "2026-11-01T12:00:00Z",
  "di": {"file": "audio/di.wav", "sha256": "…", "seconds": 20, "input_level_dbfs_rms": -18.0},
  "embedders": {
    "clap": {"name": "laion-clap music_audioset_epoch_15_esc_90.14", "dim": 512},
    "text": {"name": "all-MiniLM-L6-v2", "dim": 384}
  },
  "models": [
    {
      "id": "fender_super_reverb_a2",
      "file": "models/a2_fender_super_reverb.nam",
      "sha256": "…",
      "source": {"site": "tone3000", "tone_id": null, "license": "t3k", "modeled_by": "amalgamaudio"},
      "nam": {"architecture": "SlimmableContainer", "sample_rate": 48000, "gear_type": "amp_cab",
              "gear_make": "Fender Super Reverb 1965", "tone_type": "T3K-Null",
              "loudness": -16.39, "gain": 0.628, "input_level_dbu": 15, "output_level_dbu": 14},
      "tags": ["clean", "blues"],
      "sizes": {
        "lite": {"slim": 0.25, "mac_per_sample": 1731, "weights": 1871},
        "full": {"slim": 1.0,  "mac_per_sample": 11776, "weights": 12146}
      },
      "features": {"full": {"loudness_lufs": -17.2, "centroid_hz": 1900, "band_db": [-6.1, -4.0, -5.5, -14.2],
                            "small_signal_gain_db": 14.1, "knee_dbfs": -24, "ceiling_dbfs": -10.2,
                            "thd_pct": {"-30": 0.8, "-12": 9.5}, "gain_class": 1}},
      "text_card": "1965 Fender Super Reverb amp+cab. Clean with slight edge…",
      "emb": {"clap": "emb/clap.npy#0", "text": "emb/text.npy#0"}
    }
  ]
}
```

The values in `features` are **placeholders** that show the shape, except the `nam` block, which is
copied from our file, and the sizes, which are measured. Embeddings live in `.npy` sidecars (float16
is enough), and the index in `emb` is the row.

---

## 5. Model switching on the Pi without glitches

### 5.1 Requirements

1. The audio thread never allocates, frees, locks or makes syscalls. It does at most one atomic load
   per period to notice a pending switch.
2. Load, `Reset()` (allocation + prewarm), free and file I/O all happen on a **manager thread pinned
   to CPU 0 or 1**, at `SCHED_OTHER` or a low FIFO priority below the audio threads.
3. The switch happens at a period boundary, with a fade, at matched loudness.

### 5.2 Memory: how many models can stay preloaded

- **Measured:** `live` PSS at 128 × 2 is bypass **4.4 MB**, Lite **5.5 MB** (experiment-report, Live
  table). So one loaded A2 container, with both submodels, buffers and parse leftovers, costs about
  **1.1 MB** of PSS.
- **From weight counts:** Full 12,146 floats = **48.6 KB**, Lite 1,871 = 7.5 KB (learning/02). Most of
  the 1.1 MB is per-layer history buffers (23 layers × channels × (6,347-sample receptive field +
  block slack)) and heap left over from JSON parsing. The file is 295 KB of JSON.
- **Estimate:** about 1–1.5 MB per preloaded A2 model, so **50 models ≈ 50–75 MB**, which fits in
  1 GB. Under `mlockall(MCL_FUTURE)` (`--rt`), all of it is locked, which is good for no page faults
  but is real RAM. Measure with `pss_kb` at N = 1, 5 and 20 preloaded models.
- **Do we need to preload?** Loading on demand measured **about 0.3–0.5 s** on the Pi (`load_ms` +
  `reset_ms`, section 1.1), which is inside a 1–2 s request budget. Preload only the **current
  model plus the top few candidates** (an LRU of about 8). Preloading everything isn't needed.

### 5.3 The pipeline-worker problem (must fix first)

`patches/a2_pipeline.patch` starts the stage-B worker **in the `A2FastModel` constructor** when
`NAM_A2_PIPELINE` names that size. The worker then spins with `yield` at SCHED_FIFO on CPU 2 until
the model is destroyed. Preloading N Full models would start **N spinning FIFO threads**. Proposed
change, as a new patch:

- Construct preloaded models with the pipeline **off**. Add `A2FastModel::AttachPipeline(PipelineWorker*)`
  and `DetachPipeline()`.
- Use **one long-lived `PipelineWorker`** per process (CPU 2, FIFO, 256 KB stack as in
  `a2_pipeline_stack.patch`). It executes stage B for whichever model is attached, through the
  same two-atomic handoff it uses today.
- On a switch, the audio thread re-points the worker at block boundary *n*. The worker must first
  drain stage B of block *n−1* for the old model. That is already the protocol's invariant: the
  audio thread waits for "B(n−1) done" before writing (state-of-the-union §3).

### 5.4 Prewarm, fade and CPU budget

The new model's 132 ms of history matters. A cold model fed silence starts from a "silent" state, so
its first 132 ms differ from steady state.

| Switch | CPU during transition | Method |
|---|---|---|
| **Lite → Lite** | 2 × Lite on CPU 3: p50 2 × 236 = 472 µs of a 667 µs period at 32 × 2 (71%); p99 2 × 303 = 606 µs (91%) (**estimate** from measured single-model numbers, state-of-the-union §4) | **True equal-power crossfade, 20–50 ms.** Safe at 64 × 2; tight at 32 × 2. During the crossfade, the new model has been fed the live input for ≥ 132 ms first ("shadow run"), so it is already warm, or use the replay below. Measure both |
| **Full → Full** | 2 × Full won't fit: 73% + 73% of the period on the same 2 cores | **Dip switch:** fade the old model out over about 10 ms, swap at a boundary, fade the new model in over about 20 ms. Before the swap, the manager **replays the last 150 ms of input** (from a lock-free SPSC input ring the audio thread fills) through the new model on CPU 0/1, so its history is almost current. The fade-in hides the small mismatch from the samples that arrived during the replay |
| **Lite ↔ Full** | Pipelined Full is **one block later** than Lite (state-of-the-union §3) | Dip switch. A crossfade would mix two signals offset by 64 samples (comb filtering) unless Lite is delayed one block during the fade |
| **Same file, change `--slim`** | Same as Lite ↔ Full | Both submodels are already loaded, but `SetSlimmableSize` calls `Reset` (allocation), so it is done on the manager thread with a pre-built second instance |

The dip is a short level dip (about 30 ms) rather than silence. Whether that is audible while
playing is a listening question for Christine (section 7). It is similar to the gap when switching
channels on a real amp (**interpretation**).

### 5.5 Gain matching

- NAM files carry `loudness`, defined as the model's output loudness for a standardized input. NAM
  Core exposes it through `HasLoudness()` / `GetLoudness()`
  (`external/NeuralAmpModelerCore/NAM/dsp.h`; our file: −16.39 dB Full, −16.43 Lite).
- Better: use our **own** `loudness_lufs` from the DI render (section 4.4). It is measured with the same
  DI and level that the Pi will see.
- `trim_db = target − loudness`, clamped to [−12, +6] dB, then applied as a per-sample gain
  ramp across the fade. The audio thread only multiplies.
- Expect perceived loudness to still differ between clean and high-gain sounds at equal LUFS
  (compression and brightness). Leave a user "louder/quieter" edit (section 6.3).

### 5.6 Interfaces (C++ sketch, not code)

```cpp
// --- shared between threads (no locks) ---
struct ActiveModel {                 // built and prewarmed off the audio thread
  std::unique_ptr<nam::DSP> dsp;
  std::string id;                    // catalog id
  float trimLinear;                  // from catalog loudness
  int   latencyBlocks;               // 0 (Lite) or 1 (pipelined Full)
};
struct SwitchRequest { ActiveModel* next; int fadeOutBlocks, fadeInBlocks; bool crossfade; };

std::atomic<SwitchRequest*> gPending{nullptr};   // manager → audio (exchange at block boundary)
std::atomic<ActiveModel*>   gRetired{nullptr};   // audio → manager (frees off the RT thread)
SpscRing<float, 1<<14>      gInputHistory;       // audio → manager: last ~340 ms of gated input

// --- manager thread (CPU 0/1) ---
class ModelManager {
 public:
  bool Preload(const std::string& id);            // parse + construct (pipeline detached) + Reset
  bool Prepare(const std::string& id, double slim); // replay gInputHistory to warm history
  void RequestSwitch(const std::string& id, FadeSpec f); // publishes SwitchRequest via gPending
  void Poll();                                     // reclaims gRetired, trims LRU cache
};

// --- control plane ---
// Unix socket /run/pi-amp/ctl.sock, one JSON line per command:
//   {"cmd":"switch","id":"fender_super_reverb_a2","size":"full","fade_ms":30,"trim_db":-1.5}
//   {"cmd":"preload","ids":[...]}  {"cmd":"status"}  {"cmd":"gate","db":-50}
// Served by a ControlServer thread (CPU 0/1, poll()) that calls ModelManager.
```

The audio loop gets one addition at the top of each period: `if (auto* r = gPending.exchange(nullptr,
std::memory_order_acq_rel))` starts a fade state machine. It holds the two `ActiveModel*` pointers
during the fade, re-points the pipeline worker at the swap block, and publishes the old model to
`gRetired` after the fade. `live` keeps its CSV timing, so switch overhead shows up in `proc_*_us`.
Add a `switch_at` column to `--periods-out` so every switch can be found in the trace.

The tone **decider does not live in `live`**. A separate Python daemon (`tonectl`, CPU 0–1, nice 10)
serves the web page, holds the catalog, runs rules or embedding ranking, calls Jev or Claude in v2,
and sends one JSON line to `live`'s socket.

---

## 6. User interface

### 6.1 Text box (v0–v1)

- `tonectl` serves one static page on the Pi (e.g. `http://pi-amp.local:8080`): a text box, the
  current model name, a "why" line, ✓/✗ feedback buttons, and "back".
- **v1 option for local embeddings:** the page loads transformers.js and a CLAP text encoder
  (`Xenova/clap-htsat-unfused` is ready-made, [source](https://app.unpkg.com/@huggingface/transformers@4.2.0/files/src/models/clap/modeling_clap.js);
  a music checkpoint needs an ONNX export, **effort: unverified**). The browser caches the weights,
  so after the first load the request costs one text embedding on the phone or laptop, and the page
  POSTs `{text, clap_vec}` to the Pi.

### 6.2 Voice

- Use the phone's or Mac's own speech-to-text (OS keyboard dictation into the text box, or the
  browser's Web Speech API where available). Recognition happens on the client, never on the Pi.
- The request reaches the decider as text, so voice adds no new decider path.

### 6.3 Feedback loop: relative edits

- "Brighter", "less gain", "more like the last one but tighter": Jev's `intent` answers
  `relative_edit`. Score questions give the direction and size per axis (brightness, gain, low end,
  loudness).
- **Code**, not the model, picks the next model. It takes the nearest catalog entry to
  `current.features + Δ` on normalized descriptor axes and requires movement in the requested
  direction ("brighter" ⇒ centroid increases by at least a JND-like threshold). If nothing qualifies,
  it says so ("this is already the brightest clean capture") instead of switching. "Louder/quieter"
  changes only `trim_db`.
- Every request, choice and ✓/✗ is logged to `results/tone_requests.csv`. That log becomes the labeled
  set for section 7.

---

## 7. Evaluation plan

### 7.1 Does it pick well?

- **Labeled set.** About 60 prompts written by Christine before seeing any results (clean, blues
  breakup, classic rock crunch, modern metal, fuzz, "like <artist>", "brighter", …). For each, she
  marks the set of **acceptable** models in the catalog (there is often more than one) and one
  **best**. Hold out 20 prompts that are never used for tuning lexicons, thresholds or prompts.
- **Retrieval metrics.** Top-1 accuracy (best), **top-1 acceptable rate**, top-3 recall and MRR,
  computed per option (a, b, c, e) on the same prompts. Report Jev's calibration: accuracy vs
  `confidence` bucket, which sets the fallback threshold τ.
- **Coarse-trait test for CLAP.** Does CLAP rank "clean" prompts above "high gain" prompts for each
  model in line with the measured `gain_class`? Report Spearman ρ. This tells us whether CLAP adds
  anything beyond the descriptors.
- **Blind listening.** For 20 prompts, Christine hears 2–3 renders (the chosen model, the keyword
  baseline's choice, and a random acceptable-class model) in shuffled order, with names hidden and
  loudness matched, and picks the best fit. Win rate vs baseline.
- **Edits.** For "brighter" or "less gain", check that the descriptor moved in the requested
  direction (automatic) and that Christine agrees (listening).

### 7.2 Latency budget (targets, to be measured)

| Step | Budget | Basis |
|---|---|---|
| Browser → Pi request | < 50 ms | LAN (estimate) |
| Decide: rules / CLAP rank / Jev / Claude | < 10 ms / < 0.5 s / < 0.5 s / < 2 s | estimates; Jev vendor 70–500 ms |
| Load + prewarm (not preloaded) | 0.3–0.5 s | **measured** `load_ms` + `reset_ms`, `results-pi/summary_*.csv` |
| Replay-warm 150 ms of input | Full about 0.2 s on one core, Lite about 0.05 s (**estimate** from Full single-core about 3.4 ms per 128-sample block) | to measure |
| Fade | 20–50 ms | design |
| **Request → new sound** | **< 1.5 s p95** (v1), < 2 s with cloud (v2) | target |
| **Audio during all of the above** | Same underrun rate as without switching (Full 64 × 2: 1 / 450,000; Lite 32 × 2: 0) and p99 `proc` within +5% | state-of-the-union §4 |

The audio criterion is the one that matters. Run `live --duration 600` while a script issues a switch
every 10 s (60 switches), and compare underruns and `proc` p99 against a no-switch run of the same
length. Repeat while `tonectl` is serving requests over Ethernet, because Ethernet shares the USB
controller with the Scarlett.

---

## 8. Phased roadmap

| Phase | Scope | Effort (estimate) | Christine | Claude |
|---|---|---|---|---|
| **v0: manual switching** | Pipeline-worker refactor (5.3); `ModelManager`, control socket, fade state machine, input history ring in `live`; `ctl.py switch <id>`; **measured**: switches every 10 s for 10 min, Lite and Full, underruns / `proc` p99 / PSS vs N preloaded | 1.5–2 weeks part-time | Download 10–20 A2 captures of varied character; run the live tests; **listen to the dip vs crossfade**; write the analysis | Patch, `live` changes, scripts, CSV columns, plots |
| **v1: catalog + embeddings** | `catalog_build.py` (renders via `bench --render`, descriptors, level-transfer curves, CLAP, MiniLM, catalog JSON); `tonectl` with a rules decider and CLAP ranking; web page with in-browser text encoder; request log | 2–3 weeks | Labels for 60 prompts; blind listening; decide whether CLAP beats descriptors alone; write the DSP descriptor code herself where she wants to (FFT, centroid, loudness) | Catalog pipeline glue, embedding scripts, web page, eval harness |
| **v2: Jev chooser + feedback** | `JevDecider` (intent + choice + score), confidence gating, `ClaudeDecider` fallback with strict tool schema, relative edits, explanation line; eval of a vs c vs e on the held-out prompts | 1.5–2 weeks | Decide τ from the calibration plot; judge the explanations; API keys (her accounts) | Decider classes, prompts, eval runs, cost/latency logging |

**Interview value.** One demo covers three ML-systems topics with real measurements:
1. On-device real-time inference: a hand-optimized WaveNet at 66% of the deadline on a 2015 CPU,
   and the hard part of hot-swapping models under a deadline.
2. Retrieval: audio–text embeddings over offline renders, measured with top-k accuracy.
3. Agentic or decision-model tool use: constrained outputs, confidence-gated fallback, cost and
   latency per request.

The design questions ("why not run the LLM on the Pi?", "how do you guarantee a valid model ID?",
"how do you swap a model without a glitch?") all have measured answers.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| **Licensing of captures.** TONE3000 API terms forbid bulk download, scraping, caching and preloading the catalog. Per-tone licenses include NC and ND variants ([terms](https://www.tone3000.com/api/terms), [API](https://www.tone3000.com/api)) | User-initiated downloads only; no `.nam` in git; store `license` per entry; personal, non-commercial use; ask TONE3000 before anything public |
| **Audio contention.** Load and parse, replay-warm, Python, embedding and network all share the memory bus and L2 with the audio cores. Ethernet shares the USB controller with the Scarlett | Manager and `tonectl` stay on CPU 0–1 at low priority; the v0 test measures underruns *during* switches and requests; core isolation (open item) helps |
| **Spinning pipeline workers per preloaded model** | Single shared worker (5.3) before any preloading |
| **Two Full models exceed the CPU budget** | Dip switch for Full; true crossfade only for Lite |
| **Memory growth** | LRU of about 8 preloaded models (about 10 MB, estimate); watch `pss_kb`; note `live`'s per-period arrays already grow with `--duration` (state-of-the-union §4) |
| **Network dependence** (Jev and Claude are cloud-only) | v1 works fully offline; v2 falls back to CLAP, then rules, on timeout (e.g. 1 s) |
| **Hallucinated or invalid model IDs** | Jev Choice can only return listed options; Claude uses a strict tool with a per-request `enum`; `tonectl` and `live` still validate the ID against the catalog and the file hash |
| **Wrong but valid pick** | Confidence gating, "did you mean" alternatives, ✓/✗ logging, held-out eval |
| **Loudness jumps** | Trim from measured LUFS, ramped over the fade; clamp; user loudness edits |
| **CLAP may not separate fine amp differences** | Descriptor rules carry coarse traits; the eval decides whether CLAP earns its place |
| **Vendor risk** (Jev is early access; pricing and models can change) | Pluggable `Decider`; pin `jev-1.14.0` once τ is tuned (Pydantic AI docs recommend pinning, [link](https://pydantic.dev/docs/ai/models/typesafe/)) |
| **"jev" meant something else** | Ask Christine; the design doesn't depend on it |

---

## Sources

- TypeSafe Jev: [docs intro](https://docs.typesafe.ai/introduction), [models](https://docs.typesafe.ai/models.md),
  [Choice](https://docs.typesafe.ai/primitives/choice.md), [Score](https://docs.typesafe.ai/primitives/score.md),
  [typesafe.ai](https://typesafe.ai/), [InfoQ](https://www.infoq.com/news/2026/10/typesafe-ai-jev-released/),
  [Pydantic AI](https://pydantic.dev/docs/ai/models/typesafe/), [Anaconda](https://www.anaconda.com/blog/decision-time),
  [Jevthoven](https://github.com/cocktailpeanut/jevthoven)
- TONE3000: [API](https://www.tone3000.com/api), [API terms](https://www.tone3000.com/api/terms),
  [tone page (license example)](https://www.tone3000.com/tones/ocd-29175)
- CLAP and music embeddings: [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP),
  [larger_clap_music card mirror](https://huggingface.co/another-blue/larger_clap_music),
  [transformers.js CLAP](https://app.unpkg.com/@huggingface/transformers@4.2.0/files/src/models/clap/modeling_clap.js),
  [msclap](https://pypi.org/project/msclap), [MuQ / MuQ-MuLan](https://github.com/tencent-ailab/MuQ)
- Text embeddings and runtimes: [all-MiniLM-L6-v2](https://huggingface.co/mmine/all-MiniLM-L6-v2),
  [Superlinked](https://superlinked.com/glossary/what-is-all-minilm-l6-v2),
  [ORT Pi wheels](https://github.com/nknytk/built-onnxruntime-for-raspberrypi-linux),
  [ORT BUILD.md](https://github.com/lantiga/onnxruntime/blob/master/BUILD.md),
  [llama.cpp 32-bit Arm fix](https://app.semanticdiff.com/gh/ggml-org/llama.cpp/commit/cd644c39545aac3dca63261f99a9bfc35956cb25)
- Small LLMs on Pis: [Qwen2.5-0.5B Q4_K_M size](https://huggingface.co/ThomasBaruzier/Qwen2.5-0.5B-Instruct-GGUF/commit/d50bff78d3e58e25bf8568d9481f94538a12cb6a),
  [Arm Pi 5 learning path](https://learn.arm.com/learning-paths/embedded-and-microcontrollers/raspberry-pi-smart-home/1-overview/),
  [Better Stack Pi 1](https://betterstack.com/community/guides/ai/llm-raspberry-pi.md)
- Claude API: [pricing](https://docs.claude.com/en/docs/about-claude/pricing),
  [tool use](https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview)
- Project: README.md, docs/state-of-the-union.md, docs/experiment-report.md, learning/02, learning/10,
  `results-pi/summary_*.csv` (`load_ms`, `reset_ms`), `patches/a2_pipeline.patch`,
  `external/NeuralAmpModelerCore/NAM/dsp.h`
