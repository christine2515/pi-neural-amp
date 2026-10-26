# Connected amp: WiFi, a model store, a display, and Jev

> **2026-10-25.** Design for step 2's hardware and system integration. It builds on
> [natural-language-tone-selection.md](natural-language-tone-selection.md) (the decider, catalog
> building and glitch-free model switching), which this doc does not repeat. **Estimate** marks
> anything not measured. The Pi was offline while this was written, so Pi facts come from earlier
> sessions (state-of-the-union.md, sudo-session.md).

## 0. Summary

- **WiFi:** a USB adapter with an **in-kernel driver** (MediaTek `mt7610u` or Ralink `rt5370`), not
  a Realtek chip that needs an out-of-tree driver. Out-of-tree drivers break on kernel updates and on
  the PREEMPT_RT kernel planned in sudo-session.md step 7.
- **Power first:** the Pi already reported under-voltage with only the Scarlett attached
  (`vcgencmd get_throttled` = `0x50000`, sudo-session.md step 0). Adding WiFi and a display makes that
  worse. A **powered USB hub** for the Scarlett and the WiFi adapter is part of this step, not optional.
- **Display:** a **1.3" 128×64 I2C OLED (SH1106)**. It redraws only when something changes, uses the
  I2C bus rather than USB, and costs nothing measurable on the audio cores (to be verified, section 6).
- **"Database":** the catalog is **metadata + measured descriptors + text cards**, stored as one
  versioned `catalog.json` plus the `.nam` files in a **private bucket**. A real database earns its
  place only past Jev's 255-option limit or with more than one user (section 3.3).
- **Licensing shapes the store.** TONE3000's API terms forbid mirroring or caching their catalog and
  preloading it onto devices. The store holds only captures Christine downloaded herself, for her own
  use, never public. Direct on-demand download from TONE3000 is the cleaner route, but free-tier access
  to the model download URLs is unclear; email TONE3000 (section 3.1).
- **Jev runs from the Pi.** A small Python service (`tonectl`, on the non-audio cores, low priority) takes the
  request from a phone page, asks Jev two questions (intent, then choice over the catalog), downloads
  the file if it isn't on the SD card, and tells `live` to switch. Jev's latency is the vendor's
  70–500 ms. Its cost is about **$0.0004 per request** for a 200-model catalog (**estimate**, section 4.4).
- **Phone hotspot is the network.** The Pi joins the phone's hotspot, so the phone reaches the Pi
  page and the Pi reaches Jev and the bucket anywhere, with no home router. Home WiFi is a second saved network.

---

## 1. Do you need WiFi?

The Pi already has Ethernet, and everything in this doc works over it today. WiFi is needed to take
the amp **away from the router**: rehearsal, a gig, the couch. Both Ethernet and a USB WiFi adapter sit
behind the Pi 2's single USB 2.0 controller (`dwc_otg`, the same IRQ as the Scarlett; state-of-the-union
§2). So from the audio path's point of view, WiFi is no worse than the Ethernet traffic already
measured. Section 6 measures it anyway.

| Option | Pros | Cons |
|---|---|---|
| **USB WiFi adapter (recommended)** | One small part; works with a phone hotspot anywhere | One more USB device on the shared controller; more power draw |
| Travel router on the Pi's Ethernet | No WiFi driver on the Pi at all; the USB load is identical to today's measured setup | Another box with its own power supply |
| Newer Pi with built-in WiFi | — | Changes the measured platform; every step-1 number would need re-running |

## 2. Hardware

| Part | Pick | Why | Notes |
|---|---|---|---|
| **WiFi adapter** | Any adapter on the [in-kernel plug-and-play list](https://github.com/morrownr/USB-WiFi/blob/main/home/USB_WiFi_Adapters_that_are_supported_with_Linux_in-kernel_drivers.md). For example **Panda PAU0B** (`mt7610u`, dual-band AC600) or an `rt5370` N150 nano | In-kernel driver: survives kernel updates and the RT kernel build | Throughput is irrelevant: one A2 file is 295 KB. Avoid Realtek sticks that need out-of-kernel drivers ([list](https://github.com/morrownr/USB-WiFi)) |
| **Powered USB hub** | Any hub with its own supply | The Pi is already under-voltage with the Scarlett alone | Scarlett + WiFi on the hub. Recheck `vcgencmd get_throttled` = `0x0` |
| **Pi power supply** | 5 V / 2.5 A (official) | Same reason | sudo-session.md step 0 |
| **Display** | **1.3" SH1106 I2C OLED, 128×64** (0.96" SSD1306 also works but is small on a pedalboard) | I2C, 4 wires, readable on a dark stage; [`luma.oled`](https://luma-oled.readthedocs.io/en/stable/intro.html) supports both controllers and is packaged in Debian trixie ([package](https://packages-picconi.debian.org/ja/trixie/luma.oled-doc)) | Wiring: VCC → pin 1 (3.3 V), GND → pin 6, SDA → pin 3 (GPIO2), SCL → pin 5 (GPIO3) |
| Optional: controls | Rotary encoder with push button + one footswitch on GPIO | Scroll Jev's alternatives, confirm, "back" without the phone | Read with `libgpiod` in `tonectl` (no kernel driver) |
| Already planned | RCA → ¼" cable | Round-trip latency | sudo-session.md step 8 |

**Displays considered and rejected:**
- **SPI TFT (2.8–3.5")**: `fbtft` pushes whole frames from the CPU.
- **E-paper**: 1–2 s refresh is too slow for "switching…".
- **16×2 character LCD**: fine, but chunky.
- **HDMI screen**: no CPU cost, but size and power.

## 3. Where the models live

### 3.1 What TONE3000 allows

From the [API terms](https://www.tone3000.com/api/terms):
- "Do not bulk download, scrape, mirror, or cache the TONE3000 catalog. Only download models when a
  user requests them."
- "Do not preload catalog content onto devices or bundle it with your product without written
  permission from TONE3000."
- "Do not redistribute, republish, or make catalog content available outside your integration."

From the [API docs](https://www.tone3000.com/api):
- The whole-tone zip endpoint is partner-only.
- Individual models are fetched through `model_url`, which is valid only for tones the authenticated
  user can access.
- The free tier lists only the OAuth prompt flows and the bounded list endpoints (`favorited`,
  `downloaded`, …). It does not say whether free integrations may call the Models endpoints that
  return `model_url`. The default T3K license also forbids redistributing the file.

**Consequence:** a public or shared "database of NAM models" is off the table. Two legitimate shapes:

| | A. Private store of Christine's own downloads (start here) | B. On-demand from TONE3000 (ask them) |
|---|---|---|
| How | Christine downloads captures on the website. The Mac catalog builder (NL doc §4) uploads the `.nam` + catalog entry to a private bucket. The Pi fetches by ID | `tonectl` holds Christine's TONE3000 OAuth token and fetches `model_url` when Jev picks a model that isn't on the SD card |
| Licensing | Her files, her device, never public: not redistribution. Still keep `license` per entry and delete entries if she loses access | Exactly the "download when a user requests" pattern the terms describe |
| Blocker | None | Free-tier access to the Models endpoints is unclear. **Email support@tone3000.com** with this use case before building it |
| Catalog breadth | Only what she downloaded | Same in practice: Jev can only choose among entries the catalog describes, and describing a model needs a render (NL doc §4.3) |

Either way, a model gets **into the catalog** by being downloaded once by Christine and rendered on
the Mac. So "Jev picks from everything on the internet" isn't possible under these terms. Jev picks
from her library, which can grow to hundreds of models.

### 3.2 Store layout (option A)

```
bucket: pi-amp-models (private)
  catalog/catalog-v{N}.json      # versioned; the Pi keeps the last good one
  catalog/latest                 # text file: "v{N}"
  nam/{sha256}.nam               # content-addressed: the hash is the integrity check
```

- **Bucket:** any S3-compatible private bucket. Cloudflare R2 and Backblaze B2 are cheap at this
  size: 200 models × 295 KB ≈ 60 MB. Two keys:
  - a **write key** on the Mac;
  - a **read-only key** on the Pi in `/etc/pi-amp/secrets.env` (mode 0600, never in git).
- **Sync:** `tonectl` checks `catalog/latest` at boot and every 10 min (and on demand). It downloads a
  newer catalog atomically (write a temp file, then rename). It never pre-downloads `.nam` files
  except the LRU set below.
- **On the SD card:** `~/pi-amp-cache/nam/{sha256}.nam`, an LRU of about 50 files (≈ 15 MB). Every file
  is hash-checked against the catalog before `live` sees it.
- **Catalog entry:** the NL doc's schema (§4.5) plus `sha256`, `bytes`, `added`.

### 3.3 When to use a real database

`catalog.json` covers one user and up to Jev's 255-option limit, with simple queries done in Python.
Move to Postgres with `pgvector` (for example Supabase) when any of these happens:
- the library passes about 255 models, so a retrieval step (CLAP/text embeddings, NL doc §3e) must
  shortlist before Jev;
- more than one person or device shares the library;
- you want request and feedback logs queryable from anywhere.

Until then, a database adds a service to keep alive and gives no benefit.

## 4. Jev integration

### 4.1 Components and where they run

```
 phone (hotspot) ──HTTP──► tonectl (Pi, core 0, nice 10, Python)
   text box, ✓/✗,           │  ├─ Jev API  (HTTPS, cloud)
   alternatives             │  ├─ bucket   (HTTPS: catalog + .nam)
                            │  ├─ displayd (in-process thread → I2C OLED)
                            │  └─ GPIO encoder / footswitch
                            ▼
                    /run/pi-amp/ctl.sock (JSON lines)
                            ▼
                 live (audio on cores 3/2/1, SCHED_FIFO) + ModelManager thread (core 0)
```

**Core budget.** The lowest-latency Full setting (frame split, `--fsplit 8`) runs audio threads on
cores 3, 2 and 1 (state-of-the-union §5), so **everything in this doc shares core 0** with the OS, SSH
and the USB/WiFi interrupts: `tonectl`, Jev calls, downloads, the display, and `live`'s ModelManager
(model load + prewarm, measured 0.3–0.5 s). With Lite or the two-core pipeline, cores 0–1 are free.
The NL doc assumed cores 0–1. Under frame split, loads and switches take longer (one core instead of
two) but don't touch the audio cores. Section 6 measures whether that holds.

The audio process never touches the network, the display or Python. `tonectl` talks to it only
through the control socket in NL doc §5.6 (`preload`, `switch`, `status`, `gate`).

### 4.2 Request flow

1. **Phone → `tonectl`:** `POST /tone {"text": "warm bluesy clean, a bit of breakup"}`. Phone dictation
   works because the browser does speech-to-text (NL doc §6.2).
2. **Intent (Jev Choice):** one of `new_tone`, `relative_edit`, `undo`, `status_question`. `undo`
   and relative edits ("brighter", "less gain") are handled in code from measured descriptors
   (NL doc §6.3) and never reach step 3.
3. **Choose a model (Jev Choice):**
   - **State:** the request plus the current model's text card.
   - **Options:** every catalog entry as `id → text_card`, where a text card is a template sentence
     from the metadata and descriptors, e.g. "1965 Fender Super Reverb; clean with slight edge (gain 1/4);
     warm (centroid 1.9 kHz)".
   - Up to 255 entries go straight to Jev. Beyond that, a retrieval shortlist comes first.
   - Jev returns `choice`, `probabilities` and `confidence`
     ([Choice docs](https://docs.typesafe.ai/primitives/choice.md)).
4. **Gate on confidence:**
   - ≥ τ: act.
   - Below τ: show the top 3 by probability on the phone and the OLED and wait for a tap or the encoder.
   - τ is set from the calibration plot in the NL doc's eval (§7.1).
5. **Fetch if needed:** if the file isn't in the SD cache, download it from the bucket (option A) or
   from TONE3000 (option B), then check the hash. 295 KB over WiFi is well under a second
   (**estimate**; measure it as `fetch_ms`).
6. **Switch:** `{"cmd":"preload","ids":[id]}`, then `{"cmd":"switch","id":…,"size":…,"trim_db":…}`.
   `live`'s ModelManager loads the model on core 0 (measured 0.3–0.5 s load + reset, NL doc §1.1),
   prewarms it, and fades at a block boundary (NL doc §5.4).
7. **Report:** the OLED and phone show the new model, Jev's confidence, and ✓/✗. The request,
   decision, probabilities, timings and feedback are appended to `results/tone_requests.csv`.

**Prefetch.** After a pick, `tonectl` fetches Jev's next two alternatives into the cache in the
background. A "no, the other one" is then a local switch. This applies to option A only; under
option B it would be preloading, which the terms forbid without permission.

### 4.3 Failure handling

| Failure | Behavior |
|---|---|
| No internet / Jev timeout (1 s) | Fall back to the offline decider (rules, or CLAP once built; NL doc §3); OLED shows `offline` |
| Bucket unreachable | Choose only among cached models (pass Jev only cached options, or fall back) |
| Hash mismatch | Drop the file, keep the current model, show `bad file` |
| `live` not running / socket error | Show `amp stopped`; systemd restarts `live` |
| Jev returns a valid but wrong pick | ✗ button → show alternatives; log for the eval |

### 4.4 Cost and latency (estimates)

- **Tokens:** a text card is about 40 tokens. With 200 entries plus the request and the current
  model, that is about 8,500 input tokens per choice. The intent question adds about 200.
- **Cost:** at the vendor price of $0.042 per million input tokens
  ([InfoQ](https://www.infoq.com/news/2026/10/typesafe-ai-jev-released/)), that is **≈ $0.0004 per
  request**, about $0.04 per 100 requests. Early-access pricing can change.
- **Latency:** two Jev calls at the vendor's 70–500 ms each. Send them in parallel: the choice call's
  result is discarded if the intent turns out to be a relative edit. Then fetch (< 1 s, estimate),
  load + prewarm (0.3–0.5 s, measured) and fade. The target from NL doc §7.2 stands: **< 2 s p95
  request → new sound**.

### 4.5 Secrets and accounts (Christine's)

- **TypeSafe API key:** the Jev account.
- **Bucket read key:** on the Pi.
- **Bucket write key:** on the Mac.
- **TONE3000 OAuth token:** option B only.

All of them go in `/etc/pi-amp/secrets.env` (Pi) or the Mac keychain, never in the repo. Add
`secrets.env` to `.gitignore` defensively.

## 5. Display

### 5.1 What it shows (128 × 64, three lines plus a status bar)

```
┌────────────────────────────┐
│ Fender Super Reverb        │   model name (scrolls if long)
│ FULL  gate -50   92%       │   size, gate threshold, Jev confidence
│ ▸ warm bluesy clean        │   last request (or "1/3 Twin  2/3 AC30" when asking)
│ wifi▮▮▯  xrun 0   45°C     │   network, underruns since start, SoC temp
└────────────────────────────┘
```

Transient states: `switching…` (during load + fade), `downloading 40%`, `offline`, `amp stopped`.

### 5.2 How it's driven without hurting the audio

- A `displayd` thread inside `tonectl` (core 0) polls `live`'s `status` command at 2 Hz and redraws
  **only when the rendered text changes**.
- **Bus cost:** a full SH1106 frame is 1,024 bytes. At 400 kHz I2C that is about 23 ms of bus time
  (**estimate**: 1,024 × 9 bits / 400 kHz). It is handled by the I2C controller and its IRQ on
  core 0, not by the audio cores.
- **No `vcgencmd`:** read the SoC temperature from `/sys/class/thermal/thermal_zone0/temp`, never by
  forking `vcgencmd`. Forking `vcgencmd` caused up to 7 ms stalls in step 1 (README, "What I learned
  about measuring").
- **Enable I2C** (Christine, sudo): `sudo raspi-config nonint do_i2c 0`. Then
  `sudo usermod -aG i2c,gpio ctu` and log in again. The I2C IRQ should stay on core 0; check
  `/proc/interrupts`.

## 6. Measurements before trusting it live

Same discipline as step 1: CSV to `results/`, compared against the step-1 baselines.

| Run | Condition | Compare against |
|---|---|---|
| `wifi_idle` | Ethernet unplugged, WiFi associated, idle | `live10_full_fsplit3_48x2` (1 underrun / 600,000) |
| `wifi_fetch` | WiFi; a script fetches 20 models in a loop | same |
| `oled_2hz` | Display redrawing every 0.5 s (forced) | same |
| `tonectl_load` | A Jev request every 10 s, with fetch and switch | NL doc §7.2 switching run |
| `throttled` | `vcgencmd get_throttled` before and after each run (outside the audio run) | `0x0` |

Each run is `live --duration 600` with the recommended Full command (state-of-the-union §4 and §5), plus a
`label`. The new script `scripts/connected_interference.sh` runs the load generator alongside. The
gate is **underruns and `proc` p99 within +5% of the no-load run**. If WiFi fails it, the travel-router
option (section 1) keeps the USB load identical to what was already measured.

## 7. Phases

| Phase | Scope | Christine | Claude |
|---|---|---|---|
| **A. Bring-up** | Power supply + powered hub, WiFi on home network + phone hotspot (`nmcli`), OLED "hello", I2C enabled | Buy parts; sudo steps; wire the OLED | `scripts/oled_hello.py`, nmcli instructions, `connected_interference.sh` |
| **B. Measure** | Section 6 runs | Approve the runs | Run, CSVs, report section |
| **C. Switching (NL doc v0)** | Shared pipeline worker, ModelManager, control socket, fades | Listen to dip vs crossfade | Patch + `live` changes |
| **D. Store** | Bucket, `catalog_build.py` upload step, `tonectl` sync + SD cache + hash check; email TONE3000 about option B | Create bucket + keys; download 20–50 captures; send the email | Scripts, `tonectl` |
| **E. Jev** | `JevDecider` (intent + choice), confidence gating, phone page, OLED states, request log | TypeSafe account + key; set τ; label eval prompts | Decider, page, eval harness |
| **F. Controls** (optional) | Encoder + footswitch | Wiring | GPIO handling |

Phases A–B can start as soon as the parts arrive. C does not depend on any hardware in this doc.

## Sources

- TONE3000: [API terms](https://www.tone3000.com/api/terms), [API docs](https://www.tone3000.com/api)
- TypeSafe Jev: [Choice](https://docs.typesafe.ai/primitives/choice.md), [models](https://docs.typesafe.ai/models.md),
  [InfoQ (pricing, latency)](https://www.infoq.com/news/2026/10/typesafe-ai-jev-released/)
- USB WiFi on Linux: [morrownr/USB-WiFi](https://github.com/morrownr/USB-WiFi),
  [in-kernel plug-and-play list](https://github.com/morrownr/USB-WiFi/blob/main/home/USB_WiFi_Adapters_that_are_supported_with_Linux_in-kernel_drivers.md),
  [Pi Hut adapter (mt7601u; AP-mode caveat)](https://thepihut.com/products/usb-wifi-adapter-for-the-raspberry-pi)
- OLED: [luma.oled docs](https://luma-oled.readthedocs.io/en/stable/intro.html),
  [Debian trixie package](https://packages-picconi.debian.org/ja/trixie/luma.oled-doc)
- Project: [state-of-the-union.md](state-of-the-union.md), [../sudo-session.md](../sudo-session.md),
  [natural-language-tone-selection.md](natural-language-tone-selection.md)
