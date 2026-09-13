# PRD — SatQuery AI
**SIH 2026 · Problem Statement 26167 · ISRO/SAC, Department of Space**
**Target: Internal hackathon, Sept 10, 2026 (T‑minus 7 days) · Status: MVP build**

---

## 0. Read this first — the honest scope call

The problem statement as written (fine‑tuned VLM, single‑image VQA + captioning/grounding, bi‑temporal change VQA, optical–SAR fusion, agentic orchestration, evaluated against VRSBench/RSVQA/CDVQA + a real Cartosat‑2S/RISAT ISRO set) is a **6‑month lab project**, not a 7‑day build. Two things make it buildable by Sept 10 without lying to the judges about what was "fine‑tuned":

1. **We are not training a foundation model from scratch.** We adapt existing open, pretrained, remote‑sensing‑specific models with a small, real LoRA fine‑tune on BigEarthNet‑derived data. That fine‑tune is small but genuine — we'll have a before/after eval delta to show, which is more credible to judges than an unverifiable claim of "custom‑trained VLM."
2. **The internal round (Sept 10) is a selection/demo gate, not the final ISRO evaluation.** The ISRO benchmark scoring (VRSBench/RSVQA/CDVQA test splits + the undisclosed Cartosat‑2S/RISAT set) happens at the actual SIH grand finale, months later, with a full team and full timeline. Optimizing for benchmark accuracy this week is the wrong target. **Optimizing for a correct, explainable, live‑demoable system across all 5 mandatory capabilities on a curated set of ~15–20 examples is the right target.**

The differentiator we're leading with is **not** "we used a VLM" (every other team will say that). It's: **a hybrid architecture where classical remote‑sensing signal processing (spectral indices, SAR backscatter) grounds and cross‑verifies every VLM claim**, so the system can say *"I'm not sure"* instead of hallucinating — and the judging rubric explicitly rewards this ("estimate confidence," "auditable execution summary," "evidence‑grounded response"). That is a real, defensible, ISRO‑relevant wedge. Lead the pitch with it.

---

## 1. Objective

Build and demo, by Sept 10, an interactive web app where a user uploads 1–2 remote‑sensing images (single / optical‑SAR pair / bi‑temporal pair), asks a natural‑language question, and gets back an evidence‑grounded answer with visual overlay, a confidence score, and an auditable execution trace — covering all 5 mandatory capabilities in the problem statement.

## 2. Judging‑rubric mapping

| PS requirement (mandatory) | Where it's covered | Demo‑verifiable? |
|---|---|---|
| Remote‑sensing adaptation (fine‑tune/adapt on BigEarthNet or open data) | §6.2 — LoRA fine‑tune of GeoChat‑7B on RSVQAxBEN (BigEarthNet‑derived) | Yes — before/after eval numbers |
| Single‑image VQA (mandatory) | §6.2 — GeoChat VQA, grounded/cross‑checked by §6.1 | Yes — live query |
| + one of captioning / grounding | §6.2 (captioning) + §6.3 (CLIPSeg grounding) — **we ship both** | Yes |
| Bi‑temporal change description / change‑VQA | §6.4 | Yes |
| Optical–SAR cross‑modal analysis | §6.5 | Yes |
| Agentic orchestration (select/sequence/execute + audit trace) | §7 | Yes — trace panel + downloadable JSON |

## 3. MVP scope — must‑ship / stretch / cut

**Must ship (Sept 10, no exceptions):**
- Upload flow for single image, optical+SAR pair, bi‑temporal pair (GeoTIFF/TIFF + PNG/JPEG for benchmark samples)
- Input validator (modality/format/pairing checks) with clear error states
- Deterministic task router with visible execution trace (task, tools, params, confidence)
- VQA (GeoChat, zero‑shot or LoRA‑adapted) + captioning
- CLIPSeg‑based grounding, cross‑checked against NDWI/NDVI/NDBI masks
- Change detection (TinyCD mask) + change‑VQA answer generated from before/after classifier diff
- Optical–SAR fusion rule engine (spectral indices + SAR backscatter → combined call)
- One real LoRA fine‑tune with a measurable before/after delta on a held‑out slice
- Downloadable execution report (Markdown/JSON, PDF if time allows)
- 3‑minute live demo script + a **recorded backup demo video** (see §11)

**Stretch (only if Days 1–5 finish early):**
- PDF report styling, spatial change map export as GeoTIFF, confidence calibration plots, Hugging Face Spaces public deployment, RSVQA‑HR eval run

**Explicitly cut for this round (say so in the pitch, don't hide it):**
- Training GeoChat/any VLM from scratch
- Full VRSBench/RSVQA/CDVQA test‑split benchmark runs (do a small held‑out slice instead, honestly labeled as such)
- Multi‑user auth, persistence/DB, mobile UI, non‑English queries
- Real Cartosat‑2S/RISAT data (unavailable pre‑evaluation) — use Sentinel‑1/2 (BigEarthNet) and LEVIR‑CD/SECOND as documented stand‑ins, and **say this explicitly** in the demo rather than implying it's the real ISRO set

## 4. Users & judge psychology

Primary "user" for Sept 10 is the judging panel, not an end operator. Judges skim; they remember one wow‑moment and whether you can answer "why does this matter" and "what's next" in one breath. The wow‑moment we're engineering: ask a genuinely hard cross‑modal question live ("use optical and SAR together to find built‑up areas hidden under cloud"), and have the system **visibly lower its confidence and explain why** when the two modalities disagree on a tile. Judges have seen a hundred confident chatbots; a system that knows when it doesn't know is the memorable part.

## 5. System architecture

```mermaid
flowchart TD
    U[User: image(s) + NL query] --> V[Input Validator<br/>modality / format / pairing]
    V -->|valid| R[Deterministic Task Router]
    V -->|invalid| E[Error / guidance message]
    R --> T1[Single-image VQA/Caption]
    R --> T2[Text-guided Grounding]
    R --> T3[Bi-temporal Change]
    R --> T4[Optical-SAR Fusion]

    subgraph Perception["Deterministic + light-ML perception layer"]
      SI[Spectral indices<br/>NDVI/NDWI/NDBI]
      SAR[SAR backscatter<br/>VV/VH thresholds]
      LC[Land-cover classifier<br/>ResNet18 + BigEarthNet]
      OD[Object detector<br/>YOLOv8n, coarse]
    end

    subgraph Specialists["Specialist models"]
      GC[GeoChat-7B<br/>+ LoRA adapter]
      CS[CLIPSeg<br/>open-vocab grounding]
      CD[TinyCD<br/>change mask]
    end

    T1 --> GC
    T2 --> CS
    T2 --> SI
    T3 --> CD
    T3 --> LC
    T4 --> SI
    T4 --> SAR

    GC --> X[Cross-verification layer<br/>compare VLM claim vs deterministic signal]
    CS --> X
    SI --> X
    SAR --> X
    LC --> X
    CD --> X
    OD --> X

    X --> P[Phrasing LLM<br/>structured facts -> NL, JSON in / text out]
    P --> O[Answer + overlay + confidence + execution trace]
    O --> UI[React 3D Mission Workstation / FastAPI]
```

**Design rule (non‑negotiable):** the Phrasing LLM never sees raw pixels — only the structured JSON facts the perception/specialist layers computed. It converts facts to language; it does not invent facts. This is what keeps hallucination out of the numbers judges will probe (increase/decrease, % area changed, presence/absence).

## 6. Component decisions (with sources verified this week, not assumed)

### 6.1 Deterministic perception layer (zero training, zero cost, fully explainable)
- **NDVI / NDWI / NDBI** from Sentinel‑2 bands (Green/Red/NIR/SWIR) — pure NumPy, milliseconds, no model risk.
- **SAR backscatter rules**: low VV → water; high VV+VH (double‑bounce) → built‑up. Pure NumPy on Sentinel‑1 GRD bands.
- These are classical, textbook RS techniques — ISRO judges will recognize and respect them, and they're the fallback that keeps the whole system working even if a DL model is slow/unavailable on demo day.

### 6.2 VQA / Captioning — GeoChat‑7B + LoRA
- **Base model**: `MBZUAI/geochat-7B` (Hugging Face, Apache‑2.0) — first grounded LVLM built specifically for remote sensing, LLaVA‑1.5 architecture, zero‑shot capable on captioning, VQA, region reasoning. GitHub: `mbzuai-oryx/GeoChat`.
- **Adaptation (satisfies the mandatory requirement, honestly)**: LoRA fine‑tune (via `peft`) on `MBZUAI/GeoChat_Instruct` and/or **RSVQAxBEN** (Lobry et al., "RSVQA Meets BigEarthNet") — a VQA dataset built directly on BigEarthNet imagery, which is exactly the dataset the PS names as mandatory. Rank‑8/16 LoRA, 1–2 epochs, a few thousand examples. Run on Colab free T4 or Kaggle free P100 with 4‑bit loading + gradient checkpointing.
- **Eval**: hold out a small slice, report accuracy/BLEU before vs. after LoRA — this is your "adaptation evidence" slide.
- **Risk flag**: 7B inference on a free T4 may be slow (multi‑second per query). Mitigate with 4‑bit quantization (`bitsandbytes`), and pre‑warm/cache the exact demo images. If latency is still bad by Day 3, fall back to a smaller captioning path (BLIP‑2 OPT‑2.7B) for the live demo and keep GeoChat for the eval numbers only — decide at the Day 3 checkpoint, not on stage.

### 6.3 Grounding — CLIPSeg
- `CIDAS/clipseg-rd64` (built into `transformers`) — zero‑shot, text‑prompted segmentation, CPU‑capable, seconds per query.
- Cross‑checked against the NDWI/NDVI/NDBI mask when the query is about water/vegetation/built‑up — **this cross‑check is the confidence‑estimation feature the PS explicitly asks for.**

### 6.4 Bi‑temporal change — TinyCD + classifier diff
- `AndreaCodegoni/Tiny_model_4_CD` ("TinyCD") — deliberately small, fast change‑detection model pretrained on LEVIR‑CD; CPU‑friendly, no GPU dependency for the demo. (Heavier alternative if time allows: `open-cd`'s Changer, also LEVIR‑CD‑pretrained.)
- Change‑VQA answers ("has built‑up increased?") are generated by running the §6.2/land‑cover classifier on the before/after images and diffing the label sets — deterministic, not a VLM guess.
- Dev/test data: LEVIR‑CD (building change) and/or the CDVQA dataset (Yuan et al., built on the SECOND dataset — 2,968 pairs, ~122K QA pairs, 6 land‑cover classes) for a same‑distribution evaluation slice.

### 6.5 Optical–SAR fusion
- Rule engine combining the optical spectral‑index call and the SAR backscatter call per tile, weighted by an optical cloud/QA flag (SAR gets more weight where optical is cloud‑obscured). Output: a combined land‑cover call **with a stated reason** ("optical cloud‑masked here; SAR backscatter indicates built‑up").
- Dev/test data: BigEarthNet‑MM / BigEarthNet v2.0 already ships **co‑registered Sentinel‑1 (SAR) + Sentinel‑2 (optical)** patches — no separate SAR dataset hunt required (HF: `GFM-Bench/BigEarthNet`; Zenodo: bigearth.net).

### 6.6 Land‑cover classifier (shared by §6.4/§6.5, and the concrete "adaptation" artifact)
- ResNet18 (`torchvision`, ImageNet‑pretrained) with a new multi‑label head, fine‑tuned on a BigEarthNet‑S2 subset (19‑class CORINE‑derived scheme). A few hours on free‑tier GPU. This is your second, very concrete "fine‑tuned on BigEarthNet" artifact if the GeoChat LoRA run slips.

### 6.7 Phrasing LLM
- Small local instruct model (Phi‑3‑mini‑4k‑instruct or Llama‑3.2‑3B‑Instruct, both free via Hugging Face) converts the structured JSON facts into the final natural‑language answer. Local = no live‑demo dependency on an external API. (Claude, via the team's own access, is fine as a *dev‑time* writing aid — not as a runtime dependency for the judged demo, per the "what's your fallback if wifi dies" rule.)

## 7. Agentic controller — what "agentic" means here, precisely

Per the PS: *"only the observable execution trace... will be evaluated. Internal reasoning text is neither required nor evaluated."* That's a green light to make the router **deterministic and testable** rather than a free‑roaming LLM agent:

1. **Validator**: checks image count, modality (band count/metadata), format, co‑registration/pairing, timestamps.
2. **Router**: rule‑based classifier over query keywords + input configuration → one of `{single_vqa, single_caption, grounding, change_vqa, optical_sar_fusion}`. An LLM intent parser is a secondary signal only, used when the rule‑based confidence is low — and it must select from the same fixed enum, never freelance.
3. **Executor**: calls the selected specialist(s) in the right order, collects outputs + confidence.
4. **Trace**: emits JSON — `{task, tools_invoked: [...], parameters: {...}, confidence, timestamp}` — rendered in the UI and downloadable. This is literally the "auditable execution summary" the PS requires, and it doubles as your regression‑test fixture (see §12).

## 8. Tech stack (cost column matters)

| Layer | Choice | Cost |
|---|---|---|
| Frontend | React 19 + Vite + Tailwind + Three.js | Free |
| Hosting (demo) | Local laptop GPU (primary) / HF Spaces free tier (backup) | Free |
| Training/eval compute | Google Colab free T4 + Kaggle free P100 (parallelized) | Free |
| Model hub | Hugging Face Hub (public repos) | Free |
| Code hosting | GitHub (+ Git LFS for small LoRA weights only) | Free |
| Geo I/O | `rasterio`, `tifffile` | Free |
| DL stack | `torch`, `transformers`, `peft`, `bitsandbytes`, `ultralytics` | Free |
| Optional dev accelerants | Claude / Antigravity / Codex for pair‑programming the codebase | Existing access |

**Total required spend: $0.** Optional: Colab Pro (~$10) only if free‑tier queueing becomes the bottleneck — not required for the MVP.

## 9. 7‑day battle plan (Sept 2 → Sept 10)

Assumes a small team; treat each day's block as parallelizable workstreams — split across teammates and across Claude/Antigravity/Codex sessions rather than serialized.

| Day | Focus | Ships by end of day |
|---|---|---|
| **Day 1 (Tue)** | Repo skeleton, env setup, data download (BigEarthNet subset, LEVIR‑CD, CDVQA sample, RSVQAxBEN sample), input validator | Repo runs, images load, validator passes tests |
| **Day 2 (Wed)** | Deterministic perception layer: NDVI/NDWI/NDBI, SAR backscatter rules, land‑cover classifier training kicked off on Colab/Kaggle | Spectral‑index + SAR rules working on sample tiles; classifier training running |
| **Day 3 (Wed/Thu)** | GeoChat‑7B inference wired in (4‑bit); **checkpoint decision**: keep 7B or fall back to BLIP‑2 for live path | VQA/caption working end‑to‑end on ≥1 real query; latency measured |
| **Day 4 (Thu)** | CLIPSeg grounding + cross‑check vs spectral masks; TinyCD change mask wired in | Grounding + change mask both producing overlays |
| **Day 5 (Fri)** | LoRA fine‑tune run (GeoChat or classifier) on RSVQAxBEN slice; before/after eval numbers captured | Adaptation evidence slide ready |
| **Day 6 (Sat)** | Agentic router + execution trace + optical‑SAR fusion rule engine; React 3D UI assembled end‑to‑end | Full pipeline click‑through works for all 5 capabilities |
| **Day 7 (Sun)** | Freeze features. Curate 15–20 demo examples. Record backup demo video. Write/rehearse the 3‑minute pitch. Bug bash only — **no new features** | Submission‑ready MVP + rehearsed pitch + backup video |
| **Sept 10** | Demo day | — |

**Hard rule from the mentor lens:** if Day 5 evening arrives and the LoRA run hasn't produced a clean before/after number, cut it to the ResNet18 classifier fine‑tune (§6.6) as your adaptation evidence instead — don't burn Day 6/7 chasing it.

## 10. Verification & confidence strategy

Every user‑facing claim is tagged with how it was produced, per the ai‑failure‑mode discipline:
- **Deterministic** (spectral indices, SAR thresholds, classifier diff) → shown as "high confidence, rule‑based."
- **Model‑generated, cross‑checked** (GeoChat claim agreeing with a deterministic signal) → "high confidence, cross‑verified."
- **Model‑generated, uncross‑checked or disagreeing** (open‑ended VQA with no deterministic counterpart, or a disagreement between GeoChat and the spectral/SAR signal) → "lower confidence" surfaced explicitly in the UI, never silently smoothed over.

## 11. Demo plan & failure fallback

- Live demo path: local laptop, no Wi‑Fi dependency for inference (all models run locally/offline).
- **Backup**: a pre‑recorded 90‑second screen capture of the full flow, ready to play instantly if live inference fails on stage.
- Curated example set (15–20 images) pre‑downloaded and pre‑validated so upload/format issues can't derail the demo.

## 12. Definition of Done for Sept 10

- [ ] All 5 mandatory capabilities demonstrable live on at least 2 curated examples each
- [ ] Execution trace visibly correct and downloadable for every query
- [ ] At least one fine‑tune/adaptation run with a documented before/after delta
- [ ] Confidence shown and it visibly changes when a cross‑check disagrees
- [ ] Backup video recorded and tested on the presentation machine
- [ ] 3‑minute pitch rehearsed, hook in the first 15 seconds

## 13. Risks

| Risk | Mitigation |
|---|---|
| GeoChat‑7B too slow on free‑tier GPU for live demo | 4‑bit quant + pre‑cached demo images; BLIP‑2 fallback decided by Day 3 |
| LoRA fine‑tune doesn't converge cleanly in time | Fall back to ResNet18 classifier fine‑tune as adaptation evidence |
| Real GeoTIFF handling breaks on edge cases | Restrict live demo to pre‑validated curated files; validator rejects gracefully otherwise |
| Wi‑Fi/venue network issues | Fully local inference path + recorded backup video |
| Team over‑builds and under‑rehearses | Day 7 is feature‑frozen by design — bug bash and pitch rehearsal only |

---
*Next artifact after this: README.md (say "next" for the project folder structure + repo skeleton).*
