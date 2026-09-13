# SatQuery AI

**An agentic vision-language assistant for remote-sensing image analysis.**
Built for SIH 2026 — Problem Statement **26167** (Indian Space Research Organisation / Dept. of Space).

Upload one or two satellite images (single optical/SAR, an optical–SAR pair, or a bi-temporal pair), ask a question in plain English, and get an evidence-grounded answer: a visual overlay, a confidence score, and an auditable trace of exactly which tools were used to produce it.

> Full rationale, architecture decisions, and the 7-day build plan live in [`PRD.md`](./PRD.md). This file is the "how to run it" doc.

---

## What it does (MVP scope)

| Capability | Status |
|---|---|
| Single-image visual question answering | ✅ Mandatory, shipped |
| Image captioning / scene description | ✅ Shipped |
| Text-guided region grounding ("highlight the water body") | ✅ Shipped |
| Bi-temporal change description & change-VQA | ✅ Mandatory, shipped |
| Optical–SAR cross-modal fusion | ✅ Mandatory, shipped |
| Agentic task routing + execution trace | ✅ Mandatory, shipped |
| Confidence estimation via cross-modal/deterministic verification | ✅ Shipped — this is the differentiator, see below |
| Lunar morphological analysis (Chandrayaan-2 OHRC/TMC-2, LROC WAC/NAC) | ✅ Shipped (`satquery.lunar.lunar_pipeline`) |
| Permanently Shadowed Region (PSR) cold-trap identification | ✅ Shipped (shadow fraction + polar latitude proxy) |
| Crater ejecta mapping & continuous blanket delineation | ✅ Shipped (high-albedo statistical thresholding) |
| Boulder distribution & density estimation | ✅ Shipped (GSD-calibrated sub-meter anomaly detection) |

**The wedge:** instead of one generic VLM guessing at everything, classical remote-sensing analysis (NDVI/NDWI/NDBI spectral indices, SAR backscatter thresholds) runs alongside the learned models and **cross-checks their claims**. When the VLM and the deterministic signal disagree, the system says so and lowers its confidence — instead of answering fluently and wrong.

### Lunar Morphological Analyzer (Chandrayaan-2 & LROC)

For planetary observation, SatQuery AI supports Chandrayaan-2 (OHRC and TMC-2) and LROC (WAC and NAC) datasets via `satquery.lunar.lunar_pipeline`. Because Earth-specific spectral indices ($NDVI, NDWI$) and SAR backscatter rules do not apply on the lunar surface, lunar queries bypass Earth verification engines and return an auditable `experimental_unverified` confidence tag while executing deterministic morphological algorithms:
- **PSR Cold-Trap Candidate Identification:** Pixel-level deep shadow fraction extraction (`gray < 20` on a 0–255 scale) combined with polar latitude checks ($\ge 70^\circ$).
- **Crater Ejecta Mapping:** Statistical high-albedo thresholding ($\text{intensity} > \mu + 1.8\sigma$) isolating rays, continuous ejecta blankets, and impact melt.
- **Boulder Distribution Estimation:** Sub-meter high-contrast anomaly clustering yielding boulder counts and spatial density ($\text{boulders}/\text{km}^2$) when Ground Sampling Distance (GSD) telemetry metadata is supplied.
- **Slope & Landing Envelopes:** Photometric spatial gradient proxy bounding local slope roughness ($\le 12^\circ$ nominal envelope).
- **Planned / In Progress:** TMC-2 stereo-derived 3D Digital Elevation Models and domain-specific lunar VLM fine-tuning to replace general-purpose zero-shot VQA fallbacks.

## Architecture (short version)

```
Image(s) + query
      │
      ▼
Input Validator  →  Deterministic Task Router
      │                        │
      ▼                        ▼
Perception layer      Specialist models
(spectral indices,    (GeoChat-7B + LoRA,
 SAR backscatter,       CLIPSeg, TinyCD)
 land-cover classifier)
      │                        │
      └──────────┬─────────────┘
                  ▼
        Cross-verification layer
                  ▼
         Phrasing LLM (facts → text)
                  ▼
   Answer + overlay + confidence + trace
                  ▼
     React 19 + Three.js UI (FastAPI backend)
```

Full diagram and per-component reasoning: [`PRD.md §5–6`](./PRD.md).

## Tech stack

| Layer | Choice |
|---|---|
| UI | React 19 + Vite + Tailwind + Three.js (FastAPI backend in `app/api_server.py`; Streamlit MVP replaced) |
| VQA / captioning | `MBZUAI/geochat-7B` (LLaVA-1.5-based RS VLM) + LoRA adapter |
| Grounding | CLIPSeg (`CIDAS/clipseg-rd64`) |
| Change detection | TinyCD (`AndreaCodegoni/Tiny_model_4_CD`) |
| Land-cover classifier | ResNet18 fine-tuned on BigEarthNet |
| Phrasing | Phi-3-mini / Llama-3.2-3B-Instruct (local) |
| Geo I/O | `rasterio`, `tifffile` |
| Training | `torch`, `peft`, `bitsandbytes` |
| Compute | Google Colab (free T4) + Kaggle (free P100) |

Everything above is free-tier. See [`PRD.md §8`](./PRD.md) for the full cost breakdown.

## Datasets & pretrained models (credits)

Sentinel-1/2, BigEarthNet, and LEVIR-CD are documented stand-ins for the real Cartosat-2S/RISAT data named in the PS.

- **BigEarthNet-MM / v2.0** — Sentinel-1 + Sentinel-2 paired patches, CORINE land-cover labels. [bigearth.net](https://bigearth.net/) · [HF: GFM-Bench/BigEarthNet](https://huggingface.co/datasets/GFM-Bench/BigEarthNet)
- **VRSBench** — captioning, grounding, and VQA benchmark for RS images. [github.com/lx709/VRSBench](https://github.com/lx709/VRSBench)
- **RSVQA / RSVQAxBEN** (Lobry et al.) — VQA over Sentinel-2 and BigEarthNet imagery.
- **CDVQA** (Yuan et al., built on the SECOND dataset) — bi-temporal change VQA.
- **LEVIR-CD** — building change detection benchmark, used to pretrain TinyCD.
- **GeoChat** (Kuckreja et al., CVPR 2024) — [MBZUAI/geochat-7B](https://huggingface.co/MBZUAI/geochat-7B), Apache-2.0.
- **CLIPSeg** (Lüddecke & Ecker) — open-vocabulary segmentation, via 🤗 Transformers.
- **TinyCD** (Codegoni et al.) — [github.com/AndreaCodegoni/Tiny_model_4_CD](https://github.com/AndreaCodegoni/Tiny_model_4_CD).

We use these as-is or lightly fine-tune (LoRA) on top of them — see `PRD.md §6.2` for exactly what was adapted and why.

## Getting started

> The full repo skeleton (folder-by-folder) ships in the next artifact. Once it's in place:

```bash
# 1. Clone and set up environment
git clone <repo-url> satquery-ai && cd satquery-ai
python -m venv .venv && source .venv/bin/activate   # or: conda env create -f environment.yml
pip install -r requirements.txt

# 2. Configure
cp .env.example .env   # fill in any local paths; no API keys required for the core pipeline

# 3. Get sample data (small, curated demo set — not the full datasets)
python data/scripts/download_demo_samples.py

# 4. Run the app
# Backend (FastAPI):
uvicorn app.api_server:app --host 127.0.0.1 --port 8000 --reload

# Frontend (React 19 + Three.js, in separate terminal):
cd frontend && npm install && npm run dev
```

Training / fine-tuning notebooks (meant for Colab or Kaggle, not local CPU):

```bash
# open in Colab:
notebooks/colab_lora_finetune.ipynb
```

Evaluation on held-out benchmark slices:

```bash
python evaluation/run_rsvqa_eval.py --split held_out
python evaluation/run_cdvqa_eval.py --split held_out
```

## Evaluation & testing

### Automated Test Suite
- **Count & Coverage:** 145 passing tests (`pytest tests/`) covering deterministic task routing (`tests/test_router.py`), spectral and radar mathematics ($NDVI, NDWI, NDBI$, SAR backscatter thresholds), cross-verification conflict docking (`tests/test_cross_verification.py`), lunar morphological extraction and batch execution (`tests/test_lunar_and_batch.py`), and land-cover calibration fallbacks (`tests/test_land_cover_calibration.py`).

### Reproducible Evaluation Harness
- Every quantitative evaluation in `evaluation/` outputs a machine-readable JSON file in `evaluation/results/`.
- **Provenance Block:** Every result file carries an immutable provenance block recording `script`, UTC `timestamp`, `git_commit` hash, and `total_samples` count (zero hand-entered figures).

### Current Benchmark Results (Pre-Training Baselines)
- **Land-Cover Classification (BEN-19 / RSVQA):** **16.0%** accuracy (untrained baseline; checkpoint training in progress).
- **Captioning & VQA (RSVQA held-out, $n=50$):** BLEU-1 = **0.131** (GeoChat-7B zero-shot baseline).
- **Change-VQA (CDVQA held-out, $n=35$):** Exact Match = **0.00**, Token F1 = **0.17** (scene-level summary proxy).
- *Baseline Context:* Pre-checkpoint-training baseline; the deterministic verification layer is what the system leans on until that training is done.

### Known Limitations & Open Risks (from Repo State)
- **Land-cover classifier checkpoint still training:** When checkpoint is missing or training, inference explicitly tags output as `"untrained_fallback"` (uncalibrated) and is never displayed as verified.
- **Change-VQA question-conditioning under active repair:** `run_change_detection` currently outputs scene-level summaries from TinyCD rather than answering question-specific token conditions.
- **LoRA adapter delta over base model:** Quantitative evaluation on held-out slice shows no measurable delta over base model (both at 16.0% accuracy / 0.131 BLEU-1); re-tuning and training mixture expansion planned next pass.

## Usage

1. Upload one image (single-image tasks), two co-registered optical+SAR images (fusion), or two same-location images from different dates (change).
2. Type a question, e.g.:
   - *"Describe the land-cover and major objects visible in this image."*
   - *"Highlight the water body referred to in the query."*
   - *"What changed between these two dates, and where did the change occur?"*
   - *"Use the optical and SAR images together to identify built-up and water-covered regions."*
   - *"Has the built-up area increased, decreased, or remained unchanged?"*
3. Read the answer, the overlay, and the confidence badge. Expand the **execution trace** panel to see exactly which tools ran and why. Download the report.

## Project structure

See the next artifact for the full annotated folder tree and the actual repo skeleton.

## Team & timeline

Built for the SIH 2026 internal hackathon (Sept 10, 2026). Full day-by-day build plan: [`PRD.md §9`](./PRD.md).

## License

MIT for original code in this repo. Pretrained models and datasets retain their own upstream licenses (see credits above — GeoChat is Apache-2.0; check each dataset's terms before redistribution).
