# SatQuery AI — Repo Skeleton & Folder Structure

> Build the complete project scaffold for SIH 2026 PS 26167 (ISRO/SAC) MVP, due Sept 10.

## Context

The PRD defines 5 mandatory capabilities (single-image VQA, captioning/grounding, bi-temporal change-VQA, optical–SAR fusion, agentic orchestration) with a hybrid architecture: deterministic RS signal processing cross-verifying VLM claims. This plan creates every file and folder the team needs to start coding on Day 1.

---

## Proposed Folder Structure

```
SatQyery-AI/
│
├── PRD.md                          # The full PRD (already written)
├── README.md                       # Setup, usage, credits
├── LICENSE                         # MIT
├── .gitignore                      # Python, model weights, data
├── .env.example                    # Template — local paths, no secrets
├── requirements.txt                # Pinned pip deps
├── environment.yml                 # Conda alternative
├── setup.py                        # Optional editable install
│
├── app/                            # ── Streamlit UI ──
│   ├── __init__.py
│   ├── main.py                     # Streamlit entry point (streamlit run app/main.py)
│   ├── ui_components.py            # Reusable Streamlit widgets (upload, overlay, trace panel)
│   ├── session_state.py            # Session-state helpers
│   └── assets/                     # Static assets for the UI
│       ├── logo.png
│       └── styles.css
│
├── satquery/                       # ── Core Python package ──
│   ├── __init__.py
│   │
│   ├── validator/                  # §7.1 — Input validation
│   │   ├── __init__.py
│   │   ├── input_validator.py      # Modality/format/pairing checks
│   │   └── schemas.py              # Pydantic models for validated input
│   │
│   ├── router/                     # §7.2 — Deterministic task router
│   │   ├── __init__.py
│   │   ├── task_router.py          # Rule-based classifier → task enum
│   │   ├── intent_parser.py        # LLM-backed fallback intent parser
│   │   └── task_types.py           # TaskType enum + schemas
│   │
│   ├── perception/                 # §6.1 — Deterministic perception layer
│   │   ├── __init__.py
│   │   ├── spectral_indices.py     # NDVI / NDWI / NDBI (pure NumPy)
│   │   ├── sar_backscatter.py      # VV/VH threshold rules (pure NumPy)
│   │   └── cloud_mask.py           # Optical cloud/QA flag detector
│   │
│   ├── classifiers/                # §6.6 — Land-cover classifier
│   │   ├── __init__.py
│   │   ├── land_cover.py           # ResNet18 + BigEarthNet multi-label head
│   │   ├── predict.py              # Inference wrapper
│   │   └── train.py                # Training script (also runnable standalone)
│   │
│   ├── specialists/                # §6.2–6.4 — Specialist models
│   │   ├── __init__.py
│   │   ├── geochat_vqa.py          # GeoChat-7B VQA + captioning (4-bit)
│   │   ├── clipseg_grounding.py    # CLIPSeg open-vocab grounding
│   │   ├── tinycd_change.py        # TinyCD change detection mask
│   │   └── blip2_fallback.py       # BLIP-2 fallback if GeoChat too slow
│   │
│   ├── fusion/                     # §6.5 — Optical–SAR fusion
│   │   ├── __init__.py
│   │   └── optical_sar_fusion.py   # Rule engine: spectral + SAR + cloud weight
│   │
│   ├── cross_verification/         # §10 — Cross-verification layer
│   │   ├── __init__.py
│   │   └── verifier.py             # Compare VLM claim vs deterministic signal → confidence
│   │
│   ├── phrasing/                   # §6.7 — Phrasing LLM
│   │   ├── __init__.py
│   │   └── phrasing_llm.py         # JSON-in → NL-out (Phi-3 / Llama-3.2)
│   │
│   ├── pipeline/                   # §7.3–7.4 — Executor + trace
│   │   ├── __init__.py
│   │   ├── executor.py             # Calls specialist(s), collects outputs
│   │   ├── execution_trace.py      # Emits JSON trace + renders for UI
│   │   └── report_generator.py     # Markdown / JSON / PDF downloadable report
│   │
│   └── utils/                      # Shared utilities
│       ├── __init__.py
│       ├── geo_io.py               # rasterio/tifffile GeoTIFF loading
│       ├── image_utils.py          # Resize, normalize, band extraction
│       ├── overlay.py              # Generate visual overlays (masks on RGB)
│       └── config.py               # Central config (paths, model IDs, thresholds)
│
├── models/                         # ── Model weights & adapters (gitignored, downloaded at setup) ──
│   ├── .gitkeep
│   ├── geochat/                    # GeoChat-7B + LoRA adapter
│   │   └── .gitkeep
│   ├── clipseg/                    # CLIPSeg cached weights
│   │   └── .gitkeep
│   ├── tinycd/                     # TinyCD weights
│   │   └── .gitkeep
│   ├── land_cover/                 # ResNet18 fine-tuned checkpoint
│   │   └── .gitkeep
│   ├── phrasing/                   # Phi-3 / Llama-3.2 weights
│   │   └── .gitkeep
│   └── download_models.py          # Script to download/cache all model weights
│
├── data/                           # ── Datasets & demo samples ──
│   ├── .gitkeep
│   ├── demo_samples/               # 15–20 curated demo images (committed)
│   │   ├── single_optical/         # Single Sentinel-2 patches
│   │   ├── single_sar/             # Single Sentinel-1 patches
│   │   ├── optical_sar_pairs/      # Co-registered Sentinel-1 + Sentinel-2
│   │   ├── bitemporal_pairs/       # Before/after pairs (LEVIR-CD samples)
│   │   └── metadata.json           # Per-image metadata (bands, source, date, CRS)
│   ├── raw/                        # Full datasets (gitignored, downloaded via script)
│   │   ├── bigearth_subset/
│   │   ├── rsvqaxben/
│   │   ├── levir_cd/
│   │   ├── cdvqa/
│   │   └── .gitkeep
│   └── scripts/
│       ├── download_demo_samples.py
│       ├── download_bigearth.py
│       ├── download_levir_cd.py
│       └── prepare_rsvqaxben.py
│
├── notebooks/                      # ── Training & experimentation (Colab/Kaggle) ──
│   ├── colab_lora_finetune.ipynb   # GeoChat LoRA fine-tune on RSVQAxBEN
│   ├── colab_resnet_finetune.ipynb # ResNet18 land-cover classifier on BigEarthNet
│   ├── explore_bigearth.ipynb      # Data exploration
│   ├── explore_spectral.ipynb      # Spectral index sanity checks
│   └── latency_benchmark.ipynb     # Model latency measurements (Day 3 decision)
│
├── evaluation/                     # ── Eval scripts & results ──
│   ├── __init__.py
│   ├── run_rsvqa_eval.py           # VQA eval on held-out RSVQAxBEN slice
│   ├── run_cdvqa_eval.py           # Change-VQA eval on CDVQA slice
│   ├── run_caption_eval.py         # BLEU/METEOR captioning eval
│   ├── eval_utils.py               # Metrics helpers
│   └── results/                    # Stored eval outputs
│       ├── .gitkeep
│       ├── before_lora.json
│       └── after_lora.json
│
├── tests/                          # ── Unit & integration tests ──
│   ├── __init__.py
│   ├── test_validator.py
│   ├── test_router.py
│   ├── test_spectral_indices.py
│   ├── test_sar_backscatter.py
│   ├── test_cross_verification.py
│   ├── test_executor.py
│   ├── test_trace.py
│   ├── test_geo_io.py
│   └── fixtures/                   # Test fixture data
│       ├── sample_rgb.tif
│       ├── sample_multispectral.tif
│       ├── sample_sar.tif
│       ├── sample_trace.json       # Expected trace output
│       └── .gitkeep
│
├── demo/                           # ── Demo-day assets ──
│   ├── demo_script.md              # 3-minute pitch script
│   ├── demo_queries.json           # Pre-written queries for live demo
│   ├── backup_video/               # Recorded backup demo video
│   │   └── .gitkeep
│   └── slides/                     # Pitch deck (if needed)
│       └── .gitkeep
│
├── docs/                           # ── Documentation ──
│   ├── architecture.md             # Detailed architecture doc (from PRD §5)
│   ├── api_reference.md            # Internal API docs
│   ├── data_sources.md             # Dataset credits & licenses
│   └── deployment.md               # How to deploy (local, HF Spaces)
│
└── scripts/                        # ── Dev & ops scripts ──
    ├── setup_env.sh                # One-shot env setup (Unix)
    ├── setup_env.ps1               # One-shot env setup (Windows/PowerShell)
    ├── run_all_tests.sh            # Run full test suite
    └── lint.sh                     # Linting & formatting
```

**Total: ~80 files across 25 directories** — large enough to cover every PRD component, small enough that one person can navigate it in 30 seconds.

---

## Design Decisions

### Why this structure (not a flat scripts folder)

| Decision | Rationale |
|---|---|
| `satquery/` as an installable package | Enables `from satquery.perception import spectral_indices` everywhere — no `sys.path` hacks. Streamlit, notebooks, eval scripts, and tests all import the same code. |
| Perception ≠ Specialists ≠ Classifiers | Maps 1:1 to the PRD's "deterministic layer" vs "specialist models" vs "shared classifier" — judges can follow the architecture slide directly into the code. |
| `pipeline/` owns execution + trace | The agentic controller (router → executor → trace) is its own module. The trace JSON schema is defined here, and tests assert against it — this is literally the "auditable execution summary" the PS requires. |
| `models/` gitignored, `download_models.py` committed | No 7GB weight files in Git. One script fetches everything from HF Hub. Demo-day laptop runs it once; Colab/Kaggle notebooks call it in cell 1. |
| `data/demo_samples/` committed, `data/raw/` gitignored | The 15–20 curated demo images are small enough to commit (~50MB). Full BigEarthNet/LEVIR-CD datasets are downloaded via scripts. |
| `demo/` as a first-class directory | Demo day is the deliverable. The pitch script, pre-written queries, and backup video live here — not buried in docs. |
| `tests/fixtures/` with expected trace JSON | Regression testing the execution trace is free: save the expected JSON, assert it matches. This catches silent breakage during the Day 6 integration sprint. |

### Key files explained

| File | What it does | PRD section |
|---|---|---|
| `satquery/validator/input_validator.py` | Checks image count, band count, format (GeoTIFF/PNG/JPEG), co-registration, timestamps | §7.1 |
| `satquery/router/task_router.py` | Rule-based keyword + input config → `{single_vqa, single_caption, grounding, change_vqa, optical_sar_fusion}` | §7.2 |
| `satquery/perception/spectral_indices.py` | NDVI = (NIR−Red)/(NIR+Red), NDWI, NDBI — pure NumPy, milliseconds | §6.1 |
| `satquery/perception/sar_backscatter.py` | Low VV → water; high VV+VH → built-up. Pure NumPy. | §6.1 |
| `satquery/specialists/geochat_vqa.py` | Load GeoChat-7B (4-bit via bitsandbytes), run VQA/captioning | §6.2 |
| `satquery/specialists/clipseg_grounding.py` | Text-prompted segmentation mask | §6.3 |
| `satquery/specialists/tinycd_change.py` | Binary change mask from bi-temporal pair | §6.4 |
| `satquery/fusion/optical_sar_fusion.py` | Weighted spectral + SAR call per tile, cloud-adjusted | §6.5 |
| `satquery/cross_verification/verifier.py` | Compare VLM output vs deterministic output → confidence tag | §10 |
| `satquery/phrasing/phrasing_llm.py` | Structured JSON → natural language answer (Phi-3/Llama-3.2) | §6.7 |
| `satquery/pipeline/executor.py` | Orchestrates tool calls in order, collects results | §7.3 |
| `satquery/pipeline/execution_trace.py` | Emits `{task, tools_invoked, parameters, confidence, timestamp}` | §7.4 |

---

## What Gets Created Now

I'll scaffold every directory, every `__init__.py`, every stub file with proper docstrings and TODO markers — so on Day 1 your team can:

1. `pip install -e .` and have the `satquery` package importable
2. `streamlit run app/main.py` and see a working (empty) upload UI
3. `pytest tests/` and see all tests discovered (passing as stubs)
4. Split work: one person takes `satquery/perception/`, another takes `satquery/specialists/`, another takes `app/` — zero merge conflicts

### Files to create (~80 files):

- **Root config**: `.gitignore`, `.env.example`, `requirements.txt`, `environment.yml`, `setup.py`, `LICENSE`, `PRD.md`, `README.md`
- **App layer** (4 files): `app/main.py`, `app/ui_components.py`, `app/session_state.py`, `app/assets/styles.css`
- **Core package** (25+ files): All modules under `satquery/`
- **Model placeholders** (6 dirs + download script)
- **Data scripts** (4 scripts + demo metadata)
- **Notebooks** (5 notebooks)
- **Evaluation** (5 files)
- **Tests** (9 test files + fixtures)
- **Demo assets** (3 files)
- **Docs** (4 files)
- **Scripts** (4 dev scripts)

## Verification Plan

### Automated Tests
```bash
# After scaffolding, verify:
pip install -e .               # Package installs without error
python -c "import satquery"    # Package imports
streamlit run app/main.py      # UI launches (manual check)
pytest tests/ -v               # All test stubs discovered
```

### Manual Verification
- Folder tree matches the PRD architecture diagram 1:1
- Every PRD section (§6.1–§7.4) has a corresponding module
- `.gitignore` correctly excludes `models/`, `data/raw/`, `.env`, `__pycache__`
