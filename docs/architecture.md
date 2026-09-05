# SatQuery AI — System Architecture & Design

Built for SIH 2026 Problem Statement 26167 (ISRO/SAC).

## High-Level Architecture Flow

```
User Query + Satellite Image(s)
          │
          ▼
┌───────────────────────────────┐
│       Input Validator         │  Checks format (GeoTIFF/PNG), channels, pairing
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐
│   Deterministic Task Router   │  Rule-based keyword + configuration dispatch
└──────┬───────┬───────┬────────┘
       │       │       │
       ▼       ▼       ▼
┌───────────────────────────────┐
│     Perception Layer          │  Pure NumPy physical remote sensing signals
│  • NDVI (Vegetation)          │  • NDWI (Water bodies)
│  • NDBI (Built-up index)      │  • SAR Backscatter (VV specular, VH volume)
│  • Cloud Mask / QA flag       │  • ResNet-18 Land Cover (BigEarthNet 19-class)
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐
│     Specialist Models         │
│  • GeoChat-7B (LoRA adapted)  │  Remote Sensing VQA & Captioning
│  • CLIPSeg                    │  Open-vocabulary spatial grounding
│  • TinyCD                     │  Bi-temporal change detection mask
│  • Optical-SAR Fusion Engine  │  Dynamic cloud-weighted multi-modal rules
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐
│   Cross-Verification Layer    │  Compares VLM assertion vs physical signal
│  • Verifies consistency      │  • Detects hallucinations / divergence
│  • Calibrates confidence      │  • Assigns transparent confidence badge
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐
│         Phrasing LLM          │  Strict fact synthesis: structured JSON -> NL
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐
│   Auditable Execution Trace   │  JSON record with tool timeline, parameters
└───────────────────────────────┘
```

## Non-Negotiable Design Principles

1. **Physical Anchoring**: Deep vision-language models never operate ungrounded. Physical reflectance and microwave scattering mechanisms cross-check every claim.
2. **Transparent Failure**: The system explicitly signals low confidence and cites the contradicting physical telemetry instead of hallucinating.
3. **Auditable Execution**: Every tool invocation, parameter, duration, and output summary is recorded in the JSON execution trace.

