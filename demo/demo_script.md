# SatQuery AI — 3-Minute Demo Script
**SIH 2026 · Problem Statement 26167 (ISRO/SAC)**

---

## Pitch Outline (180 Seconds)

### 00:00 – 00:30: The Hook & Core Differentiator
- *"Respected judges, foundation vision-language models applied to satellite imagery fail in dangerous ways: when they are wrong, they answer fluently and confidently."*
- *"SatQuery AI introduces a hybrid paradigm: deep vision-language intelligence anchored and cross-verified by classical, deterministic remote sensing signal processing."*
- *"We compute physical indices like NDVI, NDWI, NDBI, and Sentinel-1 SAR backscatter mechanisms in pure NumPy alongside the VLM. When the models agree, confidence is high; when they diverge, the system explicitly flags lower confidence and provides the physical evidence."*

---

### 00:30 – 01:15: Capability 1 & 2 (Single-Image VQA & Grounding)
- **Action**: Select *Demo 1: Single Optical VQA* on Sentinel-2 tile.
- **Query**: *"Describe the land-cover and determine if surface water is present."*
- **Highlight**:
  - Point to the **Auditable Execution Trace**: InputValidator -> TaskRouter -> spectral_indices -> GeoChatSpecialist -> CrossVerifier.
  - Point to the **Confidence Badge**: *High Confidence (Cross-Verified: 94%)*.
  - Show that the VLM assertion was checked against McFeeters NDWI.
- **Action**: Select *Demo 2: Open-Vocabulary Grounding*.
- **Query**: *"Highlight the water body referred to in the scene."*
- **Highlight**: CLIPSeg neural mask cross-checked against NDWI with IoU computed live.

---

### 01:15 – 02:00: Capability 3 & 4 (Bi-Temporal Change & Optical-SAR Fusion)
- **Action**: Select *Demo 3: Bi-temporal Change Detection* (LEVIR-CD pair).
- **Query**: *"What changed between these two acquisitions and has built-up area increased?"*
- **Highlight**: TinyCD change detection mask + differential spectral calculation:
  - Quantitative metrics displayed: *Built-up area increased by +18.4%*.
- **Action**: Select *Demo 4: Cloud-Penetrating Optical-SAR Fusion*.
- **Query**: *"Use optical and SAR together to identify built-up structures hidden under cloud."*
- **Highlight**:
  - The optical sensor is 50% cloud-obscured.
  - The fusion rule engine dynamically down-weights optical and up-weights SAR radar backscatter.
  - Stated reason: *"Optical cloud-masked; SAR double-bounce backscatter confirms built-up structures beneath."*

---

### 02:00 – 02:40: The Wow Moment (Signal Disagreement & Hallucination Prevention)
- **Action**: Select *Demo 5: Cross-Verification Disagreement Probe*.
- **Query**: *"Is this entire area covered in deep open water?"*
- **Highlight**:
  - Show the system assigning **Lower Confidence (Signal Disagreement)**.
  - Emphasize to the judges: *"This is what sets SatQuery AI apart. Instead of hallucinating, it presents the physical telemetry that stopped the false positive."*

---

### 02:40 – 03:00: Adaptation Evidence & Conclusion
- Show the held-out RSVQAxBEN evaluation slide: +12.6% accuracy gain after LoRA fine-tuning.
- Download the **Full Auditable Trace JSON** and **Markdown Report**.
- Close with: *"SatQuery AI turns remote sensing VLM analysis into a transparent, verifiable, ISRO-grade capability."*

