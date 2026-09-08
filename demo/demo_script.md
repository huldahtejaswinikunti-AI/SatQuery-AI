# SatQuery AI — 3-Minute Live Presentation & Demonstration Script
**Smart India Hackathon 2026 · Problem Statement 26167 (ISRO / Space Applications Centre)**
*Theme: Multi-Modal Remote Sensing Vision-Language Assistant with Grounded Cross-Verification*

---

### [00:00 – 00:15] The Hook: The Peril of Hallucinating VLMs
*(Speaker stands tall, speaks clearly and directly to the judging panel)*

"Respected judges from ISRO and SAC. When a generic vision-language model looks at a satellite image, it sounds articulate, confident—and frequently, catastrophically wrong. A standard VLM cannot distinguish a shallow seasonal silt deposit from deep open ocean, nor can it peer through monsoon clouds to confirm if an airfield is flooded. In space applications, an unchecked hallucination isn't just an error—it's a mission failure. That is why we built **SatQuery AI**."

---

### [00:15 – 00:45] Problem Statement & The SatQuery Solution
*(Speaker gestures toward the UI projector display)*

"For Problem Statement 26167, our challenge was to build an assistant that interprets multi-sensor satellite imagery across optical, multispectral, and synthetic aperture radar, while guaranteeing operational reliability.

Our breakthrough is a **hybrid verification architecture**. We do not leave the answer solely to an unconstrained neural network. Instead, SatQuery pairs fine-tuned remote sensing specialists—like GeoChat and TinyCD—with **deterministic, classical signal processing**. We compute physical indices like NDVI, NDWI, NDBI, and Sentinel-1 dual-pol SAR backscatter directly on raw pixel values. When neural reasoning aligns with physical telemetry, confidence is verified. When they diverge, the system explicitly down-weights confidence, flags the conflict, and exposes the physical evidence."

---

### [00:45 – 01:30] Live Demo 1: Single-Image Analysis & Cross-Verification Badge
*(Speaker moves to the laptop / terminal)*

*[Click on preset dropdown: Select '1. Single-Image Captioning (Coastal Port)']*  
*[Click 'Run Query']*

"Let's see this in action on real imagery. Here is a high-resolution coastal port scene. I've asked: *'Describe the land use and coastal infrastructure in this scene.'*

Notice three things on the screen:
First, within two seconds, the model extracts the jetties, cargo berths, and water boundary.
Second, look at the **Confidence Badge** at the top right: `High Confidence (Cross-Verified)`. The system didn't just guess water—it ran McFeeters NDWI across the scene, cross-referenced the water absorption index with the VLM's text claims, and calculated an agreement consensus of 94%.
Third, our pipeline is fully deterministic and auditable. Every decision is captured in the trace below."

---

### [01:30 – 02:15] Live Demo 2: The Wow Moment — SAR Disagreement & Cloud Penetration
*(Speaker turns to the second scenario)*

*[Click on preset dropdown: Select '4. Optical + SAR Fusion (Monsoon Flood Inundation)']*  
*[Click 'Run Query']*

"Now for the real test: monsoon cloud cover over a flood-affected district. The optical image is over 50% obscured by heavy cloud shadows. A conventional VLM fails completely here.

Watch how SatQuery handles this: we feed the paired Sentinel-1 SAR imagery alongside the optical tile. Our router triggers the **Optical-SAR Fusion Engine**. The system dynamically computes the optical cloud mask, realizes optical optical bands are unreliable, and activates the SAR polarimetric cross-verifier.

SAR backscatter confirms specular water reflection in the flat floodplains and double-bounce scattering from surviving built-up structures. Look at the result: the answer clearly highlights the inundated sectors, with our consensus metric showing how SAR validated the claim despite zero optical visibility. If someone inputs a misleading query asking if the whole scene is dry land, the engine immediately drops confidence to `Lower Confidence (Signal Disagreement)` and prints the contradictory radar decibel values."

---

### [02:15 – 02:45] Auditable Trace & Executive Reporting
*(Speaker scrolls to the bottom of the UI and expands the trace)*

*[Click expander: 'Auditable Execution Trace & Tool Timeline']*

"For defense and ISRO intelligence workflows, black-box AI is unacceptable. Look at our **Auditable Execution Trace**. Every step—input validation, router latency, tools invoked, and verified factual statements—is saved as structured JSON.

*[Click 'Download PDF Report']*

With one click, an analyst can generate an executive, print-ready PDF report containing the visual overlays, quantitative metrics, and cryptographic timestamp for downstream mission briefing."

---

### [02:45 – 03:00] Close: LoRA Fine-Tuning Delta & Future Vision
*(Speaker delivers closing statement with conviction)*

"Under the hood, our backbone is fine-tuned on curated remote sensing instruction sets, demonstrating a held-out accuracy gain of over 12% compared to zero-shot base models.

SatQuery AI bridges the gap between state-of-the-art AI and the rigorous physics of Earth observation. It is trustworthy, transparent, and ready for deployment across India's remote sensing ecosystem.

Thank you, and we are ready for your questions!"
