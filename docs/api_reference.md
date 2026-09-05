# SatQuery AI — API Reference

## PipelineExecutor

`satquery.pipeline.executor.PipelineExecutor`

The primary orchestrator executing end-to-end workflows.

### Methods

#### `run(images: list[np.ndarray], metas: list[dict], query: str) -> dict[str, Any]`

- **`images`**: List of 1 or 2 NumPy arrays representing satellite images.
- **`metas`**: Metadata dictionaries containing channel count, filename, format, and CRS.
- **`query`**: Natural language question or instruction.

**Returns:**
- `answer`: Grounded natural language response.
- `confidence_score`: Float between 0.0 and 1.0.
- `confidence_tag`: Confidence category (`high_cross_verified`, `high_rule_based`, `lower_confidence_disagreement`).
- `overlay`: Visual overlay array for display.
- `trace`: Auditable execution trace dictionary.

---

## Deterministic Perception Layer

### `compute_spectral_indices(image_arr: np.ndarray) -> SpectralIndicesResult`
Computes NDVI, NDWI, and NDBI with coverage fractions in milliseconds.

### `analyze_sar_backscatter(sar_arr: np.ndarray) -> SARBackscatterResult`
Evaluates VV/VH backscatter thresholds for calm water specular reflection and urban corner reflection.

### `detect_cloud_mask(optical_arr: np.ndarray) -> CloudMaskResult`
Detects cloud cover and QA flags for dynamic sensor weighting.

---

## Optical-SAR Fusion Engine

### `OpticalSARFusionEngine.fuse(optical_arr: np.ndarray, sar_arr: np.ndarray) -> FusionResult`
Synthesizes optical spectral indices and SAR backscatter, dynamically reallocating weights under cloud obscuration.

