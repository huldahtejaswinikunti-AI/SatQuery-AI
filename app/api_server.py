"""FastAPI backend server for SatQuery AI.

Exposes REST endpoints for the React frontend while directly bridging to
app.pipeline_bridge, existing demo datasets, and specialist ML pipelines.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import sys
from pathlib import Path
from typing import Any, List, Optional

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from PIL import Image
import numpy as np

from app.pipeline_bridge import (
    run_pipeline,
    run_lunar_pipeline,
    run_batch_pipeline,
)
from app.pdf_report import generate_pdf_report
from satquery.pipeline.report_generator import generate_report
from satquery.utils.geo_io import load_image_as_array
from satquery.utils.image_utils import to_display_rgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("satquery.api")

app = FastAPI(
    title="SatQuery AI Mission API",
    description="REST backend for agentic Earth and Lunar multimodal remote sensing intelligence.",
    version="2.0.0",
)

# Enable CORS for Vite dev server and local clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEMO_BASE = _ROOT / "data" / "demo_samples"
UPLOAD_DIR = _ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
EARTH_QUERIES_PATH = _ROOT / "demo" / "demo_queries.json"
LUNAR_QUERIES_PATH = _ROOT / "demo" / "lunar_demo_queries.json"


def _load_json_catalog(path: Path) -> list[dict]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _resolve_image_path(rel_path: str, is_lunar: bool = False) -> Optional[Path]:
    p = Path(rel_path)
    if p.is_absolute() and p.exists():
        return p
    # Check upload and data directory
    if (_ROOT / "data" / rel_path).exists():
        return _ROOT / "data" / rel_path
    if (UPLOAD_DIR / rel_path).exists():
        return UPLOAD_DIR / rel_path
    base = DEMO_BASE / "lunar" if is_lunar else DEMO_BASE
    candidate = base / rel_path
    if candidate.exists():
        return candidate
    base_name = Path(rel_path).name
    if (UPLOAD_DIR / base_name).exists():
        return UPLOAD_DIR / base_name
    for subdir in ["", "single_optical", "single_sar", "optical_sar_pairs", "bitemporal_pairs", "lunar"]:
        cand = DEMO_BASE / subdir / base_name
        if cand.exists():
            return cand
    return None


# ---- Pydantic Request / Response Models ----------------------------------


class AnalyzeRequest(BaseModel):
    mode: str = "earth"  # "earth" or "lunar"
    files: List[str]
    query: str


class BatchItemRequest(BaseModel):
    files: List[str]
    query: str
    metas: Optional[List[dict]] = None


class BatchRequest(BaseModel):
    mode: str = "earth"
    items: List[BatchItemRequest]


class ExportPdfRequest(BaseModel):
    result: dict[str, Any]
    query: str = ""
    images_meta: Optional[List[dict[str, Any]]] = None


# ---- Endpoints -----------------------------------------------------------


@app.get("/api/health")
def health_check() -> dict[str, Any]:
    from satquery.classifiers.predict import get_calibration_status
    calibration = get_calibration_status()
    return {
        "status": "READY" if calibration == "calibrated" else "UNCALIBRATED",
        "classifier_calibration": calibration,
        "system": "SatQuery AI Mission Control",
        "sponsor": "ISRO / SAC & SIH 2026",
        "active_models": [
            "GeoChat-7B + LoRA",
            "CLIPSeg Open-Vocabulary Grounding",
            "TinyCD Bi-Temporal Change Detection",
            f"ResNet-18 Land Cover Classifier ({calibration})",
            "Deterministic Spectral Indices (NDVI/NDWI/NDBI)",
            "SAR Polarimetric Backscatter Engine",
            "Chandrayaan-2 OHRC/TMC-2 Lunar Pipeline",
        ],
    }


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert NumPy objects and non-standard types to JSON-serializable primitives."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(x) for x in obj]
    elif isinstance(obj, Image.Image):
        buf = io.BytesIO()
        obj.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
    return obj


@app.get("/api/scenarios")
def get_scenarios(domain: str = Query("earth", pattern="^(earth|lunar)$")) -> list[dict[str, Any]]:
    """Return curated multi-image observation scenarios for the requested domain."""
    if domain == "lunar":
        catalog = _load_json_catalog(LUNAR_QUERIES_PATH)
    else:
        catalog = _load_json_catalog(EARTH_QUERIES_PATH)
    return catalog


@app.get("/api/image/{file_path:path}")
def serve_image(file_path: str, domain: str = "earth") -> Response:
    """Serve a real satellite raster image converted to displayable JPEG/PNG."""
    is_lunar = domain.lower() == "lunar"
    resolved = _resolve_image_path(file_path, is_lunar=is_lunar)
    if not resolved or not resolved.exists():
        raise HTTPException(status_code=404, detail=f"Image file not found: {file_path}")

    # For standard browser formats, serve directly
    suffix = resolved.suffix.lower()
    if suffix in [".png", ".jpg", ".jpeg"]:
        return FileResponse(str(resolved), media_type="image/png" if suffix == ".png" else "image/jpeg")

    # For TIFF / GeoTIFF, load array and convert to RGB PNG in memory
    try:
        arr, _ = load_image_as_array(str(resolved))
        rgb = to_display_rgb(arr)
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as exc:
        logger.exception("Failed to render raster image: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/upload")
async def upload_rasters(
    files: List[UploadFile] = File(...),
    domain: str = Query("earth", pattern="^(earth|lunar)$"),
) -> dict[str, Any]:
    """Upload custom satellite raster files (GeoTIFF, TIFF, PNG, JPEG) and parse metadata."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided for upload.")

    uploaded_items = []
    for idx, uploaded in enumerate(files):
        filename = uploaded.filename or f"custom_raster_{idx}.png"
        ext = Path(filename).suffix.lower()
        if ext not in [".tif", ".tiff", ".geotiff", ".png", ".jpg", ".jpeg"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format '{ext}'. Allowed formats: .tif, .tiff, .png, .jpg, .jpeg",
            )

        dest_path = UPLOAD_DIR / filename
        content = await uploaded.read()
        dest_path.write_bytes(content)

        try:
            arr, meta = load_image_as_array(str(dest_path))
            h, w = arr.shape[:2]
            bands = meta.get("band_count", arr.shape[2] if arr.ndim == 3 else 1)
            fmt = meta.get("format", ext.replace(".", "").upper())
            crs = str(meta.get("crs") or "Local Grid / WGS 84")

            modality = "SAR" if bands <= 2 else "Optical"
            if domain == "lunar":
                modality = "Lunar OHRC/TMC-2"

            uploaded_items.append({
                "id": f"upload_{idx}_{dest_path.stem}",
                "filename": filename,
                "file": f"uploads/{filename}",
                "name": Path(filename).stem.replace("_", " ").title(),
                "sensor": f"User Custom {modality} ({fmt})",
                "resolution": "User GSD" if domain == "lunar" else ("10.0 m" if "optical" in modality.lower() else "Native Resolution"),
                "date": "Custom Dataset",
                "location": f"CRS: {crs}",
                "provenance": f"User Upload ({fmt} · {w}x{h} · {bands}b)",
                "width": w,
                "height": h,
                "band_count": bands,
                "format": fmt,
            })
        except Exception as exc:
            dest_path.unlink(missing_ok=True)
            logger.exception("Failed to parse uploaded raster: %s", exc)
            raise HTTPException(status_code=400, detail=f"Failed to decode raster '{filename}': {exc}")

    return {
        "status": "success",
        "count": len(uploaded_items),
        "items": uploaded_items,
    }


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest) -> dict[str, Any]:
    """Execute analysis pipeline (Earth or Lunar) and return structured scientific report."""
    is_lunar = req.mode.lower() == "lunar"
    images = []
    metas = []

    for f in req.files:
        resolved = _resolve_image_path(f, is_lunar=is_lunar)
        if not resolved:
            raise HTTPException(status_code=400, detail=f"Could not resolve observation file: {f}")
        arr, meta = load_image_as_array(str(resolved))
        meta["path"] = str(resolved)
        meta["filename"] = resolved.name
        images.append(arr)
        metas.append(meta)

    if not images:
        raise HTTPException(status_code=400, detail="No valid images provided.")

    logger.info("Executing %s analysis query: '%s' on %d image(s)", req.mode, req.query, len(images))

    if is_lunar:
        result = run_lunar_pipeline(images, metas, req.query)
    else:
        result = run_pipeline(images, metas, req.query)

    # Encode overlay image to base64 data URI if present (handles both PIL and NumPy arrays)
    overlay_data_uri = None
    if result.get("overlay") is not None:
        ov = result["overlay"]
        try:
            if isinstance(ov, Image.Image):
                buf = io.BytesIO()
                ov.save(buf, format="PNG")
                overlay_data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
            elif isinstance(ov, np.ndarray):
                if ov.dtype == bool:
                    ov = (ov.astype(np.uint8)) * 255
                elif np.issubdtype(ov.dtype, np.floating) and ov.max() <= 1.0:
                    ov = (ov * 255).astype(np.uint8)
                else:
                    ov = ov.astype(np.uint8)

                if ov.ndim == 2:
                    h, w = ov.shape
                    rgba = np.zeros((h, w, 4), dtype=np.uint8)
                    # Highlight change or detected masks with red/cyan overlay
                    rgba[ov > 40] = [239, 68, 68, 190]
                    pil_img = Image.fromarray(rgba, mode="RGBA")
                else:
                    pil_img = Image.fromarray(ov)
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                overlay_data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
        except Exception as exc:
            logger.warning("Could not encode overlay: %s", exc)

    # Generate detailed Markdown report
    try:
        report_md = generate_report(
            trace=result.get("trace", {}),
            phrased_answer=result.get("answer", ""),
            verified_facts=result.get("verified_facts", {}),
        )
    except Exception as exc:
        logger.warning("Could not generate markdown report: %s", exc)
        report_md = f"# SatQuery AI - Analysis Report\n\n## Summary\n\n{result.get('answer', '')}"

    clean_verified_facts = sanitize_for_json(result.get("verified_facts", {}))
    clean_trace = sanitize_for_json(result.get("trace", {}))
    clean_metas = sanitize_for_json(metas)

    # Attach model_calibration flag
    calib = (
        clean_verified_facts.get("model_calibration")
        or clean_trace.get("model_calibration")
        or result.get("model_calibration")
        or "calibrated"
    )
    clean_verified_facts["model_calibration"] = calib

    # Attach standardized spectral labels
    from satquery.perception.spectral_interpretation import (
        interpret_ndvi,
        interpret_ndwi,
        interpret_ndbi,
    )
    spectral = clean_verified_facts.get("spectral_summary") or sanitize_for_json(result.get("spectral_summary"))
    if spectral and isinstance(spectral, dict):
        if "ndvi_mean" in spectral and "ndvi_label" not in spectral:
            spectral["ndvi_label"] = interpret_ndvi(float(spectral["ndvi_mean"]))
        if "ndwi_mean" in spectral and "ndwi_label" not in spectral:
            spectral["ndwi_label"] = interpret_ndwi(float(spectral["ndwi_mean"]))
        if "ndbi_mean" in spectral and "ndbi_label" not in spectral:
            spectral["ndbi_label"] = interpret_ndbi(float(spectral["ndbi_mean"]))
        clean_verified_facts["spectral_summary"] = spectral

    return {
        "answer": result.get("answer", ""),
        "confidence": str(result.get("confidence", "0.0")),
        "confidence_tag": result.get("confidence_tag", "unverified"),
        "confidence_score": float(result["confidence_score"]) if result.get("confidence_score") is not None else None,
        "overlay": overlay_data_uri,
        "model_calibration": calib,
        "spectral_summary": spectral,
        "verified_facts": clean_verified_facts,
        "consensus_score": float(result["consensus_score"]) if result.get("consensus_score") is not None else None,
        "change_direction": result.get("change_direction"),
        "semantic_consistency": float(result["semantic_consistency"]) if result.get("semantic_consistency") is not None else None,
        "validation_failure_reason": result.get("validation_failure_reason"),
        "trace": clean_trace,
        "metas": clean_metas,
        "report_markdown": report_md,
    }


@app.post("/api/export-pdf")
def export_pdf(req: ExportPdfRequest):
    """Generate and stream an executive PDF report."""
    try:
        pdf_bytes = generate_pdf_report(req.result, req.query, req.images_meta)
        if not pdf_bytes:
            raise HTTPException(status_code=500, detail="Failed to synthesize PDF bytes")
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="SatQuery_Analysis_Report.pdf"'},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("PDF export error: %s", exc)
        raise HTTPException(status_code=500, detail=f"PDF export failed: {exc}")


@app.post("/api/batch")
def process_batch(req: BatchRequest) -> list[dict[str, Any]]:
    """Process sequential batch queue with error isolation."""
    is_lunar = req.mode.lower() == "lunar"
    batch_items = []

    for item in req.items:
        imgs, metas = [], []
        for f in item.files:
            resolved = _resolve_image_path(f, is_lunar=is_lunar)
            if resolved:
                arr, meta = load_image_as_array(str(resolved))
                meta["path"] = str(resolved)
                meta["filename"] = resolved.name
                imgs.append(arr)
                metas.append(meta)
        batch_items.append({
            "images": imgs,
            "metas": metas,
            "query": item.query,
        })

    results = run_batch_pipeline(batch_items, mode=req.mode)

    # Clean non-serializable objects (like numpy arrays) before JSON return
    sanitized = []
    for r in results:
        res = r.get("result")
        if res and "overlay" in res and isinstance(res["overlay"], Image.Image):
            buf = io.BytesIO()
            res["overlay"].save(buf, format="PNG")
            res["overlay"] = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
        sanitized.append({
            "query": r.get("query"),
            "status": r.get("status"),
            "result": res,
            "error": r.get("error"),
        })

    return sanitized


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
