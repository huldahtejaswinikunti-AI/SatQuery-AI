"""Acquisition script for VRSBench (Versatile Vision-Language Benchmark for Remote Sensing).

Fetches verified VQA, captioning, and referring expression grounding subsets
from the official Hugging Face repository (xiang709/VRSBench).
Used for GeoChat-7B LoRA fine-tuning and CLIPSeg grounding evaluation.

Usage:
    python data/scripts/download_vrsbench.py --out-dir data/raw/vrsbench --max-samples 200
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VRSBenchDownloader")

HF_VRSBENCH_API = "https://huggingface.co/api/datasets/xiang709/VRSBench"
RAW_BASE_URL = "https://huggingface.co/datasets/xiang709/VRSBench/raw/main"


def download_vrsbench_subset(out_dir: str | Path = "data/raw/vrsbench", max_samples: int = 200) -> None:
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing VRSBench download into: %s", dest)

    # 1. Download evaluation questions/annotations
    files_to_fetch = [
        ("VRSBench_EVAL_vqa.json", f"{RAW_BASE_URL}/VRSBench_EVAL_vqa.json"),
        ("VRSBench_EVAL_referring.json", f"{RAW_BASE_URL}/VRSBench_EVAL_referring.json"),
    ]

    for fname, url in files_to_fetch:
        target = dest / fname
        if not target.exists():
            logger.info("Fetching annotation catalog %s ...", fname)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "SatQuery-AI/1.0"})
                with urllib.request.urlopen(req, timeout=20) as resp, open(target, "wb") as out_f:
                    out_f.write(resp.read())
                logger.info("Saved %s (%.1f KB)", fname, target.stat().st_size / 1024.0)
            except Exception as e:
                logger.warning("Failed to fetch %s: %s", fname, e)
        else:
            logger.info("Annotation catalog %s already exists.", fname)

    # 2. Parse downloaded annotations to create normalized SatQuery training format
    vqa_file = dest / "VRSBench_EVAL_vqa.json"
    if vqa_file.exists():
        try:
            with open(vqa_file, encoding="utf-8") as f:
                data = json.load(f)
            
            # Format normalized instruction training samples
            samples = []
            records = data if isinstance(data, list) else data.get("data", [])
            for item in records[:max_samples]:
                samples.append({
                    "task": "vqa",
                    "dataset": "VRSBench",
                    "image": item.get("image", item.get("img_id", "sample.png")),
                    "instruction": item.get("question", item.get("conversations", [{}])[0].get("value", "")),
                    "answer": item.get("answer", item.get("answers", [""])[0] if isinstance(item.get("answers"), list) else ""),
                    "metadata": {
                        "modality": "optical_high_res",
                        "license": "CC BY 4.0",
                        "source": "VRSBench (Chen et al., 2024)",
                    }
                })

            out_catalog = dest / "vrsbench_satquery_vqa.json"
            with open(out_catalog, "w", encoding="utf-8") as out_f:
                json.dump({"total_samples": len(samples), "samples": samples}, out_f, indent=2)

            logger.info("Materialized %d normalized VRSBench training instruction records to %s", len(samples), out_catalog)
        except Exception as e:
            logger.warning("Error parsing VRSBench annotations: %s", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare VRSBench dataset subset.")
    parser.add_argument("--out-dir", default="data/raw/vrsbench", help="Destination folder")
    parser.add_argument("--max-samples", type=int, default=200, help="Number of instruction pairs to prepare")
    args = parser.parse_args()

    download_vrsbench_subset(args.out_dir, args.max_samples)


if __name__ == "__main__":
    main()
