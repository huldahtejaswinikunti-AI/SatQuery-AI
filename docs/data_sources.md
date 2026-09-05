# SatQuery AI — Datasets & Pretrained Model Sources

## Datasets

1. **BigEarthNet-MM / BigEarthNet v2.0**
   - Paired Sentinel-1 SAR and Sentinel-2 optical image patches.
   - CORINE Land Cover 19-class classification scheme.
   - Source: [bigearth.net](https://bigearth.net) / Microsoft TorchGeo.

2. **RSVQAxBEN**
   - Visual Question Answering pairs built directly on BigEarthNet Sentinel-2 imagery.
   - Used for LoRA fine-tuning and adaptation evaluation.

3. **LEVIR-CD & CDVQA**
   - Building change detection and bi-temporal change VQA benchmark datasets.

---

## Pretrained Specialist Models

1. **GeoChat-7B** (`MBZUAI/geochat-7B`)
   - Apache-2.0 License.
   - Grounded LVLM trained specifically on remote sensing imagery.

2. **CLIPSeg** (`CIDAS/clipseg-rd64`)
   - Open-vocabulary zero-shot segmentation model.

3. **TinyCD** (`AndreaCodegoni/Tiny_model_4_CD`)
   - Lightweight, efficient Siamese bi-temporal change detection model.

4. **Phi-3-mini-4k-instruct** (`microsoft/Phi-3-mini-4k-instruct`)
   - Lightweight local instruct model for structured fact phrasing.

