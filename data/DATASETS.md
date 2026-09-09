# SatQuery AI — Dataset Sources & Data Access Guide

This document catalogues all Earth-observation and Lunar satellite data portals, benchmark datasets, access protocols, and their corresponding demonstration samples used in SatQuery AI.

---

## 1. Indian Earth-Observation Data Portals (ISRO / NRSC / SAC)

### 1.1 Bhoonidhi (NRSC Open Data Hub)
- **Portal URL:** [https://bhoonidhi.nrsc.gov.in](https://bhoonidhi.nrsc.gov.in)
- **Operated by:** National Remote Sensing Centre (NRSC), ISRO, Hyderabad
- **Data Available:**
  - IRS / Resourcesat-2/2A (LISS-3, LISS-4, AWiFS)
  - Cartosat-1, Cartosat-2 series optical ortho imagery
  - Oceansat-2/3 scatterometer and ocean color data
  - Sentinel-1, Sentinel-2 open mirror feeds for the Indian subcontinent
- **Access Requirements:** Free registration required. Web-based ordering and batch download utility.
- **Workflow for Real Imagery:**
  1. Register at `bhoonidhi.nrsc.gov.in`.
  2. Define Area of Interest (AOI) bounding box or Indian district shapefile.
  3. Filter by sensor (e.g. `Resourcesat-2 LISS-4` at 5m resolution or `Cartosat-2`).
  4. Submit order and download via Bhoonidhi Download Client or HTTPS zip links.

### 1.2 Bhuvan Geoportal
- **Portal URL:** [https://bhuvan.nrsc.gov.in](https://bhuvan.nrsc.gov.in)
- **Operated by:** ISRO / NRSC
- **Data Available:**
  - Cartosat-1 30m / 10m Digital Elevation Models (CartoDEM)
  - Resourcesat AWiFS and LISS-III national ortho mosaics
  - 1-meter high-resolution urban imagery covering 177+ Indian municipal corporations
  - Thematic layers: Land Use / Land Cover (LULC 1:50k / 1:250k), flood inundation atlases
- **Access Requirements:** Free user account required for raster downloads; WMS/WMTS web tile services freely browsable.
- **Workflow for Real Imagery:**
  1. Access `bhuvan.nrsc.gov.in/data/index.php`.
  2. Select *Thematic Services* or *Open Data Archive*.
  3. Select tile of interest and download GeoTIFF tiles.

### 1.3 VEDAS (Visualisation of Earth Data and Archival System)
- **Portal URL:** [https://vedas.sac.gov.in](https://vedas.sac.gov.in)
- **Operated by:** Space Applications Centre (SAC), ISRO, Ahmedabad
- **Data Available:**
  - RISAT-1 / EOS-04 C-band SAR Level-1/2 backscatter products
  - Polarimetric and hybrid-pol microwave decomposition data
  - Agricultural crop monitoring, vegetation condition index (VCI), soil moisture products
- **Access Requirements:** Open public analytics; bulk raster downloads require academic/institutional login.
- **Usage in SatQuery AI:** Provides the physical SAR dual-polarization ($\sigma^0_{VV}$, $\sigma^0_{VH}$) thresholds implemented in `satquery.perception.sar_backscatter` and `satquery.fusion`.

### 1.4 MOSDAC (Meteorological & Oceanographic Satellite Data Archival Centre)
- **Portal URL:** [https://mosdac.gov.in](https://mosdac.gov.in)
- **Operated by:** Space Applications Centre (SAC), ISRO
- **Data Available:**
  - INSAT-3D/3DR meteorological imager and sounder data (thermal infrared, water vapor)
  - Severe weather, cyclone track radar, ocean wave heights, and rainfall rate estimates
- **Access Requirements:** Free user registration; automated API access for research and disaster response.

---

## 2. Chandrayaan-2 Lunar Data (ISSDC PRADAN)

### 2.1 ISSDC PRADAN & OHRC Map Browser
- **Portal URL:** [https://pradan.issdc.gov.in](https://pradan.issdc.gov.in)
- **OHRC Map Browser:** [https://chmapbrowse.issdc.gov.in](https://chmapbrowse.issdc.gov.in)
- **Operated by:** Indian Space Science Data Centre (ISSDC), ISRO, Bengaluru
- **Instruments Available:**
  - **OHRC (Orbiter High Resolution Camera):** World's highest resolution orbital lunar imager (~0.25 m to 0.32 m/pixel from 100 km orbit).
  - **TMC-2 (Terrain Mapping Camera 2):** 5-meter stereo triplet imaging for 3D digital elevation models of the lunar surface.
  - **CLASS (Chandrayaan-2 Large Area Soft X-ray Spectrometer):** Elemental composition (Mg, Al, Si, Fe).
  - **DFSAR (Dual Frequency Synthetic Aperture Radar):** L-band and S-band radar mapping of polar subsurface water-ice.
- **Format:** PDS-4 standard (`.IMG` raster with accompanying `.xml` or `.lbl` labels). Derived orthophotos and DEMs available in GeoTIFF.
- **Access Requirements:** Free registration on PRADAN.
- **Conversion Utility:** See `data/scripts/convert_pds_img.py` for converting raw PDS-4 `.IMG` products into standard GeoTIFF/PNG.
- **Mandatory ISSDC Acknowledgment Notice:**
  > *"The data used in this research/work is obtained from the Indian Space Science Data Centre (ISSDC), Indian Space Research Organisation (ISRO), Government of India, through PRADAN portal (https://pradan.issdc.gov.in)."*

---

## 3. Pre-Trained & Evaluated Academic Benchmarks (Integrated)

| Benchmark | Sensor / Modality | Primary Task | Integration in SatQuery AI |
|---|---|---|---|
| **BigEarthNet-S2** | Sentinel-2 (12-band multi-spectral) | Multi-label Land Cover Classification | Backbone for ResNet-18 19-class classifier in `satquery/classifiers/` |
| **RSVQAxBEN** | Sentinel-2 | Remote Sensing Visual Question Answering | VQA benchmark splits evaluated in `evaluation/run_vqa_eval.py` |
| **VRSBench** | High-resolution Optical | Visual Grounding & Detailed Captioning | Grounding evaluation and phrase localization in `evaluation/` |
| **LEVIR-CD** | Multi-temporal Optical (0.5m) | Building & Urban Change Detection | Benchmark bitemporal pairs in `data/demo_samples/bitemporal_pairs/` |
| **CDVQA** | Bi-temporal Remote Sensing | Change Visual Question Answering | Change VQA specialist training & evaluation |

---

## 4. Repository Demo Sample Catalog

| Category | Sample Filename | Sensor | Feature / Scenario | Source |
|---|---|---|---|---|
| **Single Optical** | `sample_coastal_port.png` | Sentinel-2A L2A | Maritime domain, piers, harbor waters | Copernicus Open Access / Bhoonidhi |
| **Single Optical** | `sample_urban_grounding.png` | Sentinel-2A L2A | Industrial units, logistics warehouses | Copernicus Open Access / Bhoonidhi |
| **Single SAR** | `sample_sar_vv_vh.png` | Sentinel-1 GRD | Dual-pol ($\sigma^0_{VV}, \sigma^0_{VH}$) double-bounce | Copernicus Sentinel-1 / VEDAS |
| **Optical+SAR Pair** | `sample_monsoon_cloudy.png` + `sample_sar_vv_vh.png` | S2 + S1 Co-registered | Monsoon cloud penetration & flood mapping | Copernicus S2/S1 / SAC VEDAS |
| **Bi-temporal Pair** | `sample_levir_pre.png` + `sample_levir_post.png` | Multi-temporal Optical | Urban encroachment & new construction | LEVIR-CD Benchmark |
| **Lunar (OHRC)** | `ch2_ohrc_crater_boguslawsky.png` | Chandrayaan-2 OHRC (0.25m) | Boguslawsky crater rim, terraces, floor | ISSDC PRADAN (CH2_OHR) |
| **Lunar (OHRC)** | `ch2_ohrc_shackleton_rim_psr.png` | Chandrayaan-2 OHRC (0.32m) | South Pole Shackleton Rim & PSR shadow | ISSDC PRADAN (CH2_OHR) |
| **Lunar (TMC-2)** | `ch2_tmc2_regolith_ejecta.png` | Chandrayaan-2 TMC-2 (5.0m) | Basaltic mare regolith & fresh ejecta rays | ISSDC PRADAN (CH2_TMC) |
