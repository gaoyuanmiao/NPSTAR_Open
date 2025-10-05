# README_DATA.md

## Overview

This document describes the datasets used to reproduce the NPSTAR (Nonpoint Source Transport and Reduction) model in the open demo package.  
The data are simplified or synthetic representations of the inputs used in the original study and are designed solely for reproducibility demonstration purposes.

---

## Data Sources

The fundamental datasets required for this study include a **Digital Elevation Model (DEM)**, **land use data**, **monthly water quality records**, **daily precipitation data**, and **crop management information**.

| Dataset | Source | Resolution / Period | Description |
|----------|---------|---------------------|-------------|
| DEM | National Geomatics Center of China ([https://www.ngcc.cn/](https://www.ngcc.cn/)) and China Geospatial Data Cloud ([https://www.gscloud.cn/](https://www.gscloud.cn/)) | 30 m × 30 m | Used for terrain and flow direction derivation |
| Land use | China Geospatial Data Cloud ([https://www.gscloud.cn/](https://www.gscloud.cn/)), reclassified using GEE ([https://earthengine.google.com/](https://earthengine.google.com/)) | 10 m × 10 m | Used for land-type dependent source generation |
| Water quality & precipitation | Chaohu Lake Management Authority | 2010–2019 | Monthly TN/TP concentrations and daily rainfall records |
| Crop management | Field surveys & statistical yearbooks | 2018–2019 | Fertilizer and irrigation data used in parameterization |
| Demo dataset | Constructed based on the Chonghu small catchment | Synthetic | Used for public demonstration and reproducibility testing |

All **demo datasets** are anonymized, spatially simplified, and non-sensitive.

---

## Data Included in This Repository

| File Name | Description |
|------------|-------------|
| `demo_PandQ.csv` | Monthly precipitation and surface flow (mm, m³/s) |
| `demo_TN_obs.xlsx` | Observed TN concentrations (mg/L) used for threshold logic |
| `demo_flow_dict.pkl` | Flow direction dictionary (`{grid_code: [upstream_cells]}`) |
| `demo_landuse.csv` | Land-use raster matrix (numeric codes: 1–8) |
| `demo_bmp.csv` | BMP layout (0 = none, 1 = vegetated strip, 2 = pond/ditch) |
| `demo_TN_dis_month1.csv` | Dissolved TN source (kg) for one demo month |
| `demo_parameter.xlsx` | Model parameter table (column `value`) |

Each raster-type file uses the same grid dimensions to ensure model consistency.

---

## Data Policy and License

- **Demo data:** openly released under **CC BY 4.0 License**  
- **Original monitoring data:** available upon reasonable request to the corresponding author  
- **Usage note:** These datasets are intended only for demonstration and reproducibility purposes; they do not represent actual field measurements in the study area.

---

**Contact:**  
Lei Chen (Beijing Normal University)  
📧 chenlei1982bnu@bnu.edu.cn  

---

**End of README_DATA**
