# How to Reproduce the NPSTAR Demo (TN, NPSTAR-only)

This document provides detailed, step-by-step instructions for reproducing the **NPSTAR model** demonstration.  
It covers all processes needed to execute the **Total Nitrogen (TN)** simulation in a minimal, synthetic setting.  
This example excludes machine learning and cost-benefit components to focus on the core NPSTAR model reproducibility.

---

## 1. Prerequisites

- **Operating System:** Windows / macOS / Linux  
- **Python:** version 3.10 or higher  
- **Conda:** version 23 or higher (recommended)

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate npstar
```

Dependencies include:
`numpy`, `pandas`, `geopandas`, `rasterio`, `matplotlib`, `tqdm`, `openpyxl`.

---

## 2. Repository Structure

Your directory should look like this:

```
.
├─ code/
│  └─ demo_run_npstar.py
├─ data/
│  ├─ demo_PandQ.csv
│  ├─ demo_TN_obs.xlsx
│  ├─ demo_flow_dict.pkl
│  ├─ demo_landuse.csv
│  ├─ demo_bmp.csv
│  ├─ demo_TN_dis_month1.csv
│  └─ demo_parameter.xlsx
├─ results/
│  └─ (automatically created)
├─ environment.yml
└─ HOWTO_REPRODUCE.md
```

---

## 3. Running the Demo Simulation

Execute the following command from the root directory:

```bash
python code/demo_run_npstar.py
```

This script performs:
1. Loading all demo input data (`data/` folder);
2. Running the NPSTAR TN module with fixed demo parameters;
3. Saving results in the `results/` folder.

Expected runtime: **< 2 minutes**.

---

## 4. Expected Outputs

After successful execution, you should see:

| Output File | Description |
|--------------|-------------|
| `results/demo_TN_outlet.csv` | Simulated TN load at the outlet (kg) |
| `results/demo_TN_matrix.csv` | Grid-based TN distribution combining surface + subsurface fluxes |

Output structure example:
```
demo_TN_outlet.csv → one value column (TN_outlet_kg)
demo_TN_matrix.csv → raster-style numeric array
```

If both files appear and contain numeric data, the run was successful.

---

## 5. Verification

You can verify the run with the following steps:

```bash
import pandas as pd
data = pd.read_csv("results/demo_TN_outlet.csv")
print(data.head())
```

Expected output: a small numeric table (e.g., TN load ≈ 10³–10⁵ kg depending on random seed).  
Visual checks can also be done using the optional plot commands in the script.

---

## 6. Troubleshooting

| Issue | Possible Cause | Solution |
|--------|----------------|-----------|
| `ModuleNotFoundError` | Conda environment not activated | Run `conda activate npstar` |
| `FileNotFoundError` | Input files missing or renamed | Check that all demo files exist under `data/` |
| Output empty / NaN | Input values all zeros | Ensure demo data contain numeric (non-zero) values |
| Runtime too long | Environment conflict | Try recreating Conda env or use `mamba` |
| Shape mismatch | Input raster dimensions differ | All raster inputs must have identical shape |

---

## 7. Reproducibility Scope

This open demo reproduces **only the TN sub-module** of NPSTAR.  
It includes:
- Hydrological routing  
- Biogeochemical legacy handling  
- Recursive upstream tracing  

It excludes:
- Machine learning BMP optimization  
- SHAP interpretation  
- Cost-benefit scenarios  

---

## 8. Data License and Access

All demo datasets are synthetic and released under **CC BY 4.0 License**.  
Real monitoring data can be provided upon reasonable request to the corresponding author.

---

## 9. Contact

**Corresponding author:**  
Lei Chen, Beijing Normal University  
chenlei1982bnu@bnu.edu.cn  

---

**End of HOWTO_REPRODUCE**
