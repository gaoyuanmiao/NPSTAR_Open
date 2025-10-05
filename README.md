# NPSTAR: Nonpoint Source Transport and Reduction Model (Open Reproducibility Package)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.TBD.svg)](https://doi.org/10.5281/zenodo.TBD)

This repository contains the **open reproducibility package** for the NPSTAR model, supporting the publication:

> *High-Resolution Remote Sensing and Machine Learning-Enhanced Distributed Hydro-Environmental Modeling for Spatially Explicit Nonpoint Source Pollution Management in Small Agricultural Catchments.*

The NPSTAR framework integrates **remote sensing**, **machine learning**, and **process-based hydro-environmental simulation** for managing total nitrogen (TN) and total phosphorus (TP) in small agricultural catchments.  
This package provides all materials required to **reproduce the TN-only NPSTAR simulation** in a simplified demonstration setting.

---

## Repository Structure

| Folder / File | Description |
|----------------|-------------|
| `code/` | Python scripts for NPSTAR simulation (`demo_run_npstar.py`) |
| `data/` | Demo datasets (synthetic example inputs) |
| `results/` | Output files generated after running the demo |
| `environment.yml` | Conda environment specification |
| `HOWTO_REPRODUCE.md` | Detailed step-by-step reproducibility guide |
| `README_DATA.md` | Dataset source and description |
| `LICENSE` | License information (MIT for code, CC-BY-4.0 for data) |

---

## Requirements

Set up the environment:

```bash
conda env create -f environment.yml
conda activate npstar
```

Dependencies include: `numpy`, `pandas`, `rasterio`, `geopandas`, `matplotlib`, `tqdm`, and `openpyxl`.

---

## Quick Start

Run the TN demo simulation:

```bash
python code/demo_run_npstar.py
```

Expected outputs:  
- `results/demo_TN_outlet.csv` — simulated TN load at the outlet (kg)  
- `results/demo_TN_matrix.csv` — grid-based TN flux distribution  

Execution time: **under 2 minutes** on a standard laptop.

---

## Reproducibility Scope

- **Included:** NPSTAR hydrologic–biogeochemical TN module  
- **Excluded:** Machine learning (BMP efficiency), SHAP, cost–benefit optimization  

All demo data are **synthetic**, non-sensitive, and serve to demonstrate workflow reproducibility.

---

## License

- **Code:** MIT License  
- **Data:** CC BY 4.0 License  
---

## Contact

**Corresponding author:**  
Lei Chen, Beijing Normal University  
chenlei1982bnu@bnu.edu.cn  

---

**End of README**
