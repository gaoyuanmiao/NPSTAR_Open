# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 11:20:28 2025

@author: 喵
"""

# -*- coding: utf-8 -*-
"""
demo_run_npstar.py
----------------------------------------------------------
Demo runner for the NPSTAR (Nonpoint Source Transport and Reduction) model.
This script reproduces a simplified Total Nitrogen (TN) simulation
using example datasets included in the open reproducibility package.

Author: NPSTAR Research Group (Beijing Normal University)
Contact: chenlei1982bnu@bnu.edu.cn
Version: 1.0
License: MIT License (code) / CC-BY 4.0 (data)
----------------------------------------------------------
"""

import os
import pandas as pd
import numpy as np
from code.npstar_core import model_run


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def load_parameters(filepath: str) -> np.ndarray:
    """Load model parameters from Excel file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Parameter file not found: {filepath}")
    params = pd.read_excel(filepath)["value"].to_numpy()
    return params


def main():
    print("🚀 Starting NPSTAR TN demo simulation...\n")

    # 1. Define input/output directories
    data_dir = "data"
    result_dir = "results"
    ensure_dir(result_dir)

    # 2. Load parameters
    param_file = os.path.join(data_dir, "demo_parameter.xlsx")
    params = load_parameters(param_file)

    # 3. Run NPSTAR model core
    print("🔄 Running NPSTAR model core...")
    sim_results = model_run(params)

    # 4. Save outlet results
    outlet_path = os.path.join(result_dir, "demo_TN_outlet.csv")
    pd.DataFrame({"TN_outlet_kg": sim_results}).to_csv(outlet_path, index=False)

    # 5. Generate example matrix (optional visualization placeholder)
    matrix = np.random.rand(30, 60) * 100  # demo data only
    matrix_path = os.path.join(result_dir, "demo_TN_matrix.csv")
    pd.DataFrame(matrix).to_csv(matrix_path, index=False)

    print("\n✅ Simulation finished successfully!")
    print(f"Results saved in:\n - {outlet_path}\n - {matrix_path}")
    print("\n🧭 Note: These are demo outputs for reproducibility demonstration only.")


if __name__ == "__main__":
    main()
