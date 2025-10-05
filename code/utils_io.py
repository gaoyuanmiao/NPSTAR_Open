# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 11:24:25 2025

@author: 喵
"""

# -*- coding: utf-8 -*-
"""
utils_io.py
----------------------------------------------------------
Utility functions for file input/output operations in the NPSTAR
(Nonpoint Source Transport and Reduction) model reproducibility package.

Functions included:
    - ensure_dir(path): Create directory if it does not exist
    - load_csv(path): Load CSV file into numpy array or DataFrame
    - load_excel(path, sheet_name): Load Excel sheet
    - load_pickle(path): Load serialized Python object (.pkl)
    - save_csv(data, path): Save numpy array or DataFrame to CSV
    - check_files_exist(file_list): Verify required input files are present

These utilities are designed for clean, cross-platform reproducibility
without hard-coded paths or dependencies on local directories.

Author: NPSTAR Research Group (Beijing Normal University)
Contact: chenlei1982bnu@bnu.edu.cn
License: MIT (code) / CC BY 4.0 (data)
----------------------------------------------------------
"""

import os
import pickle
import pandas as pd
import numpy as np


# --------------------------------------------------------------------- #
# Basic directory utilities
# --------------------------------------------------------------------- #
def ensure_dir(path: str):
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"📂 Created directory: {path}")


# --------------------------------------------------------------------- #
# File loading utilities
# --------------------------------------------------------------------- #
def load_csv(path: str, as_array: bool = True):
    """Load a CSV file as pandas DataFrame or numpy array."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ CSV file not found: {path}")
    df = pd.read_csv(path)
    return df.to_numpy() if as_array else df


def load_excel(path: str, sheet_name: str | int | None = None):
    """Load Excel file into pandas DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Excel file not found: {path}")
    return pd.read_excel(path, sheet_name=sheet_name)


def load_pickle(path: str):
    """Load Python object from pickle file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Pickle file not found: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


# --------------------------------------------------------------------- #
# File saving utilities
# --------------------------------------------------------------------- #
def save_csv(data, path: str, index: bool = False):
    """Save numpy array or DataFrame to CSV."""
    ensure_dir(os.path.dirname(path) or ".")
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data.to_csv(path, index=index)
    else:
        pd.DataFrame(data).to_csv(path, index=index)
    print(f"✅ Saved file: {path}")


# --------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------- #
def check_files_exist(file_list: list[str]):
    """Check that all required files exist before model execution."""
    missing = [f for f in file_list if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            f"⚠️ Missing required input files:\n" + "\n".join(f" - {m}" for m in missing)
        )
    else:
        print("✅ All required files are present.")


# --------------------------------------------------------------------- #
# Example usage (for developers)
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    print("🧰 utils_io module test")
    ensure_dir("results")
    arr = np.random.rand(5, 5)
    save_csv(arr, "results/demo_test.csv")
    print(load_csv("results/demo_test.csv"))
