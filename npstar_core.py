# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 11:22:47 2025

@author: 喵
"""

# -*- coding: utf-8 -*-
"""
npstar_core.py
----------------------------------------------------------
Core routines for the NPSTAR (Nonpoint Source Transport and Reduction) model.
This module provides a cleaned, self-contained implementation of the TN
simulation for open reproducibility (demo scale).

Key function:
    model_run(par: np.ndarray, data_dir: str = "data",
              outlet_code: str | None = None,
              outlet_rc: tuple[int,int] | None = (25, 40)) -> np.ndarray

Inputs expected in `data_dir`:
    - demo_PandQ.csv            : columns [pcp, surface_flow] for the demo month(s)
    - demo_TN_obs.xlsx          : first column = TN observations (used in a threshold)
    - demo_flow_dict.pkl        : { "rrrrcccc": ["rrrrcccc", ...], ... } upstream map
    - demo_landuse.csv          : land use raster (numeric classes)
    - demo_bmp.csv              : BMP raster (0 none, 1 VFS, 2 GW)
    - demo_TN_dis_month1.csv    : dissolved TN source for month 1

Notes:
- This demo runs a single month by default.
- All arrays must share the same shape (rows × cols) for rasters.
- No plotting, no absolute paths, no external side-effects.

Author: NPSTAR Research Group (Beijing Normal University)
Contact: chenlei1982bnu@bnu.edu.cn
License: MIT (code) / CC BY 4.0 (data)
----------------------------------------------------------
"""

from __future__ import annotations

import os
import pickle
import numpy as np
import pandas as pd


# ----------------------------- I/O utilities ----------------------------- #
def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _read_csv_matrix(path: str) -> np.ndarray:
    """Read a CSV grid saved with header row/col; returns a float ndarray."""
    df = pd.read_csv(path, index_col=0)
    return df.to_numpy(dtype=float)


def _check_same_shape(*arrays: np.ndarray):
    """Ensure all rasters have identical shape."""
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise ValueError(f"Raster inputs have different shapes: {shapes}")


# ---------------------------- Core model logic --------------------------- #
def model_run(
    par: np.ndarray,
    data_dir: str = "data",
    outlet_code: str | None = None,
    outlet_rc: tuple[int, int] | None = (25, 40),
) -> np.ndarray:
    """
    Run the NPSTAR TN demo (single-month) and return outlet TN (kg) as a 1D array.

    Parameters
    ----------
    par : np.ndarray
        Parameter vector; must have length >= 60 (indexes up to 59 used).
    data_dir : str
        Directory where demo inputs reside (default "data").
    outlet_code : str | None
        Grid code 'rrrrcccc' to be used as the outlet for upstream routing.
        If None, the first key found in flow dict is used.
    outlet_rc : (int, int) | None
        Row/col index used to probe the routed 2D field when aggregating
        surface outputs; clipped to raster bounds.

    Returns
    -------
    np.ndarray
        1D array of outlet TN (kg) for each simulated month (demo → length 1).
    """
    if par.size < 60:
        raise ValueError("Parameter vector must contain at least 60 values (used up to par[59]).")

    # ---- load inputs ----
    pcp_q = pd.read_csv(os.path.join(data_dir, "demo_PandQ.csv"))
    obs = np.array(pd.read_excel(os.path.join(data_dir, "demo_TN_obs.xlsx")))[:, 0]
    dir_dict: dict[str, list[str]] = _load_pickle(os.path.join(data_dir, "demo_flow_dict.pkl"))
    sink = _read_csv_matrix(os.path.join(data_dir, "demo_landuse.csv"))
    bmp = _read_csv_matrix(os.path.join(data_dir, "demo_bmp.csv"))
    source_month1 = _read_csv_matrix(os.path.join(data_dir, "demo_TN_dis_month1.csv"))

    _check_same_shape(sink, bmp, source_month1)

    rows, cols = sink.shape
    # clip outlet rc into bounds
    if outlet_rc is None:
        outlet_rc = (25, 40)
    oi = max(0, min(rows - 1, outlet_rc[0]))
    oj = max(0, min(cols - 1, outlet_rc[1]))

    # pick outlet code if not provided
    if outlet_code is None:
        # use any key from dict (demo); users can set their own outlet by name
        outlet_code = next(iter(dir_dict.keys()))

    # ---- map land use to 3 classes used by parameter allocation ----
    # This mirrors the classification in your original snippet:
    sink_classified = sink.copy()
    # Example mapping used in your code (adjust to your class scheme as needed):
    # sink_classified[sink==1] = 2  # RICE → 2
    # sink_classified[sink==2] = 1
    # sink_classified[sink==3] = 1
    # sink_classified[sink==4] = 3
    sink_classified[sink == 1] = 2
    sink_classified[sink == 2] = 1
    sink_classified[sink == 3] = 1
    sink_classified[sink == 4] = 3

    # simple slope field (demo); replace with real slope if available
    slope = np.ones_like(sink, dtype=float)

    # ---- helpers ----
    def par_alloc(p_a, p_n, p_u):
        arr = sink_classified.copy()
        arr[sink_classified == 1] = p_a
        arr[sink_classified == 2] = p_n
        arr[sink_classified == 3] = p_u
        return arr

    def path_allocation(mon, source):
        # parameters by land-use class
        source_allocation = par_alloc(par[0], par[17], par[34])
        dis_leak1 = par_alloc(par[1], par[18], par[35])
        dis_leak2 = par_alloc(par[2], par[19], par[36])
        ads_leak1 = par_alloc(par[3], par[20], par[37])
        ads_leak2 = par_alloc(par[4], par[21], par[38])

        # monthly hydro inputs
        pcp = float(pcp_q.iloc[mon, 0])  # precipitation
        surface_flow = float(pcp_q.iloc[mon, 1])

        # split into dissolved / adsorbed
        dissolved = source * source_allocation
        adsorbed = source * (1 - source_allocation)

        # allocate to surface vs legacy pools (proportional to runoff ratio)
        if pcp <= 0:
            surf_dis, surf_ads = dissolved, adsorbed
            leg_dis = np.zeros_like(dissolved)
            leg_ads = np.zeros_like(adsorbed)
        else:
            ratio = min(max(surface_flow / pcp, 0.0), 1.0)
            surf_dis = dissolved * ratio
            surf_ads = adsorbed * ratio
            leg_dis = dissolved * (1 - ratio)
            leg_ads = adsorbed * (1 - ratio)

        # legacy pools split into bgc / hyd layers
        bgc_dis = leg_dis * (1 - dis_leak1)
        hyd_dis = leg_dis * dis_leak1 * (1 - dis_leak2)
        bgc_ads = leg_ads * (1 - ads_leak1)
        hyd_ads = leg_ads * ads_leak1 * (1 - ads_leak2)
        return surf_dis, surf_ads, bgc_dis, bgc_ads, hyd_dis, hyd_ads

    def bgc_legacy(mon, new_nutrient, legacy_pool, nutrient_form: str):
        if nutrient_form == "dis":
            p_current = par_alloc(par[5], par[22], par[39])
            p_history = par_alloc(par[6], par[23], par[40])
            leak = par_alloc(par[7], par[24], par[41])
        else:
            p_current = par_alloc(par[8], par[25], par[42])
            p_history = par_alloc(par[9], par[26], par[43])
            leak = par_alloc(par[10], par[27], par[44])

        total_contrib = p_current * new_nutrient + p_history * legacy_pool
        raw_pool = legacy_pool + new_nutrient - total_contrib
        bgc_to_hyd = raw_pool * 0.1
        new_pool = raw_pool * (1 - 0.1) * (1 - leak)
        return total_contrib, new_pool, bgc_to_hyd

    def hyd_legacy(mon, new_nutrient, legacy_pool, nutrient_form: str):
        if nutrient_form == "dis":
            p_history = par_alloc(par[12], par[29], par[46])
            leak = par_alloc(par[13], par[30], par[47])
        else:
            p_history = par_alloc(par[15], par[32], par[49])
            leak = par_alloc(par[16], par[33], par[50])

        total_contrib = legacy_pool * (p_history ** mon)
        new_pool = (legacy_pool + new_nutrient - total_contrib) * (leak ** mon)
        return total_contrib, new_pool

    # recursive routing with reductions/BMP effects
    def build_tracers(obs_val):
        output_dis = None
        output_ads = None

        def trace_back_dis(code: str):
            nonlocal output_dis
            x = int(code[0:4]); y = int(code[4:])
            if code not in dir_dict:
                return total_dis_source[x, y]
            for up in dir_dict[code]:
                output_dis[x, y] += trace_back_dis(up)
                # reductions by land cover
                if sink[x, y] == 3:  # FRST
                    if output_dis[x, y] > par[53]:
                        output_dis[x, y] *= (5.889 + 0.1609 * slope[x, y] - 0.0353 + 1.007 - 0.4511 * 0.4 + 59.8298) / 100
                    else:
                        output_dis[x, y] -= par[53]
                elif sink[x, y] == 1:  # WATR
                    if output_dis[x, y] > par[56]:
                        output_dis[x, y] *= (0.0797 * np.exp(-0.00518 * (2700 / output_dis[x, y])) + 65.5432) / 100
                    else:
                        output_dis[x, y] = 0
                elif sink[x, y] == 2:
                    if output_dis[x, y] > obs_val / par[59]:
                        output_dis[x, y] *= 0.9
                # BMPs
                if bmp[x, y] == 1:  # VFS
                    if output_dis[x, y] < par[53] * 10:
                        output_dis[x, y] *= (0.3164 * 10 - 0.1201 * 10 + 0.1609 * slope[x, y] - 0.0353 * 100 - 0.4511 * 0.5 + 59.8298) / 100
                    else:
                        output_dis[x, y] -= par[53] * 1
                if bmp[x, y] == 2:  # GW
                    if output_dis[x, y] < par[53] * 6:
                        output_dis[x, y] *= (0.3164 * 10 - 0.1201 * 10 + 0.1609 * slope[x, y] - 0.0353 * 100 - 0.4511 * 0.5 + 65.8298) / 100
                    else:
                        output_dis[x, y] -= par[53] * 2
            return output_dis[x, y]

        def trace_back_ads(code: str):
            nonlocal output_ads
            x = int(code[0:4]); y = int(code[4:])
            if code not in dir_dict:
                return total_ads_source[x, y]
            for up in dir_dict[code]:
                output_ads[x, y] += trace_back_ads(up)
                if sink[x, y] == 3:
                    if output_ads[x, y] > par[53]:
                        output_ads[x, y] *= (5.889 + 0.1609 * slope[x, y] - 0.0353 + 1.007 - 0.4511 * 0.4 + 59.8298) / 100
                    else:
                        output_ads[x, y] -= par[53]
                elif sink[x, y] == 1:
                    if output_ads[x, y] > par[56]:
                        output_ads[x, y] *= (0.0797 * np.exp(-0.00518 * (2700 / output_ads[x, y])) + 65.5432) / 100
                    else:
                        output_ads[x, y] = 0
                elif sink[x, y] == 2:
                    if output_ads[x, y] > obs_val / par[59]:
                        output_ads[x, y] *= 0.9
                if bmp[x, y] == 1:
                    if output_ads[x, y] < par[53] * 10:
                        output_ads[x, y] *= (0.3164 * 10 - 0.1201 * 10 + 0.1609 * slope[x, y] - 0.0353 * 100 - 0.4511 * 0.5 + 59.8298) / 100
                    else:
                        output_ads[x, y] -= par[53] * 1
                if bmp[x, y] == 2:
                    if output_ads[x, y] < par[53] * 6:
                        output_ads[x, y] *= (0.3164 * 10 - 0.1201 * 10 + 0.1609 * slope[x, y] - 0.0353 * 100 - 0.4511 * 0.5 + 65.8298) / 100
                    else:
                        output_ads[x, y] -= par[53] * 2
            return output_ads[x, y]

        def route(total_dis, total_ads):
            nonlocal output_dis, output_ads
            output_dis = total_dis.copy()
            output_ads = total_ads.copy()
            # launch recursion from outlet
            trace_back_dis(outlet_code)
            trace_back_ads(outlet_code)
            return output_dis, output_ads

        return route

    # -------------------------- one-month demo -------------------------- #
    mon = 0

    # Source field (dissolved TN) with dry-season scaling, keep potential point-source cell if present.
    source = source_month1.copy()
    source[np.isnan(source)] = 0.0
    source *= 0.7159  # dry-season factor (kept from your code)
    # Optional point-source retention at [9, 29] if within bounds:
    if rows > 9 and cols > 29:
        ps = source_month1[9, 29]
        source[9, 29] = ps * (1 - 0.72)

    # Path allocation
    surf_dis, surf_ads, bgc_dis, bgc_ads, hyd_dis, hyd_ads = path_allocation(mon, source)

    # Initialize legacy pools
    bgc_dis_pool = np.zeros_like(sink)
    bgc_ads_pool = np.zeros_like(sink)
    hyd_dis_pool = np.zeros_like(sink)
    hyd_ads_pool = np.zeros_like(sink)

    # BGC legacy contributions
    bgc_dis_total, bgc_dis_pool, bgc2hyd_dis = bgc_legacy(mon, bgc_dis, bgc_dis_pool, "dis")
    bgc_ads_total, bgc_ads_pool, bgc2hyd_ads = bgc_legacy(mon, bgc_ads, bgc_ads_pool, "ads")

    # Total surface/interflow sources to be routed
    total_dis_source = surf_dis + bgc_dis_total
    total_ads_source = surf_ads + bgc_ads_total

    # Build routers & route to outlet (recursive backtracking)
    route = build_tracers(obs_val=float(obs[mon]))
    output_dis, output_ads = route(total_dis_source, total_ads_source)

    # Hydrologic legacy (GW-related)
    hyd_dis_total, hyd_dis_pool = hyd_legacy(mon, hyd_dis + bgc2hyd_dis, hyd_dis_pool, "dis")
    hyd_ads_total, hyd_ads_pool = hyd_legacy(mon, hyd_ads + bgc2hyd_ads, hyd_ads_pool, "ads")

    # Combine to outlet (use a probe at (oi, oj) + add hydrologic totals)
    outlet_val = (
        float(output_dis[oi, oj])
        + float(output_ads[oi, oj])
        + float(np.sum(hyd_dis_total))
        + float(np.sum(hyd_ads_total))
    )

    # Return as 1D array for consistency with multi-month interface
    return np.array([outlet_val], dtype=float)
