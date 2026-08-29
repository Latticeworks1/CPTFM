import os

import numpy as np
import pandas as pd
from . import CPTRecord

_FS_SENTINEL = -32768.0
_INDEX_SENTINEL = -999.0   # USGS_CPT_Database.xml: "not recorded" for Elevation / WT_Depth

# Physical plausibility bounds — same as BRO reader
_QC_MIN = 0.01
_QC_MAX = 120.0     # MPa; cone tip break strength in practice
_FS_MIN = 0.0
_FS_MAX = 1500.0    # kPa; beyond this is instrument saturation
_RF_MAX = 0.20      # friction ratio fs_kPa / (qc_MPa * 1000) < 20 %

_INDEX_FILENAME = "usgs_cpt_index.csv"


def _load_index(csv_path: str) -> dict:
    """Read the sibling site-level index CSV (ID, Elevation, WT_Depth, ...), if present.

    Returns a dict keyed by site ID; empty dict (with a warning) if the index
    file isn't found alongside csv_path, so callers without it still work.
    """
    index_path = os.path.join(os.path.dirname(csv_path), _INDEX_FILENAME)
    if not os.path.exists(index_path):
        print(f"  no {_INDEX_FILENAME} next to {csv_path}; "
              f"records will have no site-level metadata", flush=True)
        return {}

    idx = pd.read_csv(index_path)
    idx.columns = [c.strip().lstrip("﻿") for c in idx.columns]
    out = {}
    for _, row in idx.iterrows():
        elevation = float(row["Elevation"])
        wt_depth  = float(row["WT_Depth"])
        out[str(row["ID"])] = {
            "elevation_m":         None if elevation == _INDEX_SENTINEL else elevation,
            "water_table_depth_m": None if wt_depth == _INDEX_SENTINEL else wt_depth,
            "water_table_notes":   None if pd.isna(row.get("WT_Notes")) else str(row["WT_Notes"]),
            "date":                None if pd.isna(row.get("Date")) else str(row["Date"]),
            "county":              None if pd.isna(row.get("County")) else str(row["County"]),
            "state":               None if pd.isna(row.get("State")) else str(row["State"]),
            "operator":            None if pd.isna(row.get("Operator")) else str(row["Operator"]),
            "cone":                None if pd.isna(row.get("Cone")) else str(row["Cone"]),
        }
    return out


def load(csv_path: str) -> list[CPTRecord]:
    df = pd.read_csv(csv_path)
    index = _load_index(csv_path)
    records = []
    for site_id, grp in df.groupby("site"):
        grp = grp.sort_values("depth")
        d   = grp["depth"].values.astype(np.float32)
        qc  = grp["tip"].values.astype(np.float32)
        fs  = grp["sleeve"].values.astype(np.float32)
        lon = float(grp["lon"].iloc[0])
        lat = float(grp["lat"].iloc[0])

        keep = (
            (fs != _FS_SENTINEL)
            & np.isfinite(fs) & np.isfinite(qc)
            & (d > 0)
            & (_QC_MIN <= qc) & (qc <= _QC_MAX)
            & (_FS_MIN <= fs) & (fs <= _FS_MAX)
            & ~((qc > 0) & (fs / (qc * 1000.0) > _RF_MAX))
        )
        if keep.sum() < 2:
            continue
        records.append(CPTRecord(
            site=str(site_id), lon=lon, lat=lat,
            depth=d[keep], qc=qc[keep], fs=fs[keep],
            **index.get(str(site_id), {}),
        ))
    return records
