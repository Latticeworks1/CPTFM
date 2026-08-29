"""
Export the USGS CPT corpus as a Hugging Face datasets-style directory:
one row per site (full metadata + depth/qc/fs arrays), split into
train/validation/test parquet files under data/, with a dataset card
README.md at the root.

Usage:
    python3 build_hf_dataset.py --cpt_csv data/usgs/usgs_cpt_3d.csv \
        --out_dir data/usgs_hf
"""

import argparse
import os

import numpy as np
import pandas as pd

from cptfm.sources import usgs as usgs_reader

_SEED = 42
_VAL_FRAC  = 0.05
_TEST_FRAC = 0.05


def _records_to_frame(records) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append({
            "site":                 r.site,
            "lon":                  r.lon,
            "lat":                  r.lat,
            "elevation_m":          r.elevation_m,
            "water_table_depth_m":  r.water_table_depth_m,
            "water_table_notes":    r.water_table_notes,
            "date":                 r.date,
            "county":               r.county,
            "state":                r.state,
            "operator":             r.operator,
            "cone":                 r.cone,
            "n_points":             len(r.depth),
            "depth_m":              r.depth.tolist(),
            "qc_mpa":               r.qc.tolist(),
            "fs_kpa":               r.fs.tolist(),
        })
    return pd.DataFrame(rows)


def _split(df: pd.DataFrame):
    rng = np.random.default_rng(_SEED)
    idx = rng.permutation(len(df))
    n_val  = int(round(len(df) * _VAL_FRAC))
    n_test = int(round(len(df) * _TEST_FRAC))
    val_idx   = idx[:n_val]
    test_idx  = idx[n_val:n_val + n_test]
    train_idx = idx[n_val + n_test:]
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cpt_csv", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    records = usgs_reader.load(args.cpt_csv)
    print(f"loaded {len(records)} sites", flush=True)

    df = _records_to_frame(records)
    train_df, val_df, test_df = _split(df)

    data_dir = os.path.join(args.out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    train_df.to_parquet(os.path.join(data_dir, "train-00000-of-00001.parquet"), index=False)
    val_df.to_parquet(os.path.join(data_dir, "validation-00000-of-00001.parquet"), index=False)
    test_df.to_parquet(os.path.join(data_dir, "test-00000-of-00001.parquet"), index=False)

    print(f"train={len(train_df)}  validation={len(val_df)}  test={len(test_df)}", flush=True)
    print(f"wrote parquet splits to {data_dir}", flush=True)


if __name__ == "__main__":
    main()
