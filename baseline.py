"""
Gradient-boosted tree baseline for held-out CPT prediction.

Inputs to the model: (lat, lon, depth_m) per observation row.
Targets: qc (MPa) and fs (kPa), trained independently.

The train/val/test split uses the same seed as train.py so the 181
held-out test sites are identical. Normalization is fit on training
indices only. RMSE is reported in physical units at the same visible-depth
thresholds used by the MAE eval, where vis=0m means the model receives
only the site coordinates and must predict the full profile.
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from torch.utils.data import random_split
import torch

from cptfm.dataset import CPTDataset, DEPTH_GRID, DEPTH_STEP, PATCH_SIZE

VISIBLE_DEPTHS = [0, 5, 10, 15, 20]


def build_tables(ds, indices, raw_csv):
    """
    Build (X, y_qc, y_fs) tables for the given site indices.
    X has columns [lat, lon, depth_m].
    Returns arrays of shape (N_obs,).
    """
    raw = pd.read_csv(raw_csv)
    site_lat = raw.groupby("site")["lat"].first()
    site_lon = raw.groupby("site")["lon"].first()

    X, y_qc, y_fs = [], [], []
    for i in indices:
        site = ds.site_ids[i]
        lat  = float(site_lat[site])
        lon  = float(site_lon[site])
        pv   = ds.masks[i]           # (500,) bool — valid depth steps
        prof = ds.profiles[i]        # (500, 2) raw physical units
        for d_idx in range(len(DEPTH_GRID)):
            if not pv[d_idx]:
                continue
            X.append([lat, lon, float(DEPTH_GRID[d_idx])])
            y_qc.append(prof[d_idx, 0])
            y_fs.append(prof[d_idx, 1])

    return np.array(X, dtype=np.float32), np.array(y_qc, dtype=np.float32), np.array(y_fs, dtype=np.float32)


def eval_rmse(model_qc, model_fs, ds, test_indices, raw_csv, vis_m):
    """
    RMSE in physical units on test sites at a given visible depth.
    Only depth steps below vis_m are included in the error (same as MAE eval).
    """
    raw      = pd.read_csv(raw_csv)
    site_lat = raw.groupby("site")["lat"].first()
    site_lon = raw.groupby("site")["lon"].first()
    depth_step = PATCH_SIZE * DEPTH_STEP
    vis_idx    = int(vis_m / depth_step)

    qc_sq, fs_sq = [], []
    for i in test_indices:
        site = ds.site_ids[i]
        lat  = float(site_lat[site])
        lon  = float(site_lon[site])
        pv   = ds.masks[i]
        prof = ds.profiles[i]

        eval_mask = np.zeros(len(DEPTH_GRID), bool)
        eval_mask[vis_idx:] = True
        eval_mask = eval_mask & pv

        if eval_mask.sum() == 0:
            continue

        depths = DEPTH_GRID[eval_mask].reshape(-1, 1)
        X_site = np.column_stack([
            np.full(len(depths), lat),
            np.full(len(depths), lon),
            depths,
        ]).astype(np.float32)

        pred_qc = model_qc.predict(X_site)
        pred_fs = model_fs.predict(X_site)
        act_qc  = prof[eval_mask, 0]
        act_fs  = prof[eval_mask, 1]

        qc_sq.append((pred_qc - act_qc) ** 2)
        fs_sq.append((pred_fs - act_fs) ** 2)

    qc_rmse = float(np.sqrt(np.concatenate(qc_sq).mean())) if qc_sq else float("nan")
    fs_rmse = float(np.sqrt(np.concatenate(fs_sq).mean())) if fs_sq else float("nan")
    return qc_rmse, fs_rmse


def main(args):
    raw_csv = args.raw_csv if args.raw_csv else args.cpt_csv
    if args.cpt_csv.endswith(".npz") and not args.raw_csv:
        raise ValueError(
            "--raw_csv is required when --cpt_csv is a .npz file "
            "(baseline needs lat/lon columns from the original CSV)"
        )

    ds = CPTDataset(args.cpt_csv)

    n_test  = max(1, int(len(ds) * args.test_frac))
    n_val   = max(1, int(len(ds) * args.val_frac))
    n_train = len(ds) - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )
    ds.fit_normalization(list(train_ds.indices))
    print(f"train: {n_train}  val: {n_val}  test (held-out): {n_test}")

    print("building training table...")
    X_tr, yqc_tr, yfs_tr = build_tables(ds, list(train_ds.indices), raw_csv)
    print(f"  {len(X_tr):,} training observations  (lat, lon, depth) → (qc, fs)")

    print("fitting qc model...")
    m_qc = GradientBoostingRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        verbose=0,
    )
    m_qc.fit(X_tr, yqc_tr)

    print("fitting fs model...")
    m_fs = GradientBoostingRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        verbose=0,
    )
    m_fs.fit(X_tr, yfs_tr)

    print("\ntest-set RMSE (gradient boosting, lat/lon/depth inputs only):")
    n_t = len(test_ds)
    for vis_m in VISIBLE_DEPTHS:
        qc_r, fs_r = eval_rmse(m_qc, m_fs, ds, list(test_ds.indices), raw_csv, vis_m)
        print(f"  test (n={n_t})  vis {vis_m:2d}m  qc {qc_r:.3f} MPa  fs {fs_r:.3f} kPa")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cpt_csv",  default="data/bro_combined.npz")
    p.add_argument("--raw_csv",  default=None,
                   help="raw CPT CSV with lat/lon columns (required — .npz has no coordinates)")
    p.add_argument("--val_frac",      type=float, default=0.10)
    p.add_argument("--test_frac",     type=float, default=0.10)
    p.add_argument("--n_estimators",  type=int,   default=300)
    p.add_argument("--max_depth",     type=int,   default=5)
    main(p.parse_args())
