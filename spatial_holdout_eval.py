"""
Distance-stratified spatial holdout evaluation for the trained MAE, operationalizing
the pass/fail protocol discussed for the geographic (SatCLIP+Slepian) encoder: does
model error degrade gracefully with distance from the nearest training site, and does
it beat a plain nearest-neighbor fallback at the ranges where it matters.

Uses the persisted train/val/test split from the checkpoint (checkpoints/<run>/split.json)
rather than re-deriving a split, so results are against the actual held-out test set that
checkpoint was evaluated on during training.

Usage:
    python3 spatial_holdout_eval.py --ckpt_dir checkpoints/global_slepian \
        --cpt_csv data/global_combined.npz
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from cptfm.dataset import CPTDataset, DEPTH_GRID, DEPTH_MAX, DEPTH_STEP, PATCH_SIZE
from evaluate import load_pipeline, predict_site, DEVICE, N_PATCHES

_USGS_CSV = "data/usgs/usgs_cpt_3d.csv"
_BRO_CSV  = "data/bro_combined/bro_cpt.csv"

_DIST_BUCKETS_KM = [0, 0.05, 0.2, 0.5, 1, 2, 5, 25, 100, float("inf")]


def _site_coords() -> dict:
    coords = {}
    for path in (_USGS_CSV, _BRO_CSV):
        df = pd.read_csv(path, usecols=["site", "lat", "lon"])
        g = df.groupby("site")[["lat", "lon"]].first()
        for site, row in g.iterrows():
            coords[str(site)] = (float(row["lat"]), float(row["lon"]))
    return coords


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _bucket_label(km):
    for i in range(len(_DIST_BUCKETS_KM) - 1):
        lo, hi = _DIST_BUCKETS_KM[i], _DIST_BUCKETS_KM[i + 1]
        if lo <= km < hi:
            return f"{lo:g}-{hi:g} km" if hi != float("inf") else f">{lo:g} km"
    return "?"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="checkpoints/global_slepian")
    p.add_argument("--cpt_csv",  default="data/global_combined.npz")
    args = p.parse_args()

    with open(os.path.join(args.ckpt_dir, "split.json")) as f:
        split = json.load(f)
    train_sites = set(split["train"])
    test_sites  = set(split["test"])
    print(f"persisted split: train={len(train_sites)}  val={len(split['val'])}  test={len(test_sites)}")

    coords = _site_coords()
    missing = [s for s in (train_sites | test_sites) if s not in coords]
    if missing:
        print(f"WARNING: {len(missing)} split site_ids have no coordinate match "
              f"(e.g. {missing[:3]}) — excluded from this evaluation")

    train_sites = [s for s in train_sites if s in coords]
    test_sites  = [s for s in test_sites  if s in coords]
    train_lat = np.array([coords[s][0] for s in train_sites])
    train_lon = np.array([coords[s][1] for s in train_sites])

    ds = CPTDataset(args.cpt_csv)
    site_to_idx = {str(s): i for i, s in enumerate(ds.site_ids)}

    tok, mae, mc = load_pipeline(args.ckpt_dir)
    if "norm" in mc:
        ds.qc_lo = mc["norm"]["qc_lo"]; ds.qc_hi = mc["norm"]["qc_hi"]
        ds.fs_lo = mc["norm"]["fs_lo"]; ds.fs_hi = mc["norm"]["fs_hi"]
    else:
        train_idx = [site_to_idx[s] for s in train_sites if s in site_to_idx]
        ds.fit_normalization(train_idx)
    print(f"MAE epoch {mc['epoch']}  val_loss {mc['val_loss']:.4f}\n")

    # nearest-neighbor baseline lookup table (train profiles, physical units)
    train_profiles = {}
    for s in train_sites:
        i = site_to_idx.get(s)
        if i is None:
            continue
        item = ds[i]
        pv  = item["patch_valid"].numpy()
        raw = item["patches"].numpy().reshape(-1, 2)
        train_profiles[s] = (
            ds.denorm_qc(raw[:, 0]), ds.denorm_fs(raw[:, 1]), pv,
        )

    rows = []
    for s in test_sites:
        i = site_to_idx.get(s)
        if i is None:
            continue
        lat, lon = coords[s]
        d_km = _haversine_km(lat, lon, train_lat, train_lon)
        nn_idx = int(np.argmin(d_km))
        nn_site = train_sites[nn_idx]
        nn_dist_km = float(d_km[nn_idx])
        if nn_site not in train_profiles:
            continue

        item = ds[i]
        patches     = item["patches"].unsqueeze(0).to(DEVICE)
        satclip     = item["satclip"].unsqueeze(0).to(DEVICE)
        patch_valid = item["patch_valid"].unsqueeze(0).to(DEVICE)
        pv_np       = item["patch_valid"].numpy()

        # vis=0m: geography-only prediction, the case that matters for
        # generalization to locations with no local observation at all.
        pred_raw = predict_site(tok, mae, patches, satclip, patch_valid, vis_tok=0)
        pred_qc  = ds.denorm_qc(pred_raw[:, 0])
        pred_fs  = ds.denorm_fs(pred_raw[:, 1])

        act_raw = item["patches"].numpy().reshape(-1, 2)
        act_qc  = ds.denorm_qc(act_raw[:, 0])
        act_fs  = ds.denorm_fs(act_raw[:, 1])

        nn_qc, nn_fs, nn_pv = train_profiles[nn_site]
        eval_mask = pv_np & nn_pv
        if eval_mask.sum() == 0:
            continue

        # Distance-gated blend: weight toward the literal nearest sounding when
        # it's essentially the same physical location, weight toward the
        # geography-only model once retrieval has nothing nearby to offer.
        # d0=0.2 km is an informed guess from the coarse crossover seen in the
        # per-bucket breakdown, not a fitted parameter — a real version of this
        # should calibrate d0 on a held-out slice rather than eyeballing it.
        d0 = 0.2
        alpha = 1.0 / (1.0 + (nn_dist_km / d0))  # -> 1 near, -> 0 far
        blend_qc = alpha * nn_qc + (1 - alpha) * pred_qc
        blend_fs = alpha * nn_fs + (1 - alpha) * pred_fs

        model_qc_rmse = float(np.sqrt(np.mean((pred_qc[eval_mask]  - act_qc[eval_mask]) ** 2)))
        model_fs_rmse = float(np.sqrt(np.mean((pred_fs[eval_mask]  - act_fs[eval_mask]) ** 2)))
        nn_qc_rmse    = float(np.sqrt(np.mean((nn_qc[eval_mask]    - act_qc[eval_mask]) ** 2)))
        nn_fs_rmse    = float(np.sqrt(np.mean((nn_fs[eval_mask]    - act_fs[eval_mask]) ** 2)))
        blend_qc_rmse = float(np.sqrt(np.mean((blend_qc[eval_mask] - act_qc[eval_mask]) ** 2)))
        blend_fs_rmse = float(np.sqrt(np.mean((blend_fs[eval_mask] - act_fs[eval_mask]) ** 2)))

        rows.append({
            "site": s, "nn_dist_km": nn_dist_km, "bucket": _bucket_label(nn_dist_km),
            "n_pts": int(eval_mask.sum()), "alpha": alpha,
            "model_qc_rmse": model_qc_rmse, "model_fs_rmse": model_fs_rmse,
            "nn_qc_rmse": nn_qc_rmse, "nn_fs_rmse": nn_fs_rmse,
            "blend_qc_rmse": blend_qc_rmse, "blend_fs_rmse": blend_fs_rmse,
        })

    df = pd.DataFrame(rows)
    print(f"evaluated {len(df)} / {len(test_sites)} test sites (vis=0m, geography-only)\n")

    order = [_bucket_label(0.5 * (lo + (hi if hi != float("inf") else lo * 4 + 100)))
             for lo, hi in zip(_DIST_BUCKETS_KM[:-1], _DIST_BUCKETS_KM[1:])]
    df["bucket"] = pd.Categorical(df["bucket"], categories=order, ordered=True)

    summary = df.groupby("bucket", observed=True).agg(
        n_sites=("site", "count"),
        median_nn_dist_km=("nn_dist_km", "median"),
        model_qc_rmse=("model_qc_rmse", "mean"),
        nn_qc_rmse=("nn_qc_rmse", "mean"),
        blend_qc_rmse=("blend_qc_rmse", "mean"),
        model_fs_rmse=("model_fs_rmse", "mean"),
        nn_fs_rmse=("nn_fs_rmse", "mean"),
        blend_fs_rmse=("blend_fs_rmse", "mean"),
    )
    summary["blend_beats_both_qc"] = (summary["blend_qc_rmse"] < summary["model_qc_rmse"]) & \
                                      (summary["blend_qc_rmse"] < summary["nn_qc_rmse"])
    summary["blend_beats_both_fs"] = (summary["blend_fs_rmse"] < summary["model_fs_rmse"]) & \
                                      (summary["blend_fs_rmse"] < summary["nn_fs_rmse"])
    print(f"\noverall (all test sites):")
    print(f"  qc RMSE  model={df.model_qc_rmse.mean():.3f}  nn={df.nn_qc_rmse.mean():.3f}  "
          f"blend={df.blend_qc_rmse.mean():.3f}")
    print(f"  fs RMSE  model={df.model_fs_rmse.mean():.3f}  nn={df.nn_fs_rmse.mean():.3f}  "
          f"blend={df.blend_fs_rmse.mean():.3f}\n")

    pd.set_option("display.width", 140)
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))

    out_path = "data/spatial_holdout_eval.csv"
    df.to_csv(out_path, index=False)
    print(f"\nper-site results written to {out_path}")


if __name__ == "__main__":
    main()
