"""
Produces Table 1 (qc/fs reconstruction: R2, MAE, RMSE, MAPE by visible depth)
and Table 2 (Robertson SBT zone classification: per-zone accuracy, overall
accuracy, multiclass MCC, confusion matrix) as described in paper_draft.txt
Section 3.4, against a trained checkpoint's persisted held-out test split.

Usage:
    python3 eval_paper_metrics.py --ckpt_dir checkpoints/global_slepian \
        --cpt_csv data/global_combined.npz
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from cptfm import metrics as M
from cptfm import sbt
from cptfm.dataset import CPTDataset, DEPTH_GRID, DEPTH_STEP, PATCH_SIZE
from cptfm.sources import usgs as usgs_reader
from cptfm.sources import bro as bro_reader
from evaluate import load_pipeline, predict_site, DEVICE, N_PATCHES

VISIBLE_DEPTHS = [0, 5, 10, 15, 20]

_USGS_CSV = "data/usgs/usgs_cpt_3d.csv"
_BRO_CSV  = "data/bro_combined/bro_cpt.csv"


def _water_table_by_site() -> dict:
    """site_id -> water_table_depth_m (float or None), merged across both source corpora."""
    wt = {}
    for reader, csv_path in ((usgs_reader, _USGS_CSV), (bro_reader, _BRO_CSV)):
        if not os.path.exists(csv_path):
            continue
        for rec in reader.load(csv_path):
            wt[rec.site] = rec.water_table_depth_m
    return wt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="checkpoints/global_slepian")
    p.add_argument("--cpt_csv",  default="data/global_combined.npz")
    p.add_argument("--out_dir",  default="data")
    args = p.parse_args()

    with open(os.path.join(args.ckpt_dir, "split.json")) as f:
        split = json.load(f)
    train_sites = split["train"]
    test_sites  = set(split["test"])
    print(f"persisted split: train={len(train_sites)}  val={len(split['val'])}  test={len(test_sites)}")

    tok, mae_model, mc = load_pipeline(args.ckpt_dir)
    train_cfg = mc["args"]
    ds = CPTDataset(args.cpt_csv,
                    slepian_only=train_cfg.get("slepian_only", False),
                    satclip_only=train_cfg.get("satclip_only", False))
    site_to_idx = {str(s): i for i, s in enumerate(ds.site_ids)}

    if "norm" in mc:
        ds.qc_lo = mc["norm"]["qc_lo"]; ds.qc_hi = mc["norm"]["qc_hi"]
        ds.fs_lo = mc["norm"]["fs_lo"]; ds.fs_hi = mc["norm"]["fs_hi"]
    else:
        train_idx = [site_to_idx[s] for s in train_sites if s in site_to_idx]
        ds.fit_normalization(train_idx)
    print(f"MAE epoch {mc['epoch']}  val_loss {mc['val_loss']:.4f}\n")

    water_table = _water_table_by_site()

    test_idx = [site_to_idx[s] for s in test_sites if s in site_to_idx]
    print(f"evaluating {len(test_idx)} / {len(test_sites)} held-out test sites\n")

    table1_rows = []
    zone_cm_total = np.zeros((M.N_ZONES, M.N_ZONES), dtype=np.int64)
    table2_rows = []

    for vis_m in VISIBLE_DEPTHS:
        vis_tok    = int(vis_m / (PATCH_SIZE * DEPTH_STEP))
        mask_start = vis_tok

        qc_true_all, qc_pred_all = [], []
        fs_true_all, fs_pred_all = [], []
        zone_true_all, zone_pred_all = [], []

        for i in test_idx:
            site = ds.site_ids[i]
            item = ds[i]
            patches     = item["patches"].unsqueeze(0).to(DEVICE)
            satclip     = item["satclip"].unsqueeze(0).to(DEVICE)
            patch_valid = item["patch_valid"].unsqueeze(0).to(DEVICE)
            pv_np       = item["patch_valid"].numpy()

            pred_raw = predict_site(tok, mae_model, patches, satclip, patch_valid, vis_tok)
            pred_qc  = ds.denorm_qc(pred_raw[:, 0])
            pred_fs  = ds.denorm_fs(pred_raw[:, 1])

            act_raw = item["patches"].numpy().reshape(-1, 2)
            act_qc  = ds.denorm_qc(act_raw[:, 0])
            act_fs  = ds.denorm_fs(act_raw[:, 1])

            eval_mask = np.zeros(len(DEPTH_GRID), bool)
            eval_mask[mask_start:] = True
            eval_mask &= pv_np
            if eval_mask.sum() == 0:
                continue

            qc_true_all.append(act_qc[eval_mask]);  qc_pred_all.append(pred_qc[eval_mask])
            fs_true_all.append(act_fs[eval_mask]);  fs_pred_all.append(pred_fs[eval_mask])

            wt_m = water_table.get(site) if water_table.get(site) is not None else 0.0
            depth_eval = DEPTH_GRID[eval_mask]
            true_zone = sbt.compute(depth_eval, act_qc[eval_mask],  act_fs[eval_mask],  water_table_m=wt_m)["zone"]
            pred_zone = sbt.compute(depth_eval, pred_qc[eval_mask], pred_fs[eval_mask], water_table_m=wt_m)["zone"]
            zone_true_all.append(true_zone); zone_pred_all.append(pred_zone)

        qc_true = np.concatenate(qc_true_all); qc_pred = np.concatenate(qc_pred_all)
        fs_true = np.concatenate(fs_true_all); fs_pred = np.concatenate(fs_pred_all)
        zone_true = np.concatenate(zone_true_all); zone_pred = np.concatenate(zone_pred_all)

        table1_rows.append({
            "visible_depth_m": vis_m,
            "qc_r2": M.r2_score(qc_true, qc_pred), "qc_mae": M.mae(qc_true, qc_pred),
            "qc_rmse": M.rmse(qc_true, qc_pred),   "qc_mape": M.mape(qc_true, qc_pred),
            "fs_r2": M.r2_score(fs_true, fs_pred), "fs_mae": M.mae(fs_true, fs_pred),
            "fs_rmse": M.rmse(fs_true, fs_pred),
        })

        cm = M.confusion_matrix(zone_true, zone_pred)
        zone_cm_total += cm
        acc_k = M.per_class_accuracy(cm)
        for k in range(M.N_ZONES):
            table2_rows.append({
                "visible_depth_m": vis_m, "zone": k + 1, "zone_name": sbt.zone_name(k + 1),
                "n_true": int(cm[k].sum()), "accuracy": acc_k[k],
            })
        print(f"visible={vis_m:>2} m  overall zone accuracy={M.overall_accuracy(cm):.3f}  "
              f"MCC={M.multiclass_mcc(cm):.3f}  (n={cm.sum()})")

    t1 = pd.DataFrame(table1_rows)
    t2 = pd.DataFrame(table2_rows)

    pd.set_option("display.width", 140)
    print("\nTable 1. Held-out test reconstruction metrics by visible depth.\n")
    print(t1.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nTable 2. Robertson SBT zone classification accuracy by zone and visible depth.\n")
    pivot = t2.pivot(index="zone_name", columns="visible_depth_m", values="accuracy")
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    print(f"\nPooled over all visible depths: overall accuracy={M.overall_accuracy(zone_cm_total):.3f}  "
          f"MCC={M.multiclass_mcc(zone_cm_total):.3f}  macro OvR MCC={M.macro_ovr_mcc(zone_cm_total):.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    t1_path = os.path.join(args.out_dir, "table1_reconstruction_metrics.csv")
    t2_path = os.path.join(args.out_dir, "table2_sbt_zone_accuracy.csv")
    cm_path = os.path.join(args.out_dir, "sbt_zone_confusion_matrix.csv")
    t1.to_csv(t1_path, index=False)
    t2.to_csv(t2_path, index=False)
    pd.DataFrame(
        zone_cm_total,
        index=[f"true_{sbt.zone_name(k+1)}" for k in range(M.N_ZONES)],
        columns=[f"pred_{sbt.zone_name(k+1)}" for k in range(M.N_ZONES)],
    ).to_csv(cm_path)
    print(f"\nwrote {t1_path}\nwrote {t2_path}\nwrote {cm_path}")


if __name__ == "__main__":
    main()
