"""
Bootstrap confidence intervals on the per-site results from spatial_holdout_eval.py,
resampled at the sounding (site) level rather than the depth-row level. Each row of
that script's output CSV is already one aggregate RMSE per site, so resampling rows
with replacement is sounding-level resampling by construction -- depth positions
within a sounding are never treated as independent draws.

Reports, overall and within each distance bucket: a percentile bootstrap CI on the
model's and the nearest-neighbor baseline's mean RMSE, and a paired bootstrap CI on
their difference (nn - model), which is the quantity that actually answers whether
the model's advantage over naive spatial lookup is distinguishable from resampling
noise at that bucket's sample size, rather than just reporting two separate point
estimates and eyeballing the gap.

Usage:
    python3 bootstrap_ci.py --ckpt_dir checkpoints/region_holdout_arkansas --n_boot 5000
"""
import argparse
import os

import numpy as np
import pandas as pd


def bootstrap_ci(values, n_boot, rng, alpha=0.05):
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def bootstrap_diff_ci(a, b, n_boot, rng, alpha=0.05):
    """Paired bootstrap on mean(a) - mean(b), resampling site indices jointly
    so the pairing (same site's model and nn error) is preserved per draw."""
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_no_effect = float(min((diffs <= 0).mean(), (diffs >= 0).mean()) * 2)
    return float(a.mean() - b.mean()), float(lo), float(hi), p_no_effect


def report_metric(df, metric, n_boot, rng):
    print(f"\n--- {metric} ---")
    a = df[f"nn_{metric}"].values
    b = df[f"model_{metric}"].values

    def row(label, sub):
        na = sub[f"nn_{metric}"].values
        nb = sub[f"model_{metric}"].values
        n_mean, n_lo, n_hi = bootstrap_ci(na, n_boot, rng)
        m_mean, m_lo, m_hi = bootstrap_ci(nb, n_boot, rng)
        d_mean, d_lo, d_hi, p = bootstrap_diff_ci(na, nb, n_boot, rng)
        sig = "*" if d_lo > 0 or d_hi < 0 else " "
        print(f"{label:>12}  n={len(sub):3d}  "
              f"nn={n_mean:6.3f} [{n_lo:5.3f},{n_hi:5.3f}]  "
              f"model={m_mean:6.3f} [{m_lo:5.3f},{m_hi:5.3f}]  "
              f"diff(nn-model)={d_mean:6.3f} [{d_lo:5.3f},{d_hi:5.3f}] {sig}  p~{p:.3f}")

    row("overall", df)
    bucket_order = df.groupby("bucket")["nn_dist_km"].median().sort_values().index
    for bucket in bucket_order:
        sub = df[df["bucket"] == bucket]
        if len(sub) > 0:
            row(bucket, sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="checkpoints/region_holdout_arkansas")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    csv_path = os.path.join(args.ckpt_dir, "spatial_holdout_eval.csv")
    df = pd.read_csv(csv_path)
    print(f"loaded {len(df)} sites from {csv_path}")
    print(f"n_boot={args.n_boot}  (* marks a 95% CI on nn-model that excludes 0)")

    rng = np.random.default_rng(args.seed)
    for metric in ["qc_rmse", "fs_rmse"]:
        report_metric(df, metric, args.n_boot, rng)


if __name__ == "__main__":
    main()
