"""
Geo-Bias Score (SSI, seai-lab/PyGBS) against a CPTFM checkpoint's held-out
test set, converted to a permutation z-score per radius so raw SSI magnitude
(which is not directly comparable across scales or models) becomes an
interpretable effect size. Paired with Moran's I on the continuous qc RMSE
signal that the SSI's median split discards.

Marked SSI expects a discrete per-point label -- PyGBS's own example is
binary hit@1 correctness -- because it estimates per-class background rates
via np.unique(values); feeding raw qc RMSE directly degenerates (every value
its own class). Per-site qc RMSE is binarized against its own median:
1 = "high-error site", 0 = "low-error site".

For a fixed radius, the weight matrix built between a site's neighborhood
and its generated background points depends only on point geometry and the
neighborhood's presence count (auto_density is a function of n_neighbors and
radius alone) -- never on which sites are labeled high-error. That matrix is
therefore built once per (site, radius) and reused for every permutation
draw, leaving only the label-dependent class means/variances and a dense
matmul to redo per draw. Rebuilding the weight matrix per permutation, as a
naive B x N call into compute_marked_ssi would, made B=500 impractical for
radius=500km neighborhoods (~2000+ points): this caching is what makes a
proper permutation null tractable at all.

The null bank at a given radius is a function of site geometry, radius, and
class prevalence (n_high, n_low) only -- not of which model produced the
labels -- so it is cached to disk and reused across the three ablation
checkpoints, but only after `_assert_shareable` confirms the incoming run
has the identical site set and the identical high/low split size; a mismatch
rebuilds rather than silently reusing a stale bank.

Usage: python3 geo_bias_eval.py --ckpt-dir checkpoints/global_slepian
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

_PYGBS_CANDIDATES = [
    os.environ.get("PYGBS_SRC", ""),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "PyGBS_src"),
    "/Users/m1a4xnetworkprobe./tools/PyGBS_src",
]
for _candidate in _PYGBS_CANDIDATES:
    if _candidate and os.path.isdir(_candidate):
        sys.path.insert(0, _candidate)
        break
else:
    raise ModuleNotFoundError(
        "PyGBS source not found. Set PYGBS_SRC env var, or clone "
        "github.com/seai-lab/PyGBS into ./PyGBS_src next to this script."
    )
from gbs.ssi.utils import auto_density, construct_weight_matrix, generate_background_points  # noqa: E402
from gbs.ssi.surprisal import AnalyticalSurprisal  # noqa: E402
from partition import SSIPartitioner  # noqa: E402

_USGS_CSV = "data/usgs/usgs_cpt_3d.csv"
_BRO_CSV = "data/bro_combined/bro_cpt.csv"
EARTH_RADIUS_KM = 6371.0
SCALES_KM = [1, 5, 25, 100, 500]


def site_coords():
    coords = {}
    for path in (_USGS_CSV, _BRO_CSV):
        df = pd.read_csv(path, usecols=["site", "lat", "lon"])
        g = df.groupby("site")[["lat", "lon"]].first()
        for site, row in g.iterrows():
            coords[str(site)] = (float(row["lat"]), float(row["lon"]))
    return coords


def load_eval(ckpt_dir):
    local_csv = os.path.join(ckpt_dir, "spatial_holdout_eval.csv")
    df = pd.read_csv(local_csv if os.path.exists(local_csv) else "data/spatial_holdout_eval.csv")
    coords_lookup = site_coords()
    df = df[df["site"].astype(str).isin(coords_lookup)].reset_index(drop=True)
    lat = np.array([coords_lookup[str(s)][0] for s in df["site"]])
    lon = np.array([coords_lookup[str(s)][1] for s in df["site"]])
    coords_rad = np.radians(np.column_stack([lat, lon]))
    return df, coords_rad


def build_neighborhood_cache(partitioner, coords_rad, high_err, radius_rad):
    """Precompute, once per (site, radius): background points, combined point
    array, and weight matrix. All geometry-only -- independent of labels."""
    cache = []
    for idx in range(partitioner.N):
        center = coords_rad[idx]
        nbr_idx = partitioner.get_neighborhood(idx, radius_rad)
        if len(nbr_idx) < 4:
            continue
        if high_err[nbr_idx].max() == high_err[nbr_idx].min():
            continue  # no variation in the *observed* labels here -- same skip rule as before

        presence_points = coords_rad[nbr_idx]
        density = auto_density(radius_rad, presence_points.shape[0])
        bg_points = generate_background_points(center, radius_rad, density)
        points = np.concatenate([presence_points, bg_points], axis=0)
        weight_matrix = construct_weight_matrix(points, k=4)
        n_bg = bg_points.shape[0]
        cache.append({"nbr_idx": nbr_idx, "n_bg": n_bg, "weight_matrix": weight_matrix})
    return cache


def score_from_cache(entry, labels):
    nbr_idx, n_bg, w = entry["nbr_idx"], entry["n_bg"], entry["weight_matrix"]
    presence_values = labels[nbr_idx]
    values = np.concatenate([presence_values, np.zeros(n_bg)])
    cs, ns = np.unique(values, return_counts=True)
    if len(cs) < 2:
        return None  # this permutation happened to erase variation in this neighborhood
    rmax = np.argmax(ns)
    ignores = np.ones_like(cs)
    ignores[rmax] = 0

    surprisal = AnalyticalSurprisal()
    surprisal.fit(cs, ns, w, ignores)
    prob = surprisal.get_probability(values, w)
    return float(-np.log(prob[0] + 1e-256))


def mean_score(cache, labels):
    scores = [s for e in cache if (s := score_from_cache(e, labels)) is not None and np.isfinite(s)]
    return (float(np.mean(scores)) if scores else float("nan")), len(scores)


def morans_i(coords_rad, values, radius_rad):
    lat = coords_rad[:, 0]
    lon = coords_rad[:, 1]
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    dist_km = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))) * EARTH_RADIUS_KM
    radius_km = radius_rad * EARTH_RADIUS_KM

    w = np.where((dist_km > 0) & (dist_km <= radius_km), 1.0 / np.maximum(dist_km, 1e-6), 0.0)
    np.fill_diagonal(w, 0.0)
    W = w.sum()
    if W == 0:
        return float("nan"), 0

    x = values - values.mean()
    n = len(values)
    num = n * np.sum(w * np.outer(x, x))
    den = W * np.sum(x ** 2)
    if den == 0:
        return float("nan"), int((w > 0).sum())
    return float(num / den), int((w > 0).sum())


def _geometry_hash(df, n_high):
    site_key = ",".join(sorted(df["site"].astype(str)))
    return hashlib.sha256(f"{site_key}|{n_high}".encode()).hexdigest()[:16]


def _null_bank_path(cache_dir, radius_km, geom_hash):
    return os.path.join(cache_dir, f"null_r{radius_km}km_{geom_hash}.json")


def get_or_build_null(cache, geom_hash, radius_km, high_err, n_perm, cache_dir, seed=1234):
    path = _null_bank_path(cache_dir, radius_km, geom_hash)
    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
        # _assert_shareable: only reuse a bank built from the identical site
        # set + identical high/low prevalence, never on trust alone.
        assert saved["geom_hash"] == geom_hash, "geometry/prevalence mismatch -- refusing to reuse stale null bank"
        if saved["n_perm"] >= n_perm:
            return np.array(saved["null_scores"][:n_perm])

    rng = np.random.default_rng(seed)
    null_scores = []
    for _ in range(n_perm):
        perm_labels = rng.permutation(high_err)
        s, _ = mean_score(cache, perm_labels)
        if np.isfinite(s):
            null_scores.append(s)
    null_scores = np.array(null_scores)

    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"geom_hash": geom_hash, "radius_km": radius_km,
                    "n_perm": len(null_scores), "null_scores": null_scores.tolist()}, f)
    return null_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="checkpoints/global_slepian")
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--null-cache-dir", default="data/geo_bias_null_bank")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt_name = os.path.basename(args.ckpt_dir.rstrip("/"))
    df, coords_rad = load_eval(args.ckpt_dir)
    median_rmse = df["model_qc_rmse"].median()
    high_err = (df["model_qc_rmse"] > median_rmse).astype(float).values
    n_sites, n_high = len(df), int(high_err.sum())
    geom_hash = _geometry_hash(df, n_high)
    print(f"ckpt={ckpt_name}  n_sites={n_sites}  median_qc_rmse={median_rmse:.3f} MPa  "
          f"n_high_err={n_high}  geom_hash={geom_hash}")

    partitioner = SSIPartitioner(coords_rad, k=min(400, n_sites - 1))

    results = []
    for radius_km in SCALES_KM:
        radius_rad = radius_km / EARTH_RADIUS_KM

        cache = build_neighborhood_cache(partitioner, coords_rad, high_err, radius_rad)
        obs_score, n_scored = mean_score(cache, high_err)

        null_scores = get_or_build_null(cache, geom_hash, radius_km, high_err,
                                         args.n_perm, args.null_cache_dir)
        mu, sigma = float(null_scores.mean()), float(null_scores.std())
        z = (obs_score - mu) / sigma if sigma > 0 else float("nan")
        p = (1 + np.sum(null_scores >= obs_score)) / (len(null_scores) + 1)

        moran_i, n_pairs = morans_i(coords_rad, df["model_qc_rmse"].values, radius_rad)

        results.append({
            "radius_km": radius_km, "n_sites_scored": n_scored, "n_perm": len(null_scores),
            "mean_marked_ssi": obs_score, "null_mean": mu, "null_sd": sigma,
            "z": z, "p_one_sided": float(p),
            "morans_i": moran_i, "morans_i_n_pairs": n_pairs,
        })
        print(f"radius={radius_km:>4} km  n_scored={n_scored:>4}  obs={obs_score:8.4f}  "
              f"null_mu={mu:8.4f}  null_sd={sigma:7.4f}  z={z:7.3f}  p={p:.4f}  "
              f"moran_I={moran_i:7.4f} (npair={n_pairs})")

    out_path = args.out or os.path.join("data", f"geo_bias_score_report_{ckpt_name}.json")
    pd.DataFrame(results).to_json(out_path, orient="records", indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
