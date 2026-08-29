"""
Precompute Slepian + global-SH geographic features for each CPT site and
save them as slepian_features.npy alongside the .npz dataset file.

The output array has shape (N, D) where D = L_global^2 + n_modes_kept.
CPTDataset automatically detects this file and concatenates it with the
SatCLIP embeddings, so train.py needs no other changes beyond reading the
new embedding dimension from the dataset.

Usage (BRO dataset, Netherlands):
    python precompute_slepian.py \
        --npz  data/bro_combined.npz \
        --csv  data/bro/bro_cpt.csv \
        --source bro

Usage (USGS dataset):
    python precompute_slepian.py \
        --npz  data/usgs.npz \
        --csv  data/usgs/usgs_cpt_3d.csv \
        --source usgs

The cap parameters default to the Netherlands. Override with --cap_lon,
--cap_lat, --cap_radius, --L_slepian for other regions.
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd

try:
    import pyshtools as pysh
except ImportError:
    sys.exit("pyshtools is required: pip install pyshtools")

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Spherical harmonics  (pure-numpy, no pyshtools required for this part)
# ---------------------------------------------------------------------------

def _ylm_numpy(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Real-valued spherical harmonic Y_l^m via scipy."""
    from scipy.special import sph_harm
    # scipy sph_harm(m, l, phi, theta) uses physics convention (theta=colatitude)
    if m == 0:
        return np.real(sph_harm(0, l, phi, theta)).astype(np.float64)
    elif m > 0:
        return (math.sqrt(2) * np.real(sph_harm(m, l, phi, theta))).astype(np.float64)
    else:
        return (math.sqrt(2) * np.imag(sph_harm(-m, l, phi, theta))).astype(np.float64)


def _global_sh(lon_deg: np.ndarray, lat_deg: np.ndarray, L: int) -> np.ndarray:
    if L == 0:
        return np.zeros((len(lon_deg), 0), dtype=np.float32)
    phi   = lon_deg * math.pi / 180.0 + math.pi    # [0, 2π]
    theta = math.pi / 2.0 - lat_deg * math.pi / 180.0   # colatitude [0, π]
    theta = np.clip(theta, 1e-12, math.pi - 1e-12)
    cols  = []
    for ell in range(L):
        for m in range(-ell, ell + 1):
            cols.append(_ylm_numpy(ell, m, theta, phi))
    return np.column_stack(cols).astype(np.float32)


# ---------------------------------------------------------------------------
# Slepian features via pyshtools
# ---------------------------------------------------------------------------

def _slepian_features(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    cap_center: tuple,
    cap_radius: float,
    L_slepian: int,
    lambda_thresh: float,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (features [N, n_kept], eigenvalues [n_kept])."""
    max_modes = (L_slepian + 1) ** 2
    theta_rad = cap_radius * math.pi / 180.0
    shannon   = int((L_slepian + 1) ** 2 * (1 - math.cos(theta_rad)) / 2)
    nmax      = min(max(shannon * 2, 10), max_modes)

    t0 = time.time()
    slepian = pysh.Slepian.from_cap(theta=cap_radius, lmax=L_slepian, nmax=nmax)
    slepian.rotate(clat=cap_center[1], clon=cap_center[0], nrot=nmax)
    eigenvalues = slepian.eigenvalues[:nmax].astype(np.float32)
    keep        = eigenvalues > lambda_thresh
    n_kept      = keep.sum()
    if verbose:
        print(f"  Slepian: L={L_slepian}, cap ({cap_center[0]:.1f}°, {cap_center[1]:.1f}°) "
              f"r={cap_radius}°, nmax={nmax}, kept={n_kept}/{nmax} (λ>{lambda_thresh})")
        print(f"  Basis construction: {time.time()-t0:.1f}s")

    lon_360 = np.where(lon_deg < 0.0, lon_deg + 360.0, lon_deg)
    t1      = time.time()
    cols    = []
    for i, mode_idx in enumerate(np.where(keep)[0]):
        coeffs = slepian.to_shcoeffs(int(mode_idx))
        vals   = coeffs.expand(lon=lon_360, lat=lat_deg, degrees=True)
        cols.append(vals.astype(np.float32))
        if verbose and (i + 1) % 50 == 0:
            print(f"    evaluated {i+1}/{n_kept} modes", flush=True)

    features = np.column_stack(cols) if cols else np.zeros((len(lon_deg), 0), np.float32)
    if verbose:
        print(f"  Evaluation: {time.time()-t1:.1f}s")

    return features, eigenvalues[keep]


# ---------------------------------------------------------------------------
# Coordinate loading
# ---------------------------------------------------------------------------

def _load_coords_bro(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    first = df.groupby("site", sort=False).first().reset_index()
    return first[["site", "lon", "lat"]]


def _load_coords_usgs(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # USGS CSV has a site/id column and Longitude/Latitude or lon/lat
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("longitude", "lon", "x"):
            col_map["lon"] = c
        elif cl in ("latitude", "lat", "y"):
            col_map["lat"] = c
        elif cl in ("site", "id", "site_id", "siteid"):
            col_map["site"] = c
    missing = [k for k in ("site", "lon", "lat") if k not in col_map]
    if missing:
        raise ValueError(f"USGS CSV missing columns for: {missing}. "
                         f"Available: {list(df.columns)}")
    out = df[[col_map["site"], col_map["lon"], col_map["lat"]]].copy()
    out.columns = ["site", "lon", "lat"]
    return out.groupby("site", sort=False).first().reset_index()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_coords(csv_path, source):
    if source == "bro":
        return _load_coords_bro(csv_path)
    return _load_coords_usgs(csv_path)


def main(args):
    npz = np.load(args.npz, allow_pickle=True)
    site_ids = list(npz["site_ids"])
    N = len(site_ids)
    print(f"Dataset: {N} sites from {args.npz}")

    # Build coordinate lookup — primary CSV required, secondary optional
    coord_df = _load_coords(args.csv, args.source)
    id_to_coord = {str(row.site): (float(row.lon), float(row.lat))
                   for _, row in coord_df.iterrows()}

    if args.csv2 and args.source2:
        coord_df2 = _load_coords(args.csv2, args.source2)
        for _, row in coord_df2.iterrows():
            key = str(row.site)
            if key not in id_to_coord:
                id_to_coord[key] = (float(row.lon), float(row.lat))

    lons, lats, missing_ids = [], [], []
    for sid in site_ids:
        key = str(sid)
        if key in id_to_coord:
            lon, lat = id_to_coord[key]
            lons.append(lon)
            lats.append(lat)
        else:
            lons.append(float("nan"))
            lats.append(float("nan"))
            missing_ids.append(key)

    if missing_ids:
        print(f"WARNING: {len(missing_ids)} sites have no coordinate — "
              f"their features will be zero. First few: {missing_ids[:5]}")

    lons = np.array(lons, dtype=np.float64)
    lats = np.array(lats, dtype=np.float64)
    valid_coord = np.isfinite(lons) & np.isfinite(lats)
    valid_lons  = lons[valid_coord]
    valid_lats  = lats[valid_coord]

    # --- global spherical harmonics (valid everywhere) ---
    print(f"\nComputing global SH features (L={args.L_global}) ...")
    t0 = time.time()
    global_feats = _global_sh(lons, lats, args.L_global)
    global_feats[~valid_coord] = 0.0
    print(f"  done: {global_feats.shape[1]} dims, {time.time()-t0:.1f}s")

    # --- primary cap (cap 1) ---
    print(f"\nComputing Slepian cap 1 ({args.cap_lon:.2f}°E, {args.cap_lat:.2f}°N, r={args.cap_radius}°) ...")
    slep1_valid, eigs1 = _slepian_features(
        valid_lons, valid_lats,
        cap_center=(args.cap_lon, args.cap_lat),
        cap_radius=args.cap_radius,
        L_slepian=args.L_slepian,
        lambda_thresh=args.lambda_thresh,
        verbose=True,
    )
    slep1 = np.zeros((N, slep1_valid.shape[1]), dtype=np.float32)
    slep1[valid_coord] = slep1_valid

    parts = [global_feats, slep1]
    eigs_all = [eigs1]
    cap_info = [(args.cap_lon, args.cap_lat, args.cap_radius, args.L_slepian, slep1.shape[1])]

    # --- optional second cap ---
    if args.cap_lon2 is not None:
        print(f"\nComputing Slepian cap 2 ({args.cap_lon2:.2f}°E, {args.cap_lat2:.2f}°N, r={args.cap_radius2}°) ...")
        slep2_valid, eigs2 = _slepian_features(
            valid_lons, valid_lats,
            cap_center=(args.cap_lon2, args.cap_lat2),
            cap_radius=args.cap_radius2,
            L_slepian=args.L_slepian2,
            lambda_thresh=args.lambda_thresh,
            verbose=True,
        )
        slep2 = np.zeros((N, slep2_valid.shape[1]), dtype=np.float32)
        slep2[valid_coord] = slep2_valid
        parts.append(slep2)
        eigs_all.append(eigs2)
        cap_info.append((args.cap_lon2, args.cap_lat2, args.cap_radius2, args.L_slepian2, slep2.shape[1]))

    features = np.hstack(parts).astype(np.float32)
    dims_str  = f"global_sh={global_feats.shape[1]}, cap1={slep1.shape[1]}"
    if args.cap_lon2 is not None:
        dims_str += f", cap2={slep2.shape[1]}"
    print(f"\nFinal feature shape: {features.shape}  ({dims_str})")

    mu  = features[valid_coord].mean(0, keepdims=True)
    std = features[valid_coord].std(0, keepdims=True) + 1e-8
    features = ((features - mu) / std).astype(np.float32)
    features[~valid_coord] = 0.0

    out_dir  = os.path.dirname(os.path.abspath(args.npz))
    out_path = os.path.join(out_dir, "slepian_features.npy")
    np.save(out_path, features)
    print(f"Saved {out_path}  shape={features.shape}")

    meta_path = out_path.replace(".npy", "_meta.npz")
    np.savez(meta_path,
             eigenvalues_cap1=eigs_all[0],
             eigenvalues_cap2=(eigs_all[1] if len(eigs_all) > 1 else np.array([])),
             cap_info=np.array(cap_info),
             L_global=np.array(args.L_global),
             lambda_thresh=np.array(args.lambda_thresh),
             global_dim=np.array(global_feats.shape[1]),
             cap1_dim=np.array(slep1.shape[1]),
             cap2_dim=np.array(slep2.shape[1] if args.cap_lon2 is not None else 0),
             n_missing=np.array(len(missing_ids)))
    print(f"Saved {meta_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npz",    required=True)
    p.add_argument("--csv",    required=True,  help="Primary coordinate CSV")
    p.add_argument("--source", required=True,  choices=["bro", "usgs"])
    p.add_argument("--csv2",   default=None,   help="Secondary coordinate CSV (for combined corpus)")
    p.add_argument("--source2",default=None,   choices=["bro", "usgs"])
    # Cap 1 — Netherlands defaults
    p.add_argument("--cap_lon",      type=float, default=5.29)
    p.add_argument("--cap_lat",      type=float, default=52.13)
    p.add_argument("--cap_radius",   type=float, default=5.0)
    p.add_argument("--L_slepian",    type=int,   default=80)
    # Cap 2 — optional second region
    p.add_argument("--cap_lon2",     type=float, default=None)
    p.add_argument("--cap_lat2",     type=float, default=None)
    p.add_argument("--cap_radius2",  type=float, default=25.0,
                   help="Cap 2 radius in degrees (default 25 — continental US)")
    p.add_argument("--L_slepian2",   type=int,   default=80)
    # Shared
    p.add_argument("--L_global",     type=int,   default=10)
    p.add_argument("--lambda_thresh",type=float, default=0.1)
    main(p.parse_args())
