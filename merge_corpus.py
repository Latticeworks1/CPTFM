"""
Merge USGS and BRO CPT corpora and write a combined .npz.

Usage:
    python3 merge_corpus.py \
        --usgs_cpt  data/usgs/usgs_cpt.csv \
        --usgs_emb_npy data/usgs/satclip_embeddings.npy \
        --usgs_emb_csv data/usgs/satclip_embeddings.csv \
        --bro_cpt   data/bro/bro_cpt.csv \
        --bro_emb_npy  data/bro/bro_embeddings.npy \
        --bro_emb_csv  data/bro/bro_embeddings.csv \
        --out       data/combined/combined.npz
"""

import argparse
import os
import numpy as np
import pandas as pd

from cptfm.dataset import DEPTH_GRID
from cptfm.sources import usgs as usgs_reader
from cptfm.sources import bro  as bro_reader


def _load_source(reader, cpt_csv, emb_npy, emb_csv):
    records   = reader.load(cpt_csv)
    emb_arr   = np.load(emb_npy).astype(np.float32)
    emb_meta  = pd.read_csv(emb_csv)
    id_to_idx = {row.ID: i for i, row in emb_meta.iterrows()}
    records   = [r for r in records if r.site in id_to_idx]
    return records, emb_arr, id_to_idx


def _build_arrays(records, emb_arr, id_to_idx):
    N = len(records)
    T = len(DEPTH_GRID)
    profiles   = np.zeros((N, T, 2), dtype=np.float32)
    masks      = np.zeros((N, T),    dtype=bool)
    embeddings = np.zeros((N, emb_arr.shape[1]), dtype=np.float32)
    site_ids   = []
    skipped    = 0
    wi         = 0

    for rec in records:
        qc_grid  = np.interp(DEPTH_GRID, rec.depth, rec.qc,
                             left=np.nan, right=np.nan).astype(np.float32)
        fs_grid  = np.interp(DEPTH_GRID, rec.depth, rec.fs,
                             left=np.nan, right=np.nan).astype(np.float32)
        in_range = (DEPTH_GRID >= rec.depth.min()) & (DEPTH_GRID <= rec.depth.max())
        valid    = in_range & np.isfinite(qc_grid) & np.isfinite(fs_grid)

        if valid.sum() == 0:
            skipped += 1
            continue

        profiles[wi, :, 0] = np.where(valid, qc_grid, 0.0)
        profiles[wi, :, 1] = np.where(valid, fs_grid, 0.0)
        masks[wi]           = valid
        embeddings[wi]      = emb_arr[id_to_idx[rec.site]]
        site_ids.append(rec.site)
        wi += 1

    return profiles[:wi], masks[:wi], embeddings[:wi], site_ids, skipped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--usgs_cpt",     required=True)
    p.add_argument("--usgs_emb_npy", required=True)
    p.add_argument("--usgs_emb_csv", required=True)
    p.add_argument("--bro_cpt",      default=None)
    p.add_argument("--bro_emb_npy",  default=None)
    p.add_argument("--bro_emb_csv",  default=None)
    p.add_argument("--out",          default="data/combined/combined.npz")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"loading USGS from {args.usgs_cpt} ...", flush=True)
    usgs_recs, usgs_emb, usgs_idx = _load_source(
        usgs_reader, args.usgs_cpt, args.usgs_emb_npy, args.usgs_emb_csv)
    print(f"  {len(usgs_recs)} USGS records with embeddings", flush=True)
    u_prof, u_mask, u_emb, u_ids, u_skip = _build_arrays(usgs_recs, usgs_emb, usgs_idx)
    if u_skip:
        print(f"  skipped {u_skip} USGS sites with no valid depth steps", flush=True)

    bro_has_data = all(x is not None for x in [args.bro_cpt, args.bro_emb_npy, args.bro_emb_csv])
    if bro_has_data:
        print(f"loading BRO from {args.bro_cpt} ...", flush=True)
        bro_recs, bro_emb, bro_idx = _load_source(
            bro_reader, args.bro_cpt, args.bro_emb_npy, args.bro_emb_csv)
        print(f"  {len(bro_recs)} BRO records with embeddings", flush=True)

        assert usgs_emb.shape[1] == bro_emb.shape[1], (
            f"embedding dim mismatch: USGS {usgs_emb.shape[1]} vs BRO {bro_emb.shape[1]}"
        )

        b_prof, b_mask, b_emb, b_ids, b_skip = _build_arrays(bro_recs, bro_emb, bro_idx)
        if b_skip:
            print(f"  skipped {b_skip} BRO sites with no valid depth steps", flush=True)

        profiles   = np.concatenate([u_prof, b_prof], axis=0)
        masks      = np.concatenate([u_mask, b_mask], axis=0)
        embeddings = np.concatenate([u_emb,  b_emb],  axis=0)
        site_ids   = u_ids + b_ids
    else:
        print("BRO paths not provided — building USGS-only corpus.", flush=True)
        profiles, masks, embeddings, site_ids = u_prof, u_mask, u_emb, u_ids

    np.savez_compressed(
        args.out,
        profiles=profiles, masks=masks,
        embeddings=embeddings,
        site_ids=np.array(site_ids, dtype=object),
    )
    print(f"\nwrote {args.out}  "
          f"profiles={profiles.shape}  sites={len(site_ids)}", flush=True)


if __name__ == "__main__":
    main()
