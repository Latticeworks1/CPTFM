"""
Preprocess a CPT corpus into a single .npz for fast dataset loading.

Usage (USGS):
    python3 preprocess.py \
        --source  usgs \
        --cpt_csv data/usgs/usgs_cpt.csv \
        --emb_npy data/usgs/satclip_embeddings.npy \
        --emb_csv data/usgs/satclip_embeddings.csv \
        --out     data/usgs.npz

Usage (BRO):
    python3 preprocess.py \
        --source  bro \
        --cpt_csv data/bro/bro_cpt.csv \
        --emb_npy data/bro/bro_embeddings.npy \
        --emb_csv data/bro/bro_embeddings.csv \
        --out     data/bro.npz
"""

import argparse
import numpy as np
import pandas as pd

from cptfm.dataset import DEPTH_GRID
from cptfm.sources import usgs as usgs_reader
from cptfm.sources import bro  as bro_reader

_READERS = {"usgs": usgs_reader, "bro": bro_reader}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source",  required=True, choices=list(_READERS))
    p.add_argument("--cpt_csv", required=True)
    p.add_argument("--emb_npy", required=True)
    p.add_argument("--emb_csv", required=True)
    p.add_argument("--out",     required=True)
    args = p.parse_args()

    print(f"loading {args.cpt_csv} via {args.source} reader ...", flush=True)
    records = _READERS[args.source].load(args.cpt_csv)
    print(f"  {len(records)} records loaded", flush=True)

    emb_arr   = np.load(args.emb_npy).astype(np.float32)
    emb_meta  = pd.read_csv(args.emb_csv)
    id_to_idx = {row.ID: i for i, row in emb_meta.iterrows()}
    records   = [r for r in records if r.site in id_to_idx]
    print(f"  {len(records)} records with embeddings", flush=True)

    N = len(records)
    T = len(DEPTH_GRID)
    profiles   = np.zeros((N, T, 2), dtype=np.float32)
    masks      = np.zeros((N, T),    dtype=bool)
    embeddings = np.zeros((N, emb_arr.shape[1]), dtype=np.float32)
    site_ids   = []
    skipped    = 0
    wi         = 0

    print(f"building arrays for {N} sites ...", flush=True)
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

        if wi % 500 == 0:
            print(f"  {wi}/{N}", flush=True)

    if skipped:
        print(f"skipped {skipped} sites with no valid depth steps", flush=True)

    profiles   = profiles[:wi]
    masks      = masks[:wi]
    embeddings = embeddings[:wi]
    np.savez_compressed(
        args.out,
        profiles=profiles, masks=masks,
        embeddings=embeddings,
        site_ids=np.array(site_ids, dtype=object),
    )
    print(f"wrote {args.out}  profiles={profiles.shape}  masks={masks.shape}", flush=True)


if __name__ == "__main__":
    main()
