"""
Builds a persisted split.json for the region holdout experiment: an entire
US state (Arkansas, 199 sites, ~4.7% of the corpus) is withheld from
training and validation entirely, so its evaluation measures spatial
extrapolation to genuinely unseen geography rather than interpolation
between nearby sites, which a random per-site split cannot distinguish.

Val is drawn from the remaining non-Arkansas pool at the same 10% fraction
train.py itself would use, so model selection is not influenced by the
held-out region. Site ids not present in the state lookup (state column
missing or corpus/CSV drift) fall into the training pool by default, since
they cannot be assigned to the held-out region with confidence.

Usage: python3 build_region_holdout_split.py --ckpt_dir checkpoints/region_holdout_arkansas
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from cptfm.dataset import CPTDataset

_USGS_CSV = "data/usgs/usgs_cpt_3d.csv"
_BRO_CSV = "data/bro_combined/bro_cpt.csv"
HOLDOUT_STATE = "Arkansas"
VAL_FRAC = 0.10


def state_lookup():
    lookup = {}
    for path in (_USGS_CSV, _BRO_CSV):
        df = pd.read_csv(path, usecols=["site", "state"])
        for site, state in df.drop_duplicates("site")[["site", "state"]].itertuples(index=False):
            lookup[str(site)] = state
    return lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="checkpoints/region_holdout_arkansas")
    ap.add_argument("--cpt_csv", default="data/global_combined.npz")
    args = ap.parse_args()

    ds = CPTDataset(args.cpt_csv)
    site_ids = [str(s) for s in ds.site_ids]
    lookup = state_lookup()

    test_ids = [s for s in site_ids if lookup.get(s) == HOLDOUT_STATE]
    remaining = [s for s in site_ids if s not in test_ids]

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(remaining))
    n_val = max(1, int(len(remaining) * VAL_FRAC))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    val_ids = [remaining[i] for i in val_idx]
    train_ids = [remaining[i] for i in train_idx]

    assert not (set(test_ids) & set(val_ids) & set(train_ids)), "split overlap"
    assert len(test_ids) > 0, f"no sites found for holdout state {HOLDOUT_STATE!r}"

    os.makedirs(args.ckpt_dir, exist_ok=True)
    split_path = os.path.join(args.ckpt_dir, "split.json")
    with open(split_path, "w") as f:
        json.dump({"train": train_ids, "val": val_ids, "test": test_ids}, f)

    print(f"holdout region: {HOLDOUT_STATE}")
    print(f"train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}  "
          f"(test fraction of corpus: {len(test_ids)/len(site_ids):.1%})")
    print(f"wrote {split_path}")


if __name__ == "__main__":
    main()
