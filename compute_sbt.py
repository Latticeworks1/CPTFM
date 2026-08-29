"""
Run Robertson (1990/2010) SBT classification over a CPT source, using each
site's measured water table depth from the source index when available and
falling back to the ground-surface default (0 m) documented in cptfm/sbt.py
when it wasn't recorded.

Usage:
    python3 compute_sbt.py --source usgs --cpt_csv data/usgs/usgs_cpt_3d.csv \
        --out data/usgs/usgs_sbt.csv
"""

import argparse

import numpy as np
import pandas as pd

from cptfm import sbt
from cptfm.sources import usgs as usgs_reader
from cptfm.sources import bro as bro_reader

_READERS = {"usgs": usgs_reader, "bro": bro_reader}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source",  required=True, choices=list(_READERS))
    p.add_argument("--cpt_csv", required=True)
    p.add_argument("--out",     required=True)
    args = p.parse_args()

    records = _READERS[args.source].load(args.cpt_csv)
    print(f"loaded {len(records)} records", flush=True)

    n_measured = sum(1 for r in records if r.water_table_depth_m is not None)
    print(f"  {n_measured}/{len(records)} sites have a measured water table depth; "
          f"the rest default to the ground surface (0 m)", flush=True)

    rows = []
    for rec in records:
        wt_m = rec.water_table_depth_m if rec.water_table_depth_m is not None else 0.0
        result = sbt.compute(rec.depth, rec.qc, rec.fs, water_table_m=wt_m)
        for i in range(len(rec.depth)):
            rows.append({
                "site":                 rec.site,
                "depth_m":              rec.depth[i],
                "Ic":                   result["Ic"][i],
                "zone":                 int(result["zone"][i]),
                "zone_name":            sbt.zone_name(result["zone"][i]),
                "sigma_v0_kpa":         result["sigma_v0"][i],
                "sigma_v0_eff_kpa":     result["sigma_v0_eff"][i],
                "water_table_depth_m":  wt_m,
                "water_table_measured": rec.water_table_depth_m is not None,
            })

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(out)} rows, {out.site.nunique()} sites)", flush=True)


if __name__ == "__main__":
    main()
