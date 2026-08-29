"""
Precompute per-site Robertson Ic profiles and top-K stratigraphically similar
partner indices for the Stratigraphy-Aware Contrastive Learning (SCL) stage.

Outputs (written to the same directory as the input .npz):
  ic_profiles.npy  — (N, 500) float32, Ic at each depth for each site (0 at invalid)
  scl_partners.npy — (N, K)   int32,   indices of the K most Ic-similar sites per site

Partner similarity is measured as mean absolute Ic difference over positions where
both sites have valid measurements. The fixed hydrostatic effective stress profile
(gamma_avg=18 kN/m3, water table at surface) matches the formula used in train.py.
"""

import argparse
import os

import numpy as np

from cptfm.dataset import CPTDataset, DEPTH_GRID

_sv0_np     = (18.0 * DEPTH_GRID).astype(np.float32)
_sv0_eff_np = np.maximum(_sv0_np - 9.81 * DEPTH_GRID, 1.0).astype(np.float32)


def _ic_numpy(qc_mpa, fs_kpa, sv0, sv0_eff):
    qt_kpa = qc_mpa * 1000.0
    net    = np.maximum(qt_kpa - sv0, 1.0)
    Qt     = net / sv0_eff
    Fr     = np.clip(fs_kpa / net * 100.0, 0.01, 20.0)
    Ic     = np.sqrt(
        (3.47 - np.log10(np.maximum(Qt, 0.01))) ** 2
        + (np.log10(np.maximum(Fr, 0.01)) + 1.22) ** 2
    )
    return np.clip(Ic, 0.0, 5.0).astype(np.float32)


def main(args):
    ds = CPTDataset(args.npz)
    N  = len(ds.profiles)
    print(f"loaded {N} sites from {args.npz}")

    qc    = ds.profiles[:, :, 0]   # (N, 500) MPa
    fs    = ds.profiles[:, :, 1]   # (N, 500) kPa
    valid = ds.masks                # (N, 500) bool

    sv0     = _sv0_np[None, :]
    sv0_eff = _sv0_eff_np[None, :]
    Ic_all  = _ic_numpy(qc, fs, sv0, sv0_eff)      # (N, 500)
    Ic_save = np.where(valid, Ic_all, 0.0)          # zero at invalid positions

    print(f"Ic computed: mean {Ic_all[valid].mean():.3f}  "
          f"std {Ic_all[valid].std():.3f}")

    K       = args.K
    chunk   = args.chunk
    partners = np.full((N, K), -1, dtype=np.int32)

    Ic_clean = Ic_save  # zeros at invalid — safe for arithmetic

    print(f"finding top-{K} partners per site (chunk={chunk}) ...", flush=True)
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        C   = end - start

        ic_c = Ic_clean[start:end]   # (C, 500)
        v_c  = valid[start:end]      # (C, 500)

        diffs     = np.abs(ic_c[:, None, :] - Ic_clean[None, :, :])  # (C, N, 500)
        joint     = v_c[:, None, :] & valid[None, :, :]               # (C, N, 500)

        weighted  = (diffs * joint).sum(axis=2)                        # (C, N)
        counts    = joint.sum(axis=2).astype(np.float32)               # (C, N)
        mean_diff = weighted / (counts + 1e-6)                         # (C, N)

        # exclude pairs with insufficient joint coverage and self-comparison
        mean_diff[counts < args.min_joint] = np.inf
        for c in range(C):
            mean_diff[c, start + c] = np.inf

        # select top-K most similar (smallest mean absolute Ic difference)
        for c in range(C):
            top_k = np.argpartition(mean_diff[c], K)[:K]
            top_k = top_k[np.argsort(mean_diff[c, top_k])]
            partners[start + c] = top_k

        if start == 0 or (start // chunk) % 10 == 0:
            print(f"  {end}/{N}", flush=True)

    out_dir  = os.path.dirname(os.path.abspath(args.npz))
    ic_path  = os.path.join(out_dir, "ic_profiles.npy")
    scl_path = os.path.join(out_dir, "scl_partners.npy")
    np.save(ic_path,  Ic_save.astype(np.float32))
    np.save(scl_path, partners)

    mean_d = []
    for i in range(min(N, 500)):
        p0    = partners[i, 0]
        jv    = valid[i] & valid[p0]
        if jv.any():
            mean_d.append(np.abs(Ic_save[i, jv] - Ic_save[p0, jv]).mean())
    print(f"nearest-partner mean |ΔIc|: {np.mean(mean_d):.4f} (over {len(mean_d)} sites)")
    print(f"saved {ic_path}")
    print(f"saved {scl_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npz",   default="data/combined/combined.npz")
    p.add_argument("--K",         type=int, default=10,  help="partners per site")
    p.add_argument("--chunk",     type=int, default=32,  help="sites processed per chunk")
    p.add_argument("--min_joint", type=int, default=10,  help="minimum joint valid depth steps")
    main(p.parse_args())
