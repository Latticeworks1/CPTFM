"""
Remove exact-duplicate CPT soundings (bit-identical qc/fs sequences under
different site_ids, confirmed to originate from the same physical test
submitted multiple times in the raw USGS source) from the training corpus.

Backs up every file it touches with a .orig suffix before overwriting, and
keeps global_combined.npz, slepian_features.npy, and scl_partners.npy in
row-aligned sync (scl_partners' partner indices are remapped, and any
partner reference into a removed row is dropped, not left dangling).
"""

import shutil

import numpy as np

DATA_DIR = "data"


def find_duplicate_row_indices(profiles, masks):
    seen = {}
    drop = set()
    for i in range(len(profiles)):
        key = profiles[i][masks[i]].tobytes()
        if key in seen:
            drop.add(i)
        else:
            seen[key] = i
    return drop


def backup(path):
    bak = path + ".orig"
    if not __import__("os").path.exists(bak):
        shutil.copy2(path, bak)
        print(f"backed up {path} -> {bak}")
    else:
        print(f"backup already exists, leaving it: {bak}")


def main():
    npz_path = f"{DATA_DIR}/global_combined.npz"
    slep_path = f"{DATA_DIR}/slepian_features.npy"
    scl_path = f"{DATA_DIR}/scl_partners.npy"

    backup(npz_path)
    backup(slep_path)
    backup(scl_path)

    npz = np.load(npz_path, allow_pickle=True)
    profiles = npz["profiles"]
    masks = npz["masks"].astype(bool)
    site_ids = np.array([str(s) for s in npz["site_ids"]])
    embeddings = npz["embeddings"]
    N = len(site_ids)

    drop_idx = find_duplicate_row_indices(profiles, masks)
    print(f"dropping {len(drop_idx)} exact-duplicate rows out of {N}: "
          f"{sorted(site_ids[i] for i in drop_idx)}")

    keep_mask = np.ones(N, dtype=bool)
    keep_mask[list(drop_idx)] = False
    keep_idx = np.where(keep_mask)[0]

    old_to_new = -np.ones(N, dtype=np.int64)
    old_to_new[keep_idx] = np.arange(len(keep_idx))

    new_profiles = profiles[keep_idx]
    new_masks = npz["masks"][keep_idx]
    new_site_ids = site_ids[keep_idx]
    new_embeddings = embeddings[keep_idx]

    np.savez(
        npz_path,
        profiles=new_profiles,
        masks=new_masks,
        site_ids=new_site_ids,
        embeddings=new_embeddings,
    )
    print(f"wrote {npz_path}: {N} -> {len(keep_idx)} profiles")

    slep = np.load(slep_path)
    assert slep.shape[0] == N, f"slepian_features.npy row count {slep.shape[0]} != {N}"
    new_slep = slep[keep_idx]
    np.save(slep_path, new_slep)
    print(f"wrote {slep_path}: {slep.shape[0]} -> {new_slep.shape[0]} rows")

    scl = np.load(scl_path)
    assert scl.shape[0] == N, f"scl_partners.npy row count {scl.shape[0]} != {N}"
    scl_kept = scl[keep_idx]
    remapped = old_to_new[scl_kept]          # -1 wherever the partner pointed at a dropped row
    dropped_refs = (remapped == -1).sum() - (scl_kept == -1).sum()  # newly-invalidated refs
    np.save(scl_path, remapped.astype(np.int32))
    print(f"wrote {scl_path}: {scl.shape[0]} -> {remapped.shape[0]} rows, "
          f"{dropped_refs} partner references invalidated by the drop")

    print(f"\nfinal corpus size: {len(keep_idx)} (was {N})")


if __name__ == "__main__":
    main()
