import os

import numpy as np
import torch
from torch.utils.data import Dataset

DEPTH_MAX   = 25.0
DEPTH_STEP  = 0.05
DEPTH_GRID  = np.arange(DEPTH_STEP, DEPTH_MAX + DEPTH_STEP, DEPTH_STEP)  # (500,)
PATCH_SIZE  = 1    # one token per depth step — 500 tokens at 0.05 m resolution

assert len(DEPTH_GRID) == 500
assert DEPTH_GRID[0]  == DEPTH_STEP
assert abs(DEPTH_GRID[-1] - DEPTH_MAX) < 1e-9
assert len(DEPTH_GRID) % PATCH_SIZE == 0


class CPTDataset(Dataset):
    def __init__(self, npz_path: str, slepian_only: bool = False, satclip_only: bool = False):
        assert not (slepian_only and satclip_only), "slepian_only and satclip_only are mutually exclusive"
        if not npz_path.endswith(".npz"):
            raise ValueError(
                f"CPTDataset requires a preprocessed .npz file (got: {npz_path}). "
                "Run preprocess.py first."
            )
        data = np.load(npz_path, allow_pickle=True)
        self.profiles   = data["profiles"].astype(np.float32)   # (N, 500, 2)
        self.masks      = data["masks"].astype(bool)            # (N, 500)
        self.embeddings = data["embeddings"].astype(np.float32) # (N, D)
        self.site_ids   = list(data["site_ids"])

        self.qc_lo = self.qc_hi = self.fs_lo = self.fs_hi = None

        # Optional SCL precomputed data — present after running precompute_scl.py
        npz_dir = os.path.dirname(os.path.abspath(npz_path))
        scl_path = os.path.join(npz_dir, "scl_partners.npy")
        self.scl_partners = np.load(scl_path) if os.path.exists(scl_path) else None
        if self.scl_partners is not None:
            assert len(self.scl_partners) == len(self.site_ids), (
                "scl_partners.npy length mismatch with dataset; re-run precompute_scl.py"
            )

        # Optional Slepian geographic features — present after running precompute_slepian.py.
        # When available, concatenated with SatCLIP embeddings along the feature axis so
        # the model sees a richer geographic signal without any architectural changes.
        # embedding_blocks records (name, start, end) offsets into the final concatenated
        # vector so training can optionally regularize the streams against each other;
        # it stays None when no Slepian features are present.
        self.embedding_blocks = None
        slep_path = os.path.join(npz_dir, "slepian_features.npy")
        if satclip_only:
            self.embedding_blocks = [("satclip", 0, self.embeddings.shape[1])]
            print(f"SatCLIP-only mode: ignoring Slepian features "
                  f"({self.embeddings.shape[1]} dims)")
        elif os.path.exists(slep_path):
            slep = np.load(slep_path).astype(np.float32)
            assert len(slep) == len(self.site_ids), (
                "slepian_features.npy length mismatch; re-run precompute_slepian.py"
            )

            meta_path = slep_path.replace(".npy", "_meta.npz")
            sub_blocks = None
            if os.path.exists(meta_path):
                meta = np.load(meta_path, allow_pickle=True)
                if all(k in meta for k in ("global_dim", "cap1_dim", "cap2_dim")):
                    sub_blocks = [("global_sh", int(meta["global_dim"]))]
                    if int(meta["cap1_dim"]) > 0:
                        sub_blocks.append(("cap1", int(meta["cap1_dim"])))
                    if int(meta["cap2_dim"]) > 0:
                        sub_blocks.append(("cap2", int(meta["cap2_dim"])))
                    if sum(sz for _, sz in sub_blocks) != slep.shape[1]:
                        sub_blocks = None  # meta stale relative to the .npy — disable

            if slepian_only:
                self.embeddings = slep
                print(f"Slepian-only mode: replaced embeddings with Slepian features "
                      f"({slep.shape[1]} dims)")
                blocks = sub_blocks if sub_blocks else [("slepian", slep.shape[1])]
            else:
                satclip_dim = self.embeddings.shape[1]
                self.embeddings = np.concatenate([self.embeddings, slep], axis=1)
                print(f"Loaded Slepian features ({slep.shape[1]} dims); "
                      f"total embedding dim: {self.embeddings.shape[1]}")
                blocks = [("satclip", satclip_dim)]
                blocks += sub_blocks if sub_blocks else [("slepian", slep.shape[1])]

            offset = 0
            self.embedding_blocks = []
            for name, size in blocks:
                self.embedding_blocks.append((name, offset, offset + size))
                offset += size
        elif slepian_only:
            raise FileNotFoundError(
                f"slepian_only=True but {slep_path} not found. "
                "Run precompute_slepian.py first."
            )

    def fit_normalization(self, indices, clip_pct=(1, 99)):
        idx     = np.array(indices)
        qc_vals = self.profiles[idx, :, 0][self.masks[idx]].ravel()
        fs_vals = self.profiles[idx, :, 1][self.masks[idx]].ravel()

        self.qc_lo, self.qc_hi = np.percentile(qc_vals, clip_pct)
        self.fs_lo, self.fs_hi = np.percentile(fs_vals, clip_pct)
        self.qc_lo = float(self.qc_lo); self.qc_hi = float(self.qc_hi)
        self.fs_lo = float(self.fs_lo); self.fs_hi = float(self.fs_hi)

        self.qc_mean = float(qc_vals.mean()); self.qc_std = float(qc_vals.std() + 1e-6)
        self.fs_mean = float(fs_vals.mean()); self.fs_std = float(fs_vals.std() + 1e-6)

    def _minmax(self, arr, lo, hi):
        return np.clip((arr - lo) / (hi - lo + 1e-6), 0.0, 1.0).astype(np.float32)

    def denorm_qc(self, x):
        assert self.qc_lo is not None, "call fit_normalization() before denorm_qc()"
        return x * (self.qc_hi - self.qc_lo) + self.qc_lo

    def denorm_fs(self, x):
        assert self.fs_lo is not None, "call fit_normalization() before denorm_fs()"
        return x * (self.fs_hi - self.fs_lo) + self.fs_lo

    def __len__(self):
        return len(self.site_ids)

    def __getitem__(self, idx):
        assert self.qc_lo is not None, "call fit_normalization() before accessing samples"
        profile = self.profiles[idx].copy()
        mask    = self.masks[idx]

        qc_norm = self._minmax(profile[:, 0], self.qc_lo, self.qc_hi)
        fs_norm = self._minmax(profile[:, 1], self.fs_lo, self.fs_hi)

        normed      = np.stack([qc_norm, fs_norm], axis=-1)   # (500, 2) in [0,1]
        normed      = np.where(mask[:, None], normed, 0.0)
        patches     = normed.reshape(len(DEPTH_GRID) // PATCH_SIZE, PATCH_SIZE * 2)  # (500, 2)
        patch_mask  = mask.reshape(len(DEPTH_GRID) // PATCH_SIZE, PATCH_SIZE)
        patch_valid = patch_mask.mean(axis=1) >= 0.5

        result = {
            "patches":     torch.from_numpy(patches).float(),
            "patch_valid": torch.from_numpy(patch_valid),
            "satclip":     torch.from_numpy(self.embeddings[idx]),
            "site":        self.site_ids[idx],
        }

        if self.scl_partners is not None:
            candidates = self.scl_partners[idx]
            candidates = candidates[candidates >= 0]
            if len(candidates) > 0:
                pos_idx = int(np.random.choice(candidates))

                pos_profile = self.profiles[pos_idx].copy()
                pos_mask    = self.masks[pos_idx]

                pos_qc = self._minmax(pos_profile[:, 0], self.qc_lo, self.qc_hi)
                pos_fs = self._minmax(pos_profile[:, 1], self.fs_lo, self.fs_hi)
                pos_normed  = np.stack([pos_qc, pos_fs], axis=-1)
                pos_normed  = np.where(pos_mask[:, None], pos_normed, 0.0)
                pos_patches = pos_normed.reshape(
                    len(DEPTH_GRID) // PATCH_SIZE, PATCH_SIZE * 2
                )
                pos_pv = (pos_mask.reshape(len(DEPTH_GRID) // PATCH_SIZE, PATCH_SIZE)
                          .mean(axis=1) >= 0.5)

                result["pos_patches"]     = torch.from_numpy(pos_patches).float()
                result["pos_patch_valid"] = torch.from_numpy(pos_pv)
                result["pos_satclip"]     = torch.from_numpy(self.embeddings[pos_idx])

        return result
