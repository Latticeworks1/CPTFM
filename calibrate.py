"""
Conformal prediction calibration for the CPT MAE.

Uses the 181 validation sites to compute depth-resolved coverage quantiles,
then evaluates calibrated prediction intervals on the 181 held-out test sites.
At coverage=0.90, 90% of held-out depth-positions fall within the reported interval.

Produces:
  checkpoints/conformal_qc.npy  — (500,) per-depth qc half-widths in MPa
  checkpoints/conformal_fs.npy  — (500,) per-depth fs half-widths in kPa
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import random_split

from cptfm.dataset import CPTDataset, DEPTH_GRID, DEPTH_MAX, DEPTH_STEP, PATCH_SIZE
from cptfm.model import CPTMaskedAutoencoder
from cptfm.tokenizer import CPTTokenizer

N_PATCHES = len(DEPTH_GRID) // PATCH_SIZE


def run_inference(tok, model, ds, indices, device, vis_m):
    depth_step = PATCH_SIZE * DEPTH_STEP
    vis_tok    = int(vis_m / depth_step)

    pred_qc_all, pred_fs_all = [], []
    act_qc_all,  act_fs_all  = [], []
    depth_idx_all            = []

    tok_depth_cond = getattr(tok, "depth_cond", False)
    depth_norm_dev = torch.from_numpy(
        (DEPTH_GRID / DEPTH_MAX).astype("float32")
    ).to(device)

    model.eval()
    with torch.no_grad():
        for i in indices:
            s           = ds[i]
            patches     = s["patches"].unsqueeze(0).to(device)
            satclip     = s["satclip"].unsqueeze(0).to(device)
            patch_valid = s["patch_valid"].unsqueeze(0).to(device)
            pv_np       = s["patch_valid"].numpy()

            dn = depth_norm_dev.unsqueeze(0) if tok_depth_cond else None
            z_q_st, _, _, _ = tok(patches, patch_valid, depth_norm=dn)
            ids_keep, _     = model._prefix_mask(1, vis_tok, device)
            latent          = model.encode(z_q_st, satclip, ids_keep)
            pred_z          = model.decode(latent, satclip, N_PATCHES)
            if model.ce_loss:
                pred_z      = tok.codebook(pred_z.argmax(-1))
            pred_raw        = tok.decode(pred_z).squeeze(0).cpu().numpy()

            pred_qc = ds.denorm_qc(pred_raw[:, 0])
            pred_fs = ds.denorm_fs(pred_raw[:, 1])
            act_raw = s["patches"].numpy().reshape(-1, 2)
            act_qc  = ds.denorm_qc(act_raw[:, 0])
            act_fs  = ds.denorm_fs(act_raw[:, 1])

            eval_mask = np.zeros(N_PATCHES, bool)
            eval_mask[vis_tok:] = True
            eval_mask = eval_mask & pv_np
            if eval_mask.sum() == 0:
                continue

            d_idx = np.where(eval_mask)[0]
            pred_qc_all.append(pred_qc[d_idx])
            pred_fs_all.append(pred_fs[d_idx])
            act_qc_all.append(act_qc[d_idx])
            act_fs_all.append(act_fs[d_idx])
            depth_idx_all.append(d_idx)

    return (np.concatenate(pred_qc_all), np.concatenate(pred_fs_all),
            np.concatenate(act_qc_all),  np.concatenate(act_fs_all),
            np.concatenate(depth_idx_all))


def depth_quantiles(residuals, depth_indices, n_patches, coverage):
    global_q = np.quantile(residuals, coverage)
    q = np.full(n_patches, global_q)
    for d in range(n_patches):
        mask = depth_indices == d
        if mask.sum() >= 10:
            q[d] = np.quantile(residuals[mask], coverage)
    return q


def main(args):
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    ds = CPTDataset(args.cpt_csv)
    n_test  = max(1, int(len(ds) * args.test_frac))
    n_val   = max(1, int(len(ds) * args.val_frac))
    n_train = len(ds) - n_val - n_test
    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )
    ds.fit_normalization(list(train_ds.indices))
    print(f"train: {n_train}  val: {n_val}  test: {n_test}")

    tc = torch.load(os.path.join(args.ckpt_dir, "tokenizer.pt"),
                    map_location="cpu", weights_only=False)
    ta = tc["args"]
    tok = CPTTokenizer(d_vq=ta["d_vq"], K=ta["K"],
                       beta=ta["beta"], kernel=ta["kernel"],
                       depth_cond=ta.get("depth_cond", False)).to(device)
    tok.load_state_dict(tc["state_dict"])
    tok.eval()
    for p in tok.parameters():
        p.requires_grad_(False)

    bc = torch.load(os.path.join(args.ckpt_dir, "best.pt"),
                    map_location="cpu", weights_only=False)
    ba = bc["args"]
    satclip_dim = bc["state_dict"]["enc_satclip_proj.weight"].shape[1]
    model = CPTMaskedAutoencoder(
        patch_dim=ta["d_vq"],
        satclip_dim=satclip_dim,
        hidden_dim=ba["hidden_dim"],
        encoder_depth=ba["encoder_depth"],
        decoder_depth=ba["decoder_depth"],
        num_heads=ba["num_heads"],
        ce_loss=bc.get("ce_loss", False),
        num_codes=bc.get("num_codes", 128),
    ).to(device)
    model.load_state_dict(bc["state_dict"])
    model.eval()
    print(f"loaded checkpoint  epoch {bc['epoch']}  val_loss {bc['val_loss']:.4f}")

    print(f"calibrating on validation set (vis={args.vis_m}m)...")
    pqc, pfs, aqc, afs, didx = run_inference(
        tok, model, ds, list(val_ds.indices), device, args.vis_m)

    q_qc = depth_quantiles(np.abs(pqc - aqc), didx, N_PATCHES, args.coverage)
    q_fs = depth_quantiles(np.abs(pfs - afs), didx, N_PATCHES, args.coverage)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    np.save(os.path.join(args.ckpt_dir, "conformal_qc.npy"), q_qc)
    np.save(os.path.join(args.ckpt_dir, "conformal_fs.npy"), q_fs)
    print(f"saved  median qc ±{np.nanmedian(q_qc):.3f} MPa  "
          f"fs ±{np.nanmedian(q_fs):.3f} kPa")

    print(f"evaluating coverage on held-out test set...")
    pqc_t, pfs_t, aqc_t, afs_t, didx_t = run_inference(
        tok, model, ds, list(test_ds.indices), device, args.vis_m)

    cov_qc = float(np.mean(np.abs(pqc_t - aqc_t) <= q_qc[didx_t]))
    cov_fs = float(np.mean(np.abs(pfs_t - afs_t) <= q_fs[didx_t]))
    rmse_qc = float(np.sqrt(np.mean((pqc_t - aqc_t) ** 2)))
    rmse_fs = float(np.sqrt(np.mean((pfs_t - afs_t) ** 2)))

    print(f"  qc  RMSE {rmse_qc:.3f} MPa  coverage {cov_qc:.3f}  (target {args.coverage:.2f})")
    print(f"  fs  RMSE {rmse_fs:.3f} kPa  coverage {cov_fs:.3f}  (target {args.coverage:.2f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cpt_csv",   default="data/bro_combined.npz")
    p.add_argument("--ckpt_dir",  default="checkpoints")
    p.add_argument("--coverage",  type=float, default=0.90)
    p.add_argument("--vis_m",     type=float, default=0.0)
    p.add_argument("--val_frac",  type=float, default=0.10)
    p.add_argument("--test_frac", type=float, default=0.10)
    main(p.parse_args())
