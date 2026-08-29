import os
import torch
import numpy as np
from torch.utils.data import random_split

from cptfm.dataset import CPTDataset, DEPTH_GRID, DEPTH_MAX, DEPTH_STEP, PATCH_SIZE
from cptfm.model import CPTMaskedAutoencoder
from cptfm.tokenizer import CPTTokenizer

VISIBLE_DEPTHS = [0, 5, 10, 15, 20]
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
N_PATCHES = len(DEPTH_GRID) // PATCH_SIZE


def load_pipeline(ckpt_dir="checkpoints"):
    tok_path = os.path.join(ckpt_dir, "tokenizer.pt")
    mae_path = os.path.join(ckpt_dir, "best.pt")
    assert os.path.exists(tok_path), f"tokenizer not found: {tok_path}"
    assert os.path.exists(mae_path), f"MAE checkpoint not found: {mae_path}"

    tc = torch.load(tok_path, map_location="cpu", weights_only=False)
    ta = tc["args"]
    tok = CPTTokenizer(d_vq=ta["d_vq"], K=ta["K"],
                       beta=ta["beta"], kernel=ta["kernel"],
                       depth_cond=ta.get("depth_cond", False)).to(DEVICE)
    tok.load_state_dict(tc["state_dict"])
    tok.eval()

    mc  = torch.load(mae_path, map_location="cpu", weights_only=False)
    cfg = mc["args"]
    satclip_dim = mc["state_dict"]["enc_satclip_proj.weight"].shape[1]
    mae = CPTMaskedAutoencoder(
        patch_dim=ta["d_vq"],
        satclip_dim=satclip_dim,
        hidden_dim=cfg.get("hidden_dim", 256),
        encoder_depth=cfg.get("encoder_depth", 6),
        decoder_depth=cfg.get("decoder_depth", 3),
        num_heads=cfg.get("num_heads", 8),
        num_patches=N_PATCHES,
        ce_loss=mc.get("ce_loss", False),
        num_codes=mc.get("num_codes", 128),
    ).to(DEVICE)
    mae.load_state_dict(mc["state_dict"])
    mae.eval()

    return tok, mae, mc


_depth_norm_eval = torch.from_numpy(
    (DEPTH_GRID / DEPTH_MAX).astype("float32")
).to(DEVICE)


def predict_site(tok, mae, patches, satclip, patch_valid, vis_tok):
    """Full VQ pipeline: encode → MAE → decode → (qc,fs)"""
    tok_depth_cond = getattr(tok, "depth_cond", False)
    with torch.no_grad():
        dn = _depth_norm_eval.unsqueeze(0) if tok_depth_cond else None
        z_q_st, _, _, _ = tok(patches, patch_valid, depth_norm=dn)
        ids_keep, _     = mae._prefix_mask(1, vis_tok, DEVICE)
        latent          = mae.encode(z_q_st, satclip, ids_keep)
        pred_z          = mae.decode(latent, satclip, N_PATCHES)
        if mae.ce_loss:
            pred_z = tok.codebook(pred_z.argmax(-1))
        pred_raw        = tok.decode(pred_z)
    return pred_raw.squeeze(0).cpu().numpy()


def evaluate(ckpt_dir="checkpoints", cpt_csv="data/bro_combined.npz"):
    ds = CPTDataset(cpt_csv)
    n_test  = max(1, int(len(ds) * 0.10))
    n_val   = max(1, int(len(ds) * 0.10))
    n_train = len(ds) - n_val - n_test
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test],
                                             generator=torch.Generator().manual_seed(42))
    ds.fit_normalization(list(train_ds.indices))

    tok, mae, mc = load_pipeline(ckpt_dir)
    if "norm" in mc:
        ds.qc_lo = mc["norm"]["qc_lo"]; ds.qc_hi = mc["norm"]["qc_hi"]
        ds.fs_lo = mc["norm"]["fs_lo"]; ds.fs_hi = mc["norm"]["fs_hi"]

    print(f"MAE epoch {mc['epoch']}  val_loss {mc['val_loss']:.4f}")
    print(f"evaluating on {len(test_ds)} held-out test sites\n")

    # mean-profile baseline from training set (physical units)
    train_qc = np.zeros(len(DEPTH_GRID))
    train_fs = np.zeros(len(DEPTH_GRID))
    train_cnt = np.zeros(len(DEPTH_GRID))
    for i in train_ds.indices:
        s   = ds[i]
        pv  = s["patch_valid"].numpy()
        raw = s["patches"].numpy().reshape(-1, 2)
        train_qc  += np.where(pv, ds.denorm_qc(raw[:, 0]), 0)
        train_fs  += np.where(pv, ds.denorm_fs(raw[:, 1]), 0)
        train_cnt += pv.astype(float)
    mean_qc = np.where(train_cnt > 0, train_qc / (train_cnt + 1e-6), 0)
    mean_fs = np.where(train_cnt > 0, train_fs / (train_cnt + 1e-6), 0)

    print(f"{'visible':>10}  {'qc model (MPa)':>16}  {'qc mean (MPa)':>14}  "
          f"{'fs model (kPa)':>16}  {'fs mean (kPa)':>14}")
    print("-" * 80)

    for vis_m in VISIBLE_DEPTHS:
        vis_tok    = int(vis_m / (PATCH_SIZE * DEPTH_STEP))
        mask_start = vis_tok

        qc_sq_m, fs_sq_m = [], []
        qc_sq_b, fs_sq_b = [], []

        for i in test_ds.indices:
            s           = ds[i]
            patches     = s["patches"].unsqueeze(0).to(DEVICE)
            satclip     = s["satclip"].unsqueeze(0).to(DEVICE)
            patch_valid = s["patch_valid"].unsqueeze(0).to(DEVICE)
            pv_np       = s["patch_valid"].numpy()

            pred_raw = predict_site(tok, mae, patches, satclip, patch_valid, vis_tok)
            pred_qc  = ds.denorm_qc(pred_raw[:, 0])
            pred_fs  = ds.denorm_fs(pred_raw[:, 1])

            act_raw = s["patches"].numpy().reshape(-1, 2)
            act_qc  = ds.denorm_qc(act_raw[:, 0])
            act_fs  = ds.denorm_fs(act_raw[:, 1])

            eval_mask = np.zeros(len(DEPTH_GRID), bool)
            eval_mask[mask_start:] = True
            eval_mask = eval_mask & pv_np

            if eval_mask.sum() == 0:
                continue

            qc_sq_m.append((pred_qc[eval_mask] - act_qc[eval_mask]) ** 2)
            fs_sq_m.append((pred_fs[eval_mask] - act_fs[eval_mask]) ** 2)
            qc_sq_b.append((mean_qc[eval_mask]  - act_qc[eval_mask]) ** 2)
            fs_sq_b.append((mean_fs[eval_mask]  - act_fs[eval_mask]) ** 2)

        qc_rmse_m = np.sqrt(np.concatenate(qc_sq_m).mean())
        fs_rmse_m = np.sqrt(np.concatenate(fs_sq_m).mean())
        qc_rmse_b = np.sqrt(np.concatenate(qc_sq_b).mean())
        fs_rmse_b = np.sqrt(np.concatenate(fs_sq_b).mean())

        print(f"{vis_m:>8.0f} m  {qc_rmse_m:>12.3f} MPa  {qc_rmse_b:>10.3f} MPa  "
              f"{fs_rmse_m:>12.3f} kPa  {fs_rmse_b:>10.3f} kPa")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--cpt_csv",  default="data/bro_combined.npz")
    a = p.parse_args()
    evaluate(a.ckpt_dir, a.cpt_csv)
