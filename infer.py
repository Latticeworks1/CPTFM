import argparse
import os
import numpy as np
import torch
from cptfm.dataset import CPTDataset, DEPTH_GRID, DEPTH_MAX, DEPTH_STEP, PATCH_SIZE
from cptfm.model import CPTMaskedAutoencoder
from cptfm.tokenizer import CPTTokenizer

N_PATCHES = len(DEPTH_GRID) // PATCH_SIZE


def load_pipeline(ckpt_dir, device):
    tok_path = os.path.join(ckpt_dir, "tokenizer.pt")
    mae_path = os.path.join(ckpt_dir, "best.pt")
    assert os.path.exists(tok_path), f"tokenizer not found: {tok_path}"
    assert os.path.exists(mae_path), f"MAE checkpoint not found: {mae_path}"

    tc = torch.load(tok_path, map_location="cpu", weights_only=False)
    ta = tc["args"]
    tok = CPTTokenizer(d_vq=ta["d_vq"], K=ta["K"],
                       beta=ta["beta"], kernel=ta["kernel"],
                       depth_cond=ta.get("depth_cond", False)).to(device)
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
    ).to(device)
    mae.load_state_dict(mc["state_dict"])
    mae.eval()
    return tok, mae, mc


def main(args):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    ds = CPTDataset(args.cpt_csv)
    tok, mae, mc = load_pipeline(args.ckpt_dir, device)

    if "norm" in mc:
        ds.qc_lo = mc["norm"]["qc_lo"]; ds.qc_hi = mc["norm"]["qc_hi"]
        ds.fs_lo = mc["norm"]["fs_lo"]; ds.fs_hi = mc["norm"]["fs_hi"]

    print(f"MAE epoch {mc['epoch']}  val_loss {mc['val_loss']:.4f}")

    idx     = ds.site_ids.index(args.site) if args.site else args.idx
    site_id = ds.site_ids[idx]
    s       = ds[idx]

    patches     = s["patches"].unsqueeze(0).to(device)
    satclip     = s["satclip"].unsqueeze(0).to(device)
    patch_valid = s["patch_valid"].unsqueeze(0).to(device)
    pv_np       = s["patch_valid"].numpy()

    vis_tok   = int(args.visible_depth / (PATCH_SIZE * DEPTH_STEP))
    vis_tok   = max(0, min(vis_tok, N_PATCHES))
    visible_m = vis_tok * PATCH_SIZE * DEPTH_STEP
    print(f"site: {site_id}  showing top {visible_m:.2f} m  ({vis_tok}/{N_PATCHES} tokens)")

    depth_norm = torch.from_numpy(
        (DEPTH_GRID / DEPTH_MAX).astype("float32")
    ).to(device).unsqueeze(0)  # (1, 500)

    with torch.no_grad():
        dn = depth_norm if tok.depth_cond else None
        z_q_st, _, _, _  = tok(patches, patch_valid, depth_norm=dn)
        ids_keep, _      = mae._prefix_mask(1, vis_tok, device)
        latent           = mae.encode(z_q_st, satclip, ids_keep)
        pred_z           = mae.decode(latent, satclip, N_PATCHES)
        if mae.ce_loss:
            pred_z       = tok.codebook(pred_z.argmax(-1))
        pred_raw         = tok.decode(pred_z).squeeze(0).cpu().numpy()

    pred_qc = ds.denorm_qc(pred_raw[:, 0])
    pred_fs = ds.denorm_fs(pred_raw[:, 1])

    act_raw = s["patches"].numpy().reshape(-1, 2)
    act_qc  = ds.denorm_qc(act_raw[:, 0])
    act_fs  = ds.denorm_fs(act_raw[:, 1])

    eval_mask = np.zeros(len(DEPTH_GRID), bool)
    eval_mask[vis_tok:] = True
    eval_mask = eval_mask & pv_np

    assert eval_mask.sum() > 0, "no valid masked depth steps to evaluate"

    qc_rmse = np.sqrt(((pred_qc[eval_mask] - act_qc[eval_mask]) ** 2).mean())
    fs_rmse = np.sqrt(((pred_fs[eval_mask] - act_fs[eval_mask]) ** 2).mean())
    print(f"RMSE (masked region)  qc: {qc_rmse:.3f} MPa   fs: {fs_rmse:.3f} kPa")

    if args.plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(8, 10), sharey=True)
        for ax, actual, pred, label, unit in zip(
            axes,
            [act_qc, act_fs], [pred_qc, pred_fs],
            ["qc", "fs"], ["MPa", "kPa"],
        ):
            ax.plot(actual, DEPTH_GRID, "k-",  lw=1.0, label="actual")
            ax.plot(pred,   DEPTH_GRID, "r--", lw=1.0, label="predicted")
            if vis_tok > 0:
                ax.axhline(visible_m, color="steelblue", lw=1, ls=":",
                           label=f"cutoff ({visible_m:.1f} m)")
            ax.set_xlabel(f"{label} ({unit})")
            ax.set_ylabel("depth (m)")
            ax.invert_yaxis()
            ax.legend(fontsize=8)
            ax.set_title(label)
        fig.suptitle(f"{site_id} — top {visible_m:.1f} m visible", fontsize=11)
        plt.tight_layout()
        out = f"{site_id}_pred.png"
        plt.savefig(out, dpi=150)
        print(f"saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cpt_csv",       default="data/bro_combined.npz")
    p.add_argument("--ckpt_dir",      default="checkpoints")
    p.add_argument("--idx",           type=int,   default=0)
    p.add_argument("--site",          type=str,   default=None)
    p.add_argument("--visible_depth", type=float, default=5.0)
    p.add_argument("--plot",          action="store_true")
    main(p.parse_args())
