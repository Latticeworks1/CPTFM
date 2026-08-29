"""
Stage 1: Train the VQ-VAE tokenizer on CPT profiles.
The tokenizer is saved to checkpoints/tokenizer.pt and then used as a frozen
front-end during Stage 2 MAE pretraining (train.py).
"""

import argparse
import math
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split

from cptfm.dataset import CPTDataset, DEPTH_GRID, DEPTH_MAX
from cptfm.tokenizer import CPTTokenizer


def cosine_lr(step, warmup, total, lo, hi):
    if step < warmup:
        return hi * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return lo + 0.5 * (hi - lo) * (1 + math.cos(math.pi * t))


def main(args):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"device: {device}")

    ds = CPTDataset(args.cpt_csv)
    n_test  = max(1, int(len(ds) * 0.10))
    n_val   = max(1, int(len(ds) * 0.10))
    n_train = len(ds) - n_val - n_test
    assert n_train > 0
    train_ds, val_ds, _ = random_split(ds, [n_train, n_val, n_test],
                                       generator=torch.Generator().manual_seed(42))
    ds.fit_normalization(list(train_ds.indices))
    print(f"train: {n_train}  val: {n_val}  test (held-out): {n_test}")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    # depth_norm: (T,) tensor broadcast to (B, T) at call time
    _depth_norm = torch.from_numpy((DEPTH_GRID / DEPTH_MAX).astype(np.float32)).to(device)

    tok = CPTTokenizer(
        d_vq=args.d_vq, K=args.K, beta=args.beta,
        kernel=args.kernel, depth_cond=args.depth_cond,
    ).to(device)
    print(f"tokenizer params: {sum(p.numel() for p in tok.parameters())/1e3:.1f}K  "
          f"depth_cond: {args.depth_cond}")

    opt   = torch.optim.AdamW(tok.parameters(), lr=1.0, weight_decay=1e-4)
    steps_per_epoch = math.ceil(n_train / args.batch)
    total_steps     = args.epochs * steps_per_epoch
    warmup_steps    = 5 * steps_per_epoch

    best_val    = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        tok.train()
        tr_recon = tr_total = 0.0

        for batch in train_dl:
            lr = cosine_lr(global_step, warmup_steps, total_steps, 1e-5, args.lr)
            for pg in opt.param_groups:
                pg["lr"] = lr

            x  = batch["patches"].to(device)       # (B, 500, 2)
            pv = batch["patch_valid"].to(device)    # (B, 500)
            B  = x.shape[0]
            dn = _depth_norm.unsqueeze(0).expand(B, -1)  # (B, 500)

            opt.zero_grad()
            _, _, loss, recon = tok(x, pv, depth_norm=dn)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tok.parameters(), 1.0)
            opt.step()

            tr_total += loss.item()
            tr_recon += recon.item()
            global_step += 1

        tr_total /= len(train_dl)
        tr_recon /= len(train_dl)

        tok.eval()
        val_recon = 0.0
        with torch.no_grad():
            for batch in val_dl:
                x  = batch["patches"].to(device)
                pv = batch["patch_valid"].to(device)
                B  = x.shape[0]
                dn = _depth_norm.unsqueeze(0).expand(B, -1)
                _, _, _, recon = tok(x, pv, depth_norm=dn)
                val_recon += recon.item()
        val_recon /= len(val_dl)
        usage = tok.codebook_usage()

        print(f"epoch {epoch+1:3d}/{args.epochs}  "
              f"loss {tr_total:.4f}  train_recon {tr_recon:.4f}  "
              f"val_recon {val_recon:.4f}  codebook {usage:.0%}  lr {lr:.2e}")

        if val_recon < best_val:
            best_val = val_recon
            os.makedirs(args.ckpt_dir, exist_ok=True)
            torch.save({
                "epoch":      epoch + 1,
                "state_dict": tok.state_dict(),
                "val_recon":  best_val,
                "args": {
                    "d_vq":       args.d_vq,
                    "K":          args.K,
                    "beta":       args.beta,
                    "kernel":     args.kernel,
                    "depth_cond": args.depth_cond,
                },
                "norm": {
                    "qc_lo": ds.qc_lo, "qc_hi": ds.qc_hi,
                    "fs_lo": ds.fs_lo, "fs_hi": ds.fs_hi,
                },
            }, os.path.join(args.ckpt_dir, "tokenizer.pt"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cpt_csv",    default="data/combined/combined.npz")
    p.add_argument("--ckpt_dir",   default="checkpoints")
    p.add_argument("--epochs",     type=int,   default=80)
    p.add_argument("--batch",      type=int,   default=64)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--d_vq",       type=int,   default=32)
    p.add_argument("--K",          type=int,   default=128)
    p.add_argument("--beta",       type=float, default=0.25)
    p.add_argument("--kernel",     type=int,   default=7)
    p.add_argument("--depth_cond", action=argparse.BooleanOptionalAction, default=True)
    main(p.parse_args())
