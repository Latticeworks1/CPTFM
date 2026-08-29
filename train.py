"""
Curriculum training schedule:

  Phase 1 — foundation  (0–25% of epochs)
    mask_ratio 0.50, prefix_frac 0.00, full augmentation
    Model learns basic stratigraphic patterns from any visible subset.

  Phase 2 — alignment   (25–65% of epochs)
    mask_ratio ramps 0.50→0.75, prefix_frac ramps 0.00→0.50
    Difficulty increases while the training geometry converges toward the
    downward-continuation inference task.

  Phase 3 — refinement  (65–100% of epochs)
    mask_ratio 0.75, prefix_frac ramps 0.50→0.90, noise halved
    Model fine-tunes on the true task distribution with reduced stochasticity.

Evaluation uses a held-out test split evaluated under prefix masking at
[0, 5, 10, 15, 20] m visible depth. RMSE reported in physical units.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from cptfm.augment import augment_batch
from cptfm.dataset import CPTDataset, DEPTH_GRID, DEPTH_MAX, DEPTH_STEP, PATCH_SIZE
from cptfm.model import CPTMaskedAutoencoder
from cptfm.tokenizer import CPTTokenizer

N_PATCHES      = len(DEPTH_GRID) // PATCH_SIZE   # 500
VISIBLE_DEPTHS = [0, 5, 10, 15, 20]              # metres

# Fixed hydrostatic effective stress profile used for Ic auxiliary loss.
# gamma_avg=18 kN/m³, gamma_w=9.81 kN/m³, water table at surface.
# These are precomputed constants on CPU; moved to device once in main().
_sv0_np     = (18.0 * DEPTH_GRID).astype(np.float32)
_sv0_eff_np = np.maximum(_sv0_np - 9.81 * DEPTH_GRID, 1.0).astype(np.float32)
_depth_norm_np = (DEPTH_GRID / DEPTH_MAX).astype(np.float32)


def _ic_torch(qt_kpa, fs_kpa, sv0, sv0_eff):
    """
    Differentiable Robertson (1990) Ic from tensors on the same device.
    qt_kpa, fs_kpa: (B, T)   sv0, sv0_eff: (T,) broadcast over batch
    """
    net = (qt_kpa - sv0).clamp(min=1.0)
    Qt  = net / sv0_eff
    Fr  = (fs_kpa / net * 100.0).clamp(min=0.01, max=20.0)
    Ic  = ((3.47 - Qt.clamp(min=0.01).log10()).pow(2)
           + (Fr.clamp(min=0.01).log10() + 1.22).pow(2)).sqrt()
    return Ic.clamp(max=5.0)


def stream_decorrelation_loss(satclip_batch, weight, block_bounds):
    """Penalizes cross-block correlation in enc_satclip_proj's contribution to the
    hidden vector, one block per geographic feature stream (satclip / global_sh /
    cap1 / cap2). Mirrors the offdiag Gram-matrix penalty used to keep separately
    encoded spatial/temporal streams from collapsing onto redundant subspaces.
    """
    if len(block_bounds) < 2:
        return satclip_batch.new_zeros(())
    contribs = [satclip_batch[:, s:e] @ weight[:, s:e].T for s, e in block_bounds]
    B = satclip_batch.shape[0]
    loss = satclip_batch.new_zeros(())
    n_pairs = 0
    for i in range(len(contribs)):
        for j in range(i + 1, len(contribs)):
            cross = (contribs[i].T @ contribs[j]) / max(1, B)
            loss = loss + (cross ** 2).sum()
            n_pairs += 1
    return loss / n_pairs


def get_or_create_split(ds, args):
    """Train/val/test split persisted by site_id rather than recomputed positionally
    each run. A positional random_split(seed=42) silently redefines the partition
    whenever the underlying corpus changes size (dedup, added sites, reordering),
    which is exactly the kind of drift that leaked test examples into training
    once already. First run computes the same split random_split(seed=42) would
    have produced and writes it to ckpt_dir/split.json; every later run against
    that ckpt_dir reads it back by site_id, independent of corpus edits.
    """
    split_path = os.path.join(args.ckpt_dir, "split.json")
    site_ids = np.asarray(ds.site_ids)

    if os.path.exists(split_path):
        with open(split_path) as f:
            saved = json.load(f)
        id_to_idx = {s: i for i, s in enumerate(site_ids)}

        def resolve(name):
            ids = saved[name]
            idx = [id_to_idx[s] for s in ids if s in id_to_idx]
            missing = len(ids) - len(idx)
            if missing:
                print(f"split.json: {missing} {name} site_ids no longer in dataset (dropped)")
            return idx

        train_idx, val_idx, test_idx = resolve("train"), resolve("val"), resolve("test")
        print(f"loaded persisted split from {split_path}")
    else:
        n = len(ds)
        n_test = max(1, int(n * args.test_frac))
        n_val = max(1, int(n * args.val_frac))
        n_train = n - n_val - n_test
        assert n_train > 0

        perm = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:]

        os.makedirs(args.ckpt_dir, exist_ok=True)
        with open(split_path, "w") as f:
            json.dump({
                "train": site_ids[train_idx].tolist(),
                "val": site_ids[val_idx].tolist(),
                "test": site_ids[test_idx].tolist(),
            }, f)
        print(f"computed fresh split (seed=42), saved to {split_path}")

    return train_idx, val_idx, test_idx


def cosine_lr(step, warmup_steps, total_steps, lr_min, lr_max):
    if step < warmup_steps:
        return lr_max * step / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t))


def curriculum(epoch, total_epochs):
    p = epoch / total_epochs
    if p < 0.25:
        return 0.50, 0.00, 1.0
    elif p < 0.65:
        t = (p - 0.25) / 0.40
        return 0.50 + 0.25 * t, 0.50 * t, 1.0
    else:
        t = (p - 0.65) / 0.35
        return 0.75, 0.50 + 0.40 * t, 0.5


def test_eval(tok, model, ds, test_indices, device, depth_norm_dev, tok_depth_cond):
    model.eval()
    results    = {}
    depth_step = PATCH_SIZE * DEPTH_STEP

    with torch.no_grad():
        for vis_m in VISIBLE_DEPTHS:
            vis_tok  = int(vis_m / depth_step)
            qc_sq, fs_sq = [], []

            for i in test_indices:
                s           = ds[i]
                patches     = s["patches"].unsqueeze(0).to(device)
                satclip     = s["satclip"].unsqueeze(0).to(device)
                patch_valid = s["patch_valid"].unsqueeze(0).to(device)
                pv_np       = s["patch_valid"].numpy()

                dn = depth_norm_dev.unsqueeze(0) if tok_depth_cond else None
                z_q_st, _, _, _ = tok(patches, patch_valid, depth_norm=dn)

                ids_keep, _ = model._prefix_mask(1, vis_tok, device)
                latent      = model.encode(z_q_st, satclip, ids_keep)
                decoded     = model.decode(latent, satclip, N_PATCHES)

                if model.ce_loss:
                    pred_z = tok.codebook(decoded.argmax(-1))
                else:
                    pred_z = decoded
                pred_raw = tok.decode(pred_z).squeeze(0).cpu().numpy()

                pred_qc = ds.denorm_qc(pred_raw[:, 0])
                pred_fs = ds.denorm_fs(pred_raw[:, 1])

                act_raw = s["patches"].numpy().reshape(-1, 2)
                act_qc  = ds.denorm_qc(act_raw[:, 0])
                act_fs  = ds.denorm_fs(act_raw[:, 1])

                eval_mask             = np.zeros(N_PATCHES, bool)
                eval_mask[vis_tok:]   = True
                eval_mask             = eval_mask & pv_np

                if eval_mask.sum() == 0:
                    continue

                qc_sq.append((pred_qc[eval_mask] - act_qc[eval_mask]) ** 2)
                fs_sq.append((pred_fs[eval_mask] - act_fs[eval_mask]) ** 2)

            if qc_sq:
                results[vis_m] = (
                    float(np.sqrt(np.concatenate(qc_sq).mean())),
                    float(np.sqrt(np.concatenate(fs_sq).mean())),
                )

    return results


def main(args):
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"device: {device}")

    ds = CPTDataset(args.cpt_csv, slepian_only=args.slepian_only, satclip_only=args.satclip_only)

    train_idx, val_idx, test_idx = get_or_create_split(ds, args)
    n_train, n_val, n_test = len(train_idx), len(val_idx), len(test_idx)
    assert n_train > 0

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    test_ds = Subset(ds, test_idx)

    ds.fit_normalization(list(train_ds.indices))
    print(f"train: {n_train}  val: {n_val}  test (held-out): {n_test}")
    print(f"qc [{ds.qc_lo:.3f}, {ds.qc_hi:.3f}] MPa   "
          f"fs [{ds.fs_lo:.3f}, {ds.fs_hi:.3f}] kPa")

    train_dl = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, persistent_workers=(args.workers > 0),
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, persistent_workers=(args.workers > 0),
    )

    # Move precomputed stress / depth arrays to device once.
    sv0_dev        = torch.from_numpy(_sv0_np).to(device)         # (500,)
    sv0_eff_dev    = torch.from_numpy(_sv0_eff_np).to(device)     # (500,)
    depth_norm_dev = torch.from_numpy(_depth_norm_np).to(device)  # (500,)

    tok_ckpt = os.path.join(args.ckpt_dir, "tokenizer.pt")
    assert os.path.exists(tok_ckpt), (
        f"tokenizer checkpoint not found at {tok_ckpt}. "
        "Run train_tokenizer.py first."
    )
    tc  = torch.load(tok_ckpt, map_location="cpu", weights_only=False)
    ta  = tc["args"]
    tok_depth_cond = ta.get("depth_cond", False)
    tok = CPTTokenizer(
        d_vq=ta["d_vq"], K=ta["K"], beta=ta["beta"],
        kernel=ta["kernel"], depth_cond=tok_depth_cond,
    ).to(device)
    tok.load_state_dict(tc["state_dict"])
    tok.eval()
    for p in tok.parameters():
        p.requires_grad_(False)
    patch_dim = ta["d_vq"]
    print(f"tokenizer loaded  val_recon {tc['val_recon']:.4f}  "
          f"K={ta['K']}  d_vq={ta['d_vq']}  depth_cond={tok_depth_cond}")

    satclip_dim = ds.embeddings.shape[1]
    model = CPTMaskedAutoencoder(
        patch_dim=patch_dim,
        satclip_dim=satclip_dim,
        hidden_dim=args.hidden_dim,
        num_patches=N_PATCHES,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
        num_heads=args.num_heads,
        mask_ratio=0.75,
        probabilistic=args.probabilistic,
        ce_loss=args.ce_loss,
        num_codes=ta["K"],
    ).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    block_bounds = None
    if args.lambda_decorr > 0.0:
        if ds.embedding_blocks is None or len(ds.embedding_blocks) < 2:
            print("lambda_decorr > 0 but dataset has fewer than 2 embedding streams "
                  "(run precompute_slepian.py); decorrelation term disabled.")
        else:
            block_bounds = [(s, e) for _, s, e in ds.embedding_blocks]
            names = ", ".join(f"{n}[{s}:{e}]" for n, s, e in ds.embedding_blocks)
            print(f"decorrelation term enabled (lambda={args.lambda_decorr}): {names}")

    opt = torch.optim.AdamW(model.parameters(), lr=1.0, weight_decay=args.wd)

    steps_per_epoch = math.ceil(n_train / (args.batch * args.grad_accum))
    total_steps     = args.epochs * steps_per_epoch
    warmup_steps    = args.warmup_epochs * steps_per_epoch

    best_val      = float("inf")
    best_val_kind = "block_val"   # which task best_val was measured on: block_val or prefix_val
    global_step   = 0
    start_epoch   = 0

    def _atomic_save(obj, path):
        tmp = path + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)

    last_ckpt = os.path.join(args.ckpt_dir, "last.pt")
    if args.warm_encoder:
        best_ckpt = os.path.join(args.ckpt_dir, "best.pt")
        assert os.path.exists(best_ckpt), f"--warm_encoder requires {best_ckpt}"
        wc = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(wc["state_dict"], strict=False)
        model.to(device)
        print(f"warm encoder from epoch {wc['epoch']}  "
              f"missing={len(missing)}  unexpected={len(unexpected)}")
    elif args.resume:
        prev_ckpt = os.path.join(args.ckpt_dir, "last_prev.pt")
        ckpt_candidates = [(last_ckpt, "last.pt"), (prev_ckpt, "last_prev.pt")]
        existing = [(p, l) for p, l in ckpt_candidates if os.path.exists(p)]
        resumed  = False
        for ckpt_path, ckpt_label in ckpt_candidates:
            if not os.path.exists(ckpt_path):
                continue
            try:
                lc = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                ckpt_ce = lc.get("ce_loss", False)
                if ckpt_ce != args.ce_loss:
                    print(f"{ckpt_label}: ce_loss mismatch — skipping")
                    continue
                model.load_state_dict(lc["state_dict"])
                if lc.get("opt_state") is not None:
                    opt.load_state_dict(lc["opt_state"])
                start_epoch   = lc["epoch"]
                global_step   = lc["global_step"]
                best_val      = lc["best_val"]
                best_val_kind = lc.get("best_val_kind", "block_val")
                model.to(device)
                print(f"resumed from {ckpt_label} epoch {start_epoch}  "
                      f"best_val {best_val:.4f} ({best_val_kind})")
                resumed = True
                break
            except Exception as e:
                print(f"{ckpt_label}: load failed ({e}) — trying fallback")
        if existing and not resumed:
            # --resume was requested and checkpoint file(s) exist, but none loaded —
            # refuse to silently fall through to a freshly initialized model at
            # epoch 0, which would look like a normal run in the log.
            raise RuntimeError(
                f"--resume requested and checkpoint(s) found "
                f"({[l for _, l in existing]}) but none loaded successfully; "
                "refusing to silently restart from scratch. Fix or remove the "
                "checkpoint files, or drop --resume for an intentional fresh start."
            )
        elif not existing:
            print(f"--resume requested but no checkpoint found in {args.ckpt_dir} — starting fresh")

    # Normalization scalars on device for differentiable denorm in Ic loss.
    qc_scale = torch.tensor(ds.qc_hi - ds.qc_lo, device=device)
    qc_shift = torch.tensor(ds.qc_lo,             device=device)
    fs_scale = torch.tensor(ds.fs_hi - ds.fs_lo,  device=device)
    fs_shift = torch.tensor(ds.fs_lo,              device=device)

    for epoch in range(start_epoch, args.epochs):
        mask_ratio, prefix_frac, noise_scale = curriculum(epoch, args.epochs)

        model.train()
        train_loss_sum  = torch.zeros(1, device=device)
        grad_norm_sum   = torch.zeros(1, device=device)
        decorr_loss_sum = torch.zeros(1, device=device)
        opt.zero_grad()
        epoch_t0 = time.perf_counter()

        for j, batch in enumerate(train_dl):
            raw_patches  = batch["patches"].to(device)
            clean_patches = raw_patches.clone()
            satclip      = batch["satclip"].to(device)
            patch_valid  = batch["patch_valid"].to(device)
            B            = raw_patches.shape[0]

            diffs  = raw_patches.diff(dim=1).abs().sum(-1) * patch_valid[:, 1:].float()
            grad_w = torch.cat([diffs[:, :1], diffs], dim=1)

            raw_patches, satclip = augment_batch(
                raw_patches, satclip, patch_valid,
                noise_std=0.04 * noise_scale,
                scale_range=0.12 * noise_scale,
                satclip_std=0.02 * noise_scale,
            )

            dn = depth_norm_dev.unsqueeze(0).expand(B, -1) if tok_depth_cond else None
            with torch.no_grad():
                patches, tok_indices, _, _ = tok(raw_patches, patch_valid, depth_norm=dn)

            use_ic  = args.lambda_ic  > 0.0 and args.ce_loss
            use_scl = args.lambda_scl > 0.0 and "pos_patches" in batch

            out = model(
                patches, satclip, patch_valid,
                mask_ratio=mask_ratio, prefix_frac=prefix_frac,
                grad_weights=grad_w, token_indices=tok_indices,
                return_decoded=use_ic,
                return_latent=use_scl,
            )
            loss, mask = out[0], out[1]
            _ei = 2
            decoded    = out[_ei] if use_ic  else None; _ei += int(use_ic)
            enc_latent = out[_ei] if use_scl else None

            if use_ic:
                loss_mask = (mask * patch_valid.float())

                probs     = F.softmax(decoded, dim=-1)
                soft_z    = probs @ tok.codebook.weight
                pred_qcfs = tok.decode(soft_z)

                pred_qc   = pred_qcfs[..., 0] * qc_scale + qc_shift
                pred_fs   = pred_qcfs[..., 1] * fs_scale + fs_shift

                act_qc    = clean_patches[..., 0] * qc_scale + qc_shift
                act_fs    = clean_patches[..., 1] * fs_scale + fs_shift

                Ic_pred   = _ic_torch(pred_qc * 1000.0, pred_fs, sv0_dev, sv0_eff_dev)
                Ic_gt     = _ic_torch(act_qc  * 1000.0, act_fs,  sv0_dev, sv0_eff_dev).detach()

                lm = loss_mask.bool()
                if lm.any():
                    loss_ic = F.huber_loss(Ic_pred[lm], Ic_gt[lm], delta=0.5)
                    loss    = loss + args.lambda_ic * loss_ic

            if use_scl:
                pos_raw    = batch["pos_patches"].to(device)
                pos_sat    = batch["pos_satclip"].to(device)
                pos_valid  = batch["pos_patch_valid"].to(device)

                dn_pos = depth_norm_dev.unsqueeze(0).expand(B, -1) if tok_depth_cond else None
                with torch.no_grad():
                    pos_vq, _, _, _ = tok(pos_raw, pos_valid, depth_norm=dn_pos)

                ids_all   = torch.arange(N_PATCHES, device=device).unsqueeze(0).expand(B, -1)
                pos_enc   = model.encode(pos_vq, pos_sat, ids_all)

                h_anc = F.normalize(enc_latent[:, 0, :], dim=-1)
                h_pos = F.normalize(pos_enc[:, 0, :],    dim=-1)

                sim       = h_anc @ h_pos.T / args.tau_scl
                lbl_scl   = torch.arange(B, device=device)
                loss_scl  = F.cross_entropy(sim, lbl_scl)
                loss      = loss + args.lambda_scl * loss_scl

            if block_bounds is not None:
                loss_decorr = stream_decorrelation_loss(
                    satclip, model.enc_satclip_proj.weight, block_bounds
                )
                decorr_loss_sum = decorr_loss_sum + loss_decorr.detach()
                loss = loss + args.lambda_decorr * loss_decorr

            (loss / args.grad_accum).backward()
            train_loss_sum = train_loss_sum + loss.detach()

            last_in_epoch = (j + 1 == len(train_dl))
            if (j + 1) % args.grad_accum == 0 or last_in_epoch:
                lr = cosine_lr(global_step, warmup_steps, total_steps, args.lr_min, args.lr_max)
                for pg in opt.param_groups:
                    pg["lr"] = lr
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_norm_sum = grad_norm_sum + gn.detach()
                opt.step()
                opt.zero_grad()
                global_step += 1

        train_loss = (train_loss_sum / len(train_dl)).item()

        model.eval()
        val_loss_sum = torch.zeros(1, device=device)
        pval_loss_sum = torch.zeros(1, device=device)
        with torch.no_grad():
            for batch in val_dl:
                patches     = batch["patches"].to(device)
                satclip     = batch["satclip"].to(device)
                patch_valid = batch["patch_valid"].to(device)
                B_v         = patches.shape[0]
                dn          = depth_norm_dev.unsqueeze(0).expand(B_v, -1) if tok_depth_cond else None
                patches, tok_indices, _, _ = tok(patches, patch_valid, depth_norm=dn)
                loss, _     = model(patches, satclip, patch_valid, token_indices=tok_indices)
                val_loss_sum = val_loss_sum + loss

                if prefix_frac > 0.0:
                    ids_pv, mask_pv = model._prefix_mask(B_v, 100, device)
                    latent_pv       = model.encode(patches, satclip, ids_pv)
                    decoded_pv      = model.decode(latent_pv, satclip, N_PATCHES)
                    lm_pv = mask_pv * patch_valid.float() if patch_valid is not None else mask_pv
                    if model.ce_loss:
                        flat_pred = decoded_pv.reshape(-1, model.num_codes)
                        flat_idx  = tok_indices.reshape(-1).long()
                        flat_lm   = lm_pv.reshape(-1)
                        ploss = (F.cross_entropy(flat_pred, flat_idx, reduction='none')
                                 * flat_lm).sum() / (flat_lm.sum() + 1e-6)
                    else:
                        pred_pv = decoded_pv[0] if model.probabilistic else decoded_pv
                        ploss   = ((pred_pv - patches) ** 2).mean(dim=-1)
                        ploss   = (ploss * lm_pv).sum() / (lm_pv.sum() + 1e-6)
                    pval_loss_sum = pval_loss_sum + ploss

        val_loss  = (val_loss_sum  / len(val_dl)).item()
        pval_loss = (pval_loss_sum / len(val_dl)).item()
        ckpt_loss  = pval_loss if prefix_frac > 0.0 else val_loss
        ckpt_kind  = "prefix_val" if prefix_frac > 0.0 else "block_val"

        # pval and val measure different tasks (fixed-prefix continuation vs. random
        # block masking) and are not comparable. best_val must never be carried across
        # a change in which task it was measured on, or a checkpoint saved under the
        # easier block_val regime will silently and permanently block every future
        # prefix_val epoch from ever being recognized as an improvement.
        if ckpt_kind != best_val_kind:
            best_val = float("inf")
            best_val_kind = ckpt_kind

        epoch_secs = time.perf_counter() - epoch_t0
        n_opt_steps = max(1, len(train_dl) // args.grad_accum)
        mean_gn     = (grad_norm_sum / n_opt_steps).item()
        mps_mb      = torch.mps.current_allocated_memory() / 1e6 if device.type == "mps" else 0.0

        suffix = f"  pval {pval_loss:.4f}" if prefix_frac > 0.0 else ""
        decorr_suffix = ""
        if block_bounds is not None:
            mean_decorr = (decorr_loss_sum / len(train_dl)).item()
            decorr_suffix = f"  decorr(raw) {mean_decorr:.1f}"
        print(f"epoch {epoch+1:3d}/{args.epochs}  "
              f"train {train_loss:.4f}  val {val_loss:.4f}{suffix}{decorr_suffix}  "
              f"mr {mask_ratio:.2f}  pf {prefix_frac:.2f}  "
              f"lr {lr:.2e}  gn {mean_gn:.3f}  "
              f"t {epoch_secs:.0f}s  mps {mps_mb:.0f}MB", flush=True)

        os.makedirs(args.ckpt_dir, exist_ok=True)
        if ckpt_loss < best_val:
            best_val = ckpt_loss
            _atomic_save({
                "epoch":       epoch + 1,
                "state_dict":  model.state_dict(),
                "val_loss":    best_val,
                "ckpt_metric": ckpt_kind,
                "args":        vars(args),
                "uses_vq":     True,
                "ce_loss":     args.ce_loss,
                "num_codes":   ta["K"],
                "split": {"n_train": n_train, "n_val": n_val, "n_test": n_test, "seed": 42},
                "norm": {
                    "qc_lo": ds.qc_lo, "qc_hi": ds.qc_hi,
                    "fs_lo": ds.fs_lo, "fs_hi": ds.fs_hi,
                },
            }, os.path.join(args.ckpt_dir, "best.pt"))

        prev_ckpt = last_ckpt.replace("last.pt", "last_prev.pt")
        if os.path.exists(last_ckpt):
            os.replace(last_ckpt, prev_ckpt)
        _atomic_save({
            "epoch":       epoch + 1,
            "global_step": global_step,
            "best_val":    best_val,
            "best_val_kind": best_val_kind,
            "ckpt_loss":   ckpt_loss,
            "ce_loss":     args.ce_loss,
            "state_dict":  model.state_dict(),
            "opt_state":   opt.state_dict(),
            "norm": {
                "qc_lo": ds.qc_lo, "qc_hi": ds.qc_hi,
                "fs_lo": ds.fs_lo, "fs_hi": ds.fs_hi,
            },
        }, last_ckpt)

        # The MPS caching allocator pools memory by allocation shape and does not
        # proactively coalesce/release cached-but-unused Metal buffers on its own
        # (unlike CUDA). n_train is not a multiple of args.batch, so every epoch's
        # final batch has a different shape than the rest, and each such shape adds
        # another cached block. torch.mps.current_allocated_memory() (the "mps
        # NNNMB" figure logged below) only reflects live in-use tensors, not this
        # cache, so the driver-level physical footprint can climb far past what the
        # log shows unless the cache is flushed every epoch rather than only on the
        # (rarer) eval_interval cadence.
        if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            res = test_eval(
                tok, model, ds, list(test_ds.indices), device,
                depth_norm_dev, tok_depth_cond,
            )
            if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()
            n_t = len(test_ds)
            for vis_m in VISIBLE_DEPTHS:
                if vis_m in res:
                    qc_r, fs_r = res[vis_m]
                    print(f"  test (n={n_t})  vis {vis_m:2d}m  "
                          f"qc {qc_r:.3f} MPa  fs {fs_r:.3f} kPa")
            model.train()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cpt_csv",       default="data/bro_combined.npz")
    p.add_argument("--ckpt_dir",      default="checkpoints")
    p.add_argument("--epochs",        type=int,   default=300)
    p.add_argument("--batch",         type=int,   default=64)
    p.add_argument("--grad_accum",    type=int,   default=4)
    p.add_argument("--workers",       type=int,   default=0)
    p.add_argument("--lr_max",        type=float, default=1e-3)
    p.add_argument("--lr_min",        type=float, default=1e-5)
    p.add_argument("--wd",            type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int,   default=10)
    p.add_argument("--val_frac",      type=float, default=0.10)
    p.add_argument("--test_frac",     type=float, default=0.10)
    p.add_argument("--eval_interval", type=int,   default=25)
    p.add_argument("--lambda_ic",     type=float, default=0.15)
    p.add_argument("--lambda_scl",    type=float, default=0.10)
    p.add_argument("--tau_scl",       type=float, default=0.07)
    p.add_argument("--lambda_decorr", type=float, default=0.0,
                    help="Penalizes cross-stream correlation between satclip/global_sh/"
                         "cap1/cap2 blocks in enc_satclip_proj. 0.0 (default) disables it "
                         "and is fully checkpoint-compatible with existing runs.")
    p.add_argument("--slepian_only",  action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--satclip_only",  action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--resume",        action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--warm_encoder",  action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--probabilistic", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--ce_loss",       action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--hidden_dim",    type=int,   default=256)
    p.add_argument("--encoder_depth", type=int,   default=6)
    p.add_argument("--decoder_depth", type=int,   default=3)
    p.add_argument("--num_heads",     type=int,   default=8)
    main(p.parse_args())
