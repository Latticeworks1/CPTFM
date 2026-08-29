"""
VQ-VAE tokenizer for CPT depth profiles.

A small 1D convolutional encoder gives each depth step a local context window
before quantization, so the codebook captures short-range (qc, fs) morphology
rather than isolated 2D points. The decoder reconstructs the original (qc, fs)
values from quantized embeddings.

When depth_cond=True, the normalised depth position (z / z_max in [0, 1]) is
concatenated to the (qc, fs) pair before encoding, so codebook entries become
depth-aware and map to consistent Robertson SBT zones rather than raw-amplitude
bins. Two patches with identical (qc, fs) at different depths may now map to
different codes, which is the correct behaviour when the stress-normalised
quantities Qt and Fr differ across those depths.

Training is done independently in Stage 1 (train_tokenizer.py) before the MAE
pretraining in Stage 2 (train.py). The tokenizer is frozen during Stage 2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CPTTokenizer(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,      # qc, fs — decoder always outputs this many channels
        d_vq: int = 32,
        K: int = 128,
        beta: float = 0.25,
        kernel: int = 7,
        ema_decay: float = 0.95,
        depth_cond: bool = True,   # concatenate normalised depth before encoding
    ):
        super().__init__()
        self.d_vq       = d_vq
        self.K          = K
        self.beta       = beta
        self.depth_cond = depth_cond

        enc_in = in_channels + (1 if depth_cond else 0)
        pad    = kernel // 2

        self.encoder = nn.Sequential(
            nn.Conv1d(enc_in, d_vq, kernel_size=kernel, padding=pad),
            nn.GELU(),
            nn.Conv1d(d_vq, d_vq, kernel_size=kernel, padding=pad),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(d_vq, d_vq, kernel_size=kernel, padding=pad),
            nn.GELU(),
            nn.Conv1d(d_vq, in_channels, kernel_size=kernel, padding=pad),
        )

        self.codebook = nn.Embedding(K, d_vq)
        nn.init.uniform_(self.codebook.weight, -1.0 / K, 1.0 / K)
        self.codebook.weight.requires_grad_(False)

        self.register_buffer("ema_cluster_size", torch.ones(K) * 1e-6)
        self.register_buffer("ema_embed_sum",    self.codebook.weight.clone())
        self.ema_decay = ema_decay

    # ------------------------------------------------------------------

    def encode(self, x, depth_norm=None):
        """
        x:          (B, T, 2)  normalised (qc, fs)
        depth_norm: (B, T)     z / z_max in [0, 1]; required when depth_cond=True
        returns:    (B, T, d_vq)
        """
        if self.depth_cond:
            assert depth_norm is not None, "depth_norm required when depth_cond=True"
            x = torch.cat([x, depth_norm.unsqueeze(-1)], dim=-1)  # (B, T, 3)
        return self.encoder(x.transpose(1, 2)).transpose(1, 2)

    def quantize(self, z_e, training=False):
        """
        z_e: (B, T, d_vq)
        returns: z_q_st (straight-through), z_q, indices, loss_commit
        """
        B, T, D = z_e.shape
        z_flat  = z_e.reshape(-1, D)
        cb      = self.codebook.weight

        dist    = (z_flat.pow(2).sum(1, keepdim=True)
                   - 2.0 * z_flat @ cb.t()
                   + cb.pow(2).sum(1))
        indices = dist.argmin(1)
        z_q     = cb[indices].reshape(B, T, D)

        if training:
            self._ema_update(z_flat, indices)
            self._reinit_dead_codes(z_flat)

        z_q_st      = z_e + (z_q - z_e).detach()
        loss_commit = self.beta * F.mse_loss(z_e, z_q.detach())
        return z_q_st, z_q.detach(), indices.reshape(B, T), loss_commit

    def _ema_update(self, z_flat, indices):
        with torch.no_grad():
            one_hot = torch.zeros(len(indices), self.K, device=z_flat.device)
            one_hot.scatter_(1, indices.unsqueeze(1), 1.0)

            cluster_size = one_hot.sum(0)
            self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)

            embed_sum = one_hot.t() @ z_flat
            self.ema_embed_sum.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)

            n        = self.ema_cluster_size.sum()
            smoothed = (self.ema_cluster_size + 1e-5) / (n + self.K * 1e-5) * n
            self.codebook.weight.data.copy_(self.ema_embed_sum / smoothed.unsqueeze(1))

    def _reinit_dead_codes(self, z_flat):
        with torch.no_grad():
            usage   = self.ema_cluster_size
            dead    = (usage < 1.0).nonzero(as_tuple=True)[0]
            if len(dead) == 0:
                return
            n_reinit = min(len(dead), len(z_flat))
            perm     = torch.randperm(len(z_flat), device=z_flat.device)[:n_reinit]
            self.codebook.weight.data[dead[:n_reinit]] = z_flat[perm].detach()
            self.ema_cluster_size[dead[:n_reinit]]     = 1.0

    def decode(self, z_q):
        """z_q: (B, T, d_vq) -> (B, T, 2)"""
        return self.decoder(z_q.transpose(1, 2)).transpose(1, 2)

    def forward(self, x, patch_valid=None, depth_norm=None):
        """
        x:           (B, T, 2)   normalised (qc, fs)
        patch_valid: (B, T) bool
        depth_norm:  (B, T)      required when depth_cond=True

        returns: z_q_st, indices, loss_total, loss_recon
        """
        z_e                             = self.encode(x, depth_norm)
        z_q_st, z_q, indices, loss_commit = self.quantize(z_e, training=self.training)
        x_hat                           = self.decode(z_q_st)

        if patch_valid is not None:
            w          = patch_valid.float().unsqueeze(-1)
            loss_recon = (((x_hat - x) ** 2) * w).sum() / (w.sum() * 2 + 1e-6)
        else:
            loss_recon = F.mse_loss(x_hat, x)

        return z_q_st, indices, loss_recon + loss_commit, loss_recon

    def codebook_usage(self):
        return (self.ema_cluster_size > 1.0).float().mean().item()
