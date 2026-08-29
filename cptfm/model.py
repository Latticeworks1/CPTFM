import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class CPTMaskedAutoencoder(nn.Module):
    def __init__(
        self,
        patch_dim: int = 2,
        satclip_dim: int = 256,
        hidden_dim: int = 256,
        num_patches: int = 500,
        encoder_depth: int = 6,
        decoder_depth: int = 3,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.75,
        dropout: float = 0.0,
        probabilistic: bool = False,
        ce_loss: bool = False,
        num_codes: int = 128,
    ):
        super().__init__()
        self.hidden_dim    = hidden_dim
        self.num_patches   = num_patches
        self.mask_ratio    = mask_ratio
        self.probabilistic = probabilistic
        self.ce_loss       = ce_loss
        self.num_codes     = num_codes

        # encoder
        self.patch_proj       = nn.Linear(patch_dim, hidden_dim)
        self.enc_satclip_proj = nn.Linear(satclip_dim, hidden_dim)
        self.enc_pos_emb      = nn.Embedding(num_patches, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)

        # decoder — cross-attention: queries are all N positions, memory is encoder output
        self.mask_token       = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.dec_pos_emb      = nn.Embedding(num_patches, hidden_dim)
        self.dec_satclip_proj = nn.Linear(satclip_dim, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_depth)

        # output head: K-way classification when ce_loss, continuous reconstruction otherwise
        out_dim = num_codes if ce_loss else patch_dim
        self.decoder_pred   = nn.Linear(hidden_dim, out_dim)
        self.decoder_logvar = nn.Linear(hidden_dim, patch_dim) if probabilistic else None

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.enc_pos_emb.weight, std=0.02)
        nn.init.normal_(self.dec_pos_emb.weight, std=0.02)
        heads = [self.patch_proj, self.enc_satclip_proj, self.dec_satclip_proj, self.decoder_pred]
        if self.decoder_logvar is not None:
            heads.append(self.decoder_logvar)
        for m in heads:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # masking strategies
    # ------------------------------------------------------------------

    def _block_mask(self, B, device, mask_ratio=None, patch_valid=None, grad_weights=None):
        ratio    = mask_ratio if mask_ratio is not None else self.mask_ratio
        N        = self.num_patches
        num_keep = max(1, int(N * (1 - ratio)))
        n_blocks = max(2, int(N * ratio / 40))

        if grad_weights is not None:
            # Gumbel-max trick: samples n_blocks positions with replacement from the
            # gradient-weight distribution without leaving the MPS device.
            log_w  = (grad_weights.float() + 1e-3).log()          # (B, N)
            u      = torch.rand(B, N, n_blocks, device=device).clamp(min=1e-8)
            gumbel = -(-u.log()).log()                             # (B, N, n_blocks)
            starts = (log_w.unsqueeze(2) + gumbel).argmax(dim=1)  # (B, n_blocks)
        else:
            starts = torch.randint(0, N, (B, n_blocks), device=device)
        lengths = torch.randint(10, 80, (B, n_blocks), device=device)

        pos  = torch.arange(N, device=device)
        s    = starts.unsqueeze(2)
        l    = lengths.unsqueeze(2)
        mask = ((pos >= s) & (pos < s + l)).any(dim=1).float()

        priority = mask + torch.rand(B, N, device=device) * 1e-3
        if patch_valid is not None:
            priority = priority + (~patch_valid).float() * 2.0
        sorted_idx = priority.argsort(dim=1)
        ids_keep   = sorted_idx[:, :num_keep].sort(dim=1).values

        mask = torch.ones(B, N, device=device)
        mask.scatter_(1, ids_keep, 0.0)
        return ids_keep, mask

    def _prefix_mask(self, B, visible_patches, device):
        visible_patches = max(0, min(visible_patches, self.num_patches))
        ids_keep = torch.arange(visible_patches, device=device).unsqueeze(0).expand(B, -1)
        mask     = torch.ones(B, self.num_patches, device=device)
        if visible_patches > 0:
            mask[:, :visible_patches] = 0.0
        return ids_keep, mask

    # ------------------------------------------------------------------
    # encoder / decoder
    # ------------------------------------------------------------------

    def encode(self, patches, satclip, ids_keep):
        B, N, _ = patches.shape
        pos     = torch.arange(N, device=patches.device)
        x       = self.patch_proj(patches) + self.enc_pos_emb(pos).unsqueeze(0)
        prefix  = self.enc_satclip_proj(satclip).unsqueeze(1)

        if ids_keep.shape[1] == 0:
            return self.encoder(prefix)

        x_vis = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, self.hidden_dim))
        return self.encoder(torch.cat([prefix, x_vis], dim=1))

    def decode(self, latent, satclip, N):
        # latent: (B, 1+n_vis, hidden_dim) — encoder output including satclip prefix token
        # queries: mask token + positional embedding + geographic residual for every depth position
        B   = latent.shape[0]
        pos = torch.arange(N, device=latent.device)
        geo = self.dec_satclip_proj(satclip).unsqueeze(1)          # (B, 1, hidden_dim)
        tgt = self.mask_token.expand(B, N, -1) + self.dec_pos_emb(pos).unsqueeze(0) + geo

        # cross-attention: each depth position queries the full encoder memory
        h    = self.decoder(tgt, latent)       # (B, N, hidden_dim)
        mean = self.decoder_pred(h)

        if self.decoder_logvar is not None:
            logvar = self.decoder_logvar(h).clamp(-6.0, 6.0)
            return mean, logvar
        return mean

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, patches, satclip, patch_valid=None, mask_ratio=None, prefix_frac=0.0,
                grad_weights=None, token_indices=None, return_decoded=False, return_latent=False):
        B, N, _ = patches.shape
        device  = patches.device

        if prefix_frac > 0.0 and random.random() < prefix_frac:
            # sample vis=0 (geographic-only) 20% of prefix batches; remainder uniform [1, N-1]
            if random.random() < 0.20:
                vis = 0
            else:
                vis = torch.randint(1, N, (1,)).item()
            ids_keep, mask = self._prefix_mask(B, vis, device)
        else:
            ids_keep, mask = self._block_mask(B, device, mask_ratio, patch_valid, grad_weights)

        latent  = self.encode(patches, satclip, ids_keep)
        decoded = self.decode(latent, satclip, N)

        loss_mask = mask
        if patch_valid is not None:
            loss_mask = loss_mask * patch_valid.float()

        if self.ce_loss:
            assert token_indices is not None, "token_indices required when ce_loss=True"
            flat_pred = decoded.reshape(-1, self.num_codes)
            flat_idx  = token_indices.reshape(-1).long()
            flat_lm   = loss_mask.reshape(-1)
            loss_ce   = F.cross_entropy(flat_pred, flat_idx, reduction='none')
            loss      = (loss_ce * flat_lm).sum() / (flat_lm.sum() + 1e-6)
        elif self.probabilistic:
            pred, logvar = decoded
            loss_per = 0.5 * (logvar + ((pred - patches) ** 2) / logvar.exp()).mean(dim=-1)
            loss = (loss_per * loss_mask).sum() / (loss_mask.sum() + 1e-6)
        else:
            pred     = decoded
            loss_per = ((pred - patches) ** 2).mean(dim=-1)
            loss = (loss_per * loss_mask).sum() / (loss_mask.sum() + 1e-6)

        out = (loss, mask)
        if return_decoded:
            out = out + (decoded,)
        if return_latent:
            out = out + (latent,)
        return out
