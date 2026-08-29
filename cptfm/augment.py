import torch


def augment_batch(patches, satclip, patch_valid, noise_std=0.05, scale_range=0.15, satclip_std=0.02):
    """
    In-place-safe augmentation applied only during training.

    patches:     (B, 500, 2)  normalised patch tensor
    satclip:     (B, 256)
    patch_valid: (B, 500) bool

    Three transforms are applied independently per sample:

    1. Additive Gaussian noise on the patch values — simulates measurement
       noise and quantisation error inherent in CPT instrumentation.

    2. Per-channel amplitude jitter — multiplies the qc and fs channels by
       independent random scalars drawn from U[1-scale_range, 1+scale_range].
       This reflects calibration differences across cone tip manufacturers and
       between field campaigns, which produce proportional shifts in both channels.

    3. SatCLIP embedding noise — small isotropic Gaussian perturbation on the
       geographic embedding. Prevents the encoder from memorising exact embedding
       coordinates and improves robustness when the embedding drifts slightly
       between SatCLIP model versions.

    Invalid (padded) patch positions are zeroed after augmentation so that
    noise does not create a spurious signal in padding regions.
    """
    B, N, D = patches.shape
    device   = patches.device

    # additive patch noise
    patches = patches + noise_std * torch.randn_like(patches)

    # per-channel amplitude jitter — qc is channel 0, fs is channel 1
    scale_qc = 1.0 + (torch.rand(B, 1, 1, device=device) * 2 - 1) * scale_range
    scale_fs = 1.0 + (torch.rand(B, 1, 1, device=device) * 2 - 1) * scale_range
    half = D // 2
    patches = torch.cat([
        patches[:, :, :half] * scale_qc,
        patches[:, :, half:] * scale_fs,
    ], dim=-1)

    # zero out padding positions so augmented noise stays masked
    patches = patches * patch_valid.float().unsqueeze(-1)

    # SatCLIP embedding noise
    satclip = satclip + satclip_std * torch.randn_like(satclip)

    return patches, satclip
