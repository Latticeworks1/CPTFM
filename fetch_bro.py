"""
Fetch CPT soundings from the Dutch BRO public REST API and generate SatCLIP
embeddings for each site. Writes bro_data/bro_cpt.csv and
bro_data/bro_embeddings.{csv,npy}.

Usage:
    python3 fetch_bro.py [--max_soundings N] [--out_dir data/bro]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

from cptfm.sources.bro import fetch as bro_fetch


def load_satclip(ckpt_path, device):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "satclip", "satclip"))
    from load import get_satclip  # type: ignore
    model = get_satclip(ckpt_path, device="cpu")
    return model.float().to(device).eval()


def download_ckpt(ckpt_path):
    print(f"Downloading SatCLIP checkpoint to {ckpt_path} ...", flush=True)
    from huggingface_hub import hf_hub_download  # type: ignore
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    hf_hub_download(
        repo_id="microsoft/SatCLIP-ResNet18-L10",
        filename="satclip-resnet18-l10.ckpt",
        local_dir=os.path.dirname(ckpt_path),
    )
    print("  checkpoint saved.", flush=True)


def generate_embeddings(satclip_model, lons, lats, device, batch=256):
    coords = torch.tensor(list(zip(lons, lats)), dtype=torch.float32)
    embs   = []
    for i in range(0, len(coords), batch):
        chunk = coords[i : i + batch].to(device)
        with torch.no_grad():
            embs.append(satclip_model(chunk).cpu().numpy())
    return np.concatenate(embs, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_soundings", type=int,   default=3000)
    p.add_argument("--out_dir",       default="data/bro")
    p.add_argument("--rate_delay",    type=float, default=0.15)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join("satclip", "satclip-resnet18-l10.ckpt")
    if not os.path.exists(ckpt_path):
        download_ckpt(ckpt_path)

    device  = "mps" if torch.backends.mps.is_available() else "cpu"
    satclip = load_satclip(ckpt_path, device)

    records = bro_fetch(max_soundings=args.max_soundings, rate_delay=args.rate_delay)
    if not records:
        print("No soundings fetched — exiting.", flush=True)
        return

    cpt_rows = [
        {"site": rec.site, "lon": round(rec.lon, 6), "lat": round(rec.lat, 6),
         "depth": d, "tip": q, "sleeve": f}
        for rec in records
        for d, q, f in zip(rec.depth, rec.qc, rec.fs)
    ]
    cpt_csv = os.path.join(args.out_dir, "bro_cpt.csv")
    pd.DataFrame(cpt_rows).to_csv(cpt_csv, index=False)
    print(f"Wrote {cpt_csv}  ({len(cpt_rows)} rows)", flush=True)

    lons = [rec.lon for rec in records]
    lats = [rec.lat for rec in records]
    embs = generate_embeddings(satclip, lons, lats, device)

    emb_npy = os.path.join(args.out_dir, "bro_embeddings.npy")
    np.save(emb_npy, embs)

    emb_df = pd.DataFrame({
        "ID":        [rec.site for rec in records],
        "Longitude": lons,
        "Latitude":  lats,
        **{f"d{i}": embs[:, i] for i in range(embs.shape[1])},
    })
    emb_csv = os.path.join(args.out_dir, "bro_embeddings.csv")
    emb_df.to_csv(emb_csv, index=False)
    print(f"Wrote {emb_npy}  shape={embs.shape}", flush=True)
    print(f"Wrote {emb_csv}", flush=True)


if __name__ == "__main__":
    main()
