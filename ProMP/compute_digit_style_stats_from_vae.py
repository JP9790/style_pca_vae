#!/usr/bin/env python3
"""
Compute digit-domain style statistics (mu_style, cov_style) from a trained digit VAE.

Implements the VAE style construction:
  - Use style-related latent dims Js from an ANOVA JSON
  - For user demos i:
      zbar_u^(s) = mean_i mu_i^(s)
      sigma_u^2(s) = mean_i [ sigma_i^2(s) + (mu_i^(s) - zbar_u^(s))^2 ]
  - Sample Ns style latent vectors:
      z_u^(s) ~ N(zbar_u^(s), diag(sigma_u^2(s)))
    place them in full z (non-style dims = 0)
  - Decode each z -> w_hat, then estimate:
      mu_style = mean(w_hat)
      cov_style = cov(w_hat)

Outputs an NPZ with:
  - style_mu, style_cov, style_indices, n_style_samples
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from promp_vae import PrompVAE

np.seterr(all="ignore")


def load_style_indices(anova_json: Path) -> List[int]:
    data = json.loads(anova_json.read_text(encoding="utf-8"))
    idx = [r["component"] - 1 for r in data["results"] if r["label"] == "style"]
    return sorted(set(idx))


def sample_style_latents(
    mu_all: np.ndarray,
    sigma2_all: np.ndarray,
    style_indices: List[int],
    n_samples: int,
) -> np.ndarray:
    n_latent = mu_all.shape[1]
    z = np.zeros((n_samples, n_latent), dtype=np.float32)
    if not style_indices:
        return z
    mu_s = mu_all[:, style_indices]
    sigma2_s = sigma2_all[:, style_indices]
    zbar = np.mean(mu_s, axis=0)
    var = np.mean(sigma2_s + (mu_s - zbar) ** 2, axis=0)
    std = np.sqrt(np.maximum(var, 1e-12))
    eps = np.random.randn(n_samples, len(style_indices)).astype(np.float32)
    z[:, style_indices] = zbar.astype(np.float32) + eps * std.astype(np.float32)
    return z


def decode_weights(
    ckpt_path: Path,
    z_samples: np.ndarray,
    device: str,
) -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location=device)
    weight_dim = int(ckpt["weight_dim"])
    latent_dim = int(ckpt["latent_dim"])
    hidden_size = int(ckpt["hidden_size"])

    model = PrompVAE(input_dim=2, hidden_size=hidden_size, latent_dim=latent_dim, weight_dim=weight_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if z_samples.shape[1] != latent_dim:
        raise ValueError(f"Latent dim mismatch: z {z_samples.shape[1]} vs model {latent_dim}")

    with torch.no_grad():
        zt = torch.from_numpy(z_samples).to(device)
        w = model.decode(zt).cpu().numpy()
    return w


def main() -> None:
    p = argparse.ArgumentParser(description="Compute digit style_mu/style_cov from digit-domain VAE.")
    p.add_argument("--ckpt", type=str, default="ProMP/promp_vae_digit_all/promp_vae_digit_all.pt")
    p.add_argument("--latents", type=str, default="ProMP/promp_vae_digit_all/promp_vae_digit_all_latents.npz")
    p.add_argument("--anova", type=str, default="ProMP/vae_style_anova_results.json")
    p.add_argument("--n_style_samples", type=int, default=50)
    p.add_argument("--out", type=str, default="ProMP/digit_style_stats_user1.npz")
    p.add_argument("--target_user", type=str, default="1", help="User ID to compute style for (e.g., '1').")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    lat = np.load(Path(args.latents), allow_pickle=True)
    mu_all = lat["mu"]
    sigma2_all = lat["sigma2"] if "sigma2" in lat.files else np.exp(lat["logvar"])
    users_all = lat["users"].astype(str) if "users" in lat.files else None

    if users_all is None:
        raise SystemExit("Latents file must include 'users' when computing user-specific style from all-user training.")

    user_mask = users_all == str(args.target_user)
    if user_mask.sum() == 0:
        raise SystemExit(f"No demos found for target_user={args.target_user} in {args.latents}")
    mu_u = mu_all[user_mask]
    sigma2_u = sigma2_all[user_mask]

    style_indices = load_style_indices(Path(args.anova))
    if not style_indices:
        print("Warning: no 'style' dims in ANOVA; using all latent dims as style.")
        style_indices = list(range(mu_all.shape[1]))

    z_samples = sample_style_latents(mu_u, sigma2_u, style_indices, n_samples=args.n_style_samples)
    w_samples = decode_weights(Path(args.ckpt), z_samples, device=args.device)

    style_mu = np.mean(w_samples, axis=0)
    style_cov = np.cov(w_samples.T) if w_samples.shape[0] > 1 else np.eye(w_samples.shape[1]) * 1e-6

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        style_mu=style_mu,
        style_cov=style_cov,
        style_indices=np.array(style_indices, dtype=int),
        n_style_samples=np.array([args.n_style_samples]),
        ckpt=np.array([str(Path(args.ckpt))], dtype=object),
        latents=np.array([str(Path(args.latents))], dtype=object),
        anova=np.array([str(Path(args.anova))], dtype=object),
    )

    print(f"Saved digit style stats to: {out_path}")
    print(f"Style dims: {len(style_indices)}")


if __name__ == "__main__":
    main()

