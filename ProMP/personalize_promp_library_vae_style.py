#!/usr/bin/env python3
"""
Personalize a generic ProMP library using a style ProMP derived from the ProMP-VAE.

Method (matching the description):

1. Use the trained ProMP-VAE and the latent partitioning results (vae_style_anova)
   to construct a style-only latent vector z_style for a target user:
      - Task-related latent dimensions are set to zero
      - Style-related latent dimensions are set to the user's average latent means
        (computed from encoder outputs μ_z over that user's trajectories)
2. Sample from the style latent distribution (restricted to style dimensions),
   decode each sample through the VAE decoder to obtain ProMP weight samples
   w_style, and estimate a Gaussian style ProMP:
      N(μ_style, Σ_style) in weight space.
3. For each task in the generic ProMP library (μ_task, Σ_task), form the product
   of the task and style Gaussians in weight space:

      Σ_personalized^{-1} = Σ_task^{-1} + Σ_style^{-1}
      μ_personalized = Σ_personalized ( Σ_task^{-1} μ_task + Σ_style^{-1} μ_style )

   yielding a personalized ProMP distribution that combines task structure
   with user-specific style.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# Local import that works when this file is executed as
# `python ProMP/personalize_promp_library_vae_style.py` from the project root.
from promp_vae import PrompVAE

np.seterr(all="ignore")


def load_style_indices(vae_anova_file: Path) -> List[int]:
    """
    Load indices (0-based) of style-related latent dimensions from JSON.

    If no components are labeled strictly as "style", we fall back to also
    including "style_candidate" components. If there are still none, the
    caller can decide how to handle the empty list (we will treat all
    dimensions as style later as a final fallback so the pipeline can run).
    """
    with vae_anova_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    style_indices: List[int] = []
    # First, take components labeled explicitly as "style"
    for r in data["results"]:
        if r["label"] == "style":
            style_indices.append(r["component"] - 1)  # 0-based
    # If none, also include "style_candidate" components
    if not style_indices:
        for r in data["results"]:
            if r["label"] in ("style", "style_candidate"):
                style_indices.append(r["component"] - 1)
    return sorted(set(style_indices))


def load_user_latents(latents_file: Path) -> np.ndarray:
    """
    Load encoder means μ for a single user from *_latents.npz
    and return the per-dimension average μ̄ (shape: (M,)).
    """
    data = np.load(latents_file, allow_pickle=True)
    if "mu" not in data:
        raise ValueError(f"'mu' not found in latent file: {latents_file}")
    mu = data["mu"]  # (N, M)
    if mu.ndim != 2:
        raise ValueError(f"'mu' must be 2D (N, M), got shape {mu.shape}")
    user_mean = np.mean(mu, axis=0)
    return user_mean


def build_style_latent(
    user_mean: np.ndarray,
    style_indices: List[int],
) -> np.ndarray:
    """
    Construct full latent vector using only style-related dimensions.

    Task-related dimensions are set to zero; style dimensions are set to
    the user's average μ̄_m.
    """
    m = user_mean.shape[0]
    z = np.zeros(m, dtype=np.float32)
    if style_indices:
        z[style_indices] = user_mean[style_indices]
    return z


def sample_style_latent_distribution(
    user_mu_all: np.ndarray,
    user_sigma2_all: np.ndarray,
    style_indices: List[int],
    n_samples: int,
) -> np.ndarray:
    """
    Sample latent vectors restricted to style dimensions.

    Implements the paper's latent style distribution:

      z_u^{(s)} ~ N( z̄_u^{(s)}, diag(σ_u^{2(s)}) )

    where:
      z̄_u^{(s)} = (1/|I_u|) Σ_i μ_{z_i}^{(s)}
      σ_u^{2(s)} = (1/|I_u|) Σ_i ( σ_{z_i}^{2(s)} + (μ_{z_i}^{(s)} - z̄_u^{(s)})^2 )

    The returned array has shape (n_samples, M) with non-style dims = 0.
    """
    n_latent = user_mu_all.shape[1]
    z_samples = np.zeros((n_samples, n_latent), dtype=np.float32)

    if not style_indices:
        return z_samples

    mu_s = user_mu_all[:, style_indices]  # (N, |Js|)
    sigma2_s = user_sigma2_all[:, style_indices]  # (N, |Js|)

    zbar_s = np.mean(mu_s, axis=0)  # (|Js|,)
    var_s = np.mean(sigma2_s + (mu_s - zbar_s) ** 2, axis=0)  # (|Js|,)
    std_s = np.sqrt(np.maximum(var_s, 1e-12))

    for i in range(n_samples):
        eps = np.random.randn(len(style_indices))
        z = np.zeros(n_latent, dtype=np.float32)
        z[style_indices] = zbar_s + eps * std_s
        z_samples[i] = z

    return z_samples


def decode_style_promp(
    vae_ckpt: Path,
    latents_file: Path,
    style_indices: List[int],
    n_samples: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode a style ProMP distribution from the VAE.

    Returns:
        mu_style:  (P,)  mean ProMP weights
        cov_style: (P,P) covariance of style ProMP in weight space
    """
    ckpt = torch.load(vae_ckpt, map_location=device)

    # Recover architecture parameters
    weight_dim = int(ckpt["weight_dim"])
    latent_dim = int(ckpt["latent_dim"])
    hidden_size = int(ckpt["hidden_size"])

    model = PrompVAE(
        input_dim=2,  # trajectories are 2D (x,y)
        hidden_size=hidden_size,
        latent_dim=latent_dim,
        weight_dim=weight_dim,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load all user μ for sampling distribution
    data = np.load(latents_file, allow_pickle=True)
    if "mu" not in data:
        raise ValueError(f"'mu' not found in latent file: {latents_file}")
    mu_all = data["mu"]  # (N, M)
    if "sigma2" in data:
        sigma2_all = data["sigma2"]
    else:
        # Backwards compat (older latents saved logvar)
        if "logvar" not in data:
            raise ValueError(f"'sigma2' (or 'logvar') not found in latent file: {latents_file}")
        sigma2_all = np.exp(data["logvar"])
    if mu_all.shape[1] != latent_dim:
        raise ValueError(
            f"Latent dim mismatch: mu has {mu_all.shape[1]}, VAE expects {latent_dim}",
        )
    if sigma2_all.shape != mu_all.shape:
        raise ValueError(f"Shape mismatch: sigma2 {sigma2_all.shape} vs mu {mu_all.shape}")

    # Draw style-only latent samples
    z_samples = sample_style_latent_distribution(
        mu_all,
        sigma2_all,
        style_indices,
        n_samples=n_samples,
    )  # (n_samples, M)

    with torch.no_grad():
        z_tensor = torch.from_numpy(z_samples).to(device)
        w_style_tensor = model.decode(z_tensor)  # (n_samples, P)
        w_style = w_style_tensor.cpu().numpy()

    # Estimate Gaussian in weight space
    mu_style = np.mean(w_style, axis=0)  # (P,)
    if n_samples > 1:
        cov_style = np.cov(w_style.T)  # (P,P)
    else:
        cov_style = np.eye(weight_dim) * 1e-6

    return mu_style, cov_style


def gaussian_product(
    mu_a: np.ndarray,
    cov_a: np.ndarray,
    mu_b: np.ndarray,
    cov_b: np.ndarray,
    jitter: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Product of two Gaussians in weight space:
        Σ^{-1} = Σ_a^{-1} + Σ_b^{-1}
        μ = Σ ( Σ_a^{-1} μ_a + Σ_b^{-1} μ_b )
    """
    dim = mu_a.shape[0]
    eye = np.eye(dim)

    cov_a_reg = cov_a + jitter * eye
    cov_b_reg = cov_b + jitter * eye

    cov_a_inv = np.linalg.inv(cov_a_reg)
    cov_b_inv = np.linalg.inv(cov_b_reg)

    cov_post_inv = cov_a_inv + cov_b_inv
    cov_post = np.linalg.inv(cov_post_inv)

    mu_post = cov_post @ (cov_a_inv @ mu_a + cov_b_inv @ mu_b)
    return mu_post, cov_post


def personalize_library_with_style(
    library_file: Path,
    vae_ckpt: Path,
    latents_file: Path,
    vae_anova_file: Path,
    n_samples: int,
    device: str,
    out_dir: Path,
) -> None:
    """
    Main pipeline:
      - Load generic ProMP library
      - Build style ProMP (μ_style, Σ_style)
      - Combine with each task ProMP via Gaussian product
    """
    print(f"Loading generic ProMP library from {library_file}...")
    lib = np.load(library_file)
    tasks = lib["tasks"].astype(str)

    # Library parameters
    n_basis = int(lib["n_basis"][0])
    width = float(lib["width"][0])
    ridge = float(lib["ridge"][0])

    # Load style indices
    print(f"Loading VAE style ANOVA results from {vae_anova_file}...")
    style_indices = load_style_indices(vae_anova_file)
    print(f"Found {len(style_indices)} candidate style latent dimensions: {[i + 1 for i in style_indices]}")

    # Final fallback: if still empty, treat all latent dims as style so that
    # the personalization pipeline can run (this corresponds to assuming that
    # all latent factors are user-style related).
    if not style_indices:
        print("Warning: No style/style_candidate components in ANOVA; using all latent dimensions as style.")
        latents_data = np.load(latents_file, allow_pickle=True)
        mu_all = latents_data["mu"]
        n_latent = mu_all.shape[1]
        style_indices = list(range(n_latent))

    # Build style ProMP
    print("\nDecoding style ProMP from VAE...")
    mu_style, cov_style = decode_style_promp(
        vae_ckpt=vae_ckpt,
        latents_file=latents_file,
        style_indices=style_indices,
        n_samples=n_samples,
        device=device,
    )
    print(f"Style ProMP mean norm: {np.linalg.norm(mu_style):.4f}")

    # Personalize each task via Gaussian product
    print("\nCombining task and style ProMPs (Gaussian product)...")
    personalized_means: Dict[str, np.ndarray] = {}
    personalized_covs: Dict[str, np.ndarray] = {}

    for task in tasks:
        mu_task = lib[f"task_{task}_mean"]
        cov_task = lib[f"task_{task}_cov"]

        if mu_task.shape[0] != mu_style.shape[0]:
            raise ValueError(
                f"Dimension mismatch for task {task}: "
                f"mu_task dim {mu_task.shape[0]} vs mu_style dim {mu_style.shape[0]}",
            )

        mu_post, cov_post = gaussian_product(mu_task, cov_task, mu_style, cov_style)
        personalized_means[task] = mu_post
        personalized_covs[task] = cov_post
        print(f"  Task {task}: personalized")

    # Save personalized library
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "promp_library_personalized_vae_style.npz"

    out_data: Dict[str, np.ndarray] = {}
    for task in tasks:
        out_data[f"task_{task}_mean"] = personalized_means[task]
        out_data[f"task_{task}_cov"] = personalized_covs[task]

    out_data["tasks"] = tasks
    out_data["n_basis"] = np.array([n_basis])
    out_data["width"] = np.array([width])
    out_data["ridge"] = np.array([ridge])
    out_data["style_mu"] = mu_style
    out_data["style_cov"] = cov_style
    out_data["style_indices"] = np.array(style_indices, dtype=int)
    out_data["vae_ckpt"] = np.array([str(vae_ckpt)], dtype=object)
    out_data["latents_file"] = np.array([str(latents_file)], dtype=object)
    out_data["vae_anova_file"] = np.array([str(vae_anova_file)], dtype=object)
    out_data["n_style_samples"] = np.array([n_samples])

    np.savez(out_file, **out_data)

    print(f"\nPersonalized library saved to: {out_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Personalize ProMP library using style latent integration from ProMP-VAE.",
    )
    parser.add_argument(
        "--library",
        type=str,
        default="User_1_ProMP_library/promp_library.npz",
        help="Path to generic ProMP library (.npz).",
    )
    parser.add_argument(
        "--vae_ckpt",
        type=str,
        default="ProMP/promp_vae/promp_vae_user_1.pt",
        help="Path to trained ProMP-VAE checkpoint (.pt).",
    )
    parser.add_argument(
        "--latents",
        type=str,
        default="ProMP/promp_vae/promp_vae_user_1_latents.npz",
        help="Path to latent means file for target user.",
    )
    parser.add_argument(
        "--vae_anova",
        type=str,
        default="ProMP/vae_style_anova_results.json",
        help="Path to VAE style ANOVA JSON file.",
    )
    parser.add_argument(
        "--n_style_samples",
        type=int,
        default=50,
        help="Number of style latent samples to decode for estimating style ProMP.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="User_1_ProMP_library_personalized_vae_style",
        help="Output directory for personalized ProMP library.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    library_file = Path(args.library).expanduser().resolve()
    vae_ckpt = Path(args.vae_ckpt).expanduser().resolve()
    latents_file = Path(args.latents).expanduser().resolve()
    vae_anova_file = Path(args.vae_anova).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not library_file.exists():
        raise FileNotFoundError(f"Library file not found: {library_file}")
    if not vae_ckpt.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_ckpt}")
    if not latents_file.exists():
        raise FileNotFoundError(f"Latents file not found: {latents_file}")
    if not vae_anova_file.exists():
        raise FileNotFoundError(f"VAE style ANOVA file not found: {vae_anova_file}")

    personalize_library_with_style(
        library_file=library_file,
        vae_ckpt=vae_ckpt,
        latents_file=latents_file,
        vae_anova_file=vae_anova_file,
        n_samples=args.n_style_samples,
        device=args.device,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()

