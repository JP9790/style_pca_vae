#!/usr/bin/env python3
"""
Per-user VAE style → personalize that user's digit ProMPs → evaluate all digits.

Pipeline:
  1. Load a digit-domain VAE trained on all users (checkpoint + latents NPZ with users, tasks).
  2. Load two-factor ANOVA JSON (task+user) to get style latent indices J_s (label == "style").
  3. For each user u with demos in the latents file:
       - Build user-specific style in weight space: sample z ~ N(z̄_u, diag(σ_u^2)) on J_s,
         decode N_s times → μ_style, Σ_style (same construction as compute_digit_style_stats_from_vae).
       - For each digit d ∈ {0..9}:
           - Load generic (μ_k, Σ_k) from user{u}_digit_generic_library/task_{d}_initialpromp/promp.npz
           - Gaussian fusion with μ_style, Σ_style (optional style_strength scaling)
           - Compare each demo in user_{u}_digit_library_output/task_{d} to generic vs personalized
             trajectories (L2, IoU).
  4. Write one CSV row per (user, digit) with mean/std metrics and n_demos.

Trajectory folders follow this repo:
  - Generic library:  {root}/user{u}_digit_generic_library/task_{d}_initialpromp/promp.npz
  - User demos:       {root}/user_{u}_digit_library_output/task_{d}/*-trajectory.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist

from promp_vae import PrompVAE

np.seterr(all="ignore")


def read_trajectory_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", skip_header=1).astype(float)
    if data.ndim == 1 or data.shape[1] < 3:
        raise ValueError(f"Expected CSV columns time,x,y in {path}")
    data = data[np.isfinite(data).all(axis=1)]
    time = data[:, 0]
    y = data[:, 1:3]
    return time, y


def reconstruct_trajectory(
    weights: np.ndarray,
    n_basis: int,
    width: float,
    duration: float,
    dt: float,
) -> np.ndarray:
    d = 2
    weights_2d = weights.reshape(d, n_basis)
    t = np.arange(0, duration, dt)
    t_norm = t / duration if duration > 0 else np.linspace(0.0, 1.0, len(t))
    centers = np.linspace(0.0, 1.0, num=n_basis)
    diff = t_norm[:, None] - centers[None, :]
    basis = np.exp(-0.5 * (diff / width) ** 2)
    basis /= basis.sum(axis=1, keepdims=True)
    return basis @ weights_2d.T


def interpolate_trajectory(traj: np.ndarray, target_length: int) -> np.ndarray:
    if len(traj) == target_length:
        return traj
    original_indices = np.linspace(0, len(traj) - 1, len(traj))
    target_indices = np.linspace(0, len(traj) - 1, target_length)
    interp_func = interp1d(
        original_indices,
        traj,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    return interp_func(target_indices)


def calculate_l2_norm(traj1: np.ndarray, traj2: np.ndarray) -> float:
    max_length = max(len(traj1), len(traj2))
    t1 = interpolate_trajectory(traj1, max_length)
    t2 = interpolate_trajectory(traj2, max_length)
    return float(np.sqrt(np.sum((t1 - t2) ** 2)))


def calculate_trajectory_iou(traj1: np.ndarray, traj2: np.ndarray, threshold: float) -> float:
    max_length = max(len(traj1), len(traj2))
    t1 = interpolate_trajectory(traj1, max_length)
    t2 = interpolate_trajectory(traj2, max_length)
    distances = cdist(t1, t2)
    m1 = (np.min(distances, axis=1) <= threshold).sum()
    m2 = (np.min(distances, axis=0) <= threshold).sum()
    intersection = (m1 + m2) / 2.0
    union = max_length
    if union == 0:
        return 0.0
    return float(intersection / union)


def load_style_indices(anova_json: Path) -> List[int]:
    data = json.loads(anova_json.read_text(encoding="utf-8"))
    style_labels = {"style", "style_candidate"}
    idx = [
        r["component"] - 1
        for r in data["results"]
        if r.get("label") in style_labels
    ]
    return sorted(set(idx))


def sample_style_latents(
    mu_u: np.ndarray,
    sigma2_u: np.ndarray,
    style_indices: List[int],
    n_samples: int,
) -> np.ndarray:
    n_latent = mu_u.shape[1]
    z = np.zeros((n_samples, n_latent), dtype=np.float32)
    if not style_indices:
        return z
    mu_s = mu_u[:, style_indices]
    sigma2_s = sigma2_u[:, style_indices]
    zbar = np.mean(mu_s, axis=0)
    var = np.mean(sigma2_s + (mu_s - zbar) ** 2, axis=0)
    std = np.sqrt(np.maximum(var, 1e-12))
    eps = np.random.randn(n_samples, len(style_indices)).astype(np.float32)
    z[:, style_indices] = zbar.astype(np.float32) + eps * std.astype(np.float32)
    return z


def decode_weights(ckpt_path: Path, z_samples: np.ndarray, device: str) -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location=device)
    weight_dim = int(ckpt["weight_dim"])
    latent_dim = int(ckpt["latent_dim"])
    hidden_size = int(ckpt["hidden_size"])
    model = PrompVAE(
        input_dim=2,
        hidden_size=hidden_size,
        latent_dim=latent_dim,
        weight_dim=weight_dim,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    if z_samples.shape[1] != latent_dim:
        raise ValueError(f"Latent dim mismatch: z {z_samples.shape[1]} vs model {latent_dim}")
    with torch.no_grad():
        zt = torch.from_numpy(z_samples).to(device)
        w = model.decode(zt).cpu().numpy()
    return w


def gaussian_fusion(
    mu_task: np.ndarray,
    cov_task: np.ndarray,
    mu_style: np.ndarray,
    cov_style: np.ndarray,
    style_strength: float,
    jitter: float = 1e-6,
) -> np.ndarray:
    dim = mu_task.shape[0]
    eye = np.eye(dim)
    cov_t = cov_task + jitter * eye
    cov_s = cov_style + jitter * eye
    inv_t = np.linalg.inv(cov_t)
    inv_s = np.linalg.inv(cov_s)
    post_inv = inv_t + float(style_strength) * inv_s
    post = np.linalg.inv(post_inv)
    return post @ (inv_t @ mu_task + float(style_strength) * (inv_s @ mu_style))


def user_style_in_weight_space(
    ckpt: Path,
    mu_u: np.ndarray,
    sigma2_u: np.ndarray,
    style_indices: List[int],
    n_samples: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    z = sample_style_latents(mu_u, sigma2_u, style_indices, n_samples)
    w = decode_weights(ckpt, z, device=device)
    mu_style = np.mean(w, axis=0)
    cov_style = np.cov(w.T) if w.shape[0] > 1 else np.eye(w.shape[1]) * 1e-6
    return mu_style, cov_style


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate digit VAE personalization for all users (CSV).")
    p.add_argument("--root", type=str, default=".", help="Project root.")
    p.add_argument("--ckpt", type=str, default="ProMP/promp_vae_digit_all/promp_vae_digit_all.pt")
    p.add_argument("--latents", type=str, default="ProMP/promp_vae_digit_all/promp_vae_digit_all_latents.npz")
    p.add_argument(
        "--anova",
        type=str,
        default="ProMP/vae_style_anova_digit_results.json",
        help="JSON with per-component labels; dims labeled style/style_candidate are used.",
    )
    p.add_argument("--n_style_samples", type=int, default=50)
    p.add_argument("--style_strength", type=float, default=0.05)
    p.add_argument("--n_basis", type=int, default=20)
    p.add_argument("--width", type=float, default=0.05)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--iou_threshold", type=float, default=5.0)
    p.add_argument("--out_csv", type=str, default="ProMP/all_users_digit_vae_l2_iou.csv")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for style latent sampling.")
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    np.random.seed(args.seed)

    lat = np.load(Path(args.latents), allow_pickle=True)
    mu_all = lat["mu"]
    sigma2_all = lat["sigma2"] if "sigma2" in lat.files else np.exp(lat["logvar"])
    users_all = lat["users"].astype(str)

    style_indices = load_style_indices(Path(args.anova))
    if not style_indices:
        print("Warning: no 'style' dims in ANOVA; using all latent dims as style.")
        style_indices = list(range(mu_all.shape[1]))

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    unique_users = sorted({u for u in users_all.tolist() if u and u != "unknown"})

    out_path = Path(args.out_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []

    for uid in unique_users:
        mask = users_all == uid
        mu_u = mu_all[mask]
        sigma2_u = sigma2_all[mask]
        if mu_u.shape[0] == 0:
            continue

        print(f"User {uid}: {mu_u.shape[0]} VAE-encoded demos → computing style...")
        mu_style, cov_style = user_style_in_weight_space(
            ckpt_path,
            mu_u,
            sigma2_u,
            style_indices,
            n_samples=args.n_style_samples,
            device=args.device,
        )

        generic_root = root / f"user{uid}_digit_generic_library"
        traj_root = root / f"user_{uid}_digit_library_output"

        for d in range(10):
            ds = str(d)
            promp_file = generic_root / f"task_{ds}_initialpromp" / "promp.npz"
            task_folder = traj_root / f"task_{ds}"

            if not promp_file.exists():
                rows.append(
                    {
                        "user_id": uid,
                        "digit": ds,
                        "n_demos": 0,
                        "note": "missing_generic_promp",
                        "l2_generic_mean": "",
                        "l2_generic_std": "",
                        "l2_personalized_mean": "",
                        "l2_personalized_std": "",
                        "iou_generic_mean": "",
                        "iou_generic_std": "",
                        "iou_personalized_mean": "",
                        "iou_personalized_std": "",
                    }
                )
                continue

            pdata = np.load(promp_file)
            generic_mean = pdata["mean"]
            generic_cov = pdata["cov"]

            demos: List[Tuple[np.ndarray, np.ndarray]] = []
            if task_folder.exists():
                for csv_path in sorted(task_folder.rglob("*-trajectory.csv")):
                    try:
                        time, y = read_trajectory_csv(csv_path)
                        if time.size < 2:
                            continue
                        demos.append((time, y))
                    except Exception:
                        continue

            if not demos:
                rows.append(
                    {
                        "user_id": uid,
                        "digit": ds,
                        "n_demos": 0,
                        "note": "no_demos",
                        "l2_generic_mean": "",
                        "l2_generic_std": "",
                        "l2_personalized_mean": "",
                        "l2_personalized_std": "",
                        "iou_generic_mean": "",
                        "iou_generic_std": "",
                        "iou_personalized_mean": "",
                        "iou_personalized_std": "",
                    }
                )
                continue

            l2_g: List[float] = []
            l2_p: List[float] = []
            iou_g: List[float] = []
            iou_p: List[float] = []

            pers_mean = gaussian_fusion(
                generic_mean,
                generic_cov,
                mu_style,
                cov_style,
                style_strength=args.style_strength,
            )

            for time, y in demos:
                duration = len(y) * args.dt
                g_traj = reconstruct_trajectory(
                    generic_mean, args.n_basis, args.width, duration, args.dt
                )
                p_traj = reconstruct_trajectory(
                    pers_mean, args.n_basis, args.width, duration, args.dt
                )
                if len(g_traj) != len(y):
                    g_traj = interpolate_trajectory(g_traj, len(y))
                if len(p_traj) != len(y):
                    p_traj = interpolate_trajectory(p_traj, len(y))

                l2_g.append(calculate_l2_norm(y, g_traj))
                l2_p.append(calculate_l2_norm(y, p_traj))
                iou_g.append(calculate_trajectory_iou(y, g_traj, args.iou_threshold))
                iou_p.append(calculate_trajectory_iou(y, p_traj, args.iou_threshold))

            rows.append(
                {
                    "user_id": uid,
                    "digit": ds,
                    "n_demos": len(demos),
                    "note": "",
                    "l2_generic_mean": f"{float(np.mean(l2_g)):.6f}",
                    "l2_generic_std": f"{float(np.std(l2_g)):.6f}",
                    "l2_personalized_mean": f"{float(np.mean(l2_p)):.6f}",
                    "l2_personalized_std": f"{float(np.std(l2_p)):.6f}",
                    "iou_generic_mean": f"{float(np.mean(iou_g)):.6f}",
                    "iou_generic_std": f"{float(np.std(iou_g)):.6f}",
                    "iou_personalized_mean": f"{float(np.mean(iou_p)):.6f}",
                    "iou_personalized_std": f"{float(np.std(iou_p)):.6f}",
                }
            )

        print(f"  User {uid}: evaluated digits 0–9")

    fieldnames = [
        "user_id",
        "digit",
        "n_demos",
        "note",
        "l2_generic_mean",
        "l2_generic_std",
        "l2_personalized_mean",
        "l2_personalized_std",
        "iou_generic_mean",
        "iou_generic_std",
        "iou_personalized_mean",
        "iou_personalized_std",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
