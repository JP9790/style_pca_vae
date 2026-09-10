#!/usr/bin/env python3
"""
Apply **User 1** alphabet-domain VAE style to **User 1** generic alphabet ProMPs, evaluate on
`user_1_alphabet_library_output`, and mirror the digit VAE evaluation outputs:

- CSVs aligned with `ProMP/vis/alphabet_library_l2_norm_statistics.csv` and
  `alphabet_library_iou_statistics.csv` (Task, Generic_Mean, Generic_Std, Personalized_Mean,
  Personalized_Std, N_Demos — no trailing empty columns).
- Bar charts: generic vs VAE-personalized (L2 and IoU).
- Per-letter trajectory plot: Demo vs Generic ProMP vs VAE-style personalized (with
  mirror-then-rotate-180 display transform, same as digit VAE script). By default the
  **second** User 1 demo (`--plot_demo_index 1`) is used; falls back if only one demo exists.

Prerequisites:
  - Trained all-user alphabet VAE + latents (e.g. `python ProMP/promp_vae_alphabet_all_users.py`)
  - Optional ANOVA JSON for style dims (`ProMP/vae_style_anova_alphabet_all.json`)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist

from promp_vae import PrompVAE

_root_dir = Path(__file__).resolve().parent
_mpl_cache = _root_dir / ".mplcache"
_cache_home = _root_dir / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_home))
os.environ.setdefault("MPLBACKEND", "Agg")

np.seterr(all="ignore")


def mirror_then_rotate_180(traj: np.ndarray) -> np.ndarray:
    """Mirror x, then rotate 180° → net flip y (same as digit VAE viz)."""
    out = traj.copy()
    out[:, 0] = -out[:, 0]
    out[:, 0] = -out[:, 0]
    out[:, 1] = -out[:, 1]
    return out


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
    return sorted(
        {
            r["component"] - 1
            for r in data["results"]
            if r.get("label") in style_labels
        }
    )


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


def discover_letters(generic_lib: Path) -> List[str]:
    letters: List[str] = []
    if not generic_lib.exists():
        return letters
    for d in sorted(generic_lib.iterdir()):
        if d.is_dir() and d.name.startswith("task_") and d.name.endswith("_initialpromp"):
            letters.append(d.name.replace("task_", "").replace("_initialpromp", ""))
    return letters


def main() -> None:
    p = argparse.ArgumentParser(description="User 1 alphabet library + VAE style → CSV + plots.")
    p.add_argument("--root", type=str, default=".")
    p.add_argument(
        "--ckpt",
        type=str,
        default="ProMP/promp_vae_alphabet_all/promp_vae_alphabet_all.pt",
    )
    p.add_argument(
        "--latents",
        type=str,
        default="ProMP/promp_vae_alphabet_all/promp_vae_alphabet_all_latents.npz",
    )
    p.add_argument(
        "--anova",
        type=str,
        default="ProMP/vae_style_anova_alphabet_all.json",
    )
    p.add_argument("--style_user_id", type=str, default="1")
    p.add_argument(
        "--generic_library",
        type=str,
        default="user1_alphabet_generic_library",
    )
    p.add_argument(
        "--trajectories_dir",
        type=str,
        default="user_1_alphabet_library_output",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="ProMP/vis_vae_style_alphabet_user1",
        help="Output folder for CSVs and figures (use ProMP/vis to match legacy path).",
    )
    p.add_argument("--n_style_samples", type=int, default=50)
    p.add_argument("--style_strength", type=float, default=0.05)
    p.add_argument("--n_basis", type=int, default=20)
    p.add_argument("--width", type=float, default=0.05)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--iou_threshold", type=float, default=5.0)
    p.add_argument(
        "--plot_demo_index",
        type=int,
        default=1,
        help="0-based index of User 1 demo used for each letter's detailed plot (default 1 = second trajectory).",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    latents_path = Path(args.latents).expanduser()
    if not latents_path.is_absolute():
        latents_path = root / latents_path
    lat = np.load(latents_path.resolve(), allow_pickle=True)
    mu_all = lat["mu"]
    sigma2_all = lat["sigma2"] if "sigma2" in lat.files else np.exp(lat["logvar"])
    users_all = lat["users"].astype(str)
    suid = str(args.style_user_id).strip()
    mask = users_all == suid
    mu_u = mu_all[mask]
    sigma2_u = sigma2_all[mask]
    if mu_u.shape[0] == 0:
        raise SystemExit(f"No latents for user id {suid!r} in {latents_path}")

    anova_path = Path(args.anova).expanduser()
    if not anova_path.is_absolute():
        anova_path = root / anova_path
    anova_path = anova_path.resolve()
    style_indices: List[int] = []
    if anova_path.exists():
        style_indices = load_style_indices(anova_path)
    if not style_indices:
        print(
            "Warning: using all latent dims for style (no ANOVA style dims).",
            flush=True,
        )
        style_indices = list(range(mu_all.shape[1]))

    ckpt_path = Path(args.ckpt).expanduser()
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    ckpt_path = ckpt_path.resolve()

    print(f"User {suid}: {mu_u.shape[0]} trajectories → VAE style in weight space...", flush=True)
    mu_style, cov_style = user_style_in_weight_space(
        ckpt_path,
        mu_u,
        sigma2_u,
        style_indices,
        n_samples=args.n_style_samples,
        device=args.device,
    )

    generic_root = root / args.generic_library
    traj_root = root / args.trajectories_dir
    letters = discover_letters(generic_root)
    if not letters:
        raise SystemExit(f"No tasks under {generic_root}")

    task_stats: Dict[str, Dict[str, float]] = {}

    for letter in letters:
        promp_file = generic_root / f"task_{letter}_initialpromp" / "promp.npz"
        task_folder = traj_root / f"task_{letter}"
        if not promp_file.exists():
            print(f"  Skip {letter}: missing {promp_file}", flush=True)
            continue

        task_demos: List[Tuple[np.ndarray, np.ndarray]] = []
        if task_folder.exists():
            for csv_path in sorted(task_folder.rglob("*-trajectory.csv")):
                try:
                    time, y = read_trajectory_csv(csv_path)
                    if time.size < 2:
                        continue
                    task_demos.append((time, y))
                except Exception:
                    continue

        if not task_demos:
            print(f"  Skip {letter}: no demos in {task_folder}", flush=True)
            continue

        pdata = np.load(promp_file)
        generic_mean = pdata["mean"]
        generic_cov = pdata["cov"]
        pers_mean = gaussian_fusion(
            generic_mean,
            generic_cov,
            mu_style,
            cov_style,
            style_strength=args.style_strength,
        )

        l2_g: List[float] = []
        l2_p: List[float] = []
        iou_g: List[float] = []
        iou_p: List[float] = []

        n_d = len(task_demos)
        req_plot = int(args.plot_demo_index)
        plot_idx = min(max(0, req_plot), n_d - 1)
        if plot_idx != req_plot:
            print(
                f"  {letter}: plot_demo_index {req_plot} → {plot_idx} ({n_d} demos available)",
                flush=True,
            )

        for idx, (time, y) in enumerate(task_demos):
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

            if idx == plot_idx:
                y_pl = mirror_then_rotate_180(y)
                g_pl = mirror_then_rotate_180(g_traj)
                p_pl = mirror_then_rotate_180(p_traj)
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.plot(y_pl[:, 0], y_pl[:, 1], "k-", label="Demo", linewidth=2)
                ax.plot(g_pl[:, 0], g_pl[:, 1], "b--", label="Generic ProMP", linewidth=2)
                ax.plot(p_pl[:, 0], p_pl[:, 1], "r-", label="VAE-style personalized", linewidth=2)
                ax.set_aspect("equal", "box")
                ax.set_title(
                    f"Letter {letter.upper()} — Demo #{plot_idx + 1} vs Generic vs VAE-style"
                )
                ax.legend()
                ax.grid(True, alpha=0.3, linestyle="--")
                fig_path = out_dir / f"alphabet_task_{letter}_detailed_demo_generic_vae.png"
                plt.tight_layout()
                plt.savefig(fig_path, dpi=200, bbox_inches="tight")
                plt.close(fig)

        task_stats[letter] = {
            "n_demos": len(task_demos),
            "l2_generic_mean": float(np.mean(l2_g)),
            "l2_generic_std": float(np.std(l2_g)),
            "l2_personalized_mean": float(np.mean(l2_p)),
            "l2_personalized_std": float(np.std(l2_p)),
            "iou_generic_mean": float(np.mean(iou_g)),
            "iou_generic_std": float(np.std(iou_g)),
            "iou_personalized_mean": float(np.mean(iou_p)),
            "iou_personalized_std": float(np.std(iou_p)),
        }
        print(
            f"  {letter}: N={len(task_demos)} "
            f"L2 gen={task_stats[letter]['l2_generic_mean']:.2f} vae={task_stats[letter]['l2_personalized_mean']:.2f}",
            flush=True,
        )

    tasks = sorted(task_stats.keys())
    if not tasks:
        raise SystemExit("No task statistics computed.")

    # Bar charts
    x_pos = np.arange(len(tasks))
    wbar = 0.35
    task_labels = [t.upper() for t in tasks]

    fig1, ax1 = plt.subplots(figsize=(14, 8))
    ax1.bar(
        x_pos - wbar / 2,
        [task_stats[t]["l2_generic_mean"] for t in tasks],
        wbar,
        yerr=[task_stats[t]["l2_generic_std"] for t in tasks],
        label="Generic ProMP",
        alpha=0.8,
        color="blue",
        edgecolor="black",
        capsize=5,
    )
    ax1.bar(
        x_pos + wbar / 2,
        [task_stats[t]["l2_personalized_mean"] for t in tasks],
        wbar,
        yerr=[task_stats[t]["l2_personalized_std"] for t in tasks],
        label="VAE-style personalized",
        alpha=0.8,
        color="red",
        edgecolor="black",
        capsize=5,
    )
    ax1.set_xlabel("Letter", fontsize=14, fontweight="bold")
    ax1.set_ylabel("L2-norm (mean ± std)", fontsize=14, fontweight="bold")
    ax1.set_title(
        "Alphabet library: Generic vs VAE-style (User 1)",
        fontsize=16,
        fontweight="bold",
    )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(task_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y", linestyle="--")
    plt.tight_layout()
    p_l2_bar = out_dir / "alphabet_library_l2_norm_comparison_vae_style.png"
    plt.savefig(p_l2_bar, dpi=200, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(14, 8))
    ax2.bar(
        x_pos - wbar / 2,
        [task_stats[t]["iou_generic_mean"] for t in tasks],
        wbar,
        yerr=[task_stats[t]["iou_generic_std"] for t in tasks],
        label="Generic ProMP",
        alpha=0.8,
        color="blue",
        edgecolor="black",
        capsize=5,
    )
    ax2.bar(
        x_pos + wbar / 2,
        [task_stats[t]["iou_personalized_mean"] for t in tasks],
        wbar,
        yerr=[task_stats[t]["iou_personalized_std"] for t in tasks],
        label="VAE-style personalized",
        alpha=0.8,
        color="red",
        edgecolor="black",
        capsize=5,
    )
    ax2.set_xlabel("Letter", fontsize=14, fontweight="bold")
    ax2.set_ylabel("IoU (mean ± std)", fontsize=14, fontweight="bold")
    ax2.set_title(
        f"Alphabet library IoU: Generic vs VAE-style (User 1), threshold={args.iou_threshold}",
        fontsize=16,
        fontweight="bold",
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(task_labels)
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y", linestyle="--")
    plt.tight_layout()
    p_iou_bar = out_dir / "alphabet_library_iou_comparison_vae_style.png"
    plt.savefig(p_iou_bar, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    csv_header = [
        "Task",
        "Generic_Mean",
        "Generic_Std",
        "Personalized_Mean",
        "Personalized_Std",
        "N_Demos",
    ]

    def write_stats_csv(path: Path, key_mean_g: str, key_std_g: str, key_mean_p: str, key_std_p: str) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(csv_header)
            for t in tasks:
                st = task_stats[t]
                w.writerow(
                    [
                        t,
                        repr(float(st[key_mean_g])),
                        f"{st[key_std_g]:.4f}",
                        repr(float(st[key_mean_p])),
                        f"{st[key_std_p]:.4f}",
                        int(st["n_demos"]),
                    ],
                )

    l2_csv = out_dir / "alphabet_library_l2_norm_statistics.csv"
    iou_csv = out_dir / "alphabet_library_iou_statistics.csv"
    write_stats_csv(
        l2_csv,
        "l2_generic_mean",
        "l2_generic_std",
        "l2_personalized_mean",
        "l2_personalized_std",
    )
    write_stats_csv(
        iou_csv,
        "iou_generic_mean",
        "iou_generic_std",
        "iou_personalized_mean",
        "iou_personalized_std",
    )

    print(f"\nCSVs:  {l2_csv}\n       {iou_csv}")
    print(f"Plots: {p_l2_bar}\n       {p_iou_bar}")
    print(f"Per-letter: {out_dir}/alphabet_task_<letter>_detailed_demo_generic_vae.png")


if __name__ == "__main__":
    main()
