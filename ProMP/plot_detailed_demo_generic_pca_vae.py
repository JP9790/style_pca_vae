#!/usr/bin/env python3
"""
Detailed trajectory plots: Demo vs Generic vs PCA-personalized vs VAE-personalized.

- **Digits** (tasks 0–9): one PNG per digit under `--digit_out_dir`.
- **Letters**: one PNG per task folder under `--alphabet_out_dir`.

Plotting is split into two top-level functions so you can change layout/style per domain:
  - `plot_digit_task_demo_generic_pca_vae(...)`
  - `plot_alphabet_task_demo_generic_pca_vae(...)`

Default colors / widths (edit the constants or the plot functions):
  - Demo: black, thinner
  - Generic: red
  - PCA: blue, thicker
  - VAE: orange, thicker

PCA mean is μ_k + `pca_integration_scale` × (aggressiveness × V_s × z̄) from Method 4; default scale is 2 for stronger PCA pull.
Use `--pca_integration_scale 1.0` to match the personalize_* scripts’ offset magnitude exactly.
Digit VAE uses `style_mu` / `style_cov` from `--digit_style_npz` (Gaussian product per task).
Alphabet VAE uses all-user alphabet latents filtered to user 1 + ANOVA style dims (same as
`evaluate_alphabet_library_vae_style_user1.py`).

By default the **third** User 1 demo per task is plotted (`--demo_index 2`).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from promp_vae import PrompVAE

_root_dir = Path(__file__).resolve().parent
_mpl_cache = _root_dir / ".mplcache"
_cache_home = _root_dir / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_home))
os.environ.setdefault("MPLBACKEND", "Agg")

np.seterr(all="ignore")

# ---------------------------------------------------------------------------
# Global visual defaults (tweak here or inside the plot_* functions)
# ---------------------------------------------------------------------------
COLOR_DEMO = "black"
COLOR_GENERIC = "red"
COLOR_PCA = "blue"
COLOR_VAE = "orange"
LW_DEMO = 2.0
LW_GENERIC = 2.0
LW_PCA = 3.5
LW_VAE = 3.5


# -----------------------------------------------------------------------------
# Trajectory I/O + ProMP reconstruction
# -----------------------------------------------------------------------------


def mirror_then_rotate_180(traj: np.ndarray) -> np.ndarray:
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
    return data[:, 0], data[:, 1:3]


def normalize_time(time: np.ndarray) -> np.ndarray:
    t0, t1 = float(time[0]), float(time[-1])
    if t1 <= t0:
        return np.linspace(0.0, 1.0, num=time.shape[0])
    return (time - t0) / (t1 - t0)


def make_rbf_basis(t: np.ndarray, n_basis: int, width: float) -> np.ndarray:
    centers = np.linspace(0.0, 1.0, num=n_basis)
    diff = t[:, None] - centers[None, :]
    basis = np.exp(-0.5 * (diff / width) ** 2)
    basis /= basis.sum(axis=1, keepdims=True)
    return basis


def fit_weights(
    time: np.ndarray, y: np.ndarray, n_basis: int, width: float, ridge: float
) -> np.ndarray:
    t_norm = normalize_time(time)
    phi = make_rbf_basis(t_norm, n_basis, width)
    k, d = phi.shape[1], y.shape[1]
    weights = np.zeros((d, k))
    a = phi.T @ phi + ridge * np.eye(k)
    for dim in range(d):
        weights[dim] = np.linalg.solve(a, phi.T @ y[:, dim])
    return weights.reshape(-1)


def reconstruct_trajectory(
    weights: np.ndarray,
    n_basis: int,
    width: float,
    duration: float,
    dt: float,
) -> np.ndarray:
    d = 2
    w2 = weights.reshape(d, n_basis)
    t = np.arange(0, duration, dt)
    t_norm = t / duration if duration > 0 else np.linspace(0.0, 1.0, len(t))
    centers = np.linspace(0.0, 1.0, num=n_basis)
    diff = t_norm[:, None] - centers[None, :]
    basis = np.exp(-0.5 * (diff / width) ** 2)
    basis /= basis.sum(axis=1, keepdims=True)
    return basis @ w2.T


def interpolate_trajectory(traj: np.ndarray, target_length: int) -> np.ndarray:
    if len(traj) == target_length:
        return traj
    oi = np.linspace(0, len(traj) - 1, len(traj))
    ti = np.linspace(0, len(traj) - 1, target_length)
    return interp1d(oi, traj, axis=0, kind="linear", bounds_error=False, fill_value="extrapolate")(ti)


def list_sorted_demos(task_folder: Path) -> List[Tuple[np.ndarray, np.ndarray]]:
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    if not task_folder.exists():
        return out
    for csv_path in sorted(task_folder.rglob("*-trajectory.csv")):
        try:
            time, y = read_trajectory_csv(csv_path)
            if time.size < 2:
                continue
            out.append((time, y))
        except Exception:
            continue
    return out


def pick_demo(
    demos: List[Tuple[np.ndarray, np.ndarray]], demo_index: int
) -> Tuple[np.ndarray, np.ndarray]:
    if not demos:
        raise ValueError("no demos")
    idx = min(max(0, int(demo_index)), len(demos) - 1)
    return demos[idx]


# -----------------------------------------------------------------------------
# PCA style (User 1, Method 4)
# -----------------------------------------------------------------------------


def pca_style_indices(anova_json: Path) -> List[int]:
    data = json.loads(anova_json.read_text(encoding="utf-8"))
    return sorted(
        r["component"] - 1
        for r in data["results"]
        if r.get("label") in ("style", "style_candidate")
    )


def compute_pca_style_offset(
    pca_npz: Path,
    anova_json: Path,
    user1_style_root: Path,
    n_basis: int,
    width: float,
    ridge: float,
    aggressiveness: float,
    path_predicate: Optional[Callable[[Path], bool]] = None,
) -> np.ndarray:
    pca = np.load(pca_npz, allow_pickle=True)
    pca_mean = pca["mean"]
    pca_eigvecs = pca["eigvecs"]
    idx = pca_style_indices(anova_json)
    if not idx:
        raise SystemExit(f"No style/style_candidate components in {anova_json}")

    weights: List[np.ndarray] = []
    for csv_path in user1_style_root.rglob("*-trajectory.csv"):
        if path_predicate is not None and not path_predicate(csv_path):
            continue
        try:
            time, y = read_trajectory_csv(csv_path)
            if time.size < 2:
                continue
            weights.append(fit_weights(time, y, n_basis, width, ridge))
        except Exception:
            continue
    if not weights:
        raise SystemExit(f"No trajectories for PCA style under {user1_style_root}")

    w_arr = np.vstack(weights)
    V_s = pca_eigvecs[:, idx]
    centered = w_arr - pca_mean
    z_bar = np.mean(centered @ V_s, axis=0)
    return float(aggressiveness) * (V_s @ z_bar)


# -----------------------------------------------------------------------------
# VAE fusion (shared math)
# -----------------------------------------------------------------------------


def gaussian_fusion_mean(
    mu_task: np.ndarray,
    cov_task: np.ndarray,
    mu_style: np.ndarray,
    cov_style: np.ndarray,
    style_strength: float,
    jitter: float = 1e-6,
) -> np.ndarray:
    dim = mu_task.shape[0]
    eye = np.eye(dim)
    ct = cov_task + jitter * eye
    cs = cov_style + jitter * eye
    inv_t = np.linalg.inv(ct)
    inv_s = np.linalg.inv(cs)
    post = np.linalg.inv(inv_t + float(style_strength) * inv_s)
    return post @ (inv_t @ mu_task + float(style_strength) * (inv_s @ mu_style))


def vae_style_indices(anova_json: Path) -> List[int]:
    data = json.loads(anova_json.read_text(encoding="utf-8"))
    idx = [
        r["component"] - 1
        for r in data["results"]
        if r.get("label") in ("style", "style_candidate")
    ]
    return sorted(set(idx))


def sample_style_latents(
    mu_u: np.ndarray, sigma2_u: np.ndarray, style_idx: List[int], n_samples: int
) -> np.ndarray:
    n_latent = mu_u.shape[1]
    z = np.zeros((n_samples, n_latent), dtype=np.float32)
    if not style_idx:
        return z
    mu_s = mu_u[:, style_idx]
    s2 = sigma2_u[:, style_idx]
    zbar = np.mean(mu_s, axis=0)
    var = np.mean(s2 + (mu_s - zbar) ** 2, axis=0)
    std = np.sqrt(np.maximum(var, 1e-12))
    eps = np.random.randn(n_samples, len(style_idx)).astype(np.float32)
    z[:, style_idx] = zbar.astype(np.float32) + eps * std.astype(np.float32)
    return z


def decode_weights_vae(ckpt_path: Path, z: np.ndarray, device: str) -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = PrompVAE(
        input_dim=2,
        hidden_size=int(ckpt["hidden_size"]),
        latent_dim=int(ckpt["latent_dim"]),
        weight_dim=int(ckpt["weight_dim"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        w = model.decode(torch.from_numpy(z).to(device)).cpu().numpy()
    return w


def alphabet_user_vae_style_stats(
    ckpt: Path,
    latents_npz: Path,
    anova_json: Path,
    user_id: str,
    n_samples: int,
    device: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    lat = np.load(latents_npz, allow_pickle=True)
    mu_all = lat["mu"]
    s2 = lat["sigma2"] if "sigma2" in lat.files else np.exp(lat["logvar"])
    users = lat["users"].astype(str)
    mask = users == str(user_id).strip()
    mu_u = mu_all[mask]
    sigma_u = s2[mask]
    if mu_u.shape[0] == 0:
        raise SystemExit(f"No latents for user {user_id!r} in {latents_npz}")

    anova_path = Path(anova_json)
    si = vae_style_indices(anova_path) if anova_path.exists() else []
    if not si:
        si = list(range(mu_u.shape[1]))
    z = sample_style_latents(mu_u, sigma_u, si, n_samples)
    w = decode_weights_vae(ckpt, z, device)
    mu_s = np.mean(w, axis=0)
    cov_s = np.cov(w.T) if w.shape[0] > 1 else np.eye(w.shape[1]) * 1e-6
    return mu_s, cov_s


# -----------------------------------------------------------------------------
# Plot functions — edit these independently
# -----------------------------------------------------------------------------


def plot_digit_task_demo_generic_pca_vae(
    digit_label: str,
    y_demo: np.ndarray,
    traj_generic: np.ndarray,
    traj_pca: np.ndarray,
    traj_vae: np.ndarray,
    out_path: Path,
    *,
    apply_display_transform: bool = True,
) -> None:
    """
    Save one figure for a single digit. Modify this function to change digit-only layout,
    labels, colors, or legend.
    """
    if apply_display_transform:
        y_demo = mirror_then_rotate_180(y_demo)
        traj_generic = mirror_then_rotate_180(traj_generic)
        traj_pca = mirror_then_rotate_180(traj_pca)
        traj_vae = mirror_then_rotate_180(traj_vae)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot(y_demo[:, 0], y_demo[:, 1], "-", color=COLOR_DEMO, linewidth=LW_DEMO, label="Demo")
    ax.plot(
        traj_generic[:, 0],
        traj_generic[:, 1],
        "-",
        color=COLOR_GENERIC,
        linewidth=LW_GENERIC,
        label="Generic",
    )
    ax.plot(
        traj_pca[:, 0],
        traj_pca[:, 1],
        "-",
        color=COLOR_PCA,
        linewidth=LW_PCA,
        label="PCA",
    )
    ax.plot(
        traj_vae[:, 0],
        traj_vae[:, 1],
        "-",
        color=COLOR_VAE,
        linewidth=LW_VAE,
        label="VAE",
    )
    ax.set_aspect("equal", "box")
    ax.set_title(f"Digit {digit_label} — Demo vs Generic vs PCA vs VAE")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_alphabet_task_demo_generic_pca_vae(
    letter: str,
    y_demo: np.ndarray,
    traj_generic: np.ndarray,
    traj_pca: np.ndarray,
    traj_vae: np.ndarray,
    out_path: Path,
    *,
    apply_display_transform: bool = True,
) -> None:
    """
    Save one figure for a single letter. Modify this function to change letter-only layout.
    """
    if apply_display_transform:
        y_demo = mirror_then_rotate_180(y_demo)
        traj_generic = mirror_then_rotate_180(traj_generic)
        traj_pca = mirror_then_rotate_180(traj_pca)
        traj_vae = mirror_then_rotate_180(traj_vae)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot(y_demo[:, 0], y_demo[:, 1], "-", color=COLOR_DEMO, linewidth=LW_DEMO, label="Demo")
    ax.plot(
        traj_generic[:, 0],
        traj_generic[:, 1],
        "-",
        color=COLOR_GENERIC,
        linewidth=LW_GENERIC,
        label="Generic",
    )
    ax.plot(
        traj_pca[:, 0],
        traj_pca[:, 1],
        "-",
        color=COLOR_PCA,
        linewidth=LW_PCA,
        label="PCA",
    )
    ax.plot(
        traj_vae[:, 0],
        traj_vae[:, 1],
        "-",
        color=COLOR_VAE,
        linewidth=LW_VAE,
        label="VAE",
    )
    ax.set_aspect("equal", "box")
    ax.set_title(f"Letter {letter.upper()} — Demo vs Generic vs PCA vs VAE")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Pipelines
# -----------------------------------------------------------------------------


def run_digits(args: argparse.Namespace, root: Path) -> None:
    path_pred: Optional[Callable[[Path], bool]] = None
    if args.digit_style_only_digit_tasks:

        def _pred(p: Path) -> bool:
            s = p.as_posix()
            return any(f"/task_{i}/" in s for i in range(10))

        path_pred = _pred

    pca_off = compute_pca_style_offset(
        root / args.digit_pca_npz,
        root / args.digit_anova_json,
        root / args.user1_style_root,
        args.n_basis,
        args.width,
        args.ridge,
        args.digit_pca_aggressiveness,
        path_predicate=path_pred,
    )

    style_npz = root / args.digit_vae_style_npz
    if not style_npz.exists():
        raise SystemExit(f"Digit VAE style NPZ not found: {style_npz}")
    st = np.load(style_npz, allow_pickle=True)
    style_mu = st["style_mu"]
    style_cov = st["style_cov"]

    lib = root / args.digit_generic_library
    demos_root = root / args.digit_demos_dir
    out_dir = root / args.digit_out_dir

    for d in range(10):
        ts = str(d)
        promp_f = lib / f"task_{ts}_initialpromp" / "promp.npz"
        if not promp_f.exists():
            print(f"[digit] skip {d}: no {promp_f}")
            continue
        demos = list_sorted_demos(demos_root / f"task_{ts}")
        if not demos:
            print(f"[digit] skip {d}: no demos")
            continue
        try:
            _, y = pick_demo(demos, args.demo_index)
        except ValueError:
            continue

        pd = np.load(promp_f)
        g_mean, g_cov = pd["mean"], pd["cov"]
        p_mean = g_mean + pca_off * float(args.pca_integration_scale)
        v_mean = gaussian_fusion_mean(
            g_mean, g_cov, style_mu, style_cov, args.digit_vae_style_strength
        )

        duration = len(y) * args.dt
        tg = reconstruct_trajectory(g_mean, args.n_basis, args.width, duration, args.dt)
        tp = reconstruct_trajectory(p_mean, args.n_basis, args.width, duration, args.dt)
        tv = reconstruct_trajectory(v_mean, args.n_basis, args.width, duration, args.dt)
        if len(tg) != len(y):
            tg = interpolate_trajectory(tg, len(y))
        if len(tp) != len(y):
            tp = interpolate_trajectory(tp, len(y))
        if len(tv) != len(y):
            tv = interpolate_trajectory(tv, len(y))

        plot_digit_task_demo_generic_pca_vae(
            ts,
            y,
            tg,
            tp,
            tv,
            out_dir / f"digit_task_{ts}_detailed_demo_generic_pca_vae.png",
            apply_display_transform=not args.no_display_transform,
        )
        print(f"[digit] wrote task {d}")


def run_alphabet(args: argparse.Namespace, root: Path) -> None:
    pca_off = compute_pca_style_offset(
        root / args.alphabet_pca_npz,
        root / args.alphabet_anova_json,
        root / args.user1_style_root,
        args.n_basis,
        args.width,
        args.ridge,
        args.alphabet_pca_aggressiveness,
        path_predicate=None,
    )

    ckpt = root / args.alphabet_vae_ckpt
    lat = root / args.alphabet_vae_latents
    ajson = root / args.alphabet_vae_anova
    mu_s, cov_s = alphabet_user_vae_style_stats(
        ckpt,
        lat,
        ajson,
        str(args.alphabet_vae_user_id),
        args.alphabet_vae_n_style_samples,
        args.device,
        args.seed,
    )

    lib = root / args.alphabet_generic_library
    demos_root = root / args.alphabet_demos_dir
    out_dir = root / args.alphabet_out_dir

    letters = sorted(
        d.name.replace("task_", "").replace("_initialpromp", "")
        for d in lib.iterdir()
        if d.is_dir() and d.name.endswith("_initialpromp")
    )

    for letter in letters:
        promp_f = lib / f"task_{letter}_initialpromp" / "promp.npz"
        if not promp_f.exists():
            continue
        demos = list_sorted_demos(demos_root / f"task_{letter}")
        if not demos:
            print(f"[letter] skip {letter}: no demos")
            continue
        _, y = pick_demo(demos, args.demo_index)

        pd = np.load(promp_f)
        g_mean, g_cov = pd["mean"], pd["cov"]
        p_mean = g_mean + pca_off * float(args.pca_integration_scale)
        v_mean = gaussian_fusion_mean(
            g_mean, g_cov, mu_s, cov_s, args.alphabet_vae_style_strength
        )

        duration = len(y) * args.dt
        tg = reconstruct_trajectory(g_mean, args.n_basis, args.width, duration, args.dt)
        tp = reconstruct_trajectory(p_mean, args.n_basis, args.width, duration, args.dt)
        tv = reconstruct_trajectory(v_mean, args.n_basis, args.width, duration, args.dt)
        if len(tg) != len(y):
            tg = interpolate_trajectory(tg, len(y))
        if len(tp) != len(y):
            tp = interpolate_trajectory(tp, len(y))
        if len(tv) != len(y):
            tv = interpolate_trajectory(tv, len(y))

        plot_alphabet_task_demo_generic_pca_vae(
            letter,
            y,
            tg,
            tp,
            tv,
            out_dir / f"alphabet_task_{letter}_detailed_demo_generic_pca_vae.png",
            apply_display_transform=not args.no_display_transform,
        )
        print(f"[letter] wrote {letter}")


def main() -> None:
    p = argparse.ArgumentParser(description="Demo vs Generic vs PCA vs VAE detailed plots.")
    p.add_argument("--root", type=str, default=".")
    p.add_argument("--run_digits", action="store_true", help="Generate digit PNGs.")
    p.add_argument("--run_alphabet", action="store_true", help="Generate alphabet PNGs.")
    p.add_argument(
        "--demo_index",
        type=int,
        default=2,
        help="Which User 1 demo (0-based) after sorting CSV paths (default 2 = third).",
    )
    p.add_argument("--n_basis", type=int, default=20)
    p.add_argument("--width", type=float, default=0.05)
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--no_display_transform",
        action="store_true",
        help="Disable mirror+rotate180 on plotted trajectories.",
    )
    p.add_argument(
        "--user1_style_root",
        type=str,
        default="output",
        help="User 1 trajectories for PCA style (same as personalize_* scripts).",
    )
    p.add_argument(
        "--pca_integration_scale",
        type=float,
        default=2.0,
        help=(
            "Multiplies the PCA style offset added to generic mean (stronger > 1). "
            "Use 1.0 to match magnitude from aggressiveness alone (same as personalize_* without extra scale)."
        ),
    )

    # Digit
    p.add_argument("--digit_pca_npz", type=str, default="ProMP/promp_pca_results.npz")
    p.add_argument("--digit_anova_json", type=str, default="ProMP/style_anova_results.json")
    p.add_argument("--digit_pca_aggressiveness", type=float, default=5.0)
    p.add_argument(
        "--digit_style_only_digit_tasks",
        action="store_true",
        help="Restrict PCA style trajectories to paths containing task_0..task_9.",
    )
    p.add_argument("--digit_generic_library", type=str, default="user1_digit_generic_library")
    p.add_argument("--digit_demos_dir", type=str, default="user_1_digit_library_output")
    p.add_argument(
        "--digit_vae_style_npz",
        type=str,
        default="ProMP/digit_style_stats_user1.npz",
        help="NPZ with style_mu, style_cov (e.g. from compute_digit_style_stats_from_vae.py).",
    )
    p.add_argument("--digit_vae_style_strength", type=float, default=1.0)
    p.add_argument(
        "--digit_out_dir",
        type=str,
        default="ProMP/vis_detailed_digit_generic_pca_vae",
    )

    # Alphabet
    p.add_argument("--alphabet_pca_npz", type=str, default="ProMP/promp_pca_results.npz")
    p.add_argument("--alphabet_anova_json", type=str, default="ProMP/style_identification_results.json")
    p.add_argument("--alphabet_pca_aggressiveness", type=float, default=8.0)
    p.add_argument("--alphabet_generic_library", type=str, default="user1_alphabet_generic_library")
    p.add_argument("--alphabet_demos_dir", type=str, default="user_1_alphabet_library_output")
    p.add_argument("--alphabet_vae_ckpt", type=str, default="ProMP/promp_vae_alphabet_all/promp_vae_alphabet_all.pt")
    p.add_argument(
        "--alphabet_vae_latents",
        type=str,
        default="ProMP/promp_vae_alphabet_all/promp_vae_alphabet_all_latents.npz",
    )
    p.add_argument(
        "--alphabet_vae_anova",
        type=str,
        default="ProMP/vae_style_anova_alphabet_all.json",
    )
    p.add_argument("--alphabet_vae_user_id", type=str, default="1")
    p.add_argument("--alphabet_vae_n_style_samples", type=int, default=50)
    p.add_argument("--alphabet_vae_style_strength", type=float, default=0.05)
    p.add_argument(
        "--alphabet_out_dir",
        type=str,
        default="ProMP/vis_detailed_alphabet_generic_pca_vae",
    )

    args = p.parse_args()
    root = Path(args.root).expanduser().resolve()

    if not args.run_digits and not args.run_alphabet:
        args.run_digits = True
        args.run_alphabet = True

    if args.run_digits:
        run_digits(args, root)
    if args.run_alphabet:
        run_alphabet(args, root)


if __name__ == "__main__":
    main()
