#!/usr/bin/env python3
"""
All-user alphabet evaluation with VAE style (aligned with all_users_alphabet_evaluation.csv).

Mirrors `evaluate_all_users_alphabet_libraries.py` (PCA path) structurally:
  - Generic ProMPs: `user1_alphabet_generic_library/task_<letter>_initialpromp/promp.npz`
  - For each user U: build U's style in weight space from the all-user alphabet VAE latents
    (encoder μ, σ² on style dims from ANOVA JSON).
  - Gaussian-fuse generic (μ, Σ) with that style, then score against **U's own** alphabet demos
    under `output/User_U/task_<letter>/` (same as the PCA script's `--alphabet_trajectories`).

Output CSV matches **ProMP/all_users_alphabet_evaluation.csv**:
  - Columns: User, Task, L2_Generic, L2_Personalized, IoU_Generic, IoU_Personalized
  - Row count and order: identical to the alignment file (default: that CSV).

Default output path remains `ProMP/user1_alphabet_vae_style_transfer_l2_iou.csv` for compatibility.

Train the VAE first: `python ProMP/promp_vae_alphabet_all_users.py`
Optional ANOVA for style dims: `python ProMP/vae_style_anova.py ... --out_json ProMP/vae_style_anova_alphabet_all.json`
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


def load_alignment_rows(align_path: Path) -> List[Tuple[int, str]]:
    """(User, Task) in file order; Task is single-letter string (e.g. 'a')."""
    rows: List[Tuple[int, str]] = []
    with align_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"No header in {align_path}")
        for line in reader:
            u_raw = line.get("User", line.get("user", ""))
            t_raw = line.get("Task", line.get("task", ""))
            if u_raw is None or t_raw is None or str(u_raw).strip() == "":
                continue
            uid = int(float(str(u_raw).strip()))
            task = str(t_raw).strip()
            rows.append((uid, task))
    return rows


def fmt_metric(x: float) -> str:
    """Human-readable float similar to all_users_alphabet_evaluation.csv."""
    return repr(float(x))


def collect_demos_user_task(output_root: Path, uid: int, letter: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    task_folder = output_root / f"User_{uid}" / f"task_{letter}"
    demos: List[Tuple[np.ndarray, np.ndarray]] = []
    if not task_folder.exists():
        return demos
    for csv_path in sorted(task_folder.rglob("*-trajectory.csv")):
        try:
            time, y = read_trajectory_csv(csv_path)
            if time.size < 2:
                continue
            demos.append((time, y))
        except Exception:
            continue
    return demos


def main() -> None:
    p = argparse.ArgumentParser(
        description="VAE alphabet eval; CSV aligned with all_users_alphabet_evaluation.csv."
    )
    p.add_argument("--root", type=str, default=".", help="Project root.")
    p.add_argument(
        "--align_csv",
        type=str,
        default="ProMP/all_users_alphabet_evaluation.csv",
        help="Reference CSV: only User and Task columns define row order and count.",
    )
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
        help="Dims labeled style/style_candidate; if missing, all latents used for style.",
    )
    p.add_argument(
        "--generic_library",
        type=str,
        default="user1_alphabet_generic_library",
        help="Directory with task_<letter>_initialpromp/promp.npz",
    )
    p.add_argument(
        "--alphabet_trajectories",
        type=str,
        default="output",
        help="Root containing User_<id>/task_<letter>/ (same as PCA all-users alphabet eval).",
    )
    p.add_argument("--n_style_samples", type=int, default=50)
    p.add_argument("--style_strength", type=float, default=0.05)
    p.add_argument("--n_basis", type=int, default=20)
    p.add_argument("--width", type=float, default=0.05)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--iou_threshold", type=float, default=5.0)
    p.add_argument(
        "--out_csv",
        type=str,
        default="ProMP/user1_alphabet_vae_style_transfer_l2_iou.csv",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    np.random.seed(args.seed)

    align_path = Path(args.align_csv).expanduser()
    if not align_path.is_absolute():
        align_path = root / align_path
    align_path = align_path.resolve()
    if not align_path.exists():
        raise SystemExit(f"Alignment CSV not found: {align_path}")

    align_rows = load_alignment_rows(align_path)
    if not align_rows:
        raise SystemExit(f"No rows read from {align_path}")

    latents_path = Path(args.latents).expanduser()
    if not latents_path.is_absolute():
        latents_path = root / latents_path
    lat = np.load(latents_path.resolve(), allow_pickle=True)
    mu_all = lat["mu"]
    sigma2_all = lat["sigma2"] if "sigma2" in lat.files else np.exp(lat["logvar"])
    users_all = lat["users"].astype(str)

    anova_path = Path(args.anova).expanduser()
    if not anova_path.is_absolute():
        anova_path = root / anova_path
    anova_path = anova_path.resolve()
    style_indices: List[int] = []
    if anova_path.exists():
        style_indices = load_style_indices(anova_path)
    if not style_indices:
        print(
            "Warning: no style/style_candidate dims in ANOVA (or missing JSON); "
            "using all latent dimensions for style sampling.",
            flush=True,
        )
        style_indices = list(range(mu_all.shape[1]))

    ckpt_path = Path(args.ckpt).expanduser()
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    ckpt_path = ckpt_path.resolve()

    generic_root = root / args.generic_library
    output_root = Path(args.alphabet_trajectories).expanduser()
    if not output_root.is_absolute():
        output_root = root / output_root
    output_root = output_root.resolve()

    out_path = Path(args.out_csv).expanduser()
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Cache VAE style per user (string id matching latents)
    style_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    warned_no_latent: set[str] = set()

    def get_style(uid: int) -> Tuple[np.ndarray, np.ndarray] | None:
        su = str(int(uid))
        if su in style_cache:
            return style_cache[su]
        mask = users_all == su
        mu_u = mu_all[mask]
        sigma2_u = sigma2_all[mask]
        if mu_u.shape[0] == 0:
            if su not in warned_no_latent:
                print(f"Warning: no VAE latents for user {su}; metrics will be blank.", flush=True)
                warned_no_latent.add(su)
            return None
        print(f"User {su}: {mu_u.shape[0]} trajectories → VAE style...", flush=True)
        mu_style, cov_style = user_style_in_weight_space(
            ckpt_path,
            mu_u,
            sigma2_u,
            style_indices,
            n_samples=args.n_style_samples,
            device=args.device,
        )
        style_cache[su] = (mu_style, cov_style)
        return mu_style, cov_style

    out_rows: List[Dict[str, str]] = []
    fieldnames = ["User", "Task", "L2_Generic", "L2_Personalized", "IoU_Generic", "IoU_Personalized"]

    for uid, letter in align_rows:
        row_out: Dict[str, str] = {
            "User": str(int(uid)),
            "Task": letter,
            "L2_Generic": "",
            "L2_Personalized": "",
            "IoU_Generic": "",
            "IoU_Personalized": "",
        }

        promp_file = generic_root / f"task_{letter}_initialpromp" / "promp.npz"
        if not promp_file.exists():
            print(f"Missing generic ProMP: {promp_file} (User {uid}, task {letter})", flush=True)
            out_rows.append(row_out)
            continue

        style_pair = get_style(uid)
        if style_pair is None:
            out_rows.append(row_out)
            continue

        mu_style, cov_style = style_pair
        pdata = np.load(promp_file)
        generic_mean = pdata["mean"]
        generic_cov = pdata["cov"]

        demos = collect_demos_user_task(output_root, uid, letter)
        if not demos:
            print(f"No demos under User_{uid}/task_{letter} (User {uid}, task {letter})", flush=True)
            out_rows.append(row_out)
            continue

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

        row_out["L2_Generic"] = fmt_metric(float(np.mean(l2_g)))
        row_out["L2_Personalized"] = fmt_metric(float(np.mean(l2_p)))
        row_out["IoU_Generic"] = fmt_metric(float(np.mean(iou_g)))
        row_out["IoU_Personalized"] = fmt_metric(float(np.mean(iou_p)))
        out_rows.append(row_out)

    if len(out_rows) != len(align_rows):
        raise RuntimeError(
            f"Row count mismatch: alignment {len(align_rows)} vs output {len(out_rows)}"
        )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(f"\nWrote {len(out_rows)} rows (same order/count as {align_path}) to {out_path}", flush=True)


if __name__ == "__main__":
    main()
