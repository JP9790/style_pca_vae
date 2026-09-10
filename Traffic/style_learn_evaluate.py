#!/usr/bin/env python3
"""
Learn driving style from train segments (per driver) using ProMP + PCA and ProMP β-VAE
(handwriting-style), reconstruct validation trajectories with generic vs personalized
models, and compare statistics to ground-truth validation data.

Generic vs personalized (PCA):
  - PCA is fit on ProMP weight vectors from **all** training segments (pooled).
  - **Generic**: replace the first ``n_style`` principal coordinates of each validation
    sample with **zeros** (population-neutral style in the style subspace).
  - **Personalized**: replace those coordinates with the **mean** style coordinates of
    that driver's **training** segments.

Generic vs personalized (VAE):
  - **Global** β-VAE trained on all training trajectories.
  - **Per-driver** β-VAE trained only on that driver's training trajectories (small-data
    friendly defaults); if too few samples, falls back to the global VAE for that user.

Statistics (per validation segment, then averaged per driver):
  - Mean and variance of the lateral velocity profile (over time).
  - Peak lateral velocity |v| on lane-change segments (LCL/LCR).
  - Lane-change "duration" proxy: span (in seconds) where |v(t)| ≥ 0.5 · max|v|.
  - Mean magnitude of lateral acceleration |dv/dt|.

Outputs CSV summaries under ``--out-dir``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_PROMP = _REPO / "ProMP"
if str(_PROMP) not in sys.path:
    sys.path.insert(0, str(_PROMP))

from promp_pca import compute_pca, fit_weights, make_rbf_basis, normalize_time
from promp_vae import PrompVAE, compute_loss_with_kl

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from Traffic.config import DT_S, MNV_LCL, MNV_LCR


def _time_grid(n_steps: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_steps, dtype=np.float64)


def stack_weights_from_X(
    X: np.ndarray,
    n_basis: int,
    width: float,
) -> np.ndarray:
    """X: (N, T, 2) lateral d, v — returns W (N, P) ProMP weights."""
    t = _time_grid(X.shape[1])
    rows: list[np.ndarray] = []
    for i in range(X.shape[0]):
        y = X[i].astype(np.float64)
        w = fit_weights(t, y, n_basis, width, ridge=1e-6)
        rows.append(w)
    return np.vstack(rows)


def weights_to_traj_batch(
    W: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """W: (N, 2*n_basis), phi: (T, K) -> Y: (N, T, 2)."""
    n, p = W.shape
    k = phi.shape[1]
    assert p == 2 * k
    w3 = W.reshape(n, 2, k)
    # y[n,t,d] = sum_k phi[t,k] * w[n,d,k]
    return np.einsum("tk,ndk->ntd", phi, w3)


def pca_reconstruct_generic_personal(
    W_val: np.ndarray,
    W_train: np.ndarray,
    vehicle_id_train: np.ndarray,
    vehicle_id_val: np.ndarray,
    mean: np.ndarray,
    eigvecs: np.ndarray,
    n_style: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Z = E^T (w - mu) per column for val; generic zeros first n_style rows;
    personalized replaces first n_style rows with per-driver train means.
    """
    p = mean.shape[0]
    assert eigvecs.shape == (p, p)
    centered_val = W_val - mean
    Z = eigvecs.T @ centered_val.T  # (P, Nval)

    Z_gen = Z.copy()
    Z_gen[:n_style, :] = 0.0

    Z_per = Z.copy()
    drivers = np.unique(vehicle_id_val)
    for vid in drivers:
        cols = np.flatnonzero(vehicle_id_val == vid)
        tr_mask = vehicle_id_train == vid
        Wu = W_train[tr_mask]
        if Wu.shape[0] == 0:
            continue
        Z_tr = eigvecs.T @ (Wu - mean).T  # (P, Nu)
        z_u = np.mean(Z_tr, axis=1)  # (P,)
        Z_per[:n_style, cols] = z_u[:n_style, None]

    W_gen = mean + (eigvecs @ Z_gen).T
    W_per = mean + (eigvecs @ Z_per).T
    return W_gen, W_per


def train_vae(
    Y: np.ndarray,
    phi: torch.Tensor,
    n_basis: int,
    latent_dim: int,
    hidden_size: int,
    beta: float,
    lr: float,
    batch_size: int,
    epochs: int,
    device: str,
    seed: int,
) -> PrompVAE:
    torch.manual_seed(seed)
    n, t, d = Y.shape
    weight_dim = d * n_basis
    model = PrompVAE(
        input_dim=d,
        hidden_size=hidden_size,
        latent_dim=latent_dim,
        weight_dim=weight_dim,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = TensorDataset(torch.from_numpy(Y.astype(np.float32)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for _ in range(epochs):
        for (batch_y,) in loader:
            batch_y = batch_y.to(device)
            opt.zero_grad(set_to_none=True)
            w_hat, mu, logvar = model(batch_y)
            loss, _, _ = compute_loss_with_kl(
                batch_y, w_hat, mu, logvar, phi, beta=beta, n_basis=n_basis
            )
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def vae_reconstruct_mu(
    model: PrompVAE,
    Y: np.ndarray,
    phi: torch.Tensor,
    n_basis: int,
    device: str,
) -> np.ndarray:
    """Deterministic recon using encoder mean μ (no sampling)."""
    model.eval()
    y_t = torch.from_numpy(Y.astype(np.float32)).to(device)
    mu, _ = model.encode(y_t)
    w_hat = model.decode(mu)
    b, t, d = y_t.shape
    k = n_basis
    weights = w_hat.view(b, d, k)
    phi_b = phi.unsqueeze(0).expand(b, -1, -1)
    weights_t = weights.transpose(1, 2)
    y_hat = torch.bmm(phi_b, weights_t)
    return y_hat.cpu().numpy()


def segment_stats(
    y: np.ndarray,
    maneuver: int,
    dt: float,
) -> dict[str, float]:
    """y: (T, 2) [lateral d, lateral v]."""
    v = y[:, 1]
    v_mean_t = float(np.mean(v))
    v_var_t = float(np.var(v))
    is_lc = maneuver in (MNV_LCL, MNV_LCR)
    if is_lc and np.isfinite(v).all():
        peak_lv = float(np.max(np.abs(v)))
        vmax = peak_lv
        if vmax > 1e-9:
            mask = np.abs(v) >= 0.5 * vmax
            lc_dur = float(np.sum(mask) * dt)
        else:
            lc_dur = float(y.shape[0] * dt)
    else:
        peak_lv = float("nan")
        lc_dur = float("nan")
    a = np.diff(v) / dt
    acc_mag = float(np.mean(np.abs(a))) if a.size else float("nan")
    return {
        "v_mean_time": v_mean_t,
        "v_var_time": v_var_t,
        "lc_peak_abs_v": peak_lv,
        "lc_duration_s": lc_dur,
        "lat_acc_mag_mean": acc_mag,
    }


def mean_nan(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if a.size else float("nan")


def write_driver_wide_csv(
    driver_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    """One row per driver; columns for each statistic under each model name."""
    from collections import defaultdict

    models = [
        "original",
        "pca_generic",
        "pca_personal",
        "vae_generic",
        "vae_personal",
    ]
    stat_keys = [
        "v_mean_time",
        "v_var_time",
        "lc_peak_abs_v",
        "lc_duration_s",
        "lat_acc_mag_mean",
    ]
    by_d: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in driver_rows:
        by_d[int(r["driver_id"])][str(r["model"])] = r

    import csv

    fieldnames = ["driver_id"]
    for m in models:
        for s in stat_keys:
            fieldnames.append(f"{m}__{s}")

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for vid in sorted(by_d.keys()):
            row: dict[str, Any] = {"driver_id": vid}
            for m in models:
                mr = by_d[vid].get(m)
                for s in stat_keys:
                    key = f"{m}__{s}"
                    row[key] = "" if mr is None else mr.get(s, "")
            w.writerow(row)


def aggregate_driver_table(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group segment rows by driver_id + model name; mean numeric stats."""
    from collections import defaultdict

    acc: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        acc[(int(r["driver_id"]), str(r["model"]))].append(r)

    out: list[dict[str, Any]] = []
    for (vid, model), lst in sorted(acc.items()):
        keys = [
            "v_mean_time",
            "v_var_time",
            "lc_peak_abs_v",
            "lc_duration_s",
            "lat_acc_mag_mean",
        ]
        row: dict[str, Any] = {"driver_id": vid, "model": model, "n_segments": len(lst)}
        for k in keys:
            vals = np.array([float(x[k]) for x in lst], dtype=np.float64)
            row[k] = mean_nan(vals)
        out.append(row)
    return out


def run(
    segments_npz: Path,
    split_npz: Path,
    out_dir: Path,
    n_basis: int,
    width: float,
    n_style: int,
    vae_latent: int,
    vae_hidden: int,
    vae_beta: float,
    vae_lr: float,
    vae_epochs_global: int,
    vae_epochs_user: int,
    vae_batch: int,
    min_train_for_user_vae: int,
    device: str,
    seed: int,
) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    seg = np.load(segments_npz, allow_pickle=True)
    spl = np.load(split_npz, allow_pickle=True)
    X = seg["X"]
    y = np.asarray(seg["y"]).ravel()
    vehicle_id = np.asarray(seg["vehicle_id"]).ravel()
    train_idx = np.asarray(spl["train_idx"]).ravel()
    val_idx = np.asarray(spl["val_idx"]).ravel()

    X_tr, y_tr, vid_tr = X[train_idx], y[train_idx], vehicle_id[train_idx]
    X_va, y_va, vid_va = X[val_idx], y[val_idx], vehicle_id[val_idx]

    t_grid = _time_grid(X.shape[1])
    phi_np = make_rbf_basis(normalize_time(t_grid), n_basis=n_basis, width=width).astype(
        np.float32
    )
    phi_t = torch.from_numpy(phi_np).to(device)

    print("Fitting ProMP weights (train pool for PCA, train+val for eval)...", flush=True)
    W_train = stack_weights_from_X(X_tr, n_basis, width)
    W_val = stack_weights_from_X(X_va, n_basis, width)

    print("PCA on pooled train weights...", flush=True)
    pca = compute_pca(W_train)
    mean_w = pca["mean"]
    eigvecs = pca["eigvecs"]
    n_style = min(n_style, eigvecs.shape[1])

    W_gen, W_per = pca_reconstruct_generic_personal(
        W_val,
        W_train,
        vid_tr,
        vid_va,
        mean_w,
        eigvecs,
        n_style,
    )
    Y_pca_gen = weights_to_traj_batch(W_gen, phi_np)
    Y_pca_per = weights_to_traj_batch(W_per, phi_np)

    print("Training global VAE...", flush=True)
    model_global = train_vae(
        X_tr,
        phi_t,
        n_basis,
        vae_latent,
        vae_hidden,
        vae_beta,
        vae_lr,
        vae_batch,
        vae_epochs_global,
        device,
        seed,
    )
    Y_vae_gen = vae_reconstruct_mu(model_global, X_va, phi_t, n_basis, device)

    print("Per-driver VAE + validation reconstructions...", flush=True)
    Y_vae_per = np.empty_like(X_va)
    drivers = np.unique(vid_va)
    for vid in drivers:
        mask_va = vid_va == vid
        mask_tr = vid_tr == vid
        Xv = X_va[mask_va]
        n_tr_u = int(mask_tr.sum())
        if n_tr_u >= min_train_for_user_vae:
            sub_seed = (seed * 1_000_003 + int(vid)) % (2**31)
            m_u = train_vae(
                X_tr[mask_tr],
                phi_t,
                n_basis,
                vae_latent,
                vae_hidden,
                vae_beta,
                vae_lr,
                min(vae_batch, n_tr_u),
                vae_epochs_user,
                device,
                sub_seed,
            )
            Y_vae_per[mask_va] = vae_reconstruct_mu(m_u, Xv, phi_t, n_basis, device)
        else:
            Y_vae_per[mask_va] = vae_reconstruct_mu(model_global, Xv, phi_t, n_basis, device)

    models_y = {
        "original": X_va,
        "pca_generic": Y_pca_gen,
        "pca_personal": Y_pca_per,
        "vae_generic": Y_vae_gen,
        "vae_personal": Y_vae_per,
    }

    segment_rows: list[dict[str, Any]] = []
    for name, Ym in models_y.items():
        for j in range(Ym.shape[0]):
            st = segment_stats(Ym[j], int(y_va[j]), DT_S)
            segment_rows.append(
                {
                    "driver_id": int(vid_va[j]),
                    "maneuver": int(y_va[j]),
                    "model": name,
                    **st,
                }
            )

    driver_rows = aggregate_driver_table(segment_rows)
    out_dir.mkdir(parents=True, exist_ok=True)

    import csv

    seg_csv = out_dir / "segment_level_stats.csv"
    drv_csv = out_dir / "driver_level_mean_stats.csv"
    fieldnames = [
        "driver_id",
        "maneuver",
        "model",
        "v_mean_time",
        "v_var_time",
        "lc_peak_abs_v",
        "lc_duration_s",
        "lat_acc_mag_mean",
    ]
    with seg_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in segment_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    dfields = [
        "driver_id",
        "model",
        "n_segments",
        "v_mean_time",
        "v_var_time",
        "lc_peak_abs_v",
        "lc_duration_s",
        "lat_acc_mag_mean",
    ]
    with drv_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=dfields)
        w.writeheader()
        for r in driver_rows:
            w.writerow({k: r.get(k, "") for k in dfields})

    wide_csv = out_dir / "driver_level_wide_comparison.csv"
    write_driver_wide_csv(driver_rows, wide_csv)

    np.savez_compressed(
        out_dir / "style_eval_arrays.npz",
        Y_val_true=X_va,
        Y_pca_generic=Y_pca_gen,
        Y_pca_personal=Y_pca_per,
        Y_vae_generic=Y_vae_gen,
        Y_vae_personal=Y_vae_per,
        y_val=y_va,
        vehicle_id_val=vid_va,
        val_idx=val_idx,
        n_style=n_style,
        n_basis=n_basis,
    )

    torch.save(model_global.state_dict(), out_dir / "vae_global.pt")
    print(f"Wrote {seg_csv}", flush=True)
    print(f"Wrote {drv_csv}", flush=True)
    print(f"Wrote {wide_csv}", flush=True)
    print(f"Wrote {out_dir / 'style_eval_arrays.npz'}", flush=True)
    print(f"Wrote {out_dir / 'vae_global.pt'}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="PCA/VAE style + validation metrics (NGSIM segments).")
    p.add_argument(
        "--segments",
        type=str,
        default=str(_REPO / "Traffic/ngsim_us101_i80_segments.npz"),
        help="prepare_dataset .npz",
    )
    p.add_argument(
        "--split",
        type=str,
        default=str(_REPO / "Traffic/train_val_split.npz"),
        help="train_val_split .npz",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(_REPO / "Traffic/style_eval_results"),
    )
    p.add_argument("--n-basis", type=int, default=20)
    p.add_argument("--width", type=float, default=0.05)
    p.add_argument(
        "--n-style",
        type=int,
        default=5,
        help="First K principal directions treated as style subspace for PCA generic/personal.",
    )
    p.add_argument("--vae-latent", type=int, default=8)
    p.add_argument("--vae-hidden", type=int, default=64)
    p.add_argument("--vae-beta", type=float, default=1e-3)
    p.add_argument("--vae-lr", type=float, default=1e-3)
    p.add_argument("--vae-epochs-global", type=int, default=80)
    p.add_argument("--vae-epochs-user", type=int, default=40)
    p.add_argument("--vae-batch", type=int, default=64)
    p.add_argument("--min-train-user-vae", type=int, default=8)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run(
        Path(args.segments).expanduser().resolve(),
        Path(args.split).expanduser().resolve(),
        Path(args.out_dir).expanduser().resolve(),
        n_basis=args.n_basis,
        width=args.width,
        n_style=args.n_style,
        vae_latent=args.vae_latent,
        vae_hidden=args.vae_hidden,
        vae_beta=args.vae_beta,
        vae_lr=args.vae_lr,
        vae_epochs_global=args.vae_epochs_global,
        vae_epochs_user=args.vae_epochs_user,
        vae_batch=args.vae_batch,
        min_train_for_user_vae=args.min_train_user_vae,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
