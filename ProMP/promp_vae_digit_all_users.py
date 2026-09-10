#!/usr/bin/env python3
"""
Train a digit-domain ProMP β-VAE on *all users* digit demonstrations.

This follows your intended flow:
  1) Train the VAE using all users' digit data (task labels = 0..9, user labels vary)
  2) Partition latent dims using two-factor ANOVA (task, user) across all users
  3) Compute a specific user's style distribution (e.g., user 1) from that user's
     latent parameters only, then personalize/evaluate user 1.

Expected directory layout:
  user_1_digit_library_output/task_0/*.csv
  user_2_digit_library_output/task_0/*.csv
  ...

This script loads trajectories from all directories matching:
  user_*_digit_library_output/

Outputs:
  - checkpoint: ProMP/promp_vae_digit_all/promp_vae_digit_all.pt
  - latents:    ProMP/promp_vae_digit_all/promp_vae_digit_all_latents.npz
    (includes mu, sigma2, tasks, users, paths)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from promp_vae import PrompVAE, compute_loss_with_kl, make_rbf_basis, normalize_time, read_trajectory_csv

np.seterr(all="ignore")


@dataclass
class Demo:
    time: np.ndarray  # (T,)
    y: np.ndarray  # (T, D)
    task: str
    user: str
    path: Path


def infer_task_id_from_path(csv_path: Path) -> str:
    for part in csv_path.parts:
        if part.startswith("task_"):
            return part.replace("task_", "")
    return "unknown"


def infer_user_id_from_path(csv_path: Path) -> str:
    # user_1_digit_library_output -> 1
    for part in csv_path.parts:
        if part.startswith("user_") and "_digit_" in part:
            try:
                return part.split("user_")[1].split("_")[0]
            except Exception:  # noqa: BLE001
                return "unknown"
    return "unknown"


def resample_trajectory(time: np.ndarray, y: np.ndarray, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    if time.shape[0] < 2:
        t_resampled = np.linspace(0.0, 1.0, n_steps)
        y_resampled = np.tile(y[0], (n_steps, 1))
        return t_resampled.astype(np.float32), y_resampled.astype(np.float32)
    t_norm = normalize_time(time)
    t_resampled = np.linspace(0.0, 1.0, n_steps)
    d = y.shape[1]
    y_resampled = np.zeros((n_steps, d), dtype=np.float32)
    for dim in range(d):
        y_resampled[:, dim] = np.interp(t_resampled, t_norm, y[:, dim])
    return t_resampled.astype(np.float32), y_resampled.astype(np.float32)


def load_all_users_digit_demos(root: Path, n_steps: int) -> Tuple[List[Demo], np.ndarray]:
    demos: List[Demo] = []
    for csv_path in root.rglob("*-trajectory.csv"):
        # only consider user_*_digit_library_output
        if not any(part.startswith("user_") and "_digit_" in part for part in csv_path.parts):
            continue
        try:
            time, y = read_trajectory_csv(csv_path)
            if time.size < 2:
                continue
            t_resampled, y_resampled = resample_trajectory(time, y, n_steps)
            task = infer_task_id_from_path(csv_path)
            user = infer_user_id_from_path(csv_path)
            demos.append(Demo(time=t_resampled, y=y_resampled, task=task, user=user, path=csv_path))
        except Exception:
            continue

    demos.sort(key=lambda d: (d.user, d.task, d.path.as_posix()))
    if not demos:
        raise SystemExit(f"No digit trajectories found under {root}")
    return demos, demos[0].time


def main() -> None:
    p = argparse.ArgumentParser(description="Train digit-domain ProMP β-VAE on all users.")
    p.add_argument("--root", type=str, default=".", help="Repo root to search under.")
    p.add_argument("--out_dir", type=str, default="ProMP/promp_vae_digit_all")
    p.add_argument("--n_basis", type=int, default=20)
    p.add_argument("--width", type=float, default=0.05)
    p.add_argument("--n_steps", type=int, default=100)
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    print("ProMP digit VAE (all users) started.", flush=True)
    demos, t_resampled = load_all_users_digit_demos(root, n_steps=args.n_steps)
    print(f"Loaded {len(demos)} trajectories from {len(set(d.user for d in demos))} users", flush=True)

    phi_np = make_rbf_basis(t_resampled, n_basis=args.n_basis, width=args.width).astype(np.float32)
    phi = torch.from_numpy(phi_np).to(args.device)

    y_np = np.stack([d.y for d in demos], axis=0).astype(np.float32)
    y_tensor = torch.from_numpy(y_np)
    dataset = TensorDataset(y_tensor)

    n_total = len(dataset)
    n_val = max(1, int(round(args.val_split * n_total))) if n_total >= 2 else 0
    n_train = n_total - n_val

    rng = np.random.default_rng(args.seed)
    idx = np.arange(n_total)
    rng.shuffle(idx)
    train_idx = idx[:n_train].tolist()
    val_idx = idx[n_train:].tolist()

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False, drop_last=False) if n_val > 0 else None

    d = y_np.shape[2]
    weight_dim = d * args.n_basis
    model = PrompVAE(input_dim=d, hidden_size=args.hidden_size, latent_dim=args.latent_dim, weight_dim=weight_dim).to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        for (batch_y,) in train_loader:
            batch_y = batch_y.to(args.device)
            optim.zero_grad(set_to_none=True)
            w_hat, mu, logvar = model(batch_y)
            loss, _, _ = compute_loss_with_kl(batch_y, w_hat, mu, logvar, phi, beta=args.beta, n_basis=args.n_basis)
            loss.backward()
            optim.step()
            bs = batch_y.shape[0]
            tr_loss += loss.item() * bs
            tr_n += bs
        tr_loss /= max(tr_n, 1)

        if val_loader is not None:
            model.eval()
            va_loss = 0.0
            va_n = 0
            with torch.no_grad():
                for (batch_y,) in val_loader:
                    batch_y = batch_y.to(args.device)
                    w_hat, mu, logvar = model(batch_y)
                    loss, _, _ = compute_loss_with_kl(
                        batch_y, w_hat, mu, logvar, phi, beta=args.beta, n_basis=args.n_basis
                    )
                    bs = batch_y.shape[0]
                    va_loss += loss.item() * bs
                    va_n += bs
            va_loss /= max(va_n, 1)
            print(f"Epoch {epoch:03d} Train={tr_loss:.4f} Val={va_loss:.4f}", flush=True)
            if va_loss + 1e-9 < best_val:
                best_val = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print(f"Early stopping at epoch {epoch} (best val {best_val:.4f})", flush=True)
                    break
        else:
            print(f"Epoch {epoch:03d} Train={tr_loss:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "promp_vae_digit_all.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_basis": args.n_basis,
            "width": args.width,
            "n_steps": args.n_steps,
            "latent_dim": args.latent_dim,
            "hidden_size": args.hidden_size,
            "beta": args.beta,
            "weight_dim": weight_dim,
            "phi": phi_np,
            "t_resampled": t_resampled,
            "best_val_loss": best_val if best_state is not None else None,
        },
        ckpt_path,
    )

    # Latents for all demos
    model.eval()
    all_mu = []
    all_logvar = []
    with torch.no_grad():
        for (batch_y,) in DataLoader(dataset, batch_size=args.batch_size, shuffle=False):
            batch_y = batch_y.to(args.device)
            mu, logvar = model.encode(batch_y)
            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())
    mu_np = np.concatenate(all_mu, axis=0)
    logvar_np = np.concatenate(all_logvar, axis=0)
    sigma2_np = np.exp(logvar_np).astype(np.float32)

    tasks = np.array([d.task for d in demos])
    users = np.array([d.user for d in demos])
    paths = np.array([str(d.path) for d in demos])

    latents_path = out_dir / "promp_vae_digit_all_latents.npz"
    np.savez(
        latents_path,
        mu=mu_np,
        logvar=logvar_np,
        sigma2=sigma2_np,
        tasks=tasks,
        users=users,
        paths=paths,
        n_basis=np.array([args.n_basis]),
        width=np.array([args.width]),
        n_steps=np.array([args.n_steps]),
        latent_dim=np.array([args.latent_dim]),
        hidden_size=np.array([args.hidden_size]),
        beta=np.array([args.beta]),
        seed=np.array([args.seed]),
    )

    print(f"\nSaved checkpoint: {ckpt_path}", flush=True)
    print(f"Saved latents:    {latents_path}", flush=True)


if __name__ == "__main__":
    main()

