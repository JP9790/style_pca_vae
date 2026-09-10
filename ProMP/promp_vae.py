#!/usr/bin/env python3
"""
Learn a ProMP latent representation using a β-VAE with a GRU encoder and MLP decoder.

Method summary (matching the provided description):

- Encoder q_φ(z | τ):
  A GRU that maps an input trajectory τ = {y_1, ..., y_T} ∈ R^{T×D} to a latent
  Gaussian distribution N(z_μ, diag(z_σ^2)).

- Decoder p_θ(ŵ | z):
  A multilayer perceptron (MLP) with two hidden layers that maps latent samples
  z to ProMP weight vectors ŵ ∈ R^{K·D}.

- Reconstruction model:
  Given ŵ, we reconstruct the trajectory using standard ProMP basis functions:
      ŷ_t = Φ(t)^T ŵ
  where Φ(t) are RBF basis functions defined on normalized time.

- Objective (β-VAE):
      L = E_{q_φ(z|τ)} [ Σ_t || y_t - Φ(t)^T ŵ ||^2 ]
          + β · KL(q_φ(z|τ) || N(0, I))

This script trains the model on trajectories for a single user (default: User_1),
using the same trajectory CSV format and directory structure as the PCA pipeline.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset

np.seterr(all="ignore")


# -----------------------------------------------------------------------------
# Data utilities (mirroring existing ProMP utilities)
# -----------------------------------------------------------------------------


@dataclass
class Demo:
    time: np.ndarray  # (T_raw,)
    y: np.ndarray  # (T_raw, D)
    task: str
    user: str
    path: Path


def read_trajectory_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read trajectory CSV file and return time and y coordinates."""
    data = np.genfromtxt(path, delimiter=",", skip_header=1).astype(float)
    if data.ndim == 1 or data.shape[1] < 3:
        raise ValueError(f"Expected CSV columns time,x,y in {path}")
    data = data[np.isfinite(data).all(axis=1)]
    time = data[:, 0]
    y = data[:, 1:3]
    return time, y


def infer_task_id_from_path(csv_path: Path) -> str:
    """Extract task ID from folder structure (e.g., task_a -> a)."""
    parts = csv_path.parts
    for part in parts:
        if part.startswith("task_"):
            return part.replace("task_", "")
    return "unknown"


def infer_user_id_from_path(csv_path: Path) -> str:
    """Extract user ID from folder structure (e.g., User_1 -> 1)."""
    parts = csv_path.parts
    for part in parts:
        if part.startswith("User_"):
            return part.replace("User_", "")
    return "unknown"


def normalize_time(time: np.ndarray) -> np.ndarray:
    """Normalize time to [0, 1] range."""
    t0 = float(time[0])
    t1 = float(time[-1])
    if t1 <= t0:
        return np.linspace(0.0, 1.0, num=time.shape[0])
    return (time - t0) / (t1 - t0)


def make_rbf_basis(t: np.ndarray, n_basis: int, width: float) -> np.ndarray:
    """Create RBF basis functions Φ(t) on normalized time."""
    centers = np.linspace(0.0, 1.0, num=n_basis)
    diff = t[:, None] - centers[None, :]
    basis = np.exp(-0.5 * (diff / width) ** 2)
    basis /= basis.sum(axis=1, keepdims=True)
    return basis  # (T, K)


def resample_trajectory(
    time: np.ndarray,
    y: np.ndarray,
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample a trajectory to a fixed number of time steps using linear interpolation.

    Returns:
        t_resampled: (n_steps,) normalized time in [0, 1]
        y_resampled: (n_steps, D)
    """
    if time.shape[0] < 2:
        # Degenerate trajectory: just tile the single point
        t_resampled = np.linspace(0.0, 1.0, n_steps)
        y_resampled = np.tile(y[0], (n_steps, 1))
        return t_resampled, y_resampled

    t_norm = normalize_time(time)
    t_resampled = np.linspace(0.0, 1.0, n_steps)
    d = y.shape[1]
    y_resampled = np.zeros((n_steps, d), dtype=np.float32)
    for dim in range(d):
        y_resampled[:, dim] = np.interp(t_resampled, t_norm, y[:, dim])
    return t_resampled, y_resampled


def load_user_demos(
    output_root: Path,
    user_id: str,
    n_steps: int,
) -> Tuple[List[Demo], np.ndarray]:
    """
    Load and resample all trajectory CSVs for a given user.

    Returns:
        demos: list of Demo objects with resampled time and y
        phi:   shared RBF basis matrix Φ(t) of shape (n_steps, K) will be
               constructed later from t_resampled; here we only return the
               resampled time to keep a consistent interface.
    """
    demos: List[Demo] = []

    for csv_path in output_root.rglob("*-trajectory.csv"):
        try:
            uid = infer_user_id_from_path(csv_path)
            if uid != user_id:
                continue
            time, y = read_trajectory_csv(csv_path)
            if time.size < 2:
                continue
            t_resampled, y_resampled = resample_trajectory(time, y, n_steps)
            task_id = infer_task_id_from_path(csv_path)
            demos.append(
                Demo(
                    time=t_resampled.astype(np.float32),
                    y=y_resampled.astype(np.float32),
                    task=task_id,
                    user=uid,
                    path=csv_path,
                )
            )
        except Exception as e:  # noqa: BLE001
            print(f"Warning: Skipping {csv_path}: {e}")
            continue

    demos.sort(key=lambda d: (d.task, d.path.as_posix()))

    if not demos:
        raise SystemExit(f"No trajectory CSVs found for User {user_id} under {output_root}")

    # All demos share the same resampled time grid
    t_resampled = demos[0].time
    return demos, t_resampled


# -----------------------------------------------------------------------------
# VAE model
# -----------------------------------------------------------------------------


class PrompVAE(nn.Module):
    """
    β-VAE with:
      - GRU encoder: τ -> hidden -> (μ, logvar)
      - MLP decoder: z -> ProMP weights ŵ
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        latent_dim: int,
        weight_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, weight_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D)
        _, h_n = self.encoder(x)  # h_n: (1, B, H)
        h = h_n[-1]  # (B, H)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Returns ProMP weights ŵ of shape (B, weight_dim)
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        w_hat = self.decode(z)
        return w_hat, mu, logvar


def beta_vae_loss(
    y: torch.Tensor,
    w_hat: torch.Tensor,
    phi: torch.Tensor,
    beta: float,
    n_basis: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute β-VAE loss for a batch.

    Args:
        y:      (B, T, D) target trajectories
        w_hat:  (B, P) decoded ProMP weights, where P = D * K
        phi:    (T, K) RBF basis matrix Φ(t) shared across batch
        beta:   β coefficient
        n_basis: K, number of basis functions

    Returns:
        total_loss, recon_loss, kl_div
    """
    b, t, d = y.shape
    k = n_basis

    # Reshape weights to (B, D, K)
    weights = w_hat.view(b, d, k)

    # Compute ŷ_t = Φ(t)^T ŵ for each sample in batch
    # Phi: (1, T, K) -> (B, T, K)
    phi_batch = phi.unsqueeze(0).expand(b, -1, -1)  # (B, T, K)
    # weights_T: (B, K, D)
    weights_t = weights.transpose(1, 2)
    # y_hat: (B, T, D) = (B, T, K) @ (B, K, D)
    y_hat = torch.bmm(phi_batch, weights_t)

    # Reconstruction term: sum_t || y_t - y_hat_t ||^2
    recon = (y - y_hat) ** 2
    recon_loss = recon.sum(dim=(1, 2)).mean()

    # KL term w.r.t. standard normal prior N(0, I)
    # Note: mu and logvar must be provided separately where this function is called.
    raise RuntimeError(
        "beta_vae_loss should not be called directly; use compute_loss_with_kl "
        "which has access to mu and logvar.",
    )


def compute_loss_with_kl(
    y: torch.Tensor,
    w_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    phi: torch.Tensor,
    beta: float,
    n_basis: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Full β-VAE loss with explicit KL term.
    """
    b, t, d = y.shape
    k = n_basis

    # Reshape weights to (B, D, K)
    weights = w_hat.view(b, d, k)

    # Compute ŷ_t = Φ(t)^T ŵ for each sample in batch
    phi_batch = phi.unsqueeze(0).expand(b, -1, -1)  # (B, T, K)
    weights_t = weights.transpose(1, 2)  # (B, K, D)
    y_hat = torch.bmm(phi_batch, weights_t)  # (B, T, D)

    # Reconstruction term
    recon = (y - y_hat) ** 2
    recon_loss = recon.sum(dim=(1, 2)).mean()

    # KL term against N(0, I)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_div = kl.mean()

    total = recon_loss + beta * kl_div
    return total, recon_loss, kl_div


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def train_promp_vae(
    output_dir: Path,
    user_id: str,
    n_basis: int,
    width: float,
    n_steps: int,
    latent_dim: int,
    hidden_size: int,
    beta: float,
    batch_size: int,
    epochs: int,
    lr: float,
    val_split: float,
    patience: int,
    seed: int,
    device: str,
    out_dir: Path,
) -> None:
    """
    Train the ProMP β-VAE on trajectories from a single user.
    """
    print(f"Loading trajectories for User_{user_id} from {output_dir} ...", flush=True)
    demos, t_resampled = load_user_demos(output_dir, user_id=user_id, n_steps=n_steps)
    print(f"Loaded {len(demos)} trajectories for User_{user_id}", flush=True)

    # Build ProMP basis Φ(t) using resampled normalized time
    phi_np = make_rbf_basis(t_resampled, n_basis=n_basis, width=width).astype(np.float32)
    phi = torch.from_numpy(phi_np).to(device)  # (T, K)

    # Stack trajectories into a single tensor (N, T, D)
    y_np = np.stack([demo.y for demo in demos], axis=0).astype(np.float32)
    y_tensor = torch.from_numpy(y_np)  # (N, T, D)

    dataset = TensorDataset(y_tensor)
    n_total = len(dataset)
    n_val = max(1, int(round(val_split * n_total))) if n_total >= 2 else 0
    n_train = n_total - n_val

    rng = np.random.default_rng(seed)
    indices = np.arange(n_total)
    rng.shuffle(indices)
    train_idx = indices[:n_train].tolist()
    val_idx = indices[n_train:].tolist()

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx) if n_val > 0 else None

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = (
        DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)
        if val_set is not None
        else None
    )

    d = y_np.shape[2]
    k = n_basis
    weight_dim = d * k

    model = PrompVAE(
        input_dim=d,
        hidden_size=hidden_size,
        latent_dim=latent_dim,
        weight_dim=weight_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nStarting β-VAE training...", flush=True)
    print(
        f"  User: {user_id}, D={d}, K={k}, weight_dim={weight_dim}, "
        f"T={n_steps}, latent_dim={latent_dim}, β={beta}",
        flush=True,
    )
    if val_loader is not None:
        print(
            f"  Train/val split: {n_train}/{n_val} (val_split={val_split})  "
            f"Early stopping: patience={patience}",
            flush=True,
        )

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        print(f"  Epoch {epoch}/{epochs} starting...", flush=True)
        model.train()
        total_loss_epoch = 0.0
        recon_epoch = 0.0
        kl_epoch = 0.0
        n_batches = 0

        for (batch_y,) in train_loader:
            batch_y = batch_y.to(device)  # (B, T, D)

            optimizer.zero_grad(set_to_none=True)
            w_hat, mu, logvar = model(batch_y)
            loss, recon_loss, kl_div = compute_loss_with_kl(
                batch_y,
                w_hat,
                mu,
                logvar,
                phi,
                beta=beta,
                n_basis=n_basis,
            )
            loss.backward()
            optimizer.step()

            batch_size_actual = batch_y.shape[0]
            total_loss_epoch += loss.item() * batch_size_actual
            recon_epoch += recon_loss.item() * batch_size_actual
            kl_epoch += kl_div.item() * batch_size_actual
            n_batches += batch_size_actual

        total_loss_epoch /= n_batches
        recon_epoch /= n_batches
        kl_epoch /= n_batches

        # Validation
        if val_loader is not None:
            model.eval()
            val_total = 0.0
            val_recon = 0.0
            val_kl = 0.0
            val_n = 0
            with torch.no_grad():
                for (batch_y,) in val_loader:
                    batch_y = batch_y.to(device)
                    w_hat, mu, logvar = model(batch_y)
                    loss, recon_loss, kl_div = compute_loss_with_kl(
                        batch_y,
                        w_hat,
                        mu,
                        logvar,
                        phi,
                        beta=beta,
                        n_basis=n_basis,
                    )
                    bs = batch_y.shape[0]
                    val_total += loss.item() * bs
                    val_recon += recon_loss.item() * bs
                    val_kl += kl_div.item() * bs
                    val_n += bs
            val_total /= max(val_n, 1)
            val_recon /= max(val_n, 1)
            val_kl /= max(val_n, 1)

            print(
                f"Epoch {epoch:03d}/{epochs}  "
                f"TrainLoss={total_loss_epoch:.4f} (Recon={recon_epoch:.4f}, KL={kl_epoch:.4f})  "
                f"ValLoss={val_total:.4f} (Recon={val_recon:.4f}, KL={val_kl:.4f})",
                flush=True,
            )

            # Early stopping
            if val_total + 1e-9 < best_val:
                best_val = val_total
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(
                        f"Early stopping triggered at epoch {epoch} "
                        f"(best val loss {best_val:.4f}).",
                        flush=True,
                    )
                    break
        else:
            print(
                f"Epoch {epoch:03d}/{epochs}  "
                f"Loss={total_loss_epoch:.4f}  Recon={recon_epoch:.4f}  KL={kl_epoch:.4f}",
                flush=True,
            )

    # Restore best model (if we had validation)
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save model checkpoint and training metadata
    ckpt_path = out_dir / f"promp_vae_user_{user_id}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "user_id": user_id,
            "n_basis": n_basis,
            "width": width,
            "n_steps": n_steps,
            "latent_dim": latent_dim,
            "hidden_size": hidden_size,
            "beta": beta,
            "weight_dim": weight_dim,
            "phi": phi_np,
            "t_resampled": t_resampled,
            "val_split": val_split,
            "patience": patience,
            "seed": seed,
            "best_val_loss": best_val if best_state is not None else None,
        },
        ckpt_path,
    )

    # Export latent parameters (μ, σ^2) per trajectory for analysis
    model.eval()
    with torch.no_grad():
        all_mu = []
        all_logvar = []
        for (batch_y,) in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            batch_y = batch_y.to(device)
            mu, logvar = model.encode(batch_y)
            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())
        all_mu_np = np.concatenate(all_mu, axis=0)
        all_logvar_np = np.concatenate(all_logvar, axis=0)
        all_sigma2_np = np.exp(all_logvar_np).astype(np.float32)

    tasks = np.array([demo.task for demo in demos])
    paths = np.array([str(demo.path) for demo in demos])

    npz_path = out_dir / f"promp_vae_user_{user_id}_latents.npz"
    np.savez(
        npz_path,
        mu=all_mu_np,
        logvar=all_logvar_np,
        sigma2=all_sigma2_np,
        tasks=tasks,
        paths=paths,
        n_basis=np.array([n_basis]),
        width=np.array([width]),
        n_steps=np.array([n_steps]),
        latent_dim=np.array([latent_dim]),
        hidden_size=np.array([hidden_size]),
        beta=np.array([beta]),
        val_split=np.array([val_split]),
        patience=np.array([patience]),
        seed=np.array([seed]),
    )

    print(f"\nTraining complete. Checkpoint saved to: {ckpt_path}", flush=True)
    print(f"Latent representations saved to: {npz_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a ProMP β-VAE (GRU encoder, MLP decoder) for a single user.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Root folder containing trajectory CSVs (same as for PCA).",
    )
    parser.add_argument(
        "--user_id",
        type=str,
        default="1",
        help="User ID to train on (e.g., '1' for User_1).",
    )
    parser.add_argument(
        "--n_basis",
        type=int,
        default=20,
        help="Number of RBF basis functions K for ProMP.",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=0.05,
        help="RBF width parameter for ProMP basis.",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=100,
        help="Number of time steps T to resample each trajectory to.",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=16,
        help="Latent dimensionality of VAE.",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=128,
        help="Hidden size of GRU encoder.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="β coefficient for the KL term.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction of demos used for validation (early stopping).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs without validation improvement).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for train/val split shuffling.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer.",
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
        default="ProMP/promp_vae",
        help="Output directory for checkpoints and latent representations.",
    )
    return parser.parse_args()


def main() -> None:
    print("ProMP VAE script started.", flush=True)
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    train_promp_vae(
        output_dir=output_dir,
        user_id=args.user_id,
        n_basis=args.n_basis,
        width=args.width,
        n_steps=args.n_steps,
        latent_dim=args.latent_dim,
        hidden_size=args.hidden_size,
        beta=args.beta,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        val_split=args.val_split,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()

