from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
import torch

try:
    import ot  # type: ignore
except Exception:
    ot = None


def pairwise_sqeuclidean(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return cdist(x, y, metric="sqeuclidean")


def exact_wasserstein_uniform(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape[0] != y.shape[0]:
        raise ValueError("Exact uniform assignment currently assumes equal number of samples.")
    cost = pairwise_sqeuclidean(x, y)
    rows, cols = linear_sum_assignment(cost)
    w2_sq = cost[rows, cols].mean()
    return float(np.sqrt(max(w2_sq, 0.0)))



@torch.no_grad()
def reverse_ve_sample(
    model: torch.nn.Module,
    num_samples: int,
    seq_len: int,
    dim: int,
    device: torch.device,
    sigma_min: float,
    sigma_max: float,
    num_steps: int = 512,
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    Kept under the same name for compatibility with train.py.
    This is the VE probability flow ODE sampler.
    """
    log_sigma_ratio = math.log(sigma_max / sigma_min)

    x = torch.randn(num_samples, seq_len, dim, device=device) * sigma_max
    t_steps = torch.linspace(1.0, eps, num_steps, device=device)

    for i in range(num_steps - 1):
        t = t_steps[i]
        t_next = t_steps[i + 1]
        dt = t_next - t

        sigma = sigma_min * (sigma_max / sigma_min) ** t
        sigma_batch = sigma.expand(num_samples)

        score = model(x, sigma_batch)
        if isinstance(score, (tuple, list)):
            score = score[0]

        g2 = 2.0 * log_sigma_ratio * (sigma ** 2)
        drift = -0.5 * g2 * score
        x = x + drift * dt

    return x


def unnormalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return x * std + mean


def save_time_series_plot(real: np.ndarray, fake: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dim = real.shape[1]
    if dim == 2:
        names = ["q1", "q2"]
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.5), sharex=True)
        axs = np.atleast_1d(axs)
    elif dim == 4:
        names = ["q1", "v1", "q2", "v2"]
        fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
        axs = np.array(axs).reshape(-1)
    else:
        raise ValueError("Expected trajectory dim 2 or 4.")

    for i, ax in enumerate(axs):
        ax.plot(real[:, i], label="real", linewidth=2)
        ax.plot(fake[:, i], label="generated", linewidth=2, alpha=0.85)
        ax.set_title(names[i])

    axs[0].legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def _extract_positions(state: np.ndarray) -> Tuple[float, float]:
    state = np.asarray(state)
    if state.shape[-1] == 2:
        q1, q2 = state
    elif state.shape[-1] == 4:
        q1, _, q2, _ = state
    else:
        raise ValueError("State must have last dimension 2 or 4.")
    return float(q1), float(q2)


def render_side_by_side_frame(real_state: np.ndarray, fake_state: np.ndarray, lim: float = 3.5) -> np.ndarray:
    fig, axs = plt.subplots(2, 1, figsize=(8, 4), dpi=120)

    for ax, state, title in zip(axs, [real_state, fake_state], ["real", "generated"]):
        q1, q2 = _extract_positions(state)
        x_wall_left, x_wall_right = -3.0, 3.0

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-1.0, 1.0)
        ax.axis("off")
        ax.plot([x_wall_left, x_wall_left], [-0.6, 0.6], linewidth=3)
        ax.plot([x_wall_right, x_wall_right], [-0.6, 0.6], linewidth=3)

        def spring_points(xa: float, xb: float, y: float = 0.0, n_coils: int = 10):
            xs = np.linspace(xa, xb, 2 * n_coils + 2)
            ys = np.zeros_like(xs) + y
            if len(xs) > 2:
                ys[1:-1:2] += 0.18
                ys[2:-1:2] -= 0.18
            return xs, ys

        for xa, xb in [(x_wall_left, q1), (q1, q2), (q2, x_wall_right)]:
            xs, ys = spring_points(xa, xb)
            ax.plot(xs, ys, linewidth=2)

        ax.add_patch(plt.Rectangle((q1 - 0.22, -0.22), 0.44, 0.44))
        ax.add_patch(plt.Rectangle((q2 - 0.22, -0.22), 0.44, 0.44))
        ax.set_title(title)

    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
    plt.close(fig)
    return img


def save_side_by_side_gif(real_traj: np.ndarray, fake_traj: np.ndarray, out_path: Path, fps: int = 10) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [render_side_by_side_frame(real_traj[t], fake_traj[t]) for t in range(real_traj.shape[0])]
    imageio.mimsave(out_path, frames, fps=fps)