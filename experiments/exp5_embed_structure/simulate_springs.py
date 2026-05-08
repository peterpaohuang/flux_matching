from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SpringConfig:
    k_wall: float = 1.0
    k_couple: float = 0.7
    damping: float = 0.08
    cubic: float = 0.08
    dt: float = 0.10
    horizon: int = 50


def spring_rhs(state: np.ndarray, cfg: SpringConfig) -> np.ndarray:
    """
    state: [..., 4] = [q1, v1, q2, v2]
    """
    q1, v1, q2, v2 = np.moveaxis(state, -1, 0)

    dq1 = v1
    dq2 = v2
    dv1 = (
        -cfg.k_wall * q1
        - cfg.k_couple * (q1 - q2)
        - cfg.damping * v1
        - cfg.cubic * q1 ** 3
    )
    dv2 = (
        -cfg.k_wall * q2
        - cfg.k_couple * (q2 - q1)
        - cfg.damping * v2
        - cfg.cubic * q2 ** 3
    )
    return np.stack([dq1, dv1, dq2, dv2], axis=-1)


def rk4_step(state: np.ndarray, dt: float, cfg: SpringConfig) -> np.ndarray:
    k1 = spring_rhs(state, cfg)
    k2 = spring_rhs(state + 0.5 * dt * k1, cfg)
    k3 = spring_rhs(state + 0.5 * dt * k2, cfg)
    k4 = spring_rhs(state + dt * k3, cfg)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def sample_initial_conditions(
    n: int,
    rng: np.random.Generator,
    mode: str = "release",
    position_scale: float = 3.0,
    velocity_scale: float = 1.5,
) -> np.ndarray:
    if mode == "release":
        q = rng.normal(scale=position_scale, size=(n, 2))
        v = rng.normal(scale=velocity_scale, size=(n, 2))
    elif mode == "gaussian_all":
        q = rng.normal(scale=position_scale, size=(n, 2))
        v = rng.normal(scale=position_scale, size=(n, 2))
    else:
        raise ValueError(f"Unknown init mode: {mode}")

    x0 = np.zeros((n, 4), dtype=np.float32)
    x0[:, 0] = q[:, 0]
    x0[:, 1] = v[:, 0]
    x0[:, 2] = q[:, 1]
    x0[:, 3] = v[:, 1]
    return x0


def simulate_trajectories(
    n: int,
    cfg: SpringConfig,
    seed: int = 0,
    init_mode: str = "release",
    position_scale: float = 1.0,
    velocity_scale: float = 0.20,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state = sample_initial_conditions(
        n=n,
        rng=rng,
        mode=init_mode,
        position_scale=position_scale,
        velocity_scale=velocity_scale,
    )
    traj = np.zeros((n, cfg.horizon, 4), dtype=np.float32)
    traj[:, 0] = state
    for t in range(1, cfg.horizon):
        state = rk4_step(state, cfg.dt, cfg).astype(np.float32)
        traj[:, t] = state
    return traj


def save_dataset(
    out_path: Path,
    train: np.ndarray,
    cfg: SpringConfig,
    init_mode: str,
    position_scale: float,
    velocity_scale: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        train=train,
        k_wall=cfg.k_wall,
        k_couple=cfg.k_couple,
        damping=cfg.damping,
        cubic=cfg.cubic,
        dt=cfg.dt,
        horizon=cfg.horizon,
        init_mode=init_mode,
        position_scale=position_scale,
        velocity_scale=velocity_scale,
    )


def render_spring_frame(state: np.ndarray, lim: float = 3.5) -> np.ndarray:
    q1, _, q2, _ = state
    x_wall_left, x_wall_right = -3.0, 3.0
    x1, x2 = q1, q2

    fig, ax = plt.subplots(figsize=(8, 2.2), dpi=120)
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

    for xa, xb in [(x_wall_left, x1), (x1, x2), (x2, x_wall_right)]:
        xs, ys = spring_points(xa, xb)
        ax.plot(xs, ys, linewidth=2)

    ax.add_patch(plt.Rectangle((x1 - 0.22, -0.22), 0.44, 0.44))
    ax.add_patch(plt.Rectangle((x2 - 0.22, -0.22), 0.44, 0.44))
    ax.text(-3.3, 0.78, "two coupled nonlinear springs", fontsize=11)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
    plt.close(fig)
    return img


def save_spring_gif(traj: np.ndarray, out_path: Path, fps: int = 10) -> None:
    frames = [render_spring_frame(traj[t]) for t in range(traj.shape[0])]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-mode", type=str, default="release", choices=["release", "gaussian_all"])
    parser.add_argument("--position-scale", type=float, default=3.0)
    parser.add_argument("--velocity-scale", type=float, default=1.5)
    parser.add_argument("--out", type=str, default="data/springs_dataset.npz")
    parser.add_argument("--gif-out", type=str, default="data/example_real.gif")
    parser.add_argument("--k-wall", type=float, default=1.0)
    parser.add_argument("--k-couple", type=float, default=0.7)
    parser.add_argument("--damping", type=float, default=0.08)
    parser.add_argument("--cubic", type=float, default=0.08)
    parser.add_argument("--dt", type=float, default=0.10)
    parser.add_argument("--horizon", type=int, default=50)
    args = parser.parse_args()

    cfg = SpringConfig(
        k_wall=args.k_wall,
        k_couple=args.k_couple,
        damping=args.damping,
        cubic=args.cubic,
        dt=args.dt,
        horizon=args.horizon,
    )

    train = simulate_trajectories(
        n=args.num_samples,
        cfg=cfg,
        seed=args.seed,
        init_mode=args.init_mode,
        position_scale=args.position_scale,
        velocity_scale=args.velocity_scale,
    )
    save_dataset(
        Path(args.out),
        train=train,
        cfg=cfg,
        init_mode=args.init_mode,
        position_scale=args.position_scale,
        velocity_scale=args.velocity_scale,
    )
    save_spring_gif(train[0], Path(args.gif_out))
    print(f"Saved dataset ({train.shape}) to {args.out}")
    print(f"Saved example gif to {args.gif_out}")


if __name__ == "__main__":
    main()
