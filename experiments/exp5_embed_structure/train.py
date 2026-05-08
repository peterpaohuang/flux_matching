from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterator
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
from torch.utils.data import DataLoader, Dataset

from experiments.exp5_embed_structure.eval import reverse_ve_sample, exact_wasserstein_uniform, save_side_by_side_gif, save_time_series_plot
from experiments.exp5_embed_structure.models import ModelConfig, TrajectoryDriftModel, update_ema_model

from experiments.shared.baselines.dsm import dsm_loss_epsilon
from src.loss import flux_matching_loss

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class NoiseSchedule:
    sigma_min: float = 0.002
    sigma_max: float = 80.0


def sample_shared_sigma(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    schedule: NoiseSchedule,
) -> Tensor:
    log_sigma = torch.empty((), device=device, dtype=dtype).uniform_(
        torch.log(torch.tensor(schedule.sigma_min, device=device, dtype=dtype)),
        torch.log(torch.tensor(schedule.sigma_max, device=device, dtype=dtype)),
    )
    sigma = log_sigma.exp()
    return sigma.expand(batch_size)


class TrajectoryDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.data = torch.from_numpy(data.astype(np.float32))

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class Standardizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)   # [1, 1, 2]
        self.std = std.astype(np.float32)     # [1, 1, 2]

    @classmethod
    def from_train(cls, train: np.ndarray) -> "Standardizer":
        # Global featurewise normalization over dataset and time.
        mean = train.mean(axis=(0, 1), keepdims=True)
        std = train.std(axis=(0, 1), keepdims=True)
        std = np.maximum(std, 1e-6)
        return cls(mean=mean, std=std)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def unnormalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


def cycle(loader: DataLoader) -> Iterator[torch.Tensor]:
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_norm: np.ndarray,
    standardizer: Standardizer,
    device: torch.device,
    schedule: NoiseSchedule,
    eval_num_samples: int,
    out_dir: Path,
    step: int,
) -> Dict[str, float]:
    model.eval()
    seq_len = data_norm.shape[1]
    dim = data_norm.shape[2]

    fake = reverse_ve_sample(
        model=model,
        num_samples=eval_num_samples,
        seq_len=seq_len,
        dim=dim,
        device=device,
        sigma_min=schedule.sigma_min,
        sigma_max=schedule.sigma_max,
        num_steps=512,
    ).cpu().numpy()

    real = data_norm[:eval_num_samples]
    real_flat = real.reshape(eval_num_samples, -1)
    fake_flat = fake.reshape(eval_num_samples, -1)

    wdist = exact_wasserstein_uniform(real_flat, fake_flat)

    real_un = standardizer.unnormalize(real)
    fake_un = standardizer.unnormalize(fake)

    plot_path = out_dir / f"comparison_step_{step:07d}.png"
    gif_path = out_dir / f"comparison_step_{step:07d}.gif"
    save_time_series_plot(real_un[0], fake_un[0], plot_path)
    save_side_by_side_gif(real_un[0], fake_un[0], gif_path)

    return {"wasserstein": float(wdist)}


def build_model(args: argparse.Namespace, input_dim: int) -> TrajectoryDriftModel:
    cfg = ModelConfig(
        seq_len=args.horizon,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        n_heads=args.n_heads,
        causal=args.causal,
    )
    return TrajectoryDriftModel(cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--objective", type=str, choices=["dsm", "flux"], required=True)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.data, allow_pickle=True)
    train_raw = data["train"].astype(np.float32)
    args.horizon = int(train_raw.shape[1])
    input_dim = int(train_raw.shape[2])

    standardizer = Standardizer.from_train(train_raw)
    train = standardizer.normalize(train_raw)

    np.savez_compressed(
        out_dir / "normalization_stats.npz",
        mean=standardizer.mean,
        std=standardizer.std,
    )

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_loader = DataLoader(TrajectoryDataset(train), batch_size=256, shuffle=True, drop_last=False)
    train_iter = cycle(train_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schedule = NoiseSchedule(sigma_min=args.sigma_min, sigma_max=args.sigma_max)

    model = build_model(args, input_dim=input_dim).to(device)
    ema_model = copy.deepcopy(model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_ckpt_step: int = -1
    best_wdist: float = float("inf")

    csv_path = out_dir / "train_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "objective",
                "causal",
                "loss",
                "wasserstein",
                "main_loss",
                "aux_loss",
                "tau",
                "resp_entropy",
                "q_hat_norm",
            ],
        )
        writer.writeheader()

        for step in range(1, args.steps + 1):
            model.train()
            batch = next(train_iter).to(device)
            sigmas = sample_shared_sigma(batch.shape[0], device, batch.dtype, schedule)

            optimizer.zero_grad(set_to_none=True)
            B, _, _ = batch.shape
            if args.objective == "dsm":
                def epsilon_model(x,sigma):
                    return model(x, sigma, return_raw=True)
                loss = dsm_loss_epsilon(epsilon_model, batch, sigmas**2).mean()
                log_row = {
                    "step": step,
                    "objective": args.objective,
                    "causal": int(args.causal),
                    "loss": float(loss.detach().cpu()),
                }
            else:
                def f_theta(z): return model(z, sigmas)
                loss, t = flux_matching_loss(
                    f_theta, batch, sigmas**2,
                    q=model.importance_sampler_net, return_t=True
                )
                s_normalizer_weight = model.s_noise_normalizer(sigmas)
                is_loss = model.importance_sampler_net.reinforce_loss(
                    t, 
                    sigmas[0], 
                    loss.detach().mean() / torch.exp(s_normalizer_weight[0]).detach()
                ).expand(B)

                loss = (
                    loss / torch.exp(s_normalizer_weight) 
                    + s_normalizer_weight
                    + is_loss
                ).mean()
                
                log_row = {
                    "step": step,
                    "objective": args.objective,
                    "causal": int(args.causal),
                    "loss": float(loss.detach().cpu())
                }

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            update_ema_model(ema_model, model, decay=args.ema_decay)

            if step % args.log_every == 0:
                print(
                    f"step={step:06d} objective={args.objective} causal={args.causal} loss={float(loss.detach()):.6f}"
                )

            if step % args.eval_every == 0 or step == args.steps:
                eval_stats = evaluate_model(
                    model=ema_model,
                    data_norm=train,
                    standardizer=standardizer,
                    device=device,
                    eval_num_samples=len(train),
                    schedule=schedule,
                    out_dir=out_dir,
                    step=step,
                )
                log_row["wasserstein"] = eval_stats["wasserstein"]
                print(f"  eval wasserstein={eval_stats['wasserstein']:.6f}")

                ckpt = {
                    "model": model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                    "step": step,
                }
                ckpt_path = out_dir / f"checkpoint_step_{step:07d}.pt"
                torch.save(ckpt, ckpt_path)

                if eval_stats["wasserstein"] < best_wdist:
                    best_wdist = eval_stats["wasserstein"]
                    best_ckpt_step = step
                    torch.save(ckpt, out_dir / "checkpoint_best.pt")

            writer.writerow(log_row)
            f.flush()

    print(f"Finished. Logs saved to {csv_path}")

    # Final eval: reload best checkpoint and compute exact (non-entropic) Wasserstein distance.
    if best_ckpt_step >= 0:
        print(
            f"\nRunning final exact-Wasserstein eval on best checkpoint "
            f"(step {best_ckpt_step}, sinkhorn dist={best_wdist:.6f}) ..."
        )
        best_ckpt = torch.load(out_dir / "checkpoint_best.pt", map_location=device)
        best_ema = build_model(args, input_dim=input_dim).to(device)
        best_ema.load_state_dict(best_ckpt["ema_model"])

        final_eval_stats = evaluate_model(
            model=best_ema,
            data_norm=train,
            standardizer=standardizer,
            device=device,
            schedule=schedule,
            eval_num_samples=len(train),
            out_dir=out_dir,
            step=best_ckpt_step,
        )
        print(f"  best model exact wasserstein={final_eval_stats['wasserstein']:.6f}")

        with open(out_dir / "final_eval.json", "w") as f:
            json.dump(
                {
                    "best_ckpt_step": best_ckpt_step,
                    "sinkhorn_wasserstein": best_wdist,
                    "exact_wasserstein": final_eval_stats["wasserstein"],
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()