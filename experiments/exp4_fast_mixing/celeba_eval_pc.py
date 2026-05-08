"""
DDP evaluation script: FID vs NFE for predictor-corrector (PC) sampling on CIFAR-10.

For each PC step count in {50, 100, ..., 500}, generates 1000 images using the
VE predictor-corrector sampler and computes FID against the real CIFAR-10 stats.

Predictor : reverse-SDE Euler-Maruyama step (VE schedule)
Corrector : 1 adaptive Langevin step (Algorithm 4, Song et al. 2021)

Score is derived from the EDM x0-predictor as:
    score(x, sigma) = (D_x(x, sigma) - x) / sigma^2

Saves a single figure: x-axis = NFE, y-axis = FID.

Usage:
  torchrun --nproc_per_node=<N> celeba_eval_pc.py \\
      --config config/celeba_unconditional_EDM_flux_mixing.yaml \\
      --ckpt_path path/to/model_stepXXXXX.pth \\
      [--n_samples 1000] \\
      [--sample_batch_size 100] \\
      [--corrector_snr 0.16]
"""

import argparse
import copy
import math
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from ema_pytorch import EMA
from torchvision.utils import save_image
from tqdm import tqdm
import yaml

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.exponential import ExponentialIS
from src.normalizer import LossNormalizer
from experiments.shared.utils.fid_score import calculate_fid_given_paths, calculate_inception_score
from experiments.shared.model.unet import UNet
from experiments.exp4_fast_mixing.EDM import EDM
from experiments.shared.utils.utils import Config

PC_STEPS = list(range(10, 200, 10))  # [50, 100, ..., 500]
N_SAMPLES = 1000


# ---------------------------------------------------------------------------
# Model loading (mirrors celeba_eval.py)
# ---------------------------------------------------------------------------

def load_model(ckpt_path, opt, device, rank):
    mixing_noise_normalizer = LossNormalizer().to(device)        
    s_noise_normalizer = LossNormalizer().to(device)
    importance_sampler_net = ExponentialIS().to(device)
    
    diff = EDM(nn_model=UNet(**opt.network),
        s_noise_normalizer=s_noise_normalizer,
        importance_sampler_net=importance_sampler_net,
        mixing_noise_normalizer=mixing_noise_normalizer,
        **opt.diffusion,
        device=device,
    )
    diff.to(device)

    if rank == 0:
        ema = EMA(diff, beta=opt.ema, update_after_step=0, update_every=1)
        ema.to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        ema.load_state_dict(checkpoint['EMA'])
        ema_sd = ema.ema_model.state_dict()
    else:
        ema_sd = diff.state_dict()

    for key in ema_sd:
        tensor = ema_sd[key].to(device)
        dist.broadcast(tensor, src=0)
        ema_sd[key] = tensor

    model = copy.deepcopy(diff)
    model.load_state_dict(ema_sd)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Predictor-corrector sampler (VE, score derived from EDM x0-predictor)
# ---------------------------------------------------------------------------

def _langevin_corrector_step(x, score, snr):
    z = torch.randn_like(x)
    g_norm = score.reshape(score.shape[0], -1).norm(dim=1).mean()
    z_norm = z.reshape(z.shape[0], -1).norm(dim=1).mean()
    eps = 2.0 * (snr * z_norm / (g_norm + 1e-8)) ** 2
    return x + eps * score + (2.0 * eps).sqrt() * z


@torch.no_grad()
def pc_sample(model, n_sample, image_size, device, num_steps, corrector_snr=0.16):
    """
    Sample via VE predictor-corrector.

    Predictor: reverse-SDE Euler-Maruyama
        x_mean = x + (sigma_cur^2 - sigma_next^2) * score(x, sigma_cur)
        x      = x_mean + sqrt(sigma_cur^2 - sigma_next^2) * noise

    Corrector: 1 adaptive Langevin step at sigma_next, applied only on
               intermediate steps (not after the final predictor step)

    Final output: one final denoise at sigma_min, with no fresh noise injected

    Score: s(x, sigma) = (D_x(x, sigma) - x) / sigma^2

    Returns:
        images : [n_sample, C, H, W] in [0, 1]
        nfe    : total number of D_x evaluations
    """
    sigma_min = float(model.sigma_min)
    sigma_max = float(model.sigma_max)
    C, H, W = image_size

    sigmas = torch.exp(
        torch.linspace(
            math.log(sigma_max),
            math.log(sigma_min),
            num_steps + 1,
            device=device,
        )
    )

    x = torch.randn(n_sample, C, H, W, device=device) * sigma_max
    nfe = 0

    for i in range(num_steps):
        sigma_cur = sigmas[i]
        sigma_next = sigmas[i + 1]

        # ---------------------------------------------------------
        # Predictor: VE reverse-SDE Euler-Maruyama
        # ---------------------------------------------------------
        sigma_cur_batch = sigma_cur.expand(n_sample)
        denoised = model.D_x(x, sigma_cur_batch)
        denoised = torch.clamp(denoised, -1.0, 1.0)
        nfe += 1

        score = (denoised - x) / (sigma_cur ** 2)
        noise_var = (sigma_cur ** 2 - sigma_next ** 2).clamp(min=0.0)

        x_mean = x + noise_var * score

        # Do not inject fresh noise on the final step
        if i < num_steps - 1:
            x = x_mean + noise_var.sqrt() * torch.randn_like(x)
        else:
            x = x_mean

        # ---------------------------------------------------------
        # Corrector: adaptive Langevin at sigma_next
        # Skip it on the final step so we end noise-free
        # ---------------------------------------------------------
        if i < num_steps - 1:
            sigma_next_batch = sigma_next.expand(n_sample)
            denoised_c = model.D_x(x, sigma_next_batch)
            denoised_c = torch.clamp(denoised_c, -1.0, 1.0)
            nfe += 1

            score_c = (denoised_c - x) / (sigma_next ** 2)
            x = _langevin_corrector_step(x, score_c, snr=corrector_snr)

    # -------------------------------------------------------------
    # Final denoise at sigma_min, with no fresh noise injected
    # -------------------------------------------------------------
    sigma_final_batch = sigmas[-1].expand(n_sample)
    x = model.D_x(x, sigma_final_batch)
    x = torch.clamp(x, -1.0, 1.0)
    nfe += 1

    # Unnormalize from [-1, 1] to [0, 1]
    images = ((x + 1.0) * 0.5).clamp(0.0, 1.0)
    return images, nfe

# ---------------------------------------------------------------------------
# FID computation for one (num_steps, rank) configuration
# ---------------------------------------------------------------------------

def compute_fid_for_steps(
    model,
    image_size,
    device,
    fid_stats_path,
    rank,
    world_size,
    num_steps,
    n_samples,
    sample_batch_size,
    corrector_snr,
    save_root_dir="saved_images",
):
    samples_per_rank = n_samples // world_size
    remainder = n_samples % world_size
    my_n_samples = samples_per_rank + (1 if rank < remainder else 0)
    offset = rank * samples_per_rank + min(rank, remainder)

    total_nfe = 0
    num_batches = 0

    # Shared directory visible to all ranks
    shared_dir = os.path.join(save_root_dir, f"pc_samples_steps_{num_steps:04d}")

    # Rank 0 prepares a clean directory
    if rank == 0:
        os.makedirs(shared_dir, exist_ok=True)
        for fname in os.listdir(shared_dir):
            if fname.endswith(".png"):
                os.remove(os.path.join(shared_dir, fname))
    dist.barrier()

    generated = 0
    pbar = tqdm(
        total=my_n_samples,
        desc=f"PC steps={num_steps}",
        disable=(rank != 0),
    )

    while generated < my_n_samples:
        bs = min(sample_batch_size, my_n_samples - generated)
        imgs, nfe = pc_sample(
            model,
            bs,
            image_size,
            device,
            num_steps=num_steps,
            corrector_snr=corrector_snr,
        )
        total_nfe += nfe
        num_batches += 1

        imgs = imgs.float().cpu()
        for i, img in enumerate(imgs):
            global_idx = offset + generated + i
            save_image(img, os.path.join(shared_dir, f"sample_{global_idx:06d}.png"))

        generated += bs
        pbar.update(bs)

    pbar.close()
    dist.barrier()

    fid = 0.0
    if rank == 0:
        fid = calculate_fid_given_paths((fid_stats_path, shared_dir), device=device)

    fid_tensor = torch.tensor(fid, device=device)
    dist.broadcast(fid_tensor, src=0)
    fid = fid_tensor.item()

    # NFE per image: each batch shares one trajectory / one NFE count
    per_image_nfe = torch.tensor(total_nfe / num_batches, dtype=torch.float64, device=device)
    dist.all_reduce(per_image_nfe, op=dist.ReduceOp.SUM)
    avg_nfe = per_image_nfe.item() / world_size

    dist.barrier()
    if rank == 0:
        for fname in os.listdir(shared_dir):
            if fname.endswith(".png"):
                os.remove(os.path.join(shared_dir, fname))
        os.rmdir(shared_dir)
    dist.barrier()

    return fid, avg_nfe

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
    parser.add_argument("--sample_batch_size", type=int, default=100)
    parser.add_argument("--corrector_snr", type=float, default=0.16)
    parser.add_argument("--local_rank", type=int, default=0)
    opt = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    device = torch.device(f"cuda:{device}")

    with open(opt.config, 'r') as f:
        cfg = yaml.full_load(f)
    cfg = Config(cfg)

    if rank == 0:
        print(f"Loading checkpoint: {opt.ckpt_path}")
    model = load_model(opt.ckpt_path, cfg, device, rank)

    image_size = tuple(cfg.network['image_shape'])  # (C, H, W)
    fid_stats_path = cfg.fid_stats_path

    nfes = []
    fids = []

    for num_steps in PC_STEPS:
        if rank == 0:
            print(f"\n=== PC num_steps={num_steps} ===")

        fid, avg_nfe = compute_fid_for_steps(
            model, image_size, device, fid_stats_path,
            rank, world_size,
            num_steps=num_steps,
            n_samples=opt.n_samples,
            sample_batch_size=opt.sample_batch_size,
            corrector_snr=opt.corrector_snr,
        )

        nfes.append(avg_nfe)
        fids.append(fid)

        if rank == 0:
            print(f"  NFE={avg_nfe:.1f}  FID={fid:.4f}")

    if rank == 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(nfes, fids, marker='o', linewidth=2)
        ax.set_xlabel("NFE (number of function evaluations)")
        ax.set_ylabel("FID")
        ax.set_title("FID vs NFE — Predictor-Corrector Sampling")
        ax.grid(True, alpha=0.4)
        fig.tight_layout()

        out_path = os.path.join(os.path.dirname(opt.ckpt_path), "fid_vs_nfe_pc.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"\nFigure saved to {out_path}")

        # Also save raw numbers as CSV for reference
        csv_path = os.path.join(os.path.dirname(opt.ckpt_path), "fid_vs_nfe_pc.csv")
        with open(csv_path, "w") as f:
            f.write("pc_steps,nfe,fid\n")
            for steps, nfe, fid in zip(PC_STEPS, nfes, fids):
                f.write(f"{steps},{nfe:.2f},{fid:.4f}\n")
        print(f"Raw data saved to {csv_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
