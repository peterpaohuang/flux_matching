"""
Standalone DDP evaluation script for EDM on CelebA-64.

Metrics computed:
  1. FID-50K  — generate 50K images with adaptive Heun-Euler ODE solver, compare to real stats
  2. NLL      — Hutchinson trace estimator on the CelebA test split (bits/dim)
  3. Avg NFE  — mean number of function evaluations of the adaptive solver over 50K samples

Usage:
  torchrun --nproc_per_node=4 celeba_eval.py \
      --config config/celeba_unconditional_flux.yaml \
      --ckpt_path path/to/model_stepXXXXX.pth \
      [--sample_batch_size 500] \
      [--nll_batch_size 32] \
      [--rtol 1e-5] [--atol 1e-5]
"""

import argparse
import copy
import os
import tempfile

import numpy as np
import torch
import torch.distributed as dist
import yaml
from ema_pytorch import EMA
from torchdiffeq import odeint
from torchvision import transforms
from torchvision import datasets as tvdatasets
from torchvision.utils import save_image
from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.exponential import ExponentialIS
from src.normalizer import LossNormalizer
from experiments.shared.utils.fid_score import calculate_fid_given_paths, calculate_inception_score
from experiments.shared.model.unet import UNet
from experiments.exp3_unrestricted.EDM import EDM
from experiments.shared.utils.utils import Config

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


class Crop:
    """Center-crop matching celeba.py training preprocessing."""
    def __init__(self):
        cx, cy = 89, 121
        self.x1, self.x2 = cy - 64, cy + 64
        self.y1, self.y2 = cx - 64, cx + 64

    def __call__(self, img):
        from torchvision.transforms import functional as F
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)


def get_celeba_transform():
    return transforms.Compose([
        Crop(),
        transforms.Resize(64),
        transforms.ToTensor(),
    ])


def load_model(ckpt_path, opt, device, rank):
    """Load EDM, broadcast EMA weights from rank 0 to all ranks."""
    s_noise_normalizer = LossNormalizer().to(device)
    importance_sampler_net = ExponentialIS().to(device)
    
    try:
        diff = EDM(nn_model=UNet(**opt.network),
            s_noise_normalizer=s_noise_normalizer,
            importance_sampler_net=importance_sampler_net,
            **opt.diffusion,
            device=device,
        )
    except:
        diff = EDM(nn_model=UNet(**opt.network),
            s_noise_normalizer=None,
            importance_sampler_net=None,
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
# Adaptive Heun-Euler ODE sampler (probability flow)
# ---------------------------------------------------------------------------

def adaptive_heun_sample(model, n_sample, image_size, device,
                         rtol=1e-5, atol=1e-5):
    """
    Sample images by solving the probability-flow ODE via torchdiffeq
    with the adaptive Heun-Euler solver:
        dx/d(-sigma) = -(x - D(x, sigma)) / sigma

    We reparameterise time as tau = -sigma so that tau increases from
    -sigma_max to -sigma_min, satisfying torchdiffeq's t0 < t1 convention.

    Returns:
        images : [n_sample, C, H, W] in [0, 1]
        nfe    : int, total number of model evaluations
    """
    sigma_min = float(model.sigma_min)
    sigma_max = float(model.sigma_max)
    C, H, W = image_size

    nfe = [0]

    def drift(tau, x):
        sigma = -tau.item()
        sigma_t = torch.full((n_sample,), sigma, device=device, dtype=torch.float32)
        with torch.no_grad():
            denoised = model.D_x(x.float(), sigma_t)
            denoised = torch.clamp(denoised, -1., 1.)
        nfe[0] += 1
        return -(x - denoised.double()) / sigma

    x_init = torch.randn(n_sample, C, H, W, device=device, dtype=torch.double) * sigma_max
    t_span = torch.tensor([-sigma_max, -sigma_min], dtype=torch.double, device=device)

    x_final = odeint(drift, x_init, t_span, method="adaptive_heun", rtol=rtol, atol=atol)[-1]

    images = unnormalize_to_zero_to_one(x_final.float()).clamp(0., 1.)
    return images, nfe[0]


# ---------------------------------------------------------------------------
# FID: each rank generates its share, rank 0 computes FID
# ---------------------------------------------------------------------------

import os
import shutil
import tempfile

def compute_fid(model, image_size, device, fid_stats_path, rank, world_size,
                n_samples=50000, sample_batch_size=500, rtol=1e-5, atol=1e-5):

    samples_per_rank = n_samples // world_size
    remainder = n_samples % world_size
    my_n_samples = samples_per_rank + (1 if rank < remainder else 0)
    offset = rank * samples_per_rank + min(rank, remainder)

    total_nfe = 0
    num_batches = 0

    # Create one shared temp directory on rank 0, then broadcast its path.
    if rank == 0:
        shared_tmp_dir = tempfile.mkdtemp(prefix="fid_shared_")
    else:
        shared_tmp_dir = None

    obj = [shared_tmp_dir]
    dist.broadcast_object_list(obj, src=0)
    shared_tmp_dir = obj[0]
    dist.barrier()

    try:
        generated = 0
        pbar = tqdm(
            total=my_n_samples,
            desc="Generating samples (adaptive Heun-Euler)",
            disable=(rank != 0),
        )

        while generated < my_n_samples:
            bs = min(sample_batch_size, my_n_samples - generated)
            imgs, nfe = adaptive_heun_sample(
                model, bs, image_size, device, rtol=rtol, atol=atol
            )
            total_nfe += nfe
            num_batches += 1

            imgs = imgs.float().cpu()
            for i, img in enumerate(imgs):
                global_idx = offset + generated + i
                save_image(img, os.path.join(shared_tmp_dir, f"sample_{global_idx:06d}.png"))

            generated += bs
            pbar.update(bs)

        pbar.close()
        dist.barrier()

        fid = 0.0
        if rank == 0:
            print(f"Computing FID ({shared_tmp_dir} vs {fid_stats_path}) ...")
            fid = calculate_fid_given_paths((fid_stats_path, shared_tmp_dir), device=device)

        per_image_nfe = torch.tensor(total_nfe / num_batches, dtype=torch.float64, device=device)
        dist.all_reduce(per_image_nfe, op=dist.ReduceOp.SUM)
        avg_nfe = per_image_nfe.item() / world_size

        fid_tensor = torch.tensor(fid, device=device)
        dist.broadcast(fid_tensor, src=0)
        fid = fid_tensor.item()

    finally:
        dist.barrier()
        if rank == 0:
            shutil.rmtree(shared_tmp_dir, ignore_errors=True)
        dist.barrier()

    return fid, avg_nfe


# ---------------------------------------------------------------------------
# NLL via Hutchinson trace estimator — each rank handles a subset of batches
# ---------------------------------------------------------------------------

def compute_nll(model, test_set, device, rank, world_size, nll_batch_size,
                n_hutchinson=1, rtol=1e-5, atol=1e-5):
    """
    Estimate NLL in bits/dim. Each rank processes its own subset of the test
    set via DistributedSampler; rank 0 gathers and reports the mean.
    """
    sigma_min = float(model.sigma_min)
    sigma_max = float(model.sigma_max)

    sampler = torch.utils.data.distributed.DistributedSampler(
        test_set, num_replicas=world_size, rank=rank, shuffle=False
    )
    loader = torch.utils.data.DataLoader(
        test_set, batch_size=nll_batch_size, sampler=sampler,
        num_workers=4, pin_memory=True
    )

    local_bpds = []
    pbar = tqdm(loader, desc="Computing NLL", disable=(rank != 0))

    for batch in pbar:
        x_batch = batch[0] if isinstance(batch, (list, tuple)) else batch
        x_batch = normalize_to_neg_one_to_one(x_batch.to(device))
        B, C, H, W = x_batch.shape
        D = C * H * W

        # Fixed Rademacher vectors for Hutchinson estimator
        eps_list = [
            (torch.randint(0, 2, x_batch.shape, device=device).float() * 2 - 1).double()
            for _ in range(n_hutchinson)
        ]

        def dynamics_func(sigma_t_val, states):
            x, _ = states
            sigma = sigma_t_val.item()
            sigma_t = torch.full((B,), sigma, device=device, dtype=torch.float32)

            x_f = x.float().requires_grad_(True)
            denoised = model.D_x(x_f, sigma_t)
            drift_f = (x_f - denoised) / sigma   # dx/dsigma — no clamping, preserves gradients

            div_sum = torch.zeros(B, device=device, dtype=torch.float32)
            for eps in eps_list:
                vjp = torch.autograd.grad(
                    (drift_f * eps.float()).sum(), x_f,
                    create_graph=False, retain_graph=True
                )[0]
                div_sum += (vjp * eps.float()).reshape(B, -1).sum(dim=1)
            div_est = (div_sum / n_hutchinson).double()

            # d(log p)/dsigma = -div_x(dx/dsigma)
            return drift_f.detach().double(), -div_est

        # Encode data -> noise: start from x_0 at sigma_min, integrate forward to sigma_max
        y_init = (x_batch.double(), torch.zeros(B, device=device, dtype=torch.double))
        t_span = torch.tensor([sigma_min, sigma_max], dtype=torch.double, device=device)

        sol_x, sol_logdet = odeint(
            dynamics_func, y_init, t_span,
            method="adaptive_heun", rtol=rtol, atol=atol,
        )
        x_T = sol_x[-1]          # data encoded into noise space
        delta_logp = sol_logdet[-1]  # ∫ -div dsigma = log p_T(x_T) - log p_data(x_0)

        # log p_T(x_T) at the noise-space endpoint
        log_pT = (
            -0.5 * D * np.log(2 * np.pi * sigma_max ** 2)
            - 0.5 * (x_T.reshape(B, -1) ** 2).sum(dim=1).cpu().double().numpy() / sigma_max ** 2
        )
        # log p_data(x_0) = log p_T(x_T) - delta_logp  (since delta_logp = log p_T - log p_data)
        log_px = log_pT - delta_logp.cpu().numpy()
        bpd = -log_px / (D * np.log(2)) + 7
        local_bpds.append(bpd)

    local_bpds = np.concatenate(local_bpds)

    local_tensor = torch.tensor(local_bpds, device=device, dtype=torch.float64)
    gathered = [torch.zeros_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(gathered, local_tensor)

    if rank == 0:
        all_bpd = torch.cat(gathered).cpu().numpy()
        return float(np.mean(all_bpd)), float(np.std(all_bpd))
    return None, None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--sample_batch_size", type=int, default=500)
    parser.add_argument("--nll_batch_size", type=int, default=32)
    parser.add_argument("--n_samples_fid", type=int, default=50000)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--n_hutchinson", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--skip_fid", action="store_true", help="Skip FID+NFE computation")
    parser.add_argument("--skip_nll", action="store_true", help="Skip NLL computation")
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

    fid, avg_nfe = None, None
    nll_mean, nll_std = None, None

    # ------------------------------------------------------------------
    # 1 & 3: FID-50K + average NFE
    # ------------------------------------------------------------------
    if not opt.skip_fid:
        if rank == 0:
            print("\n=== FID-50K + Avg NFE ===")
        fid, avg_nfe = compute_fid(
            model, image_size, device, fid_stats_path, rank, world_size,
            n_samples=opt.n_samples_fid,
            sample_batch_size=opt.sample_batch_size,
            rtol=opt.rtol,
            atol=opt.atol,
        )
        if rank == 0:
            print(f"FID-50K  : {fid:.4f}")
            print(f"Avg NFE  : {avg_nfe:.2f}  (per image, over {opt.n_samples_fid} samples)")

    dist.barrier()

    # ------------------------------------------------------------------
    # 2: NLL on CelebA test split
    # ------------------------------------------------------------------
    if not opt.skip_nll:
        if rank == 0:
            print("\n=== NLL (bits/dim) on CelebA test split ===")
        tf = get_celeba_transform()
        tvdatasets.CelebA._check_integrity = lambda self: True
        test_set = tvdatasets.CelebA(
            root='../shared/data', split='test', target_type='attr',
            transform=tf, download=False
        )

        nll_mean, nll_std = compute_nll(
            model, test_set, device, rank, world_size, opt.nll_batch_size,
            n_hutchinson=opt.n_hutchinson,
            rtol=opt.rtol,
            atol=opt.atol,
        )
        if rank == 0:
            print(f"NLL (bpd): {nll_mean:.4f} ± {nll_std:.4f}")

    # ------------------------------------------------------------------
    # Summary (rank 0 only)
    # ------------------------------------------------------------------
    if rank == 0:
        print("\n======== Evaluation Summary ========")
        print(f"Checkpoint : {opt.ckpt_path}")
        if fid is not None:
            print(f"FID-50K    : {fid:.4f}")
            print(f"Avg NFE    : {avg_nfe:.2f}")
        if nll_mean is not None:
            print(f"NLL (bpd)  : {nll_mean:.4f} ± {nll_std:.4f}")
        print("====================================")

        results_path = os.path.join(os.path.dirname(opt.ckpt_path), "eval_results.txt")
        with open(results_path, 'a') as f:
            parts = [f"ckpt={opt.ckpt_path}"]
            if fid is not None:
                parts.append(f"fid50k={fid:.4f} avg_nfe={avg_nfe:.2f}")
            if nll_mean is not None:
                parts.append(f"nll_bpd={nll_mean:.4f}±{nll_std:.4f}")
            print(" ".join(parts), file=f)
        print(f"Results saved to {results_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
