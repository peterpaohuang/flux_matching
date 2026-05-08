import argparse
import os
import re
import tempfile

import torch
import torch.distributed as dist
import yaml
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import copy
from ema_pytorch import EMA
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.exponential import ExponentialIS
from src.normalizer import LossNormalizer
from experiments.shared.utils.utils import Config, reduce_tensor, DataLoaderDDP, print0
from experiments.shared.model.unet import UNet
from experiments.exp3_unrestricted.EDM import EDM

torch.backends.cuda.matmul.allow_tf32 = True

import numpy as np

def normalize_to_neg_one_to_one(img):
    # [0.0, 1.0] -> [-1.0, 1.0]
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    # [-1.0, 1.0] -> [0.0, 1.0]
    return (t + 1) * 0.5


def infinite_loader(dataloader, sampler):
    """Infinite iterator over a DataLoader, re-shuffling each epoch via sampler."""
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        for batch in dataloader:
            yield batch
        epoch += 1


def generate_and_save_images(ema_model, n_samples, sample_batch_size, image_size, device, rank, world_size, save_dir):
    """
    DDP-parallelized image generation. Each rank generates its share of images,
    saves them as PNGs to save_dir, then all ranks barrier before rank 0 computes FID.
    """
    # Divide samples across ranks
    samples_per_rank = n_samples // world_size
    remainder = n_samples % world_size
    my_n_samples = samples_per_rank + (1 if rank < remainder else 0)

    # Offset for unique filenames across ranks
    offset = rank * samples_per_rank + min(rank, remainder)

    ema_model.eval()
    with torch.no_grad():
        generated = 0
        pbar = tqdm(total=my_n_samples, desc=f"sampling ({n_samples} images)", disable=(rank != 0))
        while generated < my_n_samples:
            batch_size = min(sample_batch_size, my_n_samples - generated)
            imgs = ema_model.edm_sample(batch_size, image_size, notqdm=True)  # [B, C, H, W] in [0,1]
            imgs = imgs.float().cpu()
            for i, img in enumerate(imgs):
                idx = offset + generated + i
                save_image(img, os.path.join(save_dir, f"sample_{idx:06d}.png"))
            generated += batch_size
            pbar.update(batch_size)
        pbar.close()


# ===== training =====

def train(opt):
    yaml_path = opt.config
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    seed = opt.local_rank * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    use_dsm = True if opt.use_dsm == 'True' else False
    save_dir = opt.save_dir

    with open(yaml_path, 'r') as f:
        opt = yaml.full_load(f)
    print0(opt)
    opt = Config(opt)
    model_dir = os.path.join(save_dir, "ckpts")
    vis_dir = os.path.join(save_dir, "visual")
    if rank == 0:
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

    if not use_dsm:
        s_noise_normalizer = LossNormalizer().to(device)
        importance_sampler_net = ExponentialIS().to(device)
    else:
        s_noise_normalizer = None
        importance_sampler_net = None
        
    diff = EDM(nn_model=UNet(**opt.network),
                     s_noise_normalizer=s_noise_normalizer,
                     importance_sampler_net=importance_sampler_net,
                     **opt.diffusion,
                     device=device,
                     )
    diff.to(device)
    if rank == 0:
        ema = EMA(diff, beta=opt.ema, update_after_step=0, update_every=1)
        ema.to(device)

    diff = torch.nn.SyncBatchNorm.convert_sync_batchnorm(diff)
    diff = torch.nn.parallel.DistributedDataParallel(
        diff, device_ids=[rank])

    tf = [transforms.ToTensor()]
    if opt.flip:
        tf = [transforms.RandomHorizontalFlip()] + tf
    tf = transforms.Compose(tf)

    train_set = CIFAR10("../shared/data", train=True, transform=tf)
    image_size = train_set[0][0].shape  # (C, H, W)

    print0("CIFAR10 train dataset:", len(train_set))

    train_loader, sampler = DataLoaderDDP(train_set,
                                          batch_size=opt.batch_size,
                                          shuffle=True)

    lr = opt.lrate
    print0("Using DDP, lr = %f" % (lr))

    backbone_params = [p for n, p in diff.named_parameters()]
    optim_backbone = torch.optim.Adam(backbone_params, lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    start_step = 0
    if opt.load_step != -1:
        target = os.path.join(model_dir, f"model_step{opt.load_step}.pth")
        print0("loading model at", target)
        checkpoint = torch.load(target, map_location=torch.device(f'cuda:{device}'))
        diff.load_state_dict(checkpoint['MODEL'])
        if rank == 0:
            ema.load_state_dict(checkpoint['EMA'])
        optim_backbone.load_state_dict(checkpoint['opt_backbone'])
        start_step = checkpoint['step']

    fid_sample_batch_size = 500

    # Infinite data iterator
    data_iter = infinite_loader(train_loader, sampler)

    step = start_step
    loss_ema = None
    step_fid = []  # [(step, fid_10k), ...]

    diff.train()
    if rank == 0:
        pbar = tqdm(total=opt.n_steps, initial=step, desc="training")

    from experiments.shared.utils.fid_score import calculate_fid_given_paths, calculate_inception_score

    while step < opt.n_steps:
        # ---- LR warmup (iteration-based) ----
        current_lr = lr * min(float(step + 1) / opt.warm_steps, 1.0)
        for g in optim_backbone.param_groups:
            g['lr'] = current_lr

        x, _ = next(data_iter)
        x = x.to(device)
        loss = diff(x, use_dsm=use_dsm)

        scaler.scale(loss).backward()
        scaler.unscale_(optim_backbone)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters=diff.parameters(), max_norm=1.0)
        scaler.step(optim_backbone)
        scaler.update()
        optim_backbone.zero_grad()

        dist.barrier()
        loss_reduced = reduce_tensor(loss)

        if rank == 0:
            ema.update()
            if loss_ema is None:
                loss_ema = loss_reduced.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss_reduced.item()

            desc = f"loss: {loss_ema:.4f}"
            desc += f", grad_norm: {grad_norm.item():.4f}, lr: {current_lr:.6f}"
            pbar.set_description(desc)
            pbar.update(1)

        step += 1

        # ---- Visualization every 1000 steps (rank 0 only) ----
        if rank == 0 and step % 1000 == 0:
            ema_sample_method = ema.ema_model.edm_sample

            ema.ema_model.eval()
            with torch.no_grad():
                x_gen = ema_sample_method(opt.n_sample, image_size)
                x_real = x[:opt.n_sample].cpu()
                x_all = torch.cat([x_gen.cpu(), x_real])
                grid = make_grid(x_all, nrow=10)
                save_path = os.path.join(vis_dir, f"image_step{step}_ema.png")
                save_image(grid, save_path)
                print(f'saved image at {save_path}')
            ema.ema_model.train()

        # ---- FID evaluation + checkpoint every 50000 steps ----
        if step % 50000 == 0 or step == opt.n_steps:
            dist.barrier()

            # Pack state dict tensors for broadcast
            if rank == 0:
                ema_sd = ema.ema_model.state_dict()
            else:
                ema_sd = copy.deepcopy(diff.module.state_dict())  # same structure, will be overwritten

            for key in ema_sd:
                tensor = ema_sd[key].to(device)
                dist.broadcast(tensor, src=0)
                ema_sd[key] = tensor

            # Build a local eval model on each rank (no DDP wrapper)
            eval_model = copy.deepcopy(diff.module)
            eval_model.load_state_dict(ema_sd)
            eval_model.to(device)
            eval_model.eval()

            with tempfile.TemporaryDirectory() as tmp_dir:
                generate_and_save_images(
                    eval_model, 10000, fid_sample_batch_size, image_size, device, rank, world_size, tmp_dir
                )
                dist.barrier()

                fid_10k = 0.0
                is_mean, is_std = 0.0, 0.0
                if rank == 0:
                    fid_10k = calculate_fid_given_paths((opt.fid_stats_path, tmp_dir), device=device)
                    is_mean, is_std = calculate_inception_score(tmp_dir, device=device)
                    msg = f'step={step} fid10k={fid_10k:.4f} IS={is_mean:.4f}±{is_std:.4f}'
                    print(msg)
                    with open(os.path.join(save_dir, 'eval.log'), 'a') as f:
                        print(msg, file=f)

            fid_tensor = torch.tensor(fid_10k, device=device)
            dist.broadcast(fid_tensor, src=0)
            fid_10k = fid_tensor.item()
            step_fid.append((step, fid_10k))

            # Save checkpoint (rank 0 only)
            if rank == 0 and opt.save_model:
                checkpoint = {
                    'MODEL': diff.state_dict(),
                    'EMA': ema.state_dict(),
                    'opt_backbone': optim_backbone.state_dict(),
                    'step': step,
                }
                if s_noise_normalizer is not None:
                    checkpoint['s_noise_normalizer'] = s_noise_normalizer.state_dict()
                if importance_sampler_net is not None:
                    checkpoint['importance_sampler_net'] = importance_sampler_net.state_dict()
                save_path = os.path.join(model_dir, f"model_step{step}.pth")
                torch.save(checkpoint, save_path)
                print(f'saved checkpoint at {save_path}')

            del eval_model
            dist.barrier()

        diff.train()

    if rank == 0:
        pbar.close()

    # ---- Final 50K FID on best checkpoint ----
    print0(f"Training done. step_fid: {step_fid}")

    # Read FIDs from eval.log to find best checkpoint
    eval_log_path = os.path.join(save_dir, 'eval.log')
    step_fid_from_log = []
    if os.path.exists(eval_log_path):
        with open(eval_log_path, 'r') as f:
            for line in f:
                m = re.search(r'step=(\d+)\s+fid10k=([\d.]+)', line)
                if m:
                    step_fid_from_log.append((int(m.group(1)), float(m.group(2))))
    print0(f"step_fid from eval.log: {step_fid_from_log}")

    if len(step_fid_from_log) > 0:
        best_step = sorted(step_fid_from_log, key=lambda x: x[1])[0][0]
        print0(f"Best step (lowest 10K FID): {best_step}")

        # Load best checkpoint
        best_ckpt_path = os.path.join(model_dir, f"model_step{best_step}.pth")
        best_ckpt = torch.load(best_ckpt_path, map_location=torch.device(f'cuda:{device}'))

        # Broadcast best EMA weights to all ranks
        best_ema_sd = {}
        if rank == 0:
            # Extract EMA model state dict from the saved EMA state_dict
            tmp_ema = EMA(diff.module, beta=opt.ema, update_after_step=0, update_every=1)
            tmp_ema.to(device)
            tmp_ema.load_state_dict(best_ckpt['EMA'])
            best_ema_sd = tmp_ema.ema_model.state_dict()
            del tmp_ema
        else:
            best_ema_sd = copy.deepcopy(diff.module.state_dict())

        for key in best_ema_sd:
            tensor = best_ema_sd[key].to(device)
            dist.broadcast(tensor, src=0)
            best_ema_sd[key] = tensor

        best_eval_model = copy.deepcopy(diff.module)
        best_eval_model.load_state_dict(best_ema_sd)
        best_eval_model.to(device)
        best_eval_model.eval()

        with tempfile.TemporaryDirectory() as tmp_dir:
            generate_and_save_images(
                best_eval_model, 50000, fid_sample_batch_size, image_size, device, rank, world_size, tmp_dir
            )
            dist.barrier()

            if rank == 0:
                fid_50k = calculate_fid_given_paths((opt.fid_stats_path, tmp_dir), device=device)
                is_mean, is_std = calculate_inception_score(tmp_dir, device=device)
                msg = f'Final 50K FID (best step={best_step}): {fid_50k:.4f} IS={is_mean:.4f}±{is_std:.4f}'
                print(msg)
                with open(os.path.join(save_dir, 'eval.log'), 'a') as f:
                    print(msg, file=f)

        del best_eval_model
        dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument('--local_rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--use_dsm', type=str)
    opt = parser.parse_args()
    train(opt)
