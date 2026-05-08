# Experiment 3: Unrestricted Generative Fields (Images)

This experiment trains and evaluates an unconditional EDM model on CIFAR-10 and CelebA-64. Training uses multi-GPU DDP. Evaluation reports FID-50K, Inception Score, and NLL (bits/dim) using an adaptive Heun-Euler ODE solver.

## Data

**CIFAR-10** is downloaded automatically by torchvision on first run:

```bash
# No action needed — torchvision downloads to experiments/shared/data/ on first run
```

**CelebA** must be downloaded manually (torchvision requires it to be pre-placed due to Google Drive restrictions):

1. Download `img_align_celeba.zip` and `list_attr_celeba.txt` from the [CelebA project page](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
2. Place them under `experiments/shared/data/celeba/`

**FID statistics** must be downloaded and placed at `experiments/shared/fid_stats/`:

1. Download the `fid_stats` folder from [Google Drive](https://drive.google.com/drive/folders/1N0yzh9O3pwv8sSdO-txo5h-ZNGbc4HGe)
2. Place the downloaded folder at `experiments/shared/fid_stats/`

## Training

Training uses `torchrun` for multi-GPU DDP. Run from this directory.

**CIFAR-10:**

```bash
torchrun --nproc_per_node=4 cifar.py \
    --config config/cifar_unconditional_flux.yaml \
    --save_dir runs/cifar_flux \
    --use_dsm False
```

**CelebA-64:**

```bash
torchrun --nproc_per_node=4 celeba.py \
    --config config/celeba_unconditional_flux.yaml \
    --save_dir runs/celeba_flux \
    --use_dsm False
```

Pass `--use_dsm True` to train the DSM baseline instead.

Checkpoints are saved every 50K steps to `<save_dir>/ckpts/`. Sample grids are saved every 1K steps to `<save_dir>/visual/`. FID-10K is logged to `<save_dir>/eval.log` during training.

## Evaluation

Run from this directory. Computes FID-50K, IS, and NLL on the best checkpoint.

**CIFAR-10:**

```bash
torchrun --nproc_per_node=4 cifar_eval.py \
    --config config/cifar_unconditional_flux.yaml \
    --ckpt_path runs/cifar_flux/ckpts/model_stepXXXXX.pth
```

**CelebA-64:**

```bash
torchrun --nproc_per_node=4 celeba_eval.py \
    --config config/celeba_unconditional_flux.yaml \
    --ckpt_path runs/celeba_flux/ckpts/model_stepXXXXX.pth
```

Optional flags:

| Flag | Default | Description |
|---|---|---|
| `--sample_batch_size` | 500 | Images per GPU per sampling batch |
| `--nll_batch_size` | 64 / 32 | Batch size for NLL computation |
| `--n_samples_fid` | 50000 | Number of samples for FID |
| `--rtol` / `--atol` | 1e-5 | ODE solver tolerances |
| `--skip_fid` | — | Skip FID + NFE computation |
| `--skip_nll` | — | Skip NLL computation |

Results are appended to `eval_results.txt` in the checkpoint directory.

## References

The multi-GPU training infrastructure is based on:

> FutureXiang. Diffusion: Minimal multi-gpu implementation of diffusion models with classifier-free guidance (CFG). https://github.com/FutureXiang/Diffusion/tree/master, 2023.
