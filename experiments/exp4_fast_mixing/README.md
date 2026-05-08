# Experiment 4: Fast Mixing Generative Fields for Accelerated Sampling (Images)

This experiment extends Experiment 3 with a mixing regularization term (`mixing_lam`) added to the flux-matching loss. The goal is faster mixing during sampling, allowing high-quality generation with fewer function evaluations (NFE). Evaluation uses a predictor-corrector (PC) sampler and plots FID vs. NFE curves.

## Data

Same as exp3. See [exp3's README](../exp3_unrestricted/README.md) for CIFAR-10 and CelebA setup, and FID statistics.

## Training

Identical to exp3 but uses configs with `mixing: True` and a `mixing_lam` regularization weight. Run from this directory.

**CIFAR-10:**

```bash
torchrun --nproc_per_node=4 cifar.py \
    --config config/cifar_unconditional_flux_mixing.yaml \
    --save_dir runs/cifar_flux_mixing
```

**CelebA-64:**

```bash
torchrun --nproc_per_node=4 celeba.py \
    --config config/celeba_unconditional_flux_mixing.yaml \
    --save_dir runs/celeba_flux_mixing
```

The mixing weight is set via `mixing_lam` in the config (default: `0.01`). Note that **DSM does not support fast mixing** (since it can only learn the score). 

## Evaluation

Evaluation uses a predictor-corrector sampler and sweeps over PC step counts to produce a **FID vs. NFE** curve. Run from this directory.

**CIFAR-10:**

```bash
torchrun --nproc_per_node=4 cifar_eval_pc.py \
    --config config/cifar_unconditional_flux_mixing.yaml \
    --ckpt_path runs/cifar_flux_mixing/ckpts/model_stepXXXXX.pth
```

**CelebA-64:**

```bash
torchrun --nproc_per_node=4 celeba_eval_pc.py \
    --config config/celeba_unconditional_flux_mixing.yaml \
    --ckpt_path runs/celeba_flux_mixing/ckpts/model_stepXXXXX.pth
```

Optional flags:

| Flag | Default | Description |
|---|---|---|
| `--n_samples` | 1000 | Images per PC step count |
| `--sample_batch_size` | 100 | Images per GPU per batch |
| `--corrector_snr` | 0.16 | Signal-to-noise ratio for the Langevin corrector step |

The script sweeps PC step counts (in steps of 10) and saves a figure of FID vs. NFE to the checkpoint directory.

## References

The multi-GPU training infrastructure is based on:

> FutureXiang. Diffusion: Minimal multi-gpu implementation of diffusion models with classifier-free guidance (CFG). https://github.com/FutureXiang/Diffusion/tree/master, 2023.

PC Sampling is based on:

> Song, Yang, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. "Score-based generative modeling through stochastic differential equations." arXiv preprint arXiv:2011.13456 (2020).
