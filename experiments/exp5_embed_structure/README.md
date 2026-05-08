# Experiment 5: Embedding Structure in Generative Fields (Physics Simulation)

This experiment studies the impact of inposing causal (autoregressive) attention in the vector field model with Flux Matching vs DSM on trajectory data. The target distribution is length-50 trajectories from two coupled nonlinear springs, where each sample lives in `R^{50 x 4}` with state `(q1, v1, q2, v2)`. Four models are trained and compared via Wasserstein distance:

- DSM + noncausal attention
- DSM + causal attention
- Flux Matching loss + noncausal attention
- Flux Matching loss + causal attention

## Data

Generate the dataset before training:

```bash
python simulate_springs.py \
  --num-samples 2000 \
  --out data/springs_dataset.npz \
  --gif-out data/example_real.gif
```

A pre-generated dataset is already included at `data/springs_dataset.npz`.

## Training

Train each of the four model variants. Run from this directory.

```bash
python train.py --data data/springs_dataset.npz --outdir runs/dsm_noncausal     --objective dsm
python train.py --data data/springs_dataset.npz --outdir runs/dsm_causal        --objective dsm    --causal
python train.py --data data/springs_dataset.npz --outdir runs/flux_noncausal  --objective flux
python train.py --data data/springs_dataset.npz --outdir runs/flux_causal     --objective flux --causal
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--steps` | 10000 | Total training steps |
| `--batch-size` | 256 | Batch size |
| `--lr` | 3e-4 | Learning rate |
| `--hidden-dim` | 256 | Attention model width |
| `--depth` | 4 | Number of attention layers |
| `--eval-every` | 1000 | Steps between Wasserstein evaluations |
| `--transport` | `exact` | Wasserstein solver (`exact` or `sinkhorn`) |
| `--ema-decay` | 0.99 | EMA decay for model weights |

Each run writes to its `--outdir`:
- `train_log.csv` — loss and Wasserstein distance per eval step
- `config.json` — full configuration
- `checkpoint_best.pt` and periodic checkpoints
- Per-eval trajectory plots and GIFs

## Evaluation

Wasserstein curves for all four runs are compared in a single plot (run from this directory):

```bash
python plot_wasserstein.py
```

This reads `train_log.csv` from each subdirectory of `runs/` and saves comparison figures to the same directory.

To generate GIFs from saved trajectories:

```bash
python make_gifs.py \
  --real data/springs_dataset.npz \
  --generated runs/flux_causal/generated_trajectories.npy \
  --outdir viz/
```
