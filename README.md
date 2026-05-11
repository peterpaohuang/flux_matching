# Flux Matching

![Flux Matching](assets/flux_matching.gif)

Official implementation of [Generative Modeling with Flux Matching](https://arxiv.org/abs/2605.07319) by Peter Pao-Huang, Xiaojie Qiu, and Stefano Ermon.

**Flux Matching** is a generative modeling paradigm for learning data generating vector fields beyond the score function. `src/` contains the core Flux Matching library. `examples/` contains simple usage examples to start training and sampling with Flux Matching. `experiments/` contains all scripts and settings to setup, run, and recreate experiment results from the paper.

Please contact <peterph@stanford.edu> with any comments or issues.

---

## Installation

This project was developed with Python 3.10. Other versions are untested.

The core library requires only PyTorch:

```bash
pip install torch
```

For the examples and experiments, install additional dependencies as needed:

```bash
pip install torchdiffeq torchvision ema-pytorch scipy numpy matplotlib imageio tqdm pyyaml POT scanpy pandas
```
---

## Quickstart

```python
import torch
from src.loss import flux_matching_loss

# Define a simple vector field network
model = torch.nn.Sequential(
    torch.nn.Linear(2, 128), torch.nn.SiLU(),
    torch.nn.Linear(128, 2),
)

x = torch.randn(64, 2)
sigma2 = torch.tensor(0.1)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(1000):
    loss = flux_matching_loss(model, x, sigma2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

More comprehensive examples including noise annealed Flux Matching are in [examples/](examples/).

---

## Reproducing Experiments

Each experiment has its own self-contained directory with a dedicated README covering data preparation, training commands, and evaluation:

| Experiment | Description |
|---|---|
| [exp1_controllable](experiments/exp1_controllable/) | Controllable Generative Fields (Toy) |
| [exp2_rna_velocity](experiments/exp2_rna_velocity/) | Interpretable Generative Fields (RNA velocity) |
| [exp3_unrestricted](experiments/exp3_unrestricted/) | Unrestricted Generative Fields (Images: CIFAR-10, CelebA) |
| [exp4_fast_mixing](experiments/exp4_fast_mixing/) | Fast Mixing Fields for Accelerated Sampling (Images: CIFAR-10, CelebA) |
| [exp5_embed_structure](experiments/exp5_embed_structure/) | Embedding Structure in Generative Fields (Physics Simulation) |

See each experiment's `README.md` for full reproduction instructions.

## Citation

```
@misc{paohuang2026generativemodelingfluxmatching,
      title={Generative Modeling with Flux Matching}, 
      author={Peter Pao-Huang and Xiaojie Qiu and Stefano Ermon},
      year={2026},
      eprint={2605.07319},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```