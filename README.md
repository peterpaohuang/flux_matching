# Flux Matching

![Flux Matching](assets/flux_matching.gif)

**Flux Matching** is a generative modeling paradigm for learning data generating vector fields beyond the score function. The library (`src/`) is lightweight and dependency-free beyond PyTorch. Usage examples live in `examples/`, and each experiment in `experiments/` has its own README.

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

x = torch.randn(64, 2)          # clean samples
sigma2 = torch.tensor(0.1)      # noise variance

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
