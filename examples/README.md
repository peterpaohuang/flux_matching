# Examples

## Setup

```bash
pip install jupyter notebook
```

## Notebooks

- **flux_matching_quickstart.ipynb** — Minimal setup to train a generative vector field with Flux Matching. Define a network, call `flux_matching_loss`, sample with Langevin dynamics. Start here.
- **annealed_flux_matching_quickstart.ipynb** — Minimal setup to train over noise annealed distributions with noise annealed Flux Matching.
- **flux_vs_dsm_comparison.ipynb** — Deep dive comparing Flux Matching against DSM on a 2D ring. Covers learned vector fields, curl structure, and injecting interesting structure into the vector field (like sprials), something only Flux Matching supports.
