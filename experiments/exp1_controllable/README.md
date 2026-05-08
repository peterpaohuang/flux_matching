# Experiment 1: Controllable Generative Fields (Toy)

This experiment studies the space of vector fields consistent with a fixed stationary distribution. It computes a Pareto frontier across three geometric properties of 2D vector fields on a Gaussian mixture:

- **Mixing speed** — how quickly the field mixes probability mass
- **Triangle shape** (clockwise cycle alignment) — degree of circular flow between mixture components
- **Jacobian skewness** — asymmetry in the local flow structure

No training or dataset download is required. Everything runs analytically on CPU.

## Data

No external data. The target distribution is a 3-component Gaussian mixture in 2D, generated on the fly.

## Running
Run from this directory.
```bash
python test.py
```

Optional flags:

```bash
python test.py \
  --outdir my_output \
  --device cpu \
  --dtype float64 \
  --seed 123 \
  --sigma 0.45
```

## Output

Results are saved to `controllable_vector_field_frontier_output/` (or `--outdir`) and include:

- `compatibility_curves.png` — Pareto frontier curves across the three metrics
- `provocative_fields.png` — fixed-length vector field visualizations
- `provocative_fields_streamlines.png` — streamline visualizations
