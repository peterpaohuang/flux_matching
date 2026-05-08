# Experiment 2: Interpretable Generative Fields (RNA Velocity)

This experiment benchmarks Flux Matching velocity estimation against scVelo's dynamical model across five single-cell datasets. Each benchmark script runs the full pipeline — preprocessing, velocity computation, and evaluation — for multiple random seeds and reports mean metrics.

## Data

Most datasets are downloaded automatically via `scvelo.datasets.*`. The following datasets are used:

| Script | Dataset | Source |
|---|---|---|
| `benchmark_pancreas.py` | Pancreas endocrinogenesis | `scv.datasets.pancreas()` — auto-download |
| `benchmark_bone_marrow.py` | Human CD34+ bone marrow | `scv.datasets.bonemarrow()` — auto-download |
| `benchmark_gastrulation.py` | Mouse erythroid lineage | `scv.datasets.gastrulation_erythroid()` — auto-download |
| `benchmark_dentategyrus.py` | Dentate gyrus | `scv.datasets.dentategyrus()` — auto-download |
| `benchmark_hindbrain.py` | Hindbrain GABA/Glio | `data/h5ad_files/Hindbrain_GABA_Glio.h5ad` — must be placed manually |

For the hindbrain experiment, download `Hindbrain_GABA_Glio.h5ad` from [Google Drive](https://drive.google.com/file/d/1pd69BiS0TrtLS-lwHP9ZqJgFSEPjdwXP/view?usp=sharing) and place it at `data/h5ad_files/Hindbrain_GABA_Glio.h5ad` relative to the script.

## Running

Run each benchmark script individually from this directory:

```bash
python benchmark_pancreas.py
python benchmark_bone_marrow.py
python benchmark_gastrulation.py
python benchmark_dentategyrus.py
python benchmark_hindbrain.py
```

## Evaluation

Each script reports the following metrics (averaged across seeds):

- **Velocity confidence** — coherence of the velocity field
- **Cross-boundary correctness (CBDir)** — fraction of cells correctly directed across known lineage boundaries

Results are printed to stdout at the end of each run.

## References

The baseline velocity model and preprocessing pipeline are built on scVelo:

> Bergen, V., Lange, M., Peidli, S., Wolf, F. A., & Theis, F. J. (2020). Generalizing RNA velocity to transient cell states through dynamical modeling. *Nature Biotechnology*, 38(12), 1408–1414. https://doi.org/10.1038/s41587-020-0591-3
