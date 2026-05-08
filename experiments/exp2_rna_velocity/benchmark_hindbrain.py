import numpy as np
import pandas as pd
import scvelo as scv
import scanpy as sc
import torch

from scvelo.tools import (
    velocity_confidence,
    cross_boundary_correctness,
    flux_velocity
)

scv.settings.verbosity = 0  # reduce verbosity
scv.settings.set_figure_params("scvelo", transparent=False)

DATASET = "hindbrain"
SEED = 0

# Dataset configuration
cluster_edges = [
    ("Neural stem cells", "Proliferating VZ progenitors"),
    ("Proliferating VZ progenitors", "VZ progenitors"),
    ("VZ progenitors", "Differentiating GABA interneurons"),
    ("VZ progenitors", "Gliogenic progenitors"),
    ("Differentiating GABA interneurons", "GABA interneurons"),
]
color = "Celltype"
emb_type = "tsne"
x_key_emb = f"X_{emb_type}"

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

# ==================== Flux Matching ====================
print("Working on Flux Matching...")
adata = sc.read("data/h5ad_files/Hindbrain_GABA_Glio.h5ad", cache=True)

sc.pp.filter_cells(adata, min_counts=1)
scv.pp.filter_and_normalize(adata, min_shared_counts=20)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat", subset=True)
sc.pp.pca(adata)
sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30)
scv.pp.moments(adata, n_neighbors=None, n_pcs=None)

if x_key_emb not in adata.obsm:
    if emb_type == "umap":
        sc.tl.umap(adata)
    elif emb_type == "tsne":
        sc.tl.tsne(adata)

scv.tl.velocity(adata, mask_zero=False)
flux_velocity(
    adata,
    model_family="dynamical",
    lr=1e-3,
    epochs=100,
)

scv.tl.velocity_graph(adata, n_jobs=8)
scv.tl.velocity_embedding(adata, basis=emb_type)

flux_adata = adata.copy()

# ==================== Dynamical ====================
print("Working on Dynamical...")
adata = sc.read("data/h5ad_files/Hindbrain_GABA_Glio.h5ad", cache=True)

sc.pp.filter_cells(adata, min_counts=1)
scv.pp.filter_and_normalize(adata, min_shared_counts=20)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat", subset=True)
sc.pp.pca(adata)
sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30)
scv.pp.moments(adata, n_neighbors=None, n_pcs=None)

if x_key_emb not in adata.obsm:
    if emb_type == "umap":
        sc.tl.umap(adata)
    elif emb_type == "tsne":
        sc.tl.tsne(adata)

scv.tl.recover_dynamics(adata, n_jobs=8)
scv.tl.velocity(adata, mode="dynamical")

scv.tl.velocity_graph(adata, n_jobs=8)
scv.tl.velocity_embedding(adata, basis=emb_type)

dynamical_adata = adata.copy()

# ==================== Compute Metrics ====================
print("Computing metrics...")
vkey = "velocity"
method = "cosine"

results = []
for method_name, method_adata in [
    ("Flux Matching", flux_adata),
    ("Dynamical", dynamical_adata)
]:
    cbcs, avg_cbc = cross_boundary_correctness(
        method_adata, color, vkey, cluster_edges, return_raw=False, x_emb_key=x_key_emb
    )

    velocity_confidence(method_adata, vkey=vkey, method=method)
    consistency = method_adata.obs[vkey + '_confidence_' + method].mean()

    results.append({
        'dataset': DATASET,
        'method': method_name,
        'cross_boundary_correctness': avg_cbc,
        'consistency': consistency,
    })

    print(f"{method_name} - CBC: {avg_cbc:.4f}, Consistency: {consistency:.4f}")

df = pd.DataFrame(results)
output_file = f"results_{DATASET}.csv"
df.to_csv(output_file, index=False)
print(f"\nResults saved to {output_file}")
print(df.to_string(index=False))
