# CXR mechanistic interpretability — weekly sprints

Using ingested doctors' notes from CXR, I'll investigate errors that arise from systemic failures at the Neural-Symbolic Boundary in LLM output.

Portfolio of **1–2 week research sprints** on Qwen internals (CXR oncology notes): small questions, fast measurements, a write-up.

Live work runs on the Almera lab (Jupyter + one shared Qwen). This repo is the **public canon**: what we asked, what we ran, what we claim, and what we still do not know.

| # | Topic | Notebook | What I intend to achieve |
|---|--------|----------|--------------------------|
| 1 | Capture last-token residual | [`01_capture`](notebooks/02_observe/representation/01_capture.ipynb) | One L20 last-token vector: shape, L2, first 10 dims |
| 2 | Per-token L2 | [`02_positions`](notebooks/02_observe/representation/02_positions.ipynb) | Residual L2 at L20 for every token in the note |
| 3 | Last-token L2 by layer | [`03_layer_sweep`](notebooks/02_observe/representation/03_layer_sweep.ipynb) | How last-token L2 changes from L0 to L27 |
| 4 | Difference direction A − B | [`04_difference_vectors`](notebooks/02_observe/representation/04_difference_vectors.ipynb) | The vector `h_A − h_B` at L20 |
| 5 | Cosine to A−B | [`05_cosine_similarity`](notebooks/02_observe/representation/05_cosine_similarity.ipynb) | How aligned a note is with that direction |
| 6 | Project onto (A−B)/‖·‖ | [`06_direction_projection`](notebooks/02_observe/representation/06_direction_projection.ipynb) | A scalar score of the note on the unit A−B direction |
| 7 | PCA / UMAP | [`07_pca_umap`](notebooks/02_observe/representation/07_pca_umap.ipynb) | 2D geometry of many residuals |
| 8 | Linear probes | [`08_linear_probes`](notebooks/02_observe/representation/08_linear_probes.ipynb) | Whether a linear map can read a label out of `h` |
| 9 | Chanin L20 SAE features | [`09_sae`](notebooks/02_observe/representation/09_sae.ipynb) | Sparse features on the L20 residual |
| 10 | Attention patterns | [`10_attention_patterns`](notebooks/02_observe/representation/10_attention_patterns.ipynb) | Which tokens attend to which |
| 11 | Per-head analysis | [`11_per_head`](notebooks/02_observe/representation/11_per_head.ipynb) | The same, one attention head at a time |
| 12 | MLP features | [`12_mlp_features`](notebooks/02_observe/representation/12_mlp_features.ipynb) | What the MLP writes into the residual |
| 13 | Residual decomposition | [`13_residual_decomposition`](notebooks/02_observe/representation/13_residual_decomposition.ipynb) | Split `h` into attn + MLP + residual pieces |

Other chapters (Foundations, Intervene, Compensate, …): [`curriculum.md`](curriculum.md).
