# CXR mechanistic interpretability — weekly sprints

## Why this started

CXR reads **oncology doctors’ notes** and asks a language model questions about them: what treatment was given, whether it failed, what happens next.

When we did that, the answers were sometimes **wrong** even though the facts were sitting in the note. A typical failure: the note says FOLFOX was given, disease progressed, and FOLFOX was stopped — and the model still mixes up those events. That is a **neural–symbolic** miss: the text is in the prompt, but the model’s next-token machinery does not keep the facts straight.

So this work did not start as abstract interpretability. It started from **CXR evaluation errors**. The question became: if we can look *inside* the model (activations, layers, directions) while it reads the same notes, can we see where the story falls apart — and then change something so the answers get more reliable?

## What we aim to achieve

1. **See** what the model actually represents when it reads a CXR note (this table: capture through residual decomposition).
2. **Test** whether those internal pieces matter (later: intervene — steer, patch, ablate).
3. **Fix the pipeline** around the failures we can name (compensate: prompts, checks, gates) — and only then consider training-time changes.

This repo is the public write-up of that loop. Live runs are on the Almera lab (Jupyter + one shared Qwen). Notebooks hold the work; `sprints/` holds what we learned.

## Observe — representation

Read activations. Do not write the forward. **L2 = Euclidean norm**, not “layer 2”.

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

Other chapters: [`curriculum.md`](curriculum.md).
