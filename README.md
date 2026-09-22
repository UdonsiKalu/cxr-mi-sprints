# CXR mechanistic interpretability — weekly sprints

## Why this started

CXR reads **oncology doctors’ notes** and asks a language model questions about them (treatment given, failed, stopped). The facts are in the note. The answers were sometimes still wrong. That mismatch is what we mean by a **neural–symbolic** failure: symbols in, wrong next token out.

**Example (Qwen2.5-7B-Instruct, recorded on Almera)**

| | |
|--|--|
| **Note** | Progress note: Oxaliplatin/5-FU course ended in March. Surveillance only. |
| **Question** | Has FOLFOX stopped? Answer yes or no. |
| **What it should say** | **Yes.** FOLFOX *is* oxaliplatin plus 5-FU. The course ended; the patient is on surveillance only. |
| **What the model said** | **No.** Based on the information provided, FOLFOX has not stopped. (*“Based on the information provided, it seems that the patient has completed a…”*) |

Same pattern on the teaching note we use in the notebooks: 
>*“Patient received FOLFOX. Disease progressed. FOLFOX was discontinued.”* Question: *Has first-line FOLFOX failed?* The note is a **yes**.

The evaluation problem is the model not answering that way from the text it was given. 

This work did not start as abstract interpretability. It started from those **CXR evaluation errors**. If we look *inside* the model while it reads the same notes, can we see where the story falls apart — and then change something so the answers get more reliable?

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
