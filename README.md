# CXR mechanistic interpretability — weekly sprints

Using ingested doctors' notes from CXR, I'll investigate errors that arise from systemic failures at the Neural-Symbolic Boundary in LLM output.

Portfolio of **1–2 week research sprints** on Qwen internals (CXR oncology notes): small questions, fast measurements, a write-up.

Live work runs on the Almera lab (Jupyter + one shared Qwen). This repo is the **public canon**: what we asked, what we ran, what we claim, and what we still do not know.

Start here: [`notebooks/00_START_HERE.ipynb`](notebooks/00_START_HERE.ipynb). GitHub is for reading. Running needs the Almera lab.

## Board (JIRA-style)

**[CXR MI weekly sprints](https://github.com/users/UdonsiKalu/projects/2)**

| Column | Meaning |
|--------|---------|
| Backlog | Question is written. Not started. |
| This week | Timeboxed to the current sprint. |
| Investigating | Code / notebooks running. |
| Write-up | Numbers exist; claim text not frozen. |
| Done | Write-up in `sprints/` with a claim boundary. |

[Issues](https://github.com/UdonsiKalu/cxr-mi-sprints/issues) are the tickets. Each issue is **one question**.

## Curriculum

Observe ≠ Intervene ≠ Compensate. A representation you can see is not a mechanism you have proven.

| Chapter | Role |
|---------|------|
| [Foundations](#1-foundations) | How a transformer writes |
| [Observe — representation](#2-observe--representation) | What's in the residual (this sprint: 01→13) |
| [Observe — output](#2b-observe--output) | What the model says next |
| [Intervene](#3-intervene) | Does it matter if we write the forward? |
| [Compensate](#4-compensate) | Work around it in text / tools |
| [Case studies](#5-case-studies) | CXR oncology investigations |
| [Python skills](#6-python-skills) | Tensors, hooks, HF |
| [Research notes](#7-research-notes) | Concepts, hypotheses, log |
| [Reference suite](#8-reference-suite) | Frozen regression |
| [Interactive lab](#9-interactive-lab) | Existing GUIs in Jupyter |

---

### 1. Foundations

| # | Topic | Notebook |
|---|--------|----------|
| — | Overview | [`00_OVERVIEW`](notebooks/01_foundations/00_OVERVIEW.ipynb) |
| 1 | Tokens and the last index | [`01_tokens`](notebooks/01_foundations/01_tokens.ipynb) |
| 2 | 3584-D last-token hidden state | [`02_hidden_states`](notebooks/01_foundations/02_hidden_states.ipynb) |
| 3 | Residual stream: attn write → add → MLP write → add | [`03_residual_stream`](notebooks/01_foundations/03_residual_stream.ipynb) |
| 4 | Attention as a write into the residual | [`04_attention`](notebooks/01_foundations/04_attention.ipynb) |
| 5 | MLP as a write into the residual | [`05_mlp`](notebooks/01_foundations/05_mlp.ipynb) |
| 6 | One transformer block end-to-end | [`06_transformer_block`](notebooks/01_foundations/06_transformer_block.ipynb) |
| 7 | Next-token logits | [`07_logits`](notebooks/01_foundations/07_logits.ipynb) |
| 8 | Forward hooks (how look.py sees inside) | [`08_hooks`](notebooks/01_foundations/08_hooks.ipynb) |
| 9 | Evolution through the transformer | [`09_evolution_through_transformer`](notebooks/01_foundations/09_evolution_through_transformer.ipynb) |

### 2. Observe — representation

Read activations. **Do not write** the forward. **L2 = Euclidean norm**, not “layer 2”.

| # | Topic | Notebook | What you actually do |
|---|--------|----------|----------------------|
| 1 | Capture last-token residual | [`01_capture`](notebooks/02_observe/representation/01_capture.ipynb) | One vector at L20, last token: shape, L2, first 10 dims |
| 2 | Per-token L2 | [`02_positions`](notebooks/02_observe/representation/02_positions.ipynb) | Same layer, every token in the note |
| 3 | Last-token L2 by layer | [`03_layer_sweep`](notebooks/02_observe/representation/03_layer_sweep.ipynb) | How that last-token L2 changes L0 → L27 |
| 4 | Difference direction A − B | [`04_difference_vectors`](notebooks/02_observe/representation/04_difference_vectors.ipynb) | `h_A − h_B` at L20 |
| 5 | Cosine to A−B | [`05_cosine_similarity`](notebooks/02_observe/representation/05_cosine_similarity.ipynb) | How aligned a note is with that direction |
| 6 | Project onto (A−B)/‖·‖ | [`06_direction_projection`](notebooks/02_observe/representation/06_direction_projection.ipynb) | Scalar score on the unit difference vector |
| 7 | PCA / UMAP | [`07_pca_umap`](notebooks/02_observe/representation/07_pca_umap.ipynb) | Geometry of many residuals in 2D |
| 8 | Linear probes | [`08_linear_probes`](notebooks/02_observe/representation/08_linear_probes.ipynb) | Can a linear map read a label out of `h`? |
| 9 | Chanin L20 SAE features | [`09_sae`](notebooks/02_observe/representation/09_sae.ipynb) | Sparse features on the L20 residual |
| 10 | Attention patterns | [`10_attention_patterns`](notebooks/02_observe/representation/10_attention_patterns.ipynb) | Which tokens attend to which |
| 11 | Per-head analysis | [`11_per_head`](notebooks/02_observe/representation/11_per_head.ipynb) | Same idea, one attention head at a time |
| 12 | MLP features | [`12_mlp_features`](notebooks/02_observe/representation/12_mlp_features.ipynb) | What the MLP writes into the residual |
| 13 | Residual decomposition | [`13_residual_decomposition`](notebooks/02_observe/representation/13_residual_decomposition.ipynb) | Split `h` into attn + MLP + residual pieces |

**This sprint** is the Observe representation table (01 → 13). Executable now: 1–6 and 9. Stubs (not in `look.py` yet): 7–8 and 10–13.

### 2b. Observe — output

| # | Topic | Notebook | What you actually do |
|---|--------|----------|----------------------|
| 1 | Next-token logits | [`01_logits`](notebooks/02_observe/output/01_logits.ipynb) | Top next-token scores |
| 2 | Logit lens | [`02_logit_lens`](notebooks/02_observe/output/02_logit_lens.ipynb) | Decode the residual as if it were final |
| 3 | YES − NO logit margin | [`03_logit_margin`](notebooks/02_observe/output/03_logit_margin.ipynb) | Decision margin, not a feature claim |
| 4 | Direct logit attribution | [`04_dla`](notebooks/02_observe/output/04_dla.ipynb) | Which residual pieces move the logit (planned) |

### 3. Intervene

Write the forward. Causal evidence only after a flip.

| # | Topic | Notebook |
|---|--------|----------|
| — | Overview | [`00_OVERVIEW`](notebooks/03_intervene/00_OVERVIEW.ipynb) |
| 1 | Steering along A−B | [`01_steering`](notebooks/03_intervene/activations/01_steering.ipynb) |
| 2 | Activation patch (A → B) | [`02_activation_patching`](notebooks/03_intervene/activations/02_activation_patching.ipynb) |
| 3 | Residual site zero (block) | [`03_residual_patching`](notebooks/03_intervene/activations/03_residual_patching.ipynb) |
| 4 | SAE intervention | [`04_sae_intervention`](notebooks/03_intervene/activations/04_sae_intervention.ipynb) |
| 5 | Logit bias (Yes+/No−) | [`05_logit_bias`](notebooks/03_intervene/activations/05_logit_bias.ipynb) |
| 6 | MLP last-token zero | [`01_mlp_ablation`](notebooks/03_intervene/components/01_mlp_ablation.ipynb) |
| 7 | Attention last-token zero | [`02_attention_ablation`](notebooks/03_intervene/components/02_attention_ablation.ipynb) |
| 8 | Head ablation | [`03_head_ablation`](notebooks/03_intervene/components/03_head_ablation.ipynb) |
| 9 | Feature ablation | [`04_feature_ablation`](notebooks/03_intervene/components/04_feature_ablation.ipynb) |
| 10 | Circuits / paths | [`05_circuits_paths`](notebooks/03_intervene/components/05_circuits_paths.ipynb) |

### 4. Compensate

Text and tools around the model. Not a mechanistic proof.

| # | Topic | Notebook |
|---|--------|----------|
| — | Overview | [`00_OVERVIEW`](notebooks/04_compensate/00_OVERVIEW.ipynb) |
| 1 | Prompt scaffold | [`01_prompting`](notebooks/04_compensate/01_prompting.ipynb) |
| 2 | Retrieval prepend | [`02_retrieval`](notebooks/04_compensate/02_retrieval.ipynb) |
| 3 | Constrained JSON output | [`03_structured_output`](notebooks/04_compensate/03_structured_output.ipynb) |
| 4 | Tool-augmented prompt | [`04_tools`](notebooks/04_compensate/04_tools.ipynb) |
| 5 | Symbolic cue check | [`05_symbolic_rules`](notebooks/04_compensate/05_symbolic_rules.ipynb) |
| 6 | Overlap verifier | [`06_verifiers`](notebooks/04_compensate/06_verifiers.ipynb) |
| 7 | Policy gate → REVIEW | [`07_gate_review`](notebooks/04_compensate/07_gate_review.ipynb) |
| 8 | Extract repairs | [`08_extract_repairs`](notebooks/04_compensate/08_extract_repairs.ipynb) |

### 5. Case studies

| # | Topic | Notebook |
|---|--------|----------|
| — | Overview | [`00_OVERVIEW`](notebooks/05_case_studies/00_OVERVIEW.ipynb) |
| 1 | L24 direction d | [`01_l24_direction_d`](notebooks/05_case_studies/01_l24_direction_d.ipynb) |
| 2 | L20 MLP v | [`02_l20_mlp_v`](notebooks/05_case_studies/02_l20_mlp_v.ipynb) |
| 3 | Translate G3–G9 | [`03_translate_g3_g9`](notebooks/05_case_studies/03_translate_g3_g9.ipynb) |
| 4 | Full oncology investigation | [`04_full_oncology_investigation`](notebooks/05_case_studies/04_full_oncology_investigation.ipynb) |

### 6. Python skills

| # | Topic | Notebook |
|---|--------|----------|
| — | Overview | [`00_OVERVIEW`](notebooks/06_python_skills/00_OVERVIEW.ipynb) |
| 1 | Tensors | [`01_tensors`](notebooks/06_python_skills/01_tensors.ipynb) |
| 2 | Indexing | [`02_indexing`](notebooks/06_python_skills/02_indexing.ipynb) |
| 3 | Matrix / vector ops | [`03_matrix_vector_ops`](notebooks/06_python_skills/03_matrix_vector_ops.ipynb) |
| 4 | PyTorch hooks | [`04_pytorch_hooks`](notebooks/06_python_skills/04_pytorch_hooks.ipynb) |
| 5 | NumPy | [`05_numpy`](notebooks/06_python_skills/05_numpy.ipynb) |
| 6 | Plotting | [`06_plotting`](notebooks/06_python_skills/06_plotting.ipynb) |
| 7 | Hugging Face | [`07_huggingface`](notebooks/06_python_skills/07_huggingface.ipynb) |

### 7. Research notes

| Topic | Notebook |
|--------|----------|
| Index | [`00_INDEX`](notebooks/07_research_notes/00_INDEX.ipynb) |
| Concepts | [`concepts`](notebooks/07_research_notes/concepts.ipynb) |
| Hypotheses | [`hypotheses`](notebooks/07_research_notes/hypotheses.ipynb) |
| Experiment log | [`experiment_log`](notebooks/07_research_notes/experiment_log.ipynb) |
| Open questions | [`open_questions`](notebooks/07_research_notes/open_questions.ipynb) |

### 8. Reference suite

| Topic | Notebook |
|--------|----------|
| Overview | [`00_OVERVIEW`](notebooks/08_reference_suite/00_OVERVIEW.ipynb) |
| RUN_SUITE | [`RUN_SUITE`](notebooks/08_reference_suite/RUN_SUITE.ipynb) |

### 9. Interactive lab

| Topic | Notebook |
|--------|----------|
| Embed existing GUIs | [`12_INTERACTIVE_LAB`](notebooks/09_interactive_lab/12_INTERACTIVE_LAB.ipynb) |

## Sprint log

| Week | Issue | Question | Outcome |
|------|-------|----------|---------|
| 2026-W38 | [#1](https://github.com/UdonsiKalu/cxr-mi-sprints/issues/1) | Observe Qwen from Jupyter without a second load | Done — shared `:8270` + `look.cmd_*` |
| 2026-W39 | [#2](https://github.com/UdonsiKalu/cxr-mi-sprints/issues/2) | Observe representation 01 capture → 13 residual decomposition | This week |

How to add a week: copy [`sprints/_TEMPLATE.md`](sprints/_TEMPLATE.md) → `sprints/YYYY-Www.md`, open an issue, put it in **This week**.

## Claim rule

A sprint is not a paper. **Observe ≠ causal.** If we did not intervene, we do not say the model “uses” a feature. A Done card always has a **claim boundary**.

## Lab

Frozen public demos: [cxr-evidence-grounding-lab](https://github.com/UdonsiKalu/cxr-evidence-grounding-lab). Private working copy of notebooks + model service: Almera `cxr-mi-repeng-grounding`.
