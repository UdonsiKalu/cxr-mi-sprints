# CXR mechanistic interpretability — weekly sprints

Portfolio of **1–2 week research sprints** on Qwen internals (CXR oncology notes), small questions, fast measurements, a write-up.

Live work runs on the Almera lab (Jupyter + one shared Qwen). This repo is the **public canon**: what we asked, what we ran, what we claim, and what we still do not know.

## Board (JIRA-style)

**[CXR MI weekly sprints](https://github.com/users/UdonsiKalu/projects/2)** — columns are maturity, not vibes:

| Column | Meaning |
|--------|---------|
| Backlog | Question is written. Not started. |
| This week | Timeboxed to the current sprint. |
| Investigating | Code / notebooks running. |
| Write-up | Numbers exist; claim text not frozen. |
| Done | Write-up in `sprints/` with a claim boundary. |

[Issues](https://github.com/UdonsiKalu/cxr-mi-sprints/issues) are the tickets. Each issue is **one question**.


## Jupyter notebooks

The Observe curriculum (capture → residual decomposition) lives in [`notebooks/02_observe/representation/`](notebooks/02_observe/representation/). Start at [`notebooks/00_START_HERE.ipynb`](notebooks/00_START_HERE.ipynb).

These are the teaching notebooks from the Almera lab. They run there against the shared Qwen on `:8270`. Opening them on GitHub is for reading; running them needs that lab.

## Sprint log

| Week | Issue | Question | Outcome |
|------|-------|----------|---------|
| 2026-W38 | #1 | Can we observe Qwen activations from Jupyter without loading a second copy? | Done — shared `:8270` + `look.cmd_*` |
| 2026-W39 | #2 | Last-token residual L2 by position on the FOLFOX teaching note (L20) | This week |

How to add a week: copy [`sprints/_TEMPLATE.md`](sprints/_TEMPLATE.md) → `sprints/YYYY-Www.md`, open an issue, put it in **This week**.

## Claim rule

A sprint is not a paper. **Observe ≠ causal.** If we did not intervene, we do not say the model “uses” a feature. A Done card always has a **claim boundary**.

## Lab (private working copy)

Notebooks and the model service live in the CXR teach stack on Almera (`cxr-mi-repeng-grounding/notebooks`). Frozen public demos: [cxr-evidence-grounding-lab](https://github.com/UdonsiKalu/cxr-evidence-grounding-lab).
