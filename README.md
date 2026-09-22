# CXR mechanistic interpretability — weekly sprints

Using ingested doctors' notes from CXR, I'll investigate errors that arise from systemic failures at the Neural-Symbolic Boundary in LLM output.

Portfolio of **1–2 week research sprints** on Qwen internals (CXR oncology notes): small questions, fast measurements, a write-up.

Live work runs on the Almera lab (Jupyter + one shared Qwen). This repo is the **public canon**: what we asked, what we ran, what we claim, and what we still do not know.

Notebook tables (every chapter, every file): **[curriculum.md](curriculum.md)**.  
Start notebook: [`00_START_HERE.ipynb`](notebooks/00_START_HERE.ipynb). GitHub is for reading; running is on Almera.

## This week

**Observe — representation:** capture last-token residual through residual decomposition (notebooks 01–13).

| Activity | Status |
|----------|--------|
| Capture last-token residual | In progress |
| Per-token L2 | This week |
| Last-token L2 by layer | This week |
| Difference direction A − B | This week |
| Cosine / projection onto A−B | This week |
| SAE features (L20) | This week |
| PCA, probes, attention, heads, MLP, residual split | Planned (stubs) |

Detail and links: [Observe — representation](curriculum.md#2-observe--representation). Board: [CXR MI weekly sprints](https://github.com/users/UdonsiKalu/projects/2).

## Later sprints (high level)

| Chapter | What it is |
|---------|------------|
| Foundations | How a transformer writes (tokens → residual → logits) |
| Observe — output | Next-token logits, logit lens, margin |
| Intervene | Write the forward — does it matter? |
| Compensate | Work around it in text / tools |
| Case studies | CXR oncology investigations |
| Python skills / notes / suite | Tooling and frozen regression |

## Sprint log

| Week | Issue | Question | Outcome |
|------|-------|----------|---------|
| 2026-W38 | [#1](https://github.com/UdonsiKalu/cxr-mi-sprints/issues/1) | Observe Qwen from Jupyter without a second load | Done |
| 2026-W39 | [#2](https://github.com/UdonsiKalu/cxr-mi-sprints/issues/2) | Observe representation 01 → 13 | This week |

## Claim rule

**Observe ≠ causal.** A Done write-up always states what we can and cannot say.

Frozen demos: [cxr-evidence-grounding-lab](https://github.com/UdonsiKalu/cxr-evidence-grounding-lab).
