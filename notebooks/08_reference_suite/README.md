# 08 — Reference / regression suite

**Add-only.** Does not rewrite curriculum notebooks you’ve been editing.

## Purpose

Machine-checkable goldens for important Observe / Intervene cases:

- GUI E00 metrics (width, residual add-check)
- A/B representation sweep (`PROMPT_A` vs `PROMPT_B`)
- Causal MLP-zero candidate grid (L16…L27) — locks the 09 lesson finding: **max separation ≠ max causal effect**

## One model in memory

Run on the **same Jupyter kernel** that already holds Qwen (e.g. Connect to Existing `119f847c`).

- `cxr_suite.require_live_model()` **refuses** a fresh `from_pretrained`
- Do **not** start Streamlit `:8265` at the same time (second process = second VRAM load)

## How to run

1. Open [`RUN_SUITE.ipynb`](./RUN_SUITE.ipynb) on the live kernel  
2. Run setup (REUSE only) → run suite  
3. Results land in `artifacts/experiments/<id>.json`

## Layout

| Path | Role |
|------|------|
| `registry.json` | experiment IDs + tags |
| `expected/` | golden metrics |
| `nulls/` | known-null / failed shelf (templates) |
| `../_lib/cxr_suite.py` | runners + compare |

## Later phases (not this pass)

Steer / patch / attn-zero / SAE / controls / arbitrary-note harness / pytest CI — add new IDs only; still no overwrite of lesson notebooks.
