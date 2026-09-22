# Notebooks — CXR MI / RepEng Workbook

**Start here:** [`00_START_HERE.ipynb`](./00_START_HERE.ipynb)

## Open Jupyter Lab

```bash
cd /home/udonsi-kalu/staging/cxr-mi-repeng-grounding/notebooks
./run_jupyter.sh
```

http://127.0.0.1:8266 — kernel **CXR local Qwen (faiss_gpu1)**

## How to run a lesson

1. Open any **executable** `.ipynb`
2. Run the **setup** cell (loads local Qwen once ≈1 min)
3. Run the **mode** cell — same output as `look.py` / `intervene.py` / `process.py`

Shared helper: `_lib/cxr_boot.py` (imports `scripts/`).

## Executable map

| Folder | Backed by |
|--------|-----------|
| `01_foundations/` | tokenizer + `look` helpers / hooks + **09 L0–L27 evolution** |
| `02_observe/` | `scripts/look.py` modes |
| `03_intervene/` | `scripts/intervene.py` modes |
| `04_compensate/` | `scripts/process.py` modes |
| `08_reference_suite/` | regression goldens — run on **live** kernel only ([README](./08_reference_suite/README.md)) |
| `12_interactive_lab/` | native **ipywidgets** lab → :8270 ([12_INTERACTIVE_LAB](./12_interactive_lab/12_INTERACTIVE_LAB.ipynb)) |
| `11_model_service/` | client for **:8270** model/experiment service ([README](./11_model_service/README.md)) |
| `10_n2s_port_replays/` | N2S port A/B science (:8258/:8259/:8260/:8263) in Jupyter — **not** Streamlit ([README](./10_n2s_port_replays/README.md)) |

Still stubs (not in CLI yet): PCA, probes, per-head, DLA, circuits, extract-repairs, most case studies / python-skills.

## All-in-one legacy notebooks

| File | Role |
|------|------|
| `look_observe.ipynb` | every look mode in one notebook |
| `intervene.ipynb` | every intervene mode |
| `process.ipynb` | every process mode |

## Related

- Guided Streamlit M1: `../learning_lab/` → **:8265**
- CLI: `../scripts/look.py` etc.
