#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)/_lib${PYTHONPATH:+:$PYTHONPATH}"
# Kernel env ships @jupyter-widgets/jupyterlab-manager; include it so Lab can render ipywidgets.
_FAISS_JUPYTER="/home/udonsi-kalu/staging/cxrlabs/faiss_gpu1/share/jupyter"
export JUPYTER_PATH="${_FAISS_JUPYTER}${JUPYTER_PATH:+:$JUPYTER_PATH}"
exec /home/udonsi-kalu/miniconda3/bin/jupyter lab \
  00_START_HERE.ipynb \
  --ip=0.0.0.0 \
  --notebook-dir="$(pwd)" \
  --no-browser \
  --port=8266 \
  --ServerApp.token='' \
  --ServerApp.password=''
