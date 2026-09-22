"""Shared setup for curriculum notebooks. Import after adding notebooks/_lib to path."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def pack_root() -> Path:
    here = Path(__file__).resolve().parent  # notebooks/_lib
    return here.parent.parent  # cxr-mi-repeng-grounding


def setup(*, layer: int = 20, max_new: int = 24, load_model: bool = True) -> dict[str, Any]:
    """Offline env + scripts on path + optional Qwen load.

    Returns dict with: model, tok, NOTE, LAYER, MAX_NEW, PROMPT_A/B/TEST, look, intervene, process
    """
    root = pack_root()
    scripts = root / "scripts"
    cases_path = root / "cases" / "oncology" / "teaching_cases.json"
    if not cases_path.is_file():
        cases_path = root / "learning_lab" / "cases" / "oncology_m1.json"

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    sp = str(scripts)
    if sp not in sys.path:
        sys.path.insert(0, sp)

    import look
    import intervene
    import process
    from _common import (
        DEFAULT_MODEL,
        PROMPT_A,
        PROMPT_B,
        PROMPT_TEST,
        boot,
        describe_backend,
        device,
        encode,
        last_residual,
        full_residual,
        blocks,
    )

    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    note = cases.get("foundation_note") or cases["cases"][0]["text"]

    out: dict[str, Any] = {
        "NOTE": note,
        "LAYER": layer,
        "MAX_NEW": max_new,
        "DEFAULT_MODEL": DEFAULT_MODEL,
        "PROMPT_A": PROMPT_A,
        "PROMPT_B": PROMPT_B,
        "PROMPT_TEST": PROMPT_TEST,
        "look": look,
        "intervene": intervene,
        "process": process,
        "device": device,
        "encode": encode,
        "last_residual": last_residual,
        "full_residual": full_residual,
        "blocks": blocks,
        "model": None,
        "tok": None,
    }
    if load_model:
        existing = None
        try:
            from IPython import get_ipython

            ip = get_ipython()
            ns = ip.user_ns if ip is not None else {}
            existing = ns.get("model"), ns.get("tok")
        except Exception:
            existing = (None, None)
        if existing[0] is not None and existing[1] is not None:
            out["model"], out["tok"] = existing
            describe_backend(existing[0], existing[1], layer, reused=True)
        else:
            from _common import _gpu_busy_for_full_load
            busy = _gpu_busy_for_full_load()
            if busy:
                print("REFUSED second Qwen load.")
                print(busy)
                out["model"] = None
                out["tok"] = None
            else:
                model, tok = boot(DEFAULT_MODEL, layer)
                out["model"] = model
                out["tok"] = tok
    else:
        # load_model=False: do not from_pretrained. :8270 already holds Qwen.
        try:
            import urllib.request
            st = json.loads(urllib.request.urlopen("http://127.0.0.1:8270/status", timeout=3).read())
        except Exception:
            st = {}
        if st.get("loaded"):
            print(
                f"Qwen already loaded on :8270 pid={st.get('pid')} "
                f"({st.get('model_id')}) — not loading again."
            )
            print("look.cmd_capture(model, tok, LAYER, NOTE) will use that copy.")
        else:
            print("load_model=False and nothing is loaded on :8270 — capture will fail.")
    return out
