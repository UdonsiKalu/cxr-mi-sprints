"""Bridge to frozen N2S port experiments (:8258 / :8259 / …).

Does NOT start HTTP servers. Does NOT call n2s unload_model (would kill the Jupyter kernel).
Dual_full / ground() stay read-only — not invoked here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

LAB_ROOT = Path("/home/udonsi-kalu/staging/cxr-evidence-grounding-lab")
ARTIFACTS = LAB_ROOT / "artifacts"
DIRECTIONS_PATH = ARTIFACTS / "n2s-upstream-ua-directions.pt"
UA_MAP_PATH = ARTIFACTS / "n2s-upstream-ua-map.json"
NN_PANEL_PATH = ARTIFACTS / "n2s-nn-layer-loss-panel.json"
MAP_VIEWER_PANEL = ARTIFACTS / "n2s-map-viewer-panel.json"


def ensure_lab_on_path() -> Path:
    root = str(LAB_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return LAB_ROOT


def load_ua_directions() -> dict[str, Any]:
    if not DIRECTIONS_PATH.is_file():
        raise FileNotFoundError(DIRECTIONS_PATH)
    try:
        blob = torch.load(DIRECTIONS_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(DIRECTIONS_PATH, map_location="cpu")
    return blob


def load_ua_map() -> dict[str, Any]:
    return json.loads(UA_MAP_PATH.read_text(encoding="utf-8"))


def load_nn_panel_frozen() -> dict[str, Any]:
    """Same numbers the :8259 Analyze tab shows (CPU replay)."""
    ensure_lab_on_path()
    from n2s_lab.n2s_nn_layer_loss import build_panel, ground_rules, detect_patch_plan

    return {
        "panel": build_panel(),
        "ground_rules": ground_rules(),
        "patch_plan": detect_patch_plan(),
        "artifact": str(NN_PANEL_PATH),
    }


def default_ab_notes() -> dict[str, dict[str, str]]:
    ensure_lab_on_path()
    from n2s_lab.n2s_nn_layer_loss import DEFAULT_NOTES

    return dict(DEFAULT_NOTES)


def replay_8258_frozen_scores() -> dict[str, Any]:
    """Frozen U-A map readout — what :8258 Score shows for the lab pair."""
    m = load_ua_map()
    rows = []
    for Lkey, block in m.get("by_layer", {}).items():
        notes = block.get("notes") or {}
        t = notes.get("EX_TEMPORAL_FOLFOX") or {}
        c = notes.get("EX_CONTRA") or {}
        rows.append(
            {
                "layer": block.get("layer"),
                "mean_score_d_temporal": t.get("mean_score_d"),
                "mean_score_d_contra": c.get("mean_score_d"),
                "delta_mean_norm": block.get("delta_mean_norm"),
            }
        )
    return {
        "kind": "8258_frozen_ua_map",
        "model_id": m.get("model_id"),
        "pair": m.get("pair"),
        "layers": m.get("layers"),
        "rows": rows,
        "claim_hygiene": m.get("claim_hygiene"),
        "path": str(UA_MAP_PATH),
    }


def score_note_vs_frozen_d(
    model,
    tok,
    evidence: str,
    *,
    layers: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Live score on the *existing* Jupyter model — no n2s load/unload.

    mean_score_d ≈ mean over sequence tokens of (residual · d) at each layer.
    Correlational readout only (same claim as :8258).
    """
    from _common import encode, device, full_residual

    evidence = (evidence or "").strip()
    if not evidence:
        return {"ok": False, "error": "evidence required"}

    blob = load_ua_directions()
    by_layer: dict[str, torch.Tensor] = blob["by_layer"]
    if layers is None:
        layers = tuple(int(x) for x in blob.get("layers") or [8, 12, 16, 20, 24])

    ids = encode(tok, evidence, device(model))
    out_layers: dict[str, Any] = {}
    for L in layers:
        key = f"L{L}"
        d = by_layer.get(key)
        if d is None:
            continue
        # full sequence residual at this layer (site=block)
        H = full_residual(model, ids, L, site="block").float().cpu()  # (T, D)
        d = d.float().reshape(-1)
        scores = H @ d
        out_layers[key] = {
            "layer": L,
            "mean_score_d": float(scores.mean()),
            "max_abs_score_d": float(scores.abs().max()),
            "n_tokens": int(H.shape[0]),
        }

    best = max(out_layers.values(), key=lambda r: abs(r["mean_score_d"])) if out_layers else None
    return {
        "ok": True,
        "kind": "8258_live_vs_frozen_d",
        "evidence_preview": evidence[:160],
        "directions": str(DIRECTIONS_PATH),
        "layers": out_layers,
        "best_layer": None if best is None else best["layer"],
        "best_mean_score_d": None if best is None else best["mean_score_d"],
        "hygiene": "correlational readout only — not causal; not Dual repair",
    }


def replay_8259_cases(*ids: str) -> list[dict[str, Any]]:
    data = load_nn_panel_frozen()
    want = set(ids) if ids else {"EX_TEMPORAL_FOLFOX", "EX_CONTRA"}
    return [c for c in data["panel"]["cases"] if c["id"] in want]


def port_catalog() -> list[dict[str, str]]:
    return [
        {
            "port": "8258",
            "name": "Upstream Score",
            "notebook": "01_8258_upstream_AB_score.ipynb",
            "experiment": "Score note vs frozen U-A d=unit(μ_T−μ_C); custom pair Map",
        },
        {
            "port": "8259",
            "name": "Neural→neural layer-loss",
            "notebook": "02_8259_nn_layer_loss.ipynb",
            "experiment": "FOLFOX vs CONTRA mean_score_d traces; lost steps; patch plan",
        },
        {
            "port": "8260",
            "name": "Locator v1 (frozen)",
            "notebook": "03_8260_locator_readonly.ipynb",
            "experiment": "Route encode/compute/translate → REVIEW contain — GUI frozen",
        },
        {
            "port": "8263",
            "name": "Dual Translate map",
            "notebook": "04_8263_dual_map_readonly.ipynb",
            "experiment": "Nested none→G9 mapping readout — no Dual_full / ground() rewrite",
        },
        {
            "port": "8264",
            "name": "Mapping curriculum",
            "notebook": "(see lab curriculum-mapping; link from 04)",
            "experiment": "MAP-00…09 pedagogy around :8263",
        },
    ]
