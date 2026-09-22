"""Reference-suite runner. Reuses live model/tok from the Jupyter kernel — never a second load."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch


def pack_root() -> Path:
    here = Path(__file__).resolve().parent  # notebooks/_lib
    return here.parent.parent  # cxr-mi-repeng-grounding


def suite_root() -> Path:
    return Path(__file__).resolve().parent.parent / "08_reference_suite"


def expected_dir() -> Path:
    return suite_root() / "expected"


def artifacts_exp_dir() -> Path:
    d = pack_root() / "artifacts" / "experiments"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_registry() -> dict[str, Any]:
    path = suite_root() / "registry.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_expected(exp_id: str) -> dict[str, Any] | None:
    path = expected_dir() / f"{exp_id}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def require_live_model(ns: dict[str, Any] | None = None) -> tuple[Any, Any]:
    """Refuse a fresh load. Model must already sit in the kernel namespace."""
    if ns is None:
        try:
            from IPython import get_ipython

            ip = get_ipython()
            ns = ip.user_ns if ip is not None else {}
        except Exception:
            ns = {}
    model, tok = ns.get("model"), ns.get("tok")
    if model is None or tok is None:
        raise RuntimeError(
            "No live model/tok in this kernel. "
            "Connect to Existing (e.g. 119f847c), run a setup cell once, "
            "then re-run the suite. Do not start Streamlit as a second load."
        )
    return model, tok


def _yes_no_ids(tok) -> tuple[list[int], list[int]]:
    from _common import token_ids

    return (
        token_ids(tok, (" Yes", "Yes", " yes")),
        token_ids(tok, (" No", "No", " no")),
    )


def _margin(logits: torch.Tensor, yes_ids: list[int], no_ids: list[int]) -> float:
    yes = max(float(logits[i]) for i in yes_ids)
    no = max(float(logits[i]) for i in no_ids)
    return yes - no


def run_E00_2_capture_width(model, tok, ctx: dict[str, Any]) -> dict[str, Any]:
    from _common import encode, device, last_residual

    note = ctx["NOTE"]
    h = last_residual(model, encode(tok, note, device(model)), layer=27)
    return {
        "width": int(h.numel()),
        "l2": float(h.norm()),
        "note": note,
    }


def run_E00_3_block0_add_check(model, tok, ctx: dict[str, Any]) -> dict[str, Any]:
    from _common import blocks, unwrap, encode, device

    ids = encode(tok, ctx["NOTE"], device(model))
    block = blocks(model)[0]
    bucket: dict[str, torch.Tensor] = {}

    def pre(_m, inp):
        hs = unwrap(inp[0] if isinstance(inp, tuple) else inp)
        bucket["rin"] = hs[0, -1, :].detach().float().cpu()

    def attn_h(_m, _i, out):
        bucket["attn"] = unwrap(out)[0, -1, :].detach().float().cpu()

    def mlp_h(_m, _i, out):
        bucket["mlp"] = unwrap(out)[0, -1, :].detach().float().cpu()

    def post(_m, _i, out):
        bucket["rout"] = unwrap(out)[0, -1, :].detach().float().cpu()

    handles = [
        block.register_forward_pre_hook(pre),
        block.self_attn.register_forward_hook(attn_h),
        block.mlp.register_forward_hook(mlp_h),
        block.register_forward_hook(post),
    ]
    try:
        with torch.inference_mode():
            model(**ids)
    finally:
        for h in handles:
            h.remove()

    rin, attn, mlp, rout = bucket["rin"], bucket["attn"], bucket["mlp"], bucket["rout"]
    add_check = float((rin + attn + mlp - rout).norm())
    return {
        "resid_in_l2": float(rin.norm()),
        "attn_l2": float(attn.norm()),
        "mlp_l2": float(mlp.norm()),
        "resid_out_l2": float(rout.norm()),
        "add_check": add_check,
        "width": int(rin.numel()),
    }


def run_SWEEP_sep_PROMPT_AB(model, tok, ctx: dict[str, Any]) -> dict[str, Any]:
    from _common import last_residual, encode, device

    ids_a = encode(tok, ctx["PROMPT_A"], device(model))
    ids_b = encode(tok, ctx["PROMPT_B"], device(model))
    n = len(model.model.layers)
    sep = []
    for layer in range(n):
        h_a = last_residual(model, ids_a, layer)
        h_b = last_residual(model, ids_b, layer)
        sep.append(float((h_a - h_b).norm()))
    peak = max(range(n), key=lambda i: sep[i])
    return {
        "n_layers": n,
        "sep_by_layer": {str(i): sep[i] for i in range(n)},
        "max_sep_layer": peak,
        "max_sep": sep[peak],
        "sep_L20": sep[20],
        "sep_L0": sep[0],
        "sep_L27": sep[27],
    }


def run_CAUSAL_mlp_zero_candidates(model, tok, ctx: dict[str, Any]) -> dict[str, Any]:
    from _common import (
        encode,
        device,
        blocks,
        module,
        unwrap,
        greedy,
        last_residual,
    )

    candidates = [16, 18, 20, 22, 24, 26, 27]
    max_new = int(ctx.get("MAX_NEW", 16))
    max_new = min(max_new, 16)
    ids_test = encode(tok, ctx["PROMPT_TEST"], device(model))
    yes_ids, no_ids = _yes_no_ids(tok)

    # observational join (cheap if sep already known — still recompute candidates only)
    ids_a = encode(tok, ctx["PROMPT_A"], device(model))
    ids_b = encode(tok, ctx["PROMPT_B"], device(model))
    sep_by = {}
    for layer in candidates:
        h_a = last_residual(model, ids_a, layer)
        h_b = last_residual(model, ids_b, layer)
        sep_by[layer] = float((h_a - h_b).norm())

    def next_logits(*, on_layer: int | None = None, zero_mlp: bool = False) -> torch.Tensor:
        handles = []
        if on_layer is not None and zero_mlp:

            def hook(_m, _inp, out):
                hs = unwrap(out)
                modified = hs.clone()
                modified[0, -1, :] = 0
                if isinstance(out, tuple):
                    return (modified,) + out[1:]
                return modified

            handles.append(
                module(blocks(model)[on_layer], "mlp").register_forward_hook(hook)
            )
        try:
            with torch.inference_mode():
                out = model(**ids_test)
            return out.logits[0, -1, :].detach().float().cpu()
        finally:
            for h in handles:
                h.remove()

    base_logits = next_logits()
    base_margin = _margin(base_logits, yes_ids, no_ids)
    base_text = greedy(model, tok, ids_test, max_new)

    rows = []
    for layer in candidates:
        z_logits = next_logits(on_layer=layer, zero_mlp=True)
        z_margin = _margin(z_logits, yes_ids, no_ids)
        z_text = greedy(
            model, tok, ids_test, max_new, on_layer=layer, site="mlp", zero_last=True
        )
        d_m = z_margin - base_margin
        rows.append(
            {
                "layer": layer,
                "sep": sep_by[layer],
                "delta_margin": d_m,
                "flip": z_text.strip() != base_text.strip(),
                "zero_text": z_text,
            }
        )

    max_sep_L = max(rows, key=lambda r: r["sep"])["layer"]
    max_dm_L = max(rows, key=lambda r: abs(r["delta_margin"]))["layer"]
    l20 = next(r for r in rows if r["layer"] == 20)
    return {
        "prompt_test": ctx["PROMPT_TEST"],
        "base_margin": base_margin,
        "base_text": base_text,
        "rows": rows,
        "max_sep_layer_among_candidates": max_sep_L,
        "max_abs_delta_margin_layer": max_dm_L,
        "same_layer": max_sep_L == max_dm_L,
        "delta_margin_L20": l20["delta_margin"],
        "flip_L20": l20["flip"],
    }


RUNNERS: dict[str, Callable[[Any, Any, dict[str, Any]], dict[str, Any]]] = {
    "E00_2_capture_width": run_E00_2_capture_width,
    "E00_3_block0_add_check": run_E00_3_block0_add_check,
    "SWEEP_sep_PROMPT_AB": run_SWEEP_sep_PROMPT_AB,
    "CAUSAL_mlp_zero_candidates": run_CAUSAL_mlp_zero_candidates,
}


def _is_constraint(expect: dict) -> bool:
    keys = set(expect)
    return keys <= {"min", "max", "approx", "tol"} and bool(keys & {"min", "max", "approx"})


def _check(actual: Any, expect: Any, path: str = "") -> list[str]:
    """Return list of failure messages."""
    fails: list[str] = []
    if isinstance(expect, dict) and _is_constraint(expect):
        val = float(actual)
        if "approx" in expect:
            tol = float(expect.get("tol", 0.15))
            target = float(expect["approx"])
            if abs(val - target) > tol * max(abs(target), 1.0):
                fails.append(f"{path}: {val} not within tol {tol} of {target}")
        if "min" in expect and val < float(expect["min"]):
            fails.append(f"{path}: {val} < min {expect['min']}")
        if "max" in expect and val > float(expect["max"]):
            fails.append(f"{path}: {val} > max {expect['max']}")
        return fails
    if isinstance(expect, dict):
        for k, v in expect.items():
            if k in ("tol", "status", "id", "model", "notes", "prompt", "tolerance"):
                continue
            if isinstance(actual, dict) and k in actual:
                fails.extend(_check(actual[k], v, f"{path}.{k}" if path else k))
            else:
                fails.append(f"{path}.{k}: missing in observed" if path else f"{k}: missing in observed")
        return fails
    if actual != expect:
        fails.append(f"{path}: got {actual!r} expected {expect!r}")
    return fails


def compare_to_expected(exp_id: str, observed: dict[str, Any]) -> dict[str, Any]:
    gold = load_expected(exp_id)
    if gold is None:
        return {
            "id": exp_id,
            "status": "no_golden",
            "pass": None,
            "fails": [],
            "observed": observed,
        }
    expect = gold.get("expect", gold)
    fails = list(dict.fromkeys(_check(observed, expect)))
    return {
        "id": exp_id,
        "status": gold.get("status", "pass"),
        "pass": len(fails) == 0,
        "fails": fails,
        "observed": observed,
        "golden": gold,
    }


def run_one(exp_id: str, model, tok, ctx: dict[str, Any]) -> dict[str, Any]:
    if exp_id not in RUNNERS:
        raise KeyError(f"No runner for {exp_id}. Known: {sorted(RUNNERS)}")
    observed = RUNNERS[exp_id](model, tok, ctx)
    result = compare_to_expected(exp_id, observed)
    result["ts"] = datetime.now(timezone.utc).isoformat()
    out = artifacts_exp_dir() / f"{exp_id}.json"
    # don't dump huge sep_by_layer into fails path twice — keep full observed
    out.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    result["artifact"] = str(out)
    return result


def run_tagged(
    model,
    tok,
    ctx: dict[str, Any],
    *,
    tags: list[str] | None = None,
    ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    reg = load_registry()
    experiments = reg["experiments"]
    selected = []
    for e in experiments:
        if e.get("enabled", True) is False:
            continue
        if ids is not None and e["id"] not in ids:
            continue
        if tags is not None and not set(tags) & set(e.get("tags", [])):
            continue
        selected.append(e["id"])
    results = []
    for exp_id in selected:
        print(f"\n===== {exp_id} =====")
        r = run_one(exp_id, model, tok, ctx)
        status = "PASS" if r["pass"] else ("NO_GOLDEN" if r["pass"] is None else "FAIL")
        print(status, "fails=", r["fails"] or None)
        print("artifact:", r.get("artifact"))
        results.append(r)
    n_pass = sum(1 for r in results if r["pass"] is True)
    n_fail = sum(1 for r in results if r["pass"] is False)
    print(f"\n=== summary: {n_pass} pass / {n_fail} fail / {len(results)} run ===")
    return results
