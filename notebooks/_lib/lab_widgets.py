"""Native Jupyter lab UI (ipywidgets) — client of :8270 only. Never loads a model."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

DEFAULT_NOTE = (
    "Patient received FOLFOX. Disease progressed. FOLFOX was discontinued."
)

# Map UI labels → service experiment names
EXPERIMENT_MAP = {
    "Score (vs frozen U-A d)": "score_vs_d",
    "A/B representation": "ab_representation",
    "Layer sweep": "layer_sweep",
    "CAUSAL MLP zero (candidates)": "causal_mlp_zero_candidates",
    "Frozen nn panel (:8259)": "nn_panel_frozen",
    "Frozen U-A map (:8258)": "ua_map_frozen",
}


def _service_client(base: str = "http://127.0.0.1:8270"):
    root = Path(__file__).resolve().parent.parent.parent / "model_service"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from client import ModelServiceClient

    return ModelServiceClient(base)


def _display_result(result: dict[str, Any]) -> None:
    from IPython.display import display
    import matplotlib.pyplot as plt

    if not result.get("ok"):
        print("ERROR:", result.get("error") or result)
        return

    exp = result.get("experiment")
    print("experiment:", exp, "| served_by:", result.get("served_by"))

    if exp == "score_vs_d":
        print("best_layer", result.get("best_layer"), "mean_score_d", result.get("best_mean_score_d"))
        layers = result.get("layers") or {}
        xs, ys = [], []
        for k, row in sorted(layers.items(), key=lambda kv: int(kv[0][1:])):
            xs.append(row["layer"])
            ys.append(row["mean_score_d"])
            print(f"  L{row['layer']}: mean_score_d={row['mean_score_d']:.3f}")
        if xs:
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(xs, ys, marker="o")
            ax.set_xlabel("Layer")
            ax.set_ylabel("mean_score_d")
            ax.set_title("Score vs frozen U-A d")
            display(fig)
            plt.close(fig)
        return

    if exp == "ab_representation":
        print(
            "max_sep_layer", result.get("max_sep_layer"),
            "max_sep", result.get("max_sep"),
            "sep_L20", result.get("sep_L20"),
        )
        print(result.get("note"))
        series = result.get("series") or []
        xs = [r["layer"] for r in series]
        ys = [r["sep"] for r in series]
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(xs, ys, marker="o", markersize=3)
        ax.axvline(20, color="gray", linestyle="--", alpha=0.7)
        ax.set_xlabel("Layer")
        ax.set_ylabel("||A−B||")
        ax.set_title("A/B representation separation")
        display(fig)
        plt.close(fig)
        return

    if exp == "layer_sweep":
        print("peak_layer", result.get("peak_layer"), "peak_l2", result.get("peak_l2"))
        series = result.get("series") or []
        xs = [r["layer"] for r in series]
        ys = [r["l2"] for r in series]
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(xs, ys, marker="o", markersize=3)
        ax.set_xlabel("Layer")
        ax.set_ylabel("last-token L2")
        ax.set_title("Layer sweep (residual magnitude)")
        display(fig)
        plt.close(fig)
        return

    if exp == "causal_mlp_zero_candidates":
        obs = result.get("observed") or {}
        print("base_margin", obs.get("base_margin"))
        print("base_text", repr(obs.get("base_text")))
        print(
            "max_sep_among", obs.get("max_sep_layer_among_candidates"),
            "max_|Δmargin|", obs.get("max_abs_delta_margin_layer"),
            "same_layer?", obs.get("same_layer"),
        )
        print("Δmargin_L20", obs.get("delta_margin_L20"), "flip_L20", obs.get("flip_L20"))
        rows = result.get("rows") or []
        if rows:
            print(f"{'L':>3}  {'sep':>8}  {'Δmargin':>8}  flip")
            for r in rows:
                print(
                    f"{r['layer']:3d}  {r['sep']:8.2f}  {r['delta_margin']:8.3f}  "
                    f"{'Y' if r.get('flip') else 'n'}"
                )
            xs = [str(r["layer"]) for r in rows]
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
            axes[0].bar(xs, [r["sep"] for r in rows], color="C0")
            axes[0].set_title("||A−B||")
            axes[1].bar(xs, [r["delta_margin"] for r in rows], color="C3")
            axes[1].axhline(0, color="gray", linewidth=0.8)
            axes[1].set_title("Δmargin (MLP zero)")
            display(fig)
            plt.close(fig)
        return

    if exp in ("nn_panel_frozen", "ua_map_frozen"):
        if exp == "nn_panel_frozen":
            print("patch_plan", result.get("patch_plan"))
            for c in result.get("cases") or []:
                print(c["id"], "n_lost", c.get("n_lost_steps"), "peak", c.get("peak_layer"))
        else:
            for r in result.get("rows") or []:
                print(r)
        return

    # fallback
    print(result)


def build_interactive_lab(
    *,
    service_base: str = "http://127.0.0.1:8270",
    default_note: str = DEFAULT_NOTE,
) -> Any:
    """Render ipywidgets lab. Returns the root widget (already displayed)."""
    import ipywidgets as widgets
    from IPython.display import display

    try:
        get_ipython().run_line_magic("matplotlib", "inline")  # noqa: F821
    except Exception:
        pass

    client = _service_client(service_base)

    status_lbl = widgets.HTML(value="<b>Service:</b> …")
    note = widgets.Textarea(
        value=default_note,
        description="Note:",
        layout=widgets.Layout(width="98%", height="140px"),
        style={"description_width": "70px"},
    )
    experiment = widgets.Dropdown(
        options=list(EXPERIMENT_MAP.keys()),
        value="CAUSAL MLP zero (candidates)",
        description="Experiment:",
        layout=widgets.Layout(width="98%"),
        style={"description_width": "90px"},
    )
    layer = widgets.IntSlider(
        value=20, min=0, max=27, description="Layer:", continuous_update=False
    )
    component = widgets.Dropdown(
        options=["MLP", "attn", "block"],
        value="MLP",
        description="Component:",
        style={"description_width": "90px"},
    )
    # Component reserved for later ops; causal grid is MLP-fixed for now
    component.disabled = True

    refresh = widgets.Button(description="Refresh status", icon="refresh")
    load_btn = widgets.Button(description="Load model on :8270", button_style="warning")
    run = widgets.Button(description="RUN EXPERIMENT", button_style="primary", icon="play")
    output = widgets.Output(
        layout=widgets.Layout(border="1px solid #ccc", padding="8px", width="98%")
    )

    def refresh_status(_=None) -> None:
        h = client.health()
        st = client.status()
        if not h.get("ok"):
            status_lbl.value = (
                f"<b>Service:</b> <span style='color:#b00'>DOWN</span> — "
                f"{h.get('error') or h.get('hint') or 'start ./model_service/run_service.sh'}"
            )
            return
        loaded = st.get("loaded")
        mid = st.get("model_id") or "—"
        free = st.get("cuda0_free_GiB")
        color = "#080" if loaded else "#a60"
        state = "READY (model loaded)" if loaded else "UP (no model — click Load or free VRAM)"
        status_lbl.value = (
            f"<b>Service :</b>8270 <span style='color:{color}'>● {state}</span><br/>"
            f"<b>Model:</b> {mid} &nbsp;|&nbsp; <b>GPU free:</b> {free} GiB"
        )

    def on_load(_):
        with output:
            output.clear_output()
            print("POST /load …")
            r = client.load()
            print(r)
            refresh_status()

    def on_run(_):
        with output:
            output.clear_output()
            name = EXPERIMENT_MAP[experiment.value]
            needs_model = name not in ("nn_panel_frozen", "ua_map_frozen")
            st = client.status()
            if needs_model and not st.get("loaded"):
                print(
                    "Model not loaded on :8270. Click 'Load model on :8270' "
                    "(requires ~12 GiB free — shut Jupyter kernel Qwen first if needed)."
                )
                refresh_status()
                return
            print(f"Running {name} …")
            body: dict[str, Any] = {
                "note": note.value,
                "evidence": note.value,
                "layer": int(layer.value),
                "component": component.value,
            }
            result = client.run(name, **body)
            _display_result(result)

    refresh.on_click(refresh_status)
    load_btn.on_click(on_load)
    run.on_click(on_run)

    title = widgets.HTML(
        "<h3 style='margin:0 0 8px 0'>CXR MI / RepEng Lab</h3>"
        "<p style='margin:0 0 8px 0;color:#444'>Native Jupyter GUI → :8270 shared service. "
        "Never loads a model in this kernel.</p>"
    )
    box = widgets.VBox(
        [
            title,
            status_lbl,
            widgets.HBox([refresh, load_btn]),
            note,
            experiment,
            layer,
            component,
            run,
            widgets.HTML("<b>RESULTS</b>"),
            output,
        ]
    )
    refresh_status()
    display(box)
    # Do not return the widget — returning it double-emits text/plain __repr__
    # when the Lab frontend cannot render widget MIME (and clutters output when it can).
    return None
