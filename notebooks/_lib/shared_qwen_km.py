"""CXR kernel labels + optional share of the in-kernel Qwen PID.

JupyterLab picker text for a running session is `{session.name} ({id})`.
We tag session.name with where Qwen actually lives. Kernelspec display_name
is what the status bar and "Start Kernel" list show.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.request
from pathlib import Path

from jupyter_client.kernelspec import KernelSpecManager
from jupyter_server.services.kernels.kernelmanager import MappingKernelManager
from jupyter_server.services.sessions.sessionmanager import SessionManager

SPEC_PYTHON = "cxr-faiss-gpu1"
SPEC_8270 = "cxr-qwen-8270"
SPEC_INKERNEL = "cxr-qwen-inkernel"
OWNER_FILE = Path("/tmp/cxr-jupyter-qwen-owner.json")
GPU_HEAVY_MIB = 8000


def _nvidia_compute() -> list[tuple[int, int, str]]:
    """[(pid, mem_mib, name), ...]"""
    try:
        import subprocess

        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory,process_name",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
        )
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            mem = int(float(parts[1]))
        except ValueError:
            continue
        name = parts[2] if len(parts) > 2 else ""
        rows.append((pid, mem, name))
    return rows


def _status_8270() -> dict:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8270/status", timeout=0.6) as resp:
            data = json.loads(resp.read().decode())
        return {
            "up": True,
            "loaded": bool(data.get("loaded")),
            "model_id": data.get("model_id") or "",
            "pid": int(data["pid"]) if data.get("pid") else None,
        }
    except Exception:
        return {"up": False, "loaded": False, "model_id": "", "pid": None}


def _kernel_ids(km) -> list[str]:
    if km is None:
        return []
    try:
        return list(km.list_kernel_ids())
    except Exception:
        return list(getattr(km, "_kernels", {}) or {})


def _kernel_pid(km) -> int | None:
    try:
        prov = getattr(km, "provisioner", None)
        if prov is not None:
            pid = getattr(prov, "pid", None)
            if pid:
                return int(pid)
    except Exception:
        pass
    for attr in ("pid", "process"):
        try:
            val = getattr(km, attr, None)
            if val is None:
                continue
            if hasattr(val, "pid"):
                return int(val.pid)
            return int(val)
        except Exception:
            continue
    return None


def model_inventory(kernel_manager=None) -> dict:
    gpu = _nvidia_compute()
    heavy = [(pid, mem, name) for pid, mem, name in gpu if mem >= GPU_HEAVY_MIB]
    svc = _status_8270()
    owner = {}
    if OWNER_FILE.is_file():
        try:
            owner = json.loads(OWNER_FILE.read_text())
        except Exception:
            owner = {}

    inkernel_id = None
    inkernel_pid = None
    if kernel_manager is not None:
        gpu_pids = {pid for pid, mem, _ in heavy}
        kids = _kernel_ids(kernel_manager)
        for kid in kids:
            try:
                km = kernel_manager.get_kernel(kid)
            except Exception:
                continue
            pid = _kernel_pid(km)
            if pid and pid in gpu_pids:
                inkernel_id = kid
                inkernel_pid = pid
                break
        oid = owner.get("kernel_id")
        if inkernel_id is None and oid in kids:
            inkernel_id = oid

    svc_pid = svc.get("pid")
    svc_holds = bool(
        svc.get("loaded")
        and svc_pid
        and any(pid == svc_pid for pid, _, _ in heavy)
    )
    return {
        "svc": svc,
        "svc_holds": svc_holds,
        "inkernel_id": inkernel_id,
        "inkernel_pid": inkernel_pid,
        "heavy": heavy,
        "owner": owner,
    }


def the_loaded_spec(inv: dict) -> str:
    """Single kernelspec to advertise: whatever actually holds Qwen."""
    if inv.get("inkernel_id"):
        return SPEC_INKERNEL
    if inv.get("svc_holds"):
        return SPEC_8270
    return SPEC_PYTHON


def tag_for_kernel(kernel_id: str | None, kernel_name: str | None, inv: dict, km=None) -> str | None:
    """Connect-to-existing suffix. The one Jupyter kernel all tabs share."""
    if not kernel_id:
        return None
    if inv.get("inkernel_id") == kernel_id:
        return "● Qwen 7B loaded"
    if inv.get("svc_holds") and km is not None:
        kids = _kernel_ids(km)
        if kids and kernel_id == kids[0]:
            return "● Qwen 7B loaded"
    return None


def spec_name_for_kernel(kernel_id: str | None, started_as: str | None, inv: dict) -> str:
    return the_loaded_spec(inv)


def _short_id(kid: str | None) -> str:
    if not kid:
        return ""
    return kid.split("-")[0]


class CxrKernelSpecManager(KernelSpecManager):
    """Advertise exactly one kernelspec: the Qwen that is in GPU memory."""

    def get_all_specs(self):
        specs = super().get_all_specs()
        inv = model_inventory()
        keep = the_loaded_spec(inv)
        if keep not in specs:
            keep = SPEC_PYTHON if SPEC_PYTHON in specs else next(iter(specs), keep)
        one = {keep: specs[keep]}
        one[keep]["spec"]["display_name"] = self._display_name(keep, inv)
        return one

    def get_kernel_spec(self, kernel_name: str):
        inv = model_inventory()
        keep = the_loaded_spec(inv)
        spec = super().get_kernel_spec(keep)
        spec.display_name = self._display_name(keep, inv)
        return spec

    def _display_name(self, name: str, inv: dict) -> str:
        mid = ((inv.get("svc") or {}).get("model_id") or "Qwen2.5-7B-Instruct").split("/")[-1]
        if name == SPEC_INKERNEL:
            return f"● Qwen 7B loaded ({mid})"
        if name == SPEC_8270 and inv.get("svc_holds"):
            return f"● Qwen 7B loaded ({mid})"
        return f"● Qwen 7B loaded ({mid})" if inv.get("svc_holds") or inv.get("inkernel_id") else "CXR Python"


class SharedQwenKernelManager(MappingKernelManager):
    """One kernel for every tab. Look/NOTE/model persist when you switch notebooks."""

    def _cxr_prepare(self) -> None:
        if not hasattr(self, "_cxr_lock"):
            self._cxr_lock = asyncio.Lock()
            self._cxr_shared_id = None

    def kernel_model(self, kernel_id):
        model = super().kernel_model(kernel_id)
        inv = model_inventory(self)
        started = model.get("name")
        model["name"] = spec_name_for_kernel(kernel_id, started, inv)
        model["cxr_tag"] = tag_for_kernel(kernel_id, started, inv, km=self)
        if model.get("cxr_tag") is None:
            model.pop("cxr_tag", None)
        return model

    def _gpu_kernel_id(self) -> str | None:
        return model_inventory(self).get("inkernel_id")

    def _existing_named(self, kernel_name: str) -> str | None:
        for kid in _kernel_ids(self):
            try:
                km = self.get_kernel(kid)
            except Exception:
                continue
            if getattr(km, "kernel_name", None) == kernel_name:
                return kid
        return None

    async def _async_start_kernel(self, *, kernel_id=None, path=None, **kwargs):
        self._cxr_prepare()
        inv = model_inventory(self)
        kwargs["kernel_name"] = the_loaded_spec(inv)
        async with self._cxr_lock:
            kids = _kernel_ids(self)
            shared = self._cxr_shared_id
            if shared and (shared in kids or shared in self):
                self.log.info("Pinning %s to shared kernel %s", path, shared)
                return await super()._async_start_kernel(
                    kernel_id=shared, path=path, **kwargs
                )
            if kids:
                self._cxr_shared_id = kids[0]
                self.log.info("Pinning %s to shared kernel %s", path, kids[0])
                return await super()._async_start_kernel(
                    kernel_id=kids[0], path=path, **kwargs
                )
            kid = await super()._async_start_kernel(
                kernel_id=kernel_id, path=path, **kwargs
            )
            self._cxr_shared_id = kid
            self.log.info("Shared CXR kernel is %s (path=%s)", kid, path)
            return kid

    async def _async_shutdown_kernel(self, kernel_id, now=False, restart=False):
        if not restart:
            kids = _kernel_ids(self)
            if kids and kernel_id == kids[0]:
                self.log.info("Keeping shared CXR kernel %s alive", kernel_id)
                return None
        return await super()._async_shutdown_kernel(
            kernel_id, now=now, restart=restart
        )

    # MappingKernelManager does `start_kernel = _async_start_kernel` on ITS methods.
    # Rebind so Lab actually hits the pin logic.
    start_kernel = _async_start_kernel
    shutdown_kernel = _async_shutdown_kernel


class CxrSessionManager(SessionManager):
    """Put the Qwen tag in session.name so Connect-to-existing lists it."""

    async def row_to_model(self, row, tolerate_culled=False):
        model = await super().row_to_model(row, tolerate_culled=tolerate_culled)
        if not model:
            return model
        kernel = model.get("kernel") or {}
        inv = model_inventory(getattr(self, "kernel_manager", None))
        tag = tag_for_kernel(kernel.get("id"), kernel.get("name"), inv, km=getattr(self, "kernel_manager", None))
        base = Path(model.get("path") or model.get("name") or "notebook").name
        model["name"] = f"{base} · {tag}" if tag else base
        if "notebook" in model:
            model["notebook"] = {**model["notebook"], "name": model["name"]}
        return model
