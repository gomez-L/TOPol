"""Run persistence: save *every* intermediate object and every figure.

Design goals
------------
* One timestamped directory per run:  runs/<dataset>_<YYYYmmdd-HHMMSS>/
* Sub-folders: objects/ (arrays + pickles), tables/ (csv + parquet),
  figures/ (vectorised .pdf + .png thumbnail), and a manifest.json index.
* Arrays -> .npz (compressed, float32) ; arbitrary Python -> .pkl ; DataFrames
  -> .parquet (+ .csv) ; metadata -> .json.
* Figures are saved as PDF with editable text (pdf.fonttype = 42) so they drop
  straight into a LaTeX/AAAI submission and remain fully vectorised.

Everything is reload-able by name so downstream notebooks can pick up any object
for further exploration without recomputation.
"""
from __future__ import annotations

import os
import json
import pickle
import datetime as _dt
from typing import Any

import numpy as np


def _now_tag() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


class RunStore:
    def __init__(self, root: str, dataset: str, tag: str | None = None, resume: bool = False):
        tag = tag or _now_tag()
        self.dir = os.path.join(root, f"{dataset}_{tag}")
        self.objects = os.path.join(self.dir, "objects")
        self.tables = os.path.join(self.dir, "tables")
        self.figures = os.path.join(self.dir, "figures")
        for d in (self.objects, self.tables, self.figures):
            os.makedirs(d, exist_ok=True)
        self.manifest_path = os.path.join(self.dir, "manifest.json")
        if resume and os.path.exists(self.manifest_path):
            with open(self.manifest_path) as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {"dataset": dataset, "created": _now_tag(), "entries": {}}
            self._flush()

    # ------------------------------------------------------------------ #
    def _flush(self) -> None:
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2, default=str)

    def _register(self, name: str, kind: str, path: str, meta: dict | None = None) -> None:
        self.manifest["entries"][name] = {
            "kind": kind,
            "path": os.path.relpath(path, self.dir),
            "saved": _now_tag(),
            "meta": meta or {},
        }
        self._flush()

    # -------------------------- writers ------------------------------- #
    def save_config(self, cfg) -> None:
        p = os.path.join(self.dir, "config.json")
        cfg.to_json(p)
        self._register("config", "config", p)

    def save_array(self, name: str, arr: np.ndarray, meta: dict | None = None) -> str:
        arr = np.asarray(arr)
        p = os.path.join(self.objects, f"{name}.npz")
        np.savez_compressed(p, arr=arr.astype(np.float32) if arr.dtype.kind == "f" else arr)
        self._register(name, "array", p, {"shape": list(arr.shape), "dtype": str(arr.dtype), **(meta or {})})
        return p

    def save_array_f16(self, name: str, arr: np.ndarray, meta: dict | None = None) -> str:
        """Like `save_array`, but keeps float16 instead of upcasting to float32.

        For the per-layer activation cache (`topol_activations.py`): N docs x L
        layers x d_model in float32 is the single biggest memory cost of the
        discovery path, and cosine/distance geometry does not need f32
        precision to do useful work.
        """
        arr = np.asarray(arr)
        p = os.path.join(self.objects, f"{name}.npz")
        np.savez_compressed(p, arr=arr.astype(np.float16) if arr.dtype.kind == "f" else arr)
        self._register(name, "array_f16", p, {"shape": list(arr.shape), "dtype": "float16", **(meta or {})})
        return p

    def save_object(self, name: str, obj: Any, meta: dict | None = None) -> str:
        p = os.path.join(self.objects, f"{name}.pkl")
        with open(p, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._register(name, "pickle", p, meta)
        return p

    def save_json(self, name: str, obj: dict, meta: dict | None = None) -> str:
        p = os.path.join(self.objects, f"{name}.json")
        with open(p, "w") as f:
            json.dump(obj, f, indent=2, default=_json_default)
        self._register(name, "json", p, meta)
        return p

    def save_table(self, name: str, df, meta: dict | None = None) -> str:
        csv_p = os.path.join(self.tables, f"{name}.csv")
        df.to_csv(csv_p, index=False)
        try:
            pq_p = os.path.join(self.tables, f"{name}.parquet")
            df.to_parquet(pq_p, index=False)
        except Exception:
            pass  # parquet engine optional
        self._register(name, "table", csv_p, {"rows": len(df), "cols": list(df.columns), **(meta or {})})
        return csv_p

    def save_fig(self, name: str, fig, dpi: int = 200, meta: dict | None = None) -> str:
        """Save a matplotlib figure as a vectorised PDF (+ PNG thumbnail)."""
        import matplotlib
        matplotlib.rcParams["pdf.fonttype"] = 42   # editable text, not outlines
        matplotlib.rcParams["ps.fonttype"] = 42
        pdf_p = os.path.join(self.figures, f"{name}.pdf")
        png_p = os.path.join(self.figures, f"{name}.png")
        fig.savefig(pdf_p, bbox_inches="tight")            # vector
        fig.savefig(png_p, bbox_inches="tight", dpi=dpi)   # thumbnail
        self._register(name, "figure", pdf_p, {"png": os.path.relpath(png_p, self.dir), **(meta or {})})
        return pdf_p

    # -------------------------- readers ------------------------------- #
    def load_array(self, name: str) -> np.ndarray:
        return np.load(os.path.join(self.dir, self.manifest["entries"][name]["path"]))["arr"]

    def load_object(self, name: str) -> Any:
        with open(os.path.join(self.dir, self.manifest["entries"][name]["path"]), "rb") as f:
            return pickle.load(f)

    def load_json(self, name: str) -> dict:
        with open(os.path.join(self.dir, self.manifest["entries"][name]["path"])) as f:
            return json.load(f)

    def list_entries(self) -> dict:
        return {k: v["kind"] for k, v in self.manifest["entries"].items()}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)
