"""Shared helpers for the numbered pipeline scripts (00-08), the CLI equivalent
of notebooks/pipeline_2.ipynb -- run step by step, in order (each step resumes
the most recent run automatically):
    python scripts/00_prepare_data.py --dataset amazon_reviews   # once
    python scripts/01_extract_representations.py --dataset amazon_reviews --backend gemma
    python scripts/02_select_layer.py
    python scripts/03_cluster_manifold.py
    python scripts/04_reconstruct_field.py
    python scripts/05_topol_cache_activations.py
    python scripts/06_topol_discovery.py
    python scripts/07_topol_field_assembly.py
    python scripts/08_topol_baseline_compare.py
Pass --run-dir <path> to target a specific run explicitly.
"""
from __future__ import annotations

import os
import sys
import json
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from topolm.config import Config, default_config          # noqa: E402
from topolm.store import RunStore                          # noqa: E402
from topolm.backends import make_backend as _make_backend  # noqa: E402
from topolm.backends import make_segmentation_backend as _make_seg  # noqa: E402

_LATEST = os.path.join(REPO_ROOT, "runs", "LATEST")


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--dataset", default="synthetic_kd",
                   choices=["us_speeches", "amazon_reviews", "synthetic_1d", "synthetic_kd"])
    p.add_argument("--backend", default=None, choices=["gemma", "synthetic"],
                   help="default: synthetic for synthetic_* datasets, gemma otherwise")
    p.add_argument("--config", default=None, help="path to a config.json to load")
    p.add_argument("--tag", default=None, help="run tag (timestamp if omitted)")
    p.add_argument("--run-dir", default=None, help="resume this run directory")
    p.add_argument("--data-dir", default=os.path.join(os.path.dirname(REPO_ROOT), "data"))
    p.add_argument("--runs-dir", default=os.path.join(REPO_ROOT, "runs"))
    p.add_argument("--layer", type=int, default=None, help="fix the residual layer")
    p.add_argument("--n-permutations", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p


def build_config(args) -> Config:
    if args.config:
        cfg = Config.from_dict(json.load(open(args.config)))
    else:
        cfg = default_config(dataset=args.dataset)
    if args.backend:
        cfg.backend = args.backend
    else:
        cfg.backend = "synthetic" if args.dataset.startswith("synthetic") else "gemma"
    cfg.dataset = args.dataset
    cfg.runs_dir = args.runs_dir
    if args.layer is not None:
        cfg.model.layer = args.layer
    if args.n_permutations is not None:
        cfg.stats.n_permutations = args.n_permutations
    if args.seed is not None:
        cfg.random_state = args.seed
    return cfg


def write_latest(run_dir: str):
    os.makedirs(os.path.dirname(_LATEST), exist_ok=True)
    with open(_LATEST, "w") as f:
        f.write(run_dir)


def read_latest() -> str | None:
    if os.path.exists(_LATEST):
        return open(_LATEST).read().strip()
    return None


def open_new_store(cfg, args) -> RunStore:
    store = RunStore(cfg.runs_dir, cfg.dataset, tag=args.tag)
    store.save_config(cfg)
    write_latest(store.dir)
    return store


def open_existing_store(args) -> RunStore:
    run_dir = args.run_dir or read_latest()
    if run_dir is None or not os.path.isdir(run_dir):
        raise SystemExit("No run to resume. Run 01_extract_representations.py first "
                         "or pass --run-dir.")
    root, name = os.path.dirname(run_dir), os.path.basename(run_dir)
    dataset = name.rsplit("_", 2)[0]
    tag = name[len(dataset) + 1:]
    store = RunStore(root, dataset, tag=tag, resume=True)
    write_latest(store.dir)
    return store


def load_config_for(store) -> Config:
    return Config.from_dict(store.load_json("config")) if "config" in store.manifest["entries"] \
        else default_config()


def make_backend(cfg):
    return _make_backend(cfg)


def make_seg_backend(cfg, field_backend=None):
    """EmbeddingGemma-300M for topic segmentation (labels only)."""
    return _make_seg(cfg, field_backend)


def apply_overrides(cfg, args):
    """Apply CLI overrides onto a config loaded from a resumed run."""
    if getattr(args, "n_permutations", None) is not None:
        cfg.stats.n_permutations = args.n_permutations
    if getattr(args, "layer", None) is not None:
        cfg.model.layer = args.layer
    if getattr(args, "seed", None) is not None:
        cfg.random_state = args.seed
    return cfg
