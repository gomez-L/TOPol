#!/usr/bin/env python3
"""Step 06 - per-topic recursive-bipartition discovery + the split-half gate.

Topic assignment is the existing `cluster_labels` from 03_cluster_manifold.py
(unchanged). For every topic this runs `topol_axes.topic_axes_pipeline`
(centred 2-means bipartition, deflation, cross-layer dedup, depth profile)
independently on the full topic and on two disjoint halves, then gates every
full-topic contrast on split-half |cos| stability (`topol_gate.gate_topic`).

Needs step 05's activation cache and step 03's cluster labels from the same run.
"""
import warnings; warnings.filterwarnings("ignore")
from _common import base_parser, open_existing_store, load_config_for, apply_overrides
from topolm import pipeline as P
from topolm import data as D


def main():
    ap = base_parser("Per-topic axis discovery + split-half gate")
    ap.add_argument("--min-cluster-size", type=int, default=None)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--ari-merge", type=float, default=None)
    ap.add_argument("--stability-threshold", type=float, default=None)
    args = ap.parse_args()
    store = open_existing_store(args)
    cfg = apply_overrides(load_config_for(store), args)
    if args.min_cluster_size is not None:
        cfg.discovery.min_cluster_size = args.min_cluster_size
    if args.max_depth is not None:
        cfg.discovery.max_depth = args.max_depth
    if args.ari_merge is not None:
        cfg.discovery.ari_merge = args.ari_merge
    if args.stability_threshold is not None:
        cfg.discovery.stability_threshold = args.stability_threshold

    texts = store.load_object("texts")
    labels = store.load_array("labels")
    ratings = D.recover_ratings(cfg, texts, labels, data_dir=args.data_dir)

    results = P.step_topol_discovery(store, cfg, ratings=ratings)
    n_topics = len(results)
    n_pass = sum(sum(1 for c in r["contrasts"] if c["passed"]) for r in results.values())
    n_cand = sum(len(r["contrasts"]) for r in results.values())
    print(f"[06] {n_topics} topics analysed | {n_pass}/{n_cand} contrasts passed the gate")

if __name__ == "__main__":
    main()
