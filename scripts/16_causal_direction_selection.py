#!/usr/bin/env python3
"""Step 16 - causal ablation, part 1: identify the sentiment-related
directions per topic and build the ablated activation matrix.

Two families of direction, per topic (`topolm/causal_ablation.py`):
  1. v_CB^t -- the supervised, rating-derived CB polarity vector.
  2. Gate-passing DISCOVERED axes flagged as sentiment-related by an
     INDEPENDENT signal: a pretrained RoBERTa sentiment classifier's
     per-document polarity score (`pole_sentiment.py` / script 12's
     `topol_pv_sentiment_alignment` table), not this pipeline's own
     `corr_rating`/`cos_to_cb`.

Both are removed by orthogonal projection at ONE shared layer
(`cfg.ablation.layer`, default: this run's `topol_field_summary["ref_layer"]`),
topic by topic, so a document only ever loses directions discovered in its
OWN topic. Needs `06_topol_discovery.py`/`07_topol_field_assembly.py`'s gate
results + field summary, and `12_pole_sentiment_alignment.py`'s alignment
table, from the same run -- run that first if `topol_pv_sentiment_alignment`
is missing.
"""
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
from _common import base_parser, open_existing_store, load_config_for, apply_overrides
from topolm import causal_ablation as CA
from topolm import topol_activations as TA


def main():
    ap = base_parser("Causal ablation: select + remove sentiment-related directions")
    ap.add_argument("--sentiment-delta-threshold", type=float, default=None,
                    help="override cfg.ablation.sentiment_delta_threshold")
    ap.add_argument("--ablation-layer", type=int, default=None,
                    help="override cfg.ablation.layer (default: this run's ref_layer)")
    ap.add_argument("--no-cb", action="store_true", help="do not ablate v_CB^t")
    ap.add_argument("--no-discovered", action="store_true", help="do not ablate discovered axes")
    args = ap.parse_args()
    store = open_existing_store(args)
    cfg = apply_overrides(load_config_for(store), args)
    if args.sentiment_delta_threshold is not None:
        cfg.ablation.sentiment_delta_threshold = args.sentiment_delta_threshold
    if args.ablation_layer is not None:
        cfg.ablation.layer = args.ablation_layer
    if args.no_cb:
        cfg.ablation.include_cb = False
    if args.no_discovered:
        cfg.ablation.include_discovered_axes = False

    if "topol_pv_sentiment_alignment" not in store.manifest["entries"]:
        raise SystemExit(
            "topol_pv_sentiment_alignment not found -- run "
            "12_pole_sentiment_alignment.py on this run first.")

    cluster_labels = store.load_array("cluster_labels").astype(int)
    cb_labels = store.load_array("labels")
    gate_results = store.load_object("topol_gate_results")
    cache = TA.load_activation_cache(store)
    field_summary = store.load_json("topol_field_summary")
    layer = cfg.ablation.layer if cfg.ablation.layer is not None else field_summary["ref_layer"]
    if layer not in cache["layers"]:
        raise SystemExit(f"ablation layer {layer} is not in this run's cached layers {cache['layers']}")
    align_df = pd.read_csv(
        os.path.join(store.dir, store.manifest["entries"]["topol_pv_sentiment_alignment"]["path"]))

    print(f"[16] identifying sentiment directions at layer {layer} "
          f"(include_cb={cfg.ablation.include_cb}, "
          f"include_discovered_axes={cfg.ablation.include_discovered_axes}, "
          f"sentiment_delta_threshold={cfg.ablation.sentiment_delta_threshold})")
    result = CA.build_ablation_plan(
        gate_results, cache, cluster_labels, cb_labels, align_df, layer=layer,
        sentiment_delta_threshold=cfg.ablation.sentiment_delta_threshold,
        include_cb=cfg.ablation.include_cb, include_discovered_axes=cfg.ablation.include_discovered_axes)
    plan, summary, selected_axes = result["plan"], result["summary"], result["selected_axes"]

    store.save_object("topol_causal_plan", plan)
    store.save_table("topol_causal_summary", summary)
    store.save_table("topol_causal_selected_axes", selected_axes)
    store.save_json("topol_causal_meta", {
        "layer": int(layer), "sentiment_delta_threshold": cfg.ablation.sentiment_delta_threshold,
        "include_cb": cfg.ablation.include_cb, "include_discovered_axes": cfg.ablation.include_discovered_axes,
        "n_topics": len(plan), "n_selected_discovered_axes": len(selected_axes),
        "total_ablated_rank": int(summary["ablated_rank"].sum()),
    })
    print(f"[16] {len(selected_axes)} discovered axes selected as sentiment-related "
          f"(|sentiment_delta| >= {cfg.ablation.sentiment_delta_threshold}) "
          f"across {(summary['ablated_rank'] > 0).sum()}/{len(summary)} topics")
    print(summary.to_string(index=False))

    print("[16] building the ablated activation matrix at layer", layer)
    X_full = np.asarray(cache["acts"][layer], dtype=np.float64)
    X_ablated = CA.ablate_embedding(X_full, cluster_labels, plan, random_control=False)
    store.save_array("topol_causal_ablated_embedding", X_ablated, meta={"layer": int(layer)})
    print(f"[16] saved ablated ({X_ablated.shape}) activation matrix -> "
          f"objects/topol_causal_ablated_embedding.npz")


if __name__ == "__main__":
    main()
