#!/usr/bin/env python3
"""Step 17 - causal ablation, part 2: what changed?

Needs `16_causal_direction_selection.py`'s plan + ablated activation matrix
from the same run. Four independent before/after comparisons
(`topolm/causal_ablation.py`), each optional via its own CLI flag since the
UMAP re-clustering step is the expensive one (two full UMAP+Leiden runs over
the whole corpus):

  1. NECESSITY -- does a linear A/B probe still separate regimes on the
     ablated activations, vs. on a matched-rank random-subspace control
     ablation (`data.py`'s own stated validity goal: "collapse regime
     separability under ablation but not under a random control").
  2. v_CB^t MAGNITUDE -- direct sanity check: ||v_CB^t|| should collapse for
     topics where v_CB^t was itself ablated.
  3. EXCESS-AXIS RETENTION -- do the gate-passing axes that were NOT
     ablated (low RoBERTa sentiment_delta) keep their own discriminative
     effect size once the sentiment directions are gone, or were they more
     entangled than their low `cos_to_cb` suggested?
  4. UMAP RE-CLUSTERING -- re-run topic segmentation on the ablated
     activations and compare against a FRESH re-clustering of the
     unablated activations at the same layer (same pooling convention both
     times, so only ablation differs) via adjusted Rand index + a
     contingency heatmap + a side-by-side 2-D scatter.
"""
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
from _common import base_parser, open_existing_store, load_config_for, apply_overrides
from topolm import causal_ablation as CA
from topolm import topol_activations as TA
from topolm import manifold as M
from topolm import viz as V


def main():
    ap = base_parser("Causal ablation: before/after analysis")
    ap.add_argument("--skip-umap", action="store_true",
                    help="skip the two UMAP+Leiden re-clustering runs (slow, CPU-bound)")
    args = ap.parse_args()
    store = open_existing_store(args)
    cfg = apply_overrides(load_config_for(store), args)

    for name in ("topol_causal_plan", "topol_causal_meta", "topol_causal_ablated_embedding"):
        if name not in store.manifest["entries"]:
            raise SystemExit(f"{name} not found -- run 16_causal_direction_selection.py first.")

    meta = store.load_json("topol_causal_meta")
    layer = meta["layer"]
    plan = store.load_object("topol_causal_plan")
    summary = pd.read_csv(
        os.path.join(store.dir, store.manifest["entries"]["topol_causal_summary"]["path"]))
    selected_axes = pd.read_csv(
        os.path.join(store.dir, store.manifest["entries"]["topol_causal_selected_axes"]["path"]))
    X_ablated = store.load_array("topol_causal_ablated_embedding")

    cluster_labels = store.load_array("cluster_labels").astype(int)
    cb_labels = store.load_array("labels")
    gate_results = store.load_object("topol_gate_results")
    cache = TA.load_activation_cache(store)
    X_full = np.asarray(cache["acts"][layer], dtype=np.float64)

    # -- 1. necessity ------------------------------------------------- #
    print("[17] necessity test: A/B probe, full vs. ablated vs. random-control ablation...")
    necessity = CA.necessity_test(X_full, cluster_labels, cb_labels, plan,
                                  n_splits=cfg.ablation.probe_n_splits, C=cfg.ablation.probe_C,
                                  random_state=cfg.random_state)
    store.save_json("topol_causal_necessity", necessity)
    print(f"[17] acc_full={necessity['acc_full']:.3f}  "
          f"acc_ablate_direction={necessity['acc_ablate_direction']:.3f}  "
          f"acc_ablate_random={necessity['acc_ablate_random']:.3f}  "
          f"necessity_drop={necessity['necessity_drop']:.3f}  control_drop={necessity['control_drop']:.3f}")
    fig = V.plot_ablation(necessity)
    store.save_fig("23_causal_necessity", fig)

    # -- 2. v_CB^t magnitude ------------------------------------------ #
    print("[17] v_CB^t magnitude, before vs. after...")
    fld_geo = CA.field_geometry_before_after(X_full, X_ablated, cluster_labels, cb_labels)
    mag_df = pd.DataFrame({"topic": fld_geo["clusters"],
                           "magnitude_before": fld_geo["magnitude_before"],
                           "magnitude_after": fld_geo["magnitude_after"]})
    store.save_table("topol_causal_cb_magnitude", mag_df)
    ablated_topics = summary.loc[summary["sources"].str.contains("v_CB", na=False), "topic"]
    fig = V.plot_cb_magnitude_before_after(mag_df["topic"], mag_df["magnitude_before"],
                                           mag_df["magnitude_after"], ablated_topics=ablated_topics)
    store.save_fig("24_causal_cb_magnitude", fig)

    # -- 3. excess-axis retention -------------------------------------- #
    print("[17] excess-axis effect-size retention...")
    excess_df = CA.excess_axis_effect_size_before_after(
        gate_results, selected_axes, cluster_labels, X_full, X_ablated, layer)
    store.save_table("topol_causal_excess_axis_retention", excess_df)
    print(f"[17] {len(excess_df)} excess axes | median d_retained_frac="
          f"{excess_df['d_retained_frac'].median():.3f}  "
          f"mean={excess_df['d_retained_frac'].mean():.3f}")
    fig = V.plot_excess_axis_retention(excess_df)
    store.save_fig("25_causal_excess_axis_retention", fig)

    # -- 4. UMAP re-clustering ------------------------------------------ #
    if args.skip_umap:
        print("[17] --skip-umap: skipping re-clustering comparison")
        return

    from sklearn.metrics import adjusted_rand_score
    print("[17] re-clustering before ablation (fresh UMAP+Leiden on unablated activations)...")
    cl_before = M.cluster_documents(X_full, cfg)
    print("[17] re-clustering after ablation...")
    cl_after = M.cluster_documents(X_ablated, cfg)

    ari = float(adjusted_rand_score(cl_before["labels"], cl_after["labels"]))
    ari_vs_original = float(adjusted_rand_score(cluster_labels, cl_before["labels"]))
    store.save_json("topol_causal_reclustering", {
        "ari_before_vs_after": ari, "ari_fresh_before_vs_original_clustering": ari_vs_original,
        "n_clusters_before": cl_before["n_clusters"], "n_clusters_after": cl_after["n_clusters"],
    })
    print(f"[17] ARI(before,after)={ari:.3f}  n_clusters before={cl_before['n_clusters']} "
          f"after={cl_after['n_clusters']}  "
          f"(sanity: ARI(fresh-before, original Part-2 clustering)={ari_vs_original:.3f})")

    fig = V.plot_umap_before_after(cl_before["coords_2d"], cl_after["coords_2d"], cluster_labels,
                                   label_name="original topic")
    store.save_fig("26_causal_umap_before_after", fig)
    fig = V.plot_cluster_agreement(cl_before["labels"], cl_after["labels"], ari=ari)
    store.save_fig("27_causal_cluster_agreement", fig)


if __name__ == "__main__":
    main()
