#!/usr/bin/env python3
"""Step 14 - semantic (meaning-based) family clustering.

Replaces the depth-profile-correlation family clustering that
`07_topol_field_assembly.py` used to compute alongside the geometric
(|cos|-direction) one (see `topolm/field.py`'s module docstring for why that
comparison was removed: it produced 543 singleton families and chance-level
agreement with the geometric clustering, which is a degenerate result, not
an informative one). This step answers the "do these axes mean the same
thing" question directly, from each axis's own differential TF-IDF
vocabulary (`09_pole_tfidf_wordclouds.py`) rather than from raw geometry.

Needs `09_pole_tfidf_wordclouds.py`'s `topol_pv_tfidf_terms` table and
`topol_nla_axis_index` (or `topol_pv_tfidf_summary`) for the axis_id -> topic
lookup, plus (for the agreement check) `topol_field_summary.json`'s existing
geometric family labels. No GPU, no Gemma reload.
"""
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
from _common import base_parser, open_existing_store, load_config_for, apply_overrides
from topolm import axis_semantics as SEM


def main():
    ap = base_parser("Semantic (TF-IDF-vocabulary) axis family clustering")
    ap.add_argument("--merge-threshold", type=float, default=0.3)
    ap.add_argument("--top-n", type=int, default=8)
    args = ap.parse_args()
    store = open_existing_store(args)
    apply_overrides(load_config_for(store), args)

    def load_table(name):
        return pd.read_csv(os.path.join(store.dir, store.manifest["entries"][name]["path"]))

    if "topol_pv_tfidf_terms" not in store.manifest["entries"]:
        raise SystemExit("topol_pv_tfidf_terms not found -- run 09_pole_tfidf_wordclouds.py first")
    tfidf_long = load_table("topol_pv_tfidf_terms")
    axis_index = load_table("topol_nla_axis_index") if "topol_nla_axis_index" in store.manifest["entries"] \
        else load_table("topol_pv_tfidf_summary")[["axis_id", "topic", "contrast_index"]]

    print("[14] building signed per-axis term vectors from differential TF-IDF vocabulary...")
    M, axis_ids, terms = SEM.build_term_vectors(tfidf_long)
    print(f"[14] {M.shape[0]} axes over {M.shape[1]} shared terms")

    S = SEM.semantic_similarity_matrix(M)
    iu = np.triu_indices(len(axis_ids), k=1)
    vals = S[iu]
    print(f"[14] pairwise |cos| over vocabulary vectors: median={np.median(vals):.3f} "
          f"p90={np.quantile(vals, 0.90):.3f} p95={np.quantile(vals, 0.95):.3f} "
          f"p99={np.quantile(vals, 0.99):.3f} max={vals.max():.3f}")

    fam_df = SEM.semantic_families_table(M, axis_ids, terms, axis_index,
                                         merge_threshold=args.merge_threshold, top_n=args.top_n)
    store.save_table("topol_sem_families", fam_df)
    n_cross_topic = int((fam_df["n_topics"] > 1).sum())
    print(f"[14] {len(fam_df)} semantic families ({n_cross_topic} span >1 topic) "
          f"at merge_threshold={args.merge_threshold}")

    per_axis = pd.DataFrame({"axis_id": axis_ids,
                             "sem_family_id": SEM.semantic_family_clustering(
                                 M, merge_threshold=args.merge_threshold)})
    store.save_table("topol_sem_family_per_axis", per_axis)

    # Agreement with the existing geometric (|cos|-direction) family clustering.
    if "topol_field_summary" in store.manifest["entries"]:
        import json
        field_summary = json.load(open(os.path.join(
            store.dir, store.manifest["entries"]["topol_field_summary"]["path"])))
        geo_labels_flat = field_summary["families"]["cos_family_labels"]
        # cos_family_labels is in pooled_field_assembly's iteration order (topic
        # order, then contrast order, gate-passing only) -- topol_nla_axis_index
        # is built by prepare_axis_dataset in the SAME order (topic order, then
        # contrast order among that topic's passing contrasts), so position i in
        # both lists refers to the same axis. Verified by construction, not by
        # matching axis_id values (cos_family_labels carries no axis_id of its own).
        if len(geo_labels_flat) == len(axis_ids):
            ari = SEM.agreement_with_geometric(per_axis["sem_family_id"].to_numpy(),
                                               np.asarray(geo_labels_flat))
            print(f"[14] agreement with geometric (|cos|-direction) family clustering: ARI={ari:.3f}")
            store.save_json("topol_sem_geo_agreement", {"ari": ari, "n_axes": len(axis_ids)})
        else:
            print(f"[14] WARNING: geometric family label count ({len(geo_labels_flat)}) != "
                 f"axis count ({len(axis_ids)}) -- skipping agreement check")

    print("\n[14] top 10 semantic families by topic span:")
    with pd.option_context("display.max_colwidth", 80):
        print(fam_df.head(10)[["family_id", "n_axes", "n_topics", "topics",
                               "pole_neg_terms", "pole_pos_terms"]].to_string())


if __name__ == "__main__":
    main()
