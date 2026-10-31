#!/usr/bin/env python3
"""Step 08 - CB baseline comparison (spec step 8) + final reporting (step 9).

Per topic, compares the discovered/gate-passing axes against v_CB^t (the
existing diff-in-means estimator, `field.reconstruct_field` -- computed here
via the SAME depth-profile machinery so it gets an l* and effect size on the
same footing):

  identification    max_j |cos(a_j, v_CB^t)| at l_ref, and which axis achieves
                    it -- is v_CB^t represented among the discovered axes at all.
  sentiment alignment   Pearson correlation between each axis's own projection
                    and the CONTINUOUS star rating: `cb_corr_rating` for v_CB^t
                    itself (does the human-coded CB axis actually track rating
                    as a continuum in this topic, not just the two-class label
                    used to build it), and `corr_rating` per discovered axis
                    (already attached by `topol_gate.nuisance_diagnostics`).
                    `most_sentiment_aligned_index` is the passing axis with the
                    largest |corr_rating|; `step_topol_baseline_compare` warns
                    if v_CB^t's own correlation is weak, or notes when the
                    identification match is NOT the most sentiment-aligned axis
                    -- a measured replacement for an earlier, unfounded
                    assumption that identification should land at the topic's
                    first recursion depth (see that function's docstring).
  excess structure  count of gate-passing axes with |cos| < excess_cos_threshold
                    to v_CB^t -- the headline number.
  layer agreement   l* of the CB-matched axis vs l* of the excess axes -- the
                    direct test of whether "extra" structure lives at a
                    different depth than sentiment.

Then assembles the full per-axis report (topic, l*, layers found, effect
size, split-half |cos|, nuisance diagnostics, 10 highest/lowest-projecting
review texts per pole), the analogous v_CB^t-only report, and logs every
rejected split-half contrast.

Needs steps 05-07's cache / gate results / field summary from the same run.
"""
import warnings; warnings.filterwarnings("ignore")
from _common import base_parser, open_existing_store, load_config_for, apply_overrides
from topolm import pipeline as P
from topolm import data as D
from topolm import topol_activations as TA
from topolm import topol_report as TR


def main():
    ap = base_parser("CB baseline comparison + final report")
    ap.add_argument("--n-extreme", type=int, default=10,
                    help="highest/lowest-projecting texts per pole")
    args = ap.parse_args()
    store = open_existing_store(args)
    cfg = apply_overrides(load_config_for(store), args)

    texts = store.load_object("texts")
    labels = store.load_array("labels")
    ratings = D.recover_ratings(cfg, texts, labels, data_dir=args.data_dir)

    cmp_result = P.step_topol_baseline_compare(store, cfg, ratings=ratings, texts=texts,
                                               n_extreme=args.n_extreme)
    for t, r in cmp_result.items():
        print(f"[08] topic {t}: identification |cos|={r['identification_cos']:.3f} "
              f"(l*={r['identification_l_star']}) | cb_corr_rating={r['cb_corr_rating']:.3f} | "
              f"most_sentiment_aligned_matches_identification="
              f"{r['identification_matches_most_sentiment_aligned']} | "
              f"excess={r['excess_count']}/{r['n_passing']} "
              f"excess_l*={r['excess_axis_l_stars']} | CB l*={r['cb_l_star']}")

    gate_results = store.load_object("topol_gate_results")
    cache = TA.load_activation_cache(store)
    cluster_labels = store.load_array("cluster_labels")
    report = TR.build_report(gate_results, texts, cluster_labels, cache, n_extreme=args.n_extreme)
    TR.save_report(store, report)
    print(f"[08] report: {len(report['axes'])} surviving axes, "
          f"{len(report['rejected'])} rejected contrasts logged")

if __name__ == "__main__":
    main()
