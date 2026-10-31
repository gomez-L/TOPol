#!/usr/bin/env python3
"""Part 2, step 12 - closing the loop: does a polar vector's DIRECTION carry
sentiment information?

Scores every document once with a Hugging Face sentiment classifier
(continuous polarity = p_pos - p_neg), aggregates it per axis/pole, and
tests whether `|cos(axis, v_CB^t)|` (direction similarity to the human-coded
sentiment reference) predicts sentiment alignment -- both against the
NEW HF-sentiment-based `sentiment_delta` and against the ALREADY-computed
`corr_rating` (continuous Amazon star-rating correlation, attached to every
gate-passing contrast by `topol_gate.nuisance_diagnostics` back in
`06_topol_discovery.py`). See `topolm/pole_sentiment.py`'s module docstring
for the full rationale.

Needs `06_topol_discovery.py`/`07_topol_field_assembly.py`'s gate results,
cache and field summary from the same run.
"""
import warnings; warnings.filterwarnings("ignore")
import json
from _common import base_parser, open_existing_store, load_config_for, apply_overrides
from topolm import pole_sentiment as PS
from topolm import topol_activations as TA
from topolm import viz as V
import matplotlib.pyplot as plt


def main():
    ap = base_parser("Sentiment scoring + direction-vs-CB alignment test")
    ap.add_argument("--sentiment-model", default="cardiffnlp/twitter-roberta-base-sentiment-latest")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    store = open_existing_store(args)
    apply_overrides(load_config_for(store), args)

    texts = store.load_object("texts")
    cluster_labels = store.load_array("cluster_labels").astype(int)
    cb_labels = store.load_array("labels")
    gate_results = store.load_object("topol_gate_results")
    cache = TA.load_activation_cache(store)
    field_summary = store.load_json("topol_field_summary")
    ref_layer = field_summary["ref_layer"]

    import pandas as pd, os
    axis_index = pd.read_csv(
        os.path.join(store.dir, store.manifest["entries"]["topol_nla_axis_index"]["path"]))

    if "topol_doc_sentiment" in store.manifest["entries"]:
        print("[12] topol_doc_sentiment found on disk, loading")
        doc_sentiment = store.load_array("topol_doc_sentiment")
    else:
        print(f"[12] scoring {len(texts)} documents with {args.sentiment_model} (one pass, whole corpus)...")
        tokenizer, model, device = PS.load_sentiment_model(args.sentiment_model)
        doc_sentiment = PS.polarity_scalar(tokenizer, model, texts, device, batch_size=args.batch_size)
        store.save_array("topol_doc_sentiment", doc_sentiment, meta={"model": args.sentiment_model})

    print("[12] building the per-axis alignment table (cos-to-CB, corr_rating, sentiment_delta)...")
    align_df = PS.build_alignment_table(gate_results, cache, cluster_labels, cb_labels, ref_layer,
                                        axis_index, doc_sentiment)
    store.save_table("topol_pv_sentiment_alignment", align_df)

    results = PS.test_alignment_hypothesis(align_df)
    store.save_json("topol_pv_alignment_test", results)
    for metric, r in results.items():
        print(f"[12] |cos_to_cb| vs |{metric}| (n={r['n']}): "
              f"pearson r={r['pearson_r']:.3f} (p={r['pearson_p']:.4g})  "
              f"spearman r={r['spearman_r']:.3f} (p={r['spearman_p']:.4g})")

    for metric, label in (("abs_corr_rating", "corr(projection, rating)"),
                         ("abs_sentiment_delta", "sentiment_delta (HF model)")):
        r = results[metric]
        fig = V.plot_direction_sentiment_alignment(align_df, metric, label,
                                                   r=r["pearson_r"], p=r["pearson_p"])
        store.save_fig(f"pv_alignment_{metric}", fig)
        plt.close(fig)
    print("[12] alignment scatter plots saved -> figures/pv_alignment_abs_corr_rating, "
          "figures/pv_alignment_abs_sentiment_delta")


if __name__ == "__main__":
    main()
