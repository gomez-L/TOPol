#!/usr/bin/env python3
"""Part 2, step 09 - full per-axis text export + per-topic differential
TF-IDF vocabulary + word clouds.

For every gate-passing axis (topic, contrast) from `06_topol_discovery.py`/
Part 1's `topol_gate_results`: materialize every document behind it (both
poles) as its own CSV, then -- ONE TF-IDF fit per topic, reused across every
axis in that topic (see `topolm/pole_vocab.py`'s module docstring for why
this is the right granularity, not per-axis or global) -- rank the terms
most distinctive of pole A vs pole B, and render a word-cloud pair per axis.

Needs steps 05-07's cache / gate results from the same run (`06_topol_discovery.py`,
`07_topol_field_assembly.py`) plus Part 1's NLA vocab (notebook Part 8 /
`topol_nla_vocab`).
"""
import warnings; warnings.filterwarnings("ignore")
from _common import base_parser, open_existing_store, load_config_for, apply_overrides
from topolm import nla as NLA
from topolm import pole_vocab as PV
from topolm import topol_activations as TA
from topolm import viz as V


def main():
    ap = base_parser("Per-axis full-text export + per-topic TF-IDF + word clouds")
    ap.add_argument("--top-n", type=int, default=20, help="distinctive terms per pole")
    ap.add_argument("--max-features", type=int, default=3000)
    args = ap.parse_args()
    store = open_existing_store(args)
    apply_overrides(load_config_for(store), args)

    texts = store.load_object("texts")
    cluster_labels = store.load_array("cluster_labels").astype(int)
    gate_results = store.load_object("topol_gate_results")
    cache = TA.load_activation_cache(store)
    vocab = store.load_object("topol_nla_vocab")

    print("[09] rebuilding the per-axis document dataset (prepare_axis_dataset, cheap/CPU)...")
    data, axis_index = NLA.prepare_axis_dataset(gate_results, texts, cluster_labels, cache, vocab, max_len=48)

    print("[09] full per-axis text export...")
    full_texts = PV.build_pv_full_text_table(data, axis_index)
    store.save_table("topol_pv_full_texts", full_texts)
    print(f"[09] {len(full_texts)} (axis, document) rows saved -> topol_pv_full_texts")

    print("[09] per-topic TF-IDF differential vocabulary...")
    long_df, summary_df = PV.build_pv_tfidf_table(gate_results, texts, cluster_labels, axis_index,
                                                   top_n=args.top_n, max_features=args.max_features)
    store.save_table("topol_pv_tfidf_terms", long_df)
    store.save_table("topol_pv_tfidf_summary", summary_df)
    print(f"[09] {len(summary_df)} axes summarized -> topol_pv_tfidf_summary / topol_pv_tfidf_terms")

    print("[09] rendering word-cloud pairs for every axis...")
    for axis_id in summary_df["axis_id"]:
        freqs_A = PV.term_frequencies_for_wordcloud(long_df, axis_id, "A")
        freqs_B = PV.term_frequencies_for_wordcloud(long_df, axis_id, "B")
        row = summary_df[summary_df["axis_id"] == axis_id].iloc[0]
        fig = V.plot_pole_wordclouds(freqs_A, freqs_B,
                                     title=f"topic {row['topic']}, axis {axis_id} (l*={row['l_star']})")
        store.save_fig(f"pv_wordcloud_axis{int(axis_id):03d}", fig)
        import matplotlib.pyplot as plt
        plt.close(fig)
    print(f"[09] {len(summary_df)} word-cloud pairs saved under figures/pv_wordcloud_axis*")


if __name__ == "__main__":
    main()
