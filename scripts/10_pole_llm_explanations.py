#!/usr/bin/env python3
"""Part 2, step 10 - a natural-language gloss for every polar vector, via the
SAME gemma-3-4b-pt checkpoint already loaded as this pipeline's `backend`
(no new model download -- see `topolm/pole_llm.py`'s module docstring for
why the prompt is a grounded completion, not an instruction, given `-pt` is
a base/pretrained checkpoint).

Needs `09_pole_tfidf_wordclouds.py` to have run first if you want each axis's
top TF-IDF words folded into its prompt (optional -- falls back to snippets
only if that table isn't present yet).
"""
import warnings; warnings.filterwarnings("ignore")
import os
from _common import base_parser, open_existing_store, load_config_for, apply_overrides, make_backend
from topolm import nla as NLA
from topolm import pole_llm as PL
from topolm import topol_activations as TA


def main():
    ap = base_parser("Gemma-based natural-language pole explanations")
    ap.add_argument("--n-snippets", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    store = open_existing_store(args)
    cfg = apply_overrides(load_config_for(store), args)

    texts = store.load_object("texts")
    cluster_labels = store.load_array("cluster_labels").astype(int)
    gate_results = store.load_object("topol_gate_results")
    cache = TA.load_activation_cache(store)
    vocab = store.load_object("topol_nla_vocab")

    print("[10] rebuilding the per-axis document dataset...")
    data, axis_index = NLA.prepare_axis_dataset(gate_results, texts, cluster_labels, cache, vocab, max_len=48)

    tfidf_summary = None
    if "topol_pv_tfidf_summary" in store.manifest["entries"]:
        import pandas as pd
        tfidf_summary = pd.read_csv(
            os.path.join(store.dir, store.manifest["entries"]["topol_pv_tfidf_summary"]["path"]))
        print("[10] using topol_pv_tfidf_summary's top words as extra prompt anchors")
    else:
        print("[10] WARNING: topol_pv_tfidf_summary not found -- run 09_pole_tfidf_wordclouds.py first "
              "for richer prompts; continuing with snippets only")

    print("[10] loading gemma-3-4b-pt backend (this is the model already used throughout this pipeline)...")
    backend = make_backend(cfg)

    print(f"[10] generating explanations for {len(axis_index)} axes "
          f"(batch_size={args.batch_size}, max_new_tokens={args.max_new_tokens})...")
    explanations = PL.explain_axes(backend, data, axis_index, tfidf_summary=tfidf_summary,
                                   n_snippets=args.n_snippets, max_new_tokens=args.max_new_tokens,
                                   batch_size=args.batch_size, random_state=cfg.random_state)
    store.save_table("topol_pv_llm_explanations", explanations)
    print(f"[10] {len(explanations)} explanations saved -> topol_pv_llm_explanations")


if __name__ == "__main__":
    main()
