#!/usr/bin/env python3
"""Step 15 - instruction-tuned comparison: gemma-3-4b-it pole explanations.

`10_pole_llm_explanations.py` glosses every axis with the SAME `-pt`
(pretrained, non-instruction-tuned) checkpoint used for the field, via a
grounded COMPLETION prompt -- necessary because `-pt` does not reliably obey
an instruction, but empirically limited: 84.9% of its 543 glosses reduce to
a generic "Group A/B reviews are more positive/negative" regardless of
whether the axis is actually about sentiment (cross-check against
`topol_pv_tfidf_summary` and `topol_pv_sentiment_alignment`).

This script asks the SAME question, grounded in the SAME evidence, of
`google/gemma-3-4b-it` -- the instruction-tuned sibling checkpoint -- via a
direct instruction rather than a completion (`pole_llm.explain_axes_it`).
Read its output next to the `-pt` gloss AND the TF-IDF vocabulary
(`10_pole_llm_explanations.py`'s docstring); agreement across TF-IDF and
BOTH LLM routes is the trustworthy signal, not either LLM route alone -- an
instruction-tuned model can confabulate a fluent, wrong contrast just as
easily as a base model can default to a generic one.

NEEDS A SECOND ~4B-PARAMETER MODEL RESIDENT ON THE GPU. If VRAM is tight,
run this AFTER the rest of the pipeline (steps 05-13) has produced every
cached artifact it needs, so the `-pt` field backend does not need to stay
loaded simultaneously -- this script does not load it at all, only `-it`.
"""
import warnings; warnings.filterwarnings("ignore")
import os
import glob
from _common import base_parser, open_existing_store, load_config_for, apply_overrides
from topolm import nla as NLA
from topolm import pole_llm as PL
from topolm import topol_activations as TA
from topolm.backends import make_backend
from topolm.config import ModelConfig


def _find_or_download(model_id: str, cache_dir: str) -> str:
    """Same local-snapshot-first pattern the notebooks use for `-pt` --
    checks `cache_dir` for a complete snapshot before downloading."""
    from huggingface_hub import snapshot_download
    os.makedirs(cache_dir, exist_ok=True)
    folder_name = "models--" + model_id.replace("/", "--")
    matches = glob.glob(os.path.join(cache_dir, folder_name, "snapshots", "*"))
    if matches:
        print(f"[15] [cache] {model_id} already present in {cache_dir}")
        return matches[0]
    print(f"[15] {model_id} not found in {cache_dir} -- downloading now (one-time)...")
    path = snapshot_download(repo_id=model_id, cache_dir=cache_dir)
    print(f"[15] downloaded to: {path}")
    return path


def main():
    ap = base_parser("Gemma-3-4B-IT pole explanations (comparison route)")
    ap.add_argument("--it-model", default="google/gemma-3-4b-it")
    ap.add_argument("--hf-cache-dir", default=os.path.expanduser("~/scratch/hf_cache"))
    ap.add_argument("--n-snippets", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    store = open_existing_store(args)
    cfg = apply_overrides(load_config_for(store), args)

    texts = store.load_object("texts")
    cluster_labels = store.load_array("cluster_labels").astype(int)
    gate_results = store.load_object("topol_gate_results")
    cache = TA.load_activation_cache(store)
    vocab = store.load_object("topol_nla_vocab")

    print("[15] rebuilding the per-axis document dataset...")
    data, axis_index = NLA.prepare_axis_dataset(gate_results, texts, cluster_labels, cache, vocab, max_len=48)

    tfidf_summary = None
    if "topol_pv_tfidf_summary" in store.manifest["entries"]:
        import pandas as pd
        tfidf_summary = pd.read_csv(
            os.path.join(store.dir, store.manifest["entries"]["topol_pv_tfidf_summary"]["path"]))
        print("[15] using topol_pv_tfidf_summary's top words as extra prompt anchors")
    else:
        print("[15] WARNING: topol_pv_tfidf_summary not found -- run 09_pole_tfidf_wordclouds.py first "
              "for richer prompts; continuing with snippets only")

    print(f"[15] loading {args.it_model} (a SECOND ~4B-parameter model, separate from the -pt field model)...")
    it_snapshot = _find_or_download(args.it_model, args.hf_cache_dir)
    it_cfg = ModelConfig(name=it_snapshot, layer=cfg.model.layer, pooling=cfg.model.pooling,
                         dtype=cfg.model.dtype, device=cfg.model.device,
                         max_length=cfg.model.max_length, batch_size=cfg.model.batch_size)
    it_cfg_full = cfg
    it_cfg_full.model = it_cfg
    backend_it = make_backend(it_cfg_full)
    print(f"[15] model: {args.it_model} | d_model {backend_it.d_model}")

    print(f"[15] generating instruction-style explanations for {len(axis_index)} axes "
          f"(batch_size={args.batch_size}, max_new_tokens={args.max_new_tokens})...")
    explanations_it = PL.explain_axes_it(backend_it, data, axis_index, tfidf_summary=tfidf_summary,
                                         n_snippets=args.n_snippets, max_new_tokens=args.max_new_tokens,
                                         batch_size=args.batch_size, random_state=cfg.random_state)
    store.save_table("topol_pv_llm_explanations_it", explanations_it)
    print(f"[15] {len(explanations_it)} axes explained -> topol_pv_llm_explanations_it")

    # Quick side-by-side sanity read against the -pt route, if present.
    if "topol_pv_llm_explanations" in store.manifest["entries"]:
        import pandas as pd
        pt_df = pd.read_csv(os.path.join(
            store.dir, store.manifest["entries"]["topol_pv_llm_explanations"]["path"]))
        merged = explanations_it.merge(pt_df[["axis_id", "explanation"]], on="axis_id", how="left")
        store.save_table("topol_pv_llm_pt_vs_it", merged)
        generic = merged["explanation"].str.contains("positive|negative", case=False, na=False)
        generic_it = merged["explanation_it"].str.contains("positive|negative", case=False, na=False) & \
                    (merged["explanation_it"].str.len() < 90)
        print(f"[15] -pt generic-sentiment-gloss rate: {generic.mean():.1%}  "
              f"-it generic-sentiment-gloss rate: {generic_it.mean():.1%}")


if __name__ == "__main__":
    main()
