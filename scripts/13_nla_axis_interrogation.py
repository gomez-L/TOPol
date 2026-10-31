#!/usr/bin/env python3
"""Step 13 - NLA/AR interrogation of every gate-passing axis.

This is the CPU-only companion analysis `pipeline_2.ipynb` always intended to
run (see `topolm/nla.py`'s module docstring) but that earlier runs of this
project stopped short of: train a small natural-language autoencoder (NLA)
and an activation-reconstruction (AR) head against the SAME cached activations
`06_topol_discovery.py`/`07_topol_field_assembly.py` already produced, then
decode a natural-language prototype and an A->B interpolation sweep for every
one of the 543 gate-passing axes -- an internals-grounded reading of each
axis's meaning, complementary to (not a replacement for) the differential
TF-IDF vocabulary and LLM-completion glosses from `09_pole_tfidf_wordclouds.py`
/`10_pole_llm_explanations.py`.

Needs no GPU and no Gemma reload: everything here trains on the pooled,
already-cached per-document activations and texts (see `nla.py`'s module
docstring for why that is a deliberate scope decision, not an oversight).

COMPUTE-BUDGET NOTE: `NLA.prepare_axis_dataset` pools every gate-passing
axis's FULL topic every time it appears (see that function's docstring), so
the raw pooled dataset is ~40x the corpus size (393,207 rows for this run's
9,971 documents / 543 axes) -- far more than a small GRU decoder needs to
converge, and slow to train on CPU. `NLA.subsample_axis_dataset` caps each
(axis, pole) cell at `--max-per-pole` documents for TRAINING only; prototype
decoding and the A->B sweep are computed afterwards against every axis's
FULL, unsampled document set, so centroids and nearest-real-doc grounding are
never limited by the training cap.
"""
import warnings; warnings.filterwarnings("ignore")
import os
import time
import numpy as np
from _common import base_parser, open_existing_store, load_config_for, apply_overrides
from topolm import nla as NLA
from topolm import topol_activations as TA


def main():
    ap = base_parser("NLA/AR axis interrogation")
    ap.add_argument("--max-per-pole", type=int, default=25,
                    help="training-set cap per (axis, pole) cell; prototypes/sweeps "
                         "still use the full, unsampled document set")
    ap.add_argument("--latent-dim", type=int, default=256)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--ar-splits", type=int, default=5)
    ap.add_argument("--ar-epochs", type=int, default=60)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    args = ap.parse_args()
    store = open_existing_store(args)
    cfg = apply_overrides(load_config_for(store), args)

    import torch
    torch.set_num_threads(max(1, args.threads))

    texts = store.load_object("texts")
    cluster_labels = store.load_array("cluster_labels").astype(int)
    gate_results = store.load_object("topol_gate_results")
    cache = TA.load_activation_cache(store)

    if "topol_nla_vocab" in store.manifest["entries"]:
        print("[13] [cache] topol_nla_vocab found on disk, loading")
        vocab = store.load_object("topol_nla_vocab")
    else:
        print("[13] building NLA vocabulary from corpus text...")
        vocab = NLA.build_vocab(texts, max_size=4000)
        store.save_object("topol_nla_vocab", vocab)
    print(f"[13] vocab size: {len(vocab)}")

    print("[13] pooling every gate-passing axis into one NLA dataset "
          "(prepare_axis_dataset, CPU, no GPU)...")
    t0 = time.time()
    data, axis_index = NLA.prepare_axis_dataset(gate_results, texts, cluster_labels, cache,
                                                vocab, max_len=48)
    print(f"[13] {len(data['texts'])} (axis, document) rows across "
          f"{axis_index['axis_id'].nunique()} axes  [{time.time() - t0:.1f}s]")
    store.save_table("topol_nla_axis_index", axis_index)

    print(f"[13] subsampling to <= {args.max_per_pole} docs/pole/axis for training...")
    small = NLA.subsample_axis_dataset(data, max_per_pole=args.max_per_pole,
                                       random_state=cfg.random_state)
    print(f"[13] training rows: {len(small['texts'])} "
          f"(full: {len(data['texts'])}, {len(data['texts']) / len(small['texts']):.1f}x reduction)")

    if store.manifest["entries"].get("topol_nla_encoder") and \
       store.manifest["entries"].get("topol_nla_decoder"):
        print("[13] [cache] topol_nla_encoder/decoder found on disk, loading")
        nla_result = {
            "encoder": store.load_object("topol_nla_encoder"),
            "decoder": store.load_object("topol_nla_decoder"),
            "history": store.load_object("topol_nla_history"),
            "latent_dim": args.latent_dim, "n_topics": data["n_topics"],
            "n_layers": int(data["layer_ids"].max()) + 1,
        }
    else:
        print(f"[13] training NLA (encoder+decoder), {args.epochs} epochs, "
              f"batch_size={args.batch_size}, device={args.device}...")
        t0 = time.time()
        nla_result = NLA.train_nla(small, vocab, latent_dim=args.latent_dim,
                                   hidden_dim=args.hidden_dim, epochs=args.epochs,
                                   batch_size=args.batch_size, device=args.device,
                                   random_state=cfg.random_state)
        print(f"[13] NLA training done  [{time.time() - t0:.1f}s]")
        store.save_object("topol_nla_encoder", nla_result["encoder"])
        store.save_object("topol_nla_decoder", nla_result["decoder"])
        store.save_object("topol_nla_history", nla_result["history"])

    if store.manifest["entries"].get("topol_nla_ar_summary"):
        print("[13] [cache] topol_nla_ar_summary found on disk, loading")
        ar_summary = store.load_json("topol_nla_ar_summary")
        rmse_per_doc = store.load_array("topol_nla_ar_rmse_per_doc")
    else:
        print(f"[13] training AR head ({args.ar_splits}-fold CV, {args.ar_epochs} "
              f"epochs/fold) on the SAME training subsample...")
        t0 = time.time()
        ar_result = NLA.train_ar_cv(nla_result, small, n_splits=args.ar_splits,
                                    epochs=args.ar_epochs, device=args.device,
                                    random_state=cfg.random_state)
        print(f"[13] AR CV done  overall RMSE={ar_result['overall_rmse']:.4f}  "
              f"R^2={ar_result['overall_r2']:.4f}  [{time.time() - t0:.1f}s]")
        rmse_per_doc = ar_result["rmse_per_doc"]
        ar_summary = {"overall_rmse": ar_result["overall_rmse"], "overall_r2": ar_result["overall_r2"],
                      "n_train_rows": len(small["texts"]), "max_per_pole": args.max_per_pole}
        store.save_array("topol_nla_ar_rmse_per_doc", rmse_per_doc)
        store.save_json("topol_nla_ar_summary", ar_summary)

    # pole_contrast_table: AR fidelity gap between an axis's own pole A and
    # pole B, on the SAME training subsample rmse_per_doc was computed on --
    # "cluster" here is axis_id, "cb_labels" is each row's own pole label.
    print("[13] AR fidelity contrast per axis (pole A vs pole B)...")
    pole_contrast = NLA.pole_contrast_table(rmse_per_doc, small["topic_ids"], small["pole_labels"])
    pole_contrast = pole_contrast.rename(columns={"cluster": "axis_id"})
    store.save_table("topol_nla_pole_contrast", pole_contrast)

    # Prototypes + sweep: computed against the FULL, unsampled dataset so
    # every axis's centroid and nearest-real-docs draw on all of its
    # documents, not just the training subsample.
    layer_id_map = NLA.axis_layer_id_map(axis_index)
    print("[13] embedding the FULL (unsampled) dataset through the trained encoder...")
    t0 = time.time()
    proto_df = NLA.pole_prototypes_table(nla_result, data, vocab, data["texts"],
                                         data["topic_ids"], data["pole_labels"],
                                         n_neighbors=3, layer_id_map=layer_id_map)
    proto_df = proto_df.rename(columns={"cluster": "axis_id"}).merge(
        axis_index[["axis_id", "topic", "contrast_index", "l_star", "effect_size_at_lstar"]],
        on="axis_id", how="left")
    store.save_table("topol_nla_pole_prototypes", proto_df)
    print(f"[13] pole prototypes: {len(proto_df)} rows  [{time.time() - t0:.1f}s]")

    t0 = time.time()
    sweep_df = NLA.direction_sweep_table(nla_result, data, vocab, data["topic_ids"],
                                         data["pole_labels"], n_steps=5, layer_id_map=layer_id_map)
    store.save_table("topol_nla_direction_sweep", sweep_df)
    print(f"[13] direction sweep: {len(sweep_df)} rows  [{time.time() - t0:.1f}s]")

    summary_df = NLA.pole_direction_summary(
        proto_df.rename(columns={"axis_id": "cluster"}), sweep_df, max_words=8)
    summary_df = summary_df.rename(columns={"cluster": "axis_id"}).merge(
        axis_index[["axis_id", "topic", "contrast_index", "l_star", "effect_size_at_lstar"]],
        on="axis_id", how="left")
    store.save_table("topol_nla_pole_direction_summary", summary_df)
    print(f"[13] pole direction summary: {len(summary_df)} axes")

    print("[13] done. New entries:")
    for k, kind in sorted(store.list_entries().items()):
        if k.startswith("topol_nla"):
            print(f"  [{kind:9s}] {k}")


if __name__ == "__main__":
    main()
