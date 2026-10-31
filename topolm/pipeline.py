"""Pipeline steps, shared by the CLI scripts and pipeline_2.ipynb.

Each `step_*` function takes the open `RunStore`, the `Config`, and the `Backend`,
reads any prerequisites it needs from the store, computes its piece, saves every
resulting object + figure, and returns the results in-memory.  The scripts call
these one per process (reloading prerequisites from disk); the notebook calls them
in sequence keeping everything live.  Same code path either way.

Pruned to exactly what `scripts/00-08_*.py` and `notebooks/pipeline_2.ipynb` use:
corpus extraction, layer selection, topic clustering, the CB field, and the V1
axis-discovery steps. The CB estimator's OWN steps (`step_extract` through
`step_field`) are unchanged from the original pipeline; SAE decomposition,
causal validation (steering/ablation) and calibration benchmarking were a
different, now-removed branch of this codebase and are not needed here.
"""
from __future__ import annotations

import numpy as np

from . import data as D
from . import representations as R
from . import manifold as M
from . import layer_select as LS
from . import field as F
from . import viz as V
from . import topol_activations as TA
from . import topol_axes as TAX
from . import topol_gate as TG
from . import topol_report as TR


# ------------------------------------------------------------------ #
def _get_state(store):
    try:
        return store.load_json("state")
    except Exception:
        return {}

def _set_state(store, **kv):
    st = _get_state(store)
    st.update(kv)
    store.save_json("state", st)
    return st


# ------------------------------------------------------------------ #
def step_extract(store, cfg, backend, seg_backend=None, data_dir="data",
                 subsample_layers=400):
    """Load the corpus and build BOTH representations.

    SEGMENTATION (EmbeddingGemma-300M) -> the vectors UMAP/Leiden operate on.
    FIELD        (Gemma 3 12B residual) -> where polarity vectors and every causal
                                           test live.

    Two models because the jobs have incompatible requirements: clustering needs
    similarity geometry, the field needs intervenability. Only cluster LABELS cross
    from the first to the second.
    """
    texts, labels, topics, meta = D.load_dataset(cfg, data_dir=data_dir, backend=backend)
    print(f"  corpus: {meta.get('corpus')} | n={len(texts)} "
          f"| A={int((labels==0).sum())} B={int((labels==1).sum())} "
          f"| CB rule: {meta.get('rule')}")

    # segmentation representations (labels only)
    if seg_backend is not None and seg_backend is not backend:
        seg_emb = seg_backend.embed(texts)
        store.save_array("seg_embeddings", seg_emb,
                         meta={"model": getattr(seg_backend, "name", "segmentation")})
        print(f"  segmentation embeddings: {seg_emb.shape} "
              f"({getattr(seg_backend, 'name', 'n/a')})")

    acts_all, idx = R.extract_all_layers(backend, texts, pooling=cfg.model.pooling,
                                         subsample=subsample_layers, random_state=cfg.random_state)
    store.save_object("texts", texts)
    store.save_array("labels", labels)
    store.save_object("topics", None if topics is None else np.asarray(topics))
    store.save_array("acts_all_layers", acts_all, meta={"subsample_idx": "layer_subsample_idx"})
    store.save_array("layer_subsample_idx", idx)
    store.save_json("data_meta", meta)
    return {"texts": texts, "labels": labels, "topics": topics,
            "acts_all_layers": acts_all, "layer_subsample_idx": idx, "meta": meta}


def step_select_layer(store, cfg, backend, extract=None):
    """Pick the residual layer by A/B probe separability, then embed all docs there."""
    ex = extract or {
        "texts": store.load_object("texts"),
        "labels": store.load_array("labels"),
        "acts_all_layers": store.load_array("acts_all_layers"),
        "layer_subsample_idx": store.load_array("layer_subsample_idx"),
    }
    # If the wide (4-depth) dictionaries are wanted, restrict the layer sweep to
    # the depths they cover; otherwise leave selection free and use an "_all" site.
    restrict = None
    sc = getattr(cfg, "sae", None)
    if sc is not None and getattr(sc, "constrain_layer_selection", False):
        try:
            from .gemmascope2 import subset_layers
            n_layers = ex["acts_all_layers"].shape[1] - 1
            restrict = [l for l in subset_layers(n_layers) if 0 < l <= n_layers]
            print(f"  layer selection restricted to Gemma Scope 2 wide depths: {restrict}")
        except Exception:
            restrict = None

    if cfg.model.layer is not None:
        best = cfg.model.layer
        sel = {"per_layer": {}, "best_layer": best, "best_acc": float("nan"),
               "note": "layer fixed in config"}
    else:
        sel = LS.probe_separability_by_layer(
            ex["acts_all_layers"], ex["labels"][ex["layer_subsample_idx"]],
            random_state=cfg.random_state)
        if restrict:
            cand = {l: sel["per_layer"][l] for l in restrict if l in sel["per_layer"]}
            if cand:
                best = max(cand, key=lambda l: cand[l]["acc_mean"])
                sel["best_layer"] = int(best)
                sel["best_acc"] = cand[best]["acc_mean"]
                sel["restricted_to"] = restrict
        best = sel["best_layer"]
    emb = R.extract_layer(backend, ex["texts"], layer=best, pooling=cfg.model.pooling)
    fig = V.plot_layer_selection(sel) if sel["per_layer"] else None
    store.save_json("layer_selection", sel)
    store.save_array("embeddings", emb, meta={"layer": int(best)})
    _set_state(store, layer=int(best))
    if fig is not None:
        store.save_fig("01_layer_selection", fig)
    return {"selection": sel, "layer": int(best), "embeddings": emb, "fig": fig}


def step_cluster(store, cfg, embeddings=None):
    """UMAP + Leiden topic segmentation, on the SEGMENTATION representations.

    Clusters `seg_embeddings` (EmbeddingGemma) when available, falling back to the
    field representations when `segmentation.source == "field"` or on synthetic
    data. Only the resulting LABELS are used downstream.

    When `segmentation.boundary_independent` is set, the partition is fitted on
    regime A alone and regime B is assigned by nearest centroid, which removes the
    selection bias of Proposition 3.
    """
    labels = store.load_array("labels")
    sc = getattr(cfg, "segmentation", None)
    use_seg = (sc is None or getattr(sc, "source", "embedding") == "embedding")
    seg_emb = None
    if use_seg and "seg_embeddings" in store.manifest["entries"]:
        seg_emb = store.load_array("seg_embeddings")
    clustering_space = "segmentation" if seg_emb is not None else "field"
    emb_for_cluster = seg_emb if seg_emb is not None else (
        embeddings if embeddings is not None else store.load_array("embeddings"))

    if sc is not None and getattr(sc, "boundary_independent", False):
        cl = M.cluster_boundary_independent(emb_for_cluster, labels, cfg)
    else:
        cl = M.cluster_documents(emb_for_cluster, cfg)

    # operating-window diagnostics, in the FIELD space where the field is computed
    field_emb = embeddings if embeddings is not None else store.load_array("embeddings")
    diag = M.spanning_fraction(cl["labels"], labels)
    diag["clustering_space"] = clustering_space
    diag["boundary_independent"] = bool(sc is not None and
                                        getattr(sc, "boundary_independent", False))
    diag["shift_to_scatter"] = M.shift_to_scatter(field_emb, cl["labels"], labels)

    # did the partition survive the change of space?
    if seg_emb is not None:
        try:
            from .backends.segmentation import cross_space_coherence
            diag["cross_space_coherence"] = cross_space_coherence(
                seg_emb, field_emb, cl["labels"])
        except Exception as e:
            diag["cross_space_coherence_error"] = str(e)

    fig = V.plot_field_2d(cl["coords_2d"], cl["labels"], labels)
    store.save_array("cluster_labels", cl["labels"])
    store.save_array("coords_2d", cl["coords_2d"])
    store.save_json("segmentation_diagnostics", diag)
    store.save_fig("02_field_2d", fig)
    print(f"  clustered in {clustering_space} space | k={cl['n_clusters']} "
          f"| spanning {diag['n_spanning']}/{diag['n_clusters']}")
    if diag["spanning_fraction"] < 1.0:
        print("  WARNING: spanning fraction < 1 -- topics are being dropped and "
              "mean drift is biased DOWNWARD. Consider "
              "segmentation.boundary_independent = True (Prop. 3).")
    return {"cluster": cl, "diagnostics": diag, "fig": fig}


def step_field(store, cfg, embeddings=None, cluster_labels=None):
    """Reconstruct the polarity field (full space) + probe directions + stats."""
    emb = embeddings if embeddings is not None else store.load_array("embeddings")
    cl = cluster_labels if cluster_labels is not None else store.load_array("cluster_labels")
    labels = store.load_array("labels")
    fld = F.reconstruct_field(emb, cl, labels)
    W = F.probe_directions(emb, cl, labels, fld["clusters"], random_state=cfg.random_state)
    fs = F.field_stats(fld["V"])
    # agreement between diff-in-means and probe directions
    agree = [float(fld["V"][i] / (np.linalg.norm(fld["V"][i]) + 1e-12) @ W[i])
             for i in range(len(fld["clusters"]))] if len(W) else []
    store.save_object("field", fld)
    store.save_array("probe_directions", W)
    store.save_json("field_stats", {"m_bar": fs["m_bar"], "s_bar": fs["s_bar"],
                                    "diffmean_probe_cos": agree,
                                    "clusters": fld["clusters"].tolist(),
                                    "magnitudes": fld["magnitudes"].tolist()})
    return {"field": fld, "probe_directions": W, "stats": fs, "agreement": agree}


# ==================================================================== #
#  V1 axis discovery: activation caching -> per-topic recursive bipartition
#  -> split-half gate -> cross-topic field assembly -> CB baseline comparison.
#
#  Additive to everything above: reads `texts`, `labels` (CB) and
#  `cluster_labels` the same way the existing steps do, but writes to its own
#  object names (`topol_*`) and never overwrites `acts_all_layers`,
#  `embeddings`, or `field` -- the CB path stays bit-identical.
# ==================================================================== #
def _selected_layer(store):
    """The CB-chosen residual layer, read from `layer_selection.json`.

    NOT `_get_state(store)["layer"]`: that key is only populated by calling
    `step_select_layer` itself. `pipeline_2.ipynb`'s Part 2 instead calls
    `layer_select.probe_separability_by_layer` directly and threads the
    chosen layer as a local Python variable (`L`) between cells, the same
    way `02_select_layer.py` and `step_select_layer` do internally --
    `layer_selection.json` is saved by both paths, so it is the one place
    this doesn't matter regardless of which one produced it.
    """
    if "layer_selection" in store.manifest["entries"]:
        return int(store.load_json("layer_selection")["best_layer"])
    return int(_get_state(store)["layer"])


def step_topol_cache_activations(store, cfg, backend, texts=None):
    """Cache the full residual stack (subject to `discovery.layer_stride`),
    excluding BOS/special/chat-template tokens from the pool -- the input
    `topol_axes.py` needs and the CB path's single-layer `embeddings` does not
    provide (see `topol_activations.py`).

    Force-includes the layer `layer_selection.json` records (if that step has
    already run), since `discovery.ref_layer` defaults to it and it need not
    land on the stride grid otherwise.
    """
    texts = texts if texts is not None else store.load_object("texts")
    try:
        extra = [_selected_layer(store)]
    except Exception:
        extra = None
    cache = TA.cache_activations(backend, texts, layer_stride=cfg.discovery.layer_stride,
                                 extra_layers=extra)
    TA.save_activation_cache(store, cache)
    return cache


def step_topol_discovery(store, cfg, cache=None, cluster_labels=None, texts=None,
                         ratings=None, product_ids=None):
    """Per-topic recursive-bipartition discovery + split-half gate (spec
    steps 2-5). Returns {topic_id: gate_result}; also saved to the store.

    `ratings`/`product_ids` are OPTIONAL, full-corpus-length arrays for the
    nuisance diagnostics (recorded, never gated on) -- the current Amazon
    corpus prep does not surface either through `data.load_dataset`'s return
    tuple, so callers that want them must recover them out-of-band (see
    `18_topol_baseline_compare.py`) and pass them in here; both default to
    None, in which case the corresponding diagnostic is NaN with a note.
    """
    cache = cache if cache is not None else TA.load_activation_cache(store)
    cl = cluster_labels if cluster_labels is not None else store.load_array("cluster_labels")
    texts = texts if texts is not None else store.load_object("texts")
    lengths = np.array([len(t) for t in texts])
    layers = cache["layers"]

    results = {}
    for t in sorted(np.unique(cl).tolist()):
        doc_idx = np.where(cl == t)[0]
        if len(doc_idx) < 2 * cfg.discovery.min_cluster_size:
            print(f"  topic {t}: {len(doc_idx)} docs < 2*min_cluster_size, skipped")
            continue
        acts_by_layer = {l: cache["acts"][l][doc_idx].astype(np.float64) for l in layers}
        gate_res = TG.gate_topic(
            acts_by_layer, layers, cfg.discovery, random_state=cfg.random_state,
            lengths=lengths[doc_idx],
            ratings=ratings[doc_idx] if ratings is not None else None,
            product_ids=product_ids[doc_idx] if product_ids is not None else None)
        results[int(t)] = gate_res
        n_pass = sum(1 for c in gate_res["contrasts"] if c["passed"])
        print(f"  topic {t}: {len(doc_idx)} docs | {n_pass}/{len(gate_res['contrasts'])} "
              f"contrasts passed the gate (threshold={cfg.discovery.stability_threshold})")

    store.save_object("topol_gate_results", results)
    return results


def _resolve_ref_layer(ref_layer, cached_layers, store):
    """Fall back to the layer `layer_selection.json` records when `ref_layer`
    is unset; snap to the nearest CACHED layer if that layer (or an explicit
    override) is not actually in the cache -- e.g. it predates a change to
    `discovery.layer_stride`, or an explicit `--ref-layer` was passed that
    was never force-included at caching time (see `step_topol_cache_activations`)."""
    if ref_layer is None:
        ref_layer = _selected_layer(store)
    if ref_layer not in cached_layers:
        nearest = min(cached_layers, key=lambda l: abs(l - ref_layer))
        print(f"  WARNING: ref_layer={ref_layer} is not cached ({cached_layers}); "
              f"snapping to nearest cached layer {nearest}")
        ref_layer = nearest
    return int(ref_layer)


def step_topol_field(store, cfg, gate_results=None, cache=None):
    """Cross-topic field assembly at the fixed reference layer (spec step 7)."""
    gate_results = gate_results if gate_results is not None else store.load_object("topol_gate_results")
    cache = cache if cache is not None else TA.load_activation_cache(store)
    ref_layer = _resolve_ref_layer(cfg.discovery.ref_layer, cache["layers"], store)

    pooled = F.pooled_field_assembly(gate_results, ref_layer, cache["layers"])
    families = F.axis_families(pooled, cos_merge=cfg.discovery.ari_merge)
    per_topic_rank = {int(tid): F.topic_rank_estimate(res) for tid, res in gate_results.items()}

    store.save_json("topol_field_summary", {
        "ref_layer": int(ref_layer), "pr": pooled["pr"],
        "spectrum": np.asarray(pooled["spectrum"]).tolist(),
        "k": pooled["k"], "families": families, "per_topic_rank": per_topic_rank,
    })
    print(f"  pooled field: k={pooled['k']} gate-passing axes across "
          f"{len(gate_results)} topics | PR={pooled['pr']:.3f} @ layer {ref_layer}")
    return {"pooled": pooled, "families": families, "per_topic_rank": per_topic_rank}


def step_topol_baseline_compare(store, cfg, gate_results=None, cache=None, cluster_labels=None,
                                ratings=None, texts=None, n_extreme=10):
    """Baseline comparison (spec step 8): identification, excess structure and
    layer agreement between the discovered axes and the CB estimator v_CB^t --
    PLUS sentiment alignment (spec step 8b, see below).

    IDENTIFICATION alone (max_j |cos(a_j, v_CB^t)|) only tells you whether some
    discovered axis points the same way as v_CB^t; it says nothing about
    whether v_CB^t itself, or the axis it matched, actually tracks sentiment.
    An earlier version of this function ASSUMED the identification match must
    land at the topic's first recursion depth (reasoning: dropping the middle
    band should make sentiment dominate the topic cloud's variance) and
    printed a warning keyed on `found_depth != 1`. That assumption does not
    hold empirically and was never something to gate interpretation on --
    recursion depth is an artifact of how much OTHER structure got deflated
    out first, not a measure of how sentiment-aligned a direction is. It has
    been replaced by a direct, measured test:

      * `cb_corr_rating`     -- Pearson correlation between v_CB^t's own
                                (topic-mean-centred) document projection and
                                the CONTINUOUS star rating (not the two-class
                                label used to build v_CB^t itself, which would
                                be circular) -- does the human-coded CB axis
                                actually track sentiment as a continuum in
                                THIS topic? `ratings=None` (not recovered, or
                                a non-Amazon dataset) reports NaN.
      * per-axis `corr_rating` (already attached to every gate-passing
        contrast by `topol_gate.nuisance_diagnostics`, PROVIDED discovery was
        run with `ratings` threaded through -- see `step_topol_discovery`) --
        the same continuous-rating correlation for every OTHER discovered
        axis in the topic, which is what "how sentiment-aligned are the other
        vectors" actually asks.
      * `most_sentiment_aligned_index` -- which passing axis has the largest
        |corr_rating|, compared against `identification`'s cosine-matched
        axis (`identification_matches_most_sentiment_aligned`): if they
        disagree, the axis that LOOKS like v_CB^t by direction is not the one
        that best explains rating -- informative, not necessarily broken.

    `texts`, if given, additionally builds `topol_report.cb_axis_report` per
    topic (v_CB^t's own highest/lowest-projecting reviews) -- the qualitative
    half of "is the CB axis sentiment-aligned": reading the extremes should
    show obviously negative vs. obviously positive reviews if it is.
    """
    gate_results = gate_results if gate_results is not None else store.load_object("topol_gate_results")
    cache = cache if cache is not None else TA.load_activation_cache(store)
    cl = cluster_labels if cluster_labels is not None else store.load_array("cluster_labels")
    cb_labels = store.load_array("labels")
    layers = cache["layers"]
    ref_layer = _resolve_ref_layer(cfg.discovery.ref_layer, layers, store)
    excess_thr = cfg.discovery.excess_cos_threshold
    sentiment_thr = cfg.discovery.sentiment_corr_threshold

    per_topic = {}
    cb_reports = {}
    for t, res in gate_results.items():
        doc_idx = np.where(cl == t)[0]
        cb_local = cb_labels[doc_idx]
        if len(np.unique(cb_local)) < 2:
            continue
        acts_by_layer = {l: cache["acts"][l][doc_idx].astype(np.float64) for l in layers}
        cb_profile = TAX.depth_profile(cb_local.astype(int), acts_by_layer, layers)
        v_cb_ref = cb_profile["profile"][ref_layer]["v"]
        v_cb_ref_hat = v_cb_ref / (np.linalg.norm(v_cb_ref) + 1e-12)

        X_ref = acts_by_layer[ref_layer]
        proj_cb_ref = (X_ref - X_ref.mean(0)) @ v_cb_ref_hat
        cb_corr_rating = TG.pearsonr(proj_cb_ref, ratings[doc_idx] if ratings is not None else None)

        passing = [c for c in res["contrasts"] if c["passed"]]
        cos_to_cb = []
        for c in passing:
            v = c["profile"][ref_layer]["v"]
            cos_to_cb.append(abs(float((v / (np.linalg.norm(v) + 1e-12)) @ v_cb_ref_hat)))

        best_j = int(np.argmax(cos_to_cb)) if cos_to_cb else None
        excess_j = [j for j, c in enumerate(cos_to_cb) if j != best_j and c < excess_thr]

        abs_corr_rating = [abs(c["corr_rating"]) if c.get("corr_rating") is not None
                           and not np.isnan(c["corr_rating"]) else float("nan") for c in passing]
        most_idx = (int(np.nanargmax(abs_corr_rating))
                   if passing and not np.all(np.isnan(abs_corr_rating)) else None)

        entry = {
            "cb_l_star": cb_profile["l_star"], "cb_effect_size_at_lstar": cb_profile["d_at_lstar"],
            "cb_corr_rating": cb_corr_rating,
            "identification_cos": cos_to_cb[best_j] if best_j is not None else float("nan"),
            "identification_found_depth": passing[best_j]["found_depth"] if best_j is not None else None,
            "identification_l_star": passing[best_j]["l_star"] if best_j is not None else None,
            "identification_corr_rating": passing[best_j].get("corr_rating") if best_j is not None else float("nan"),
            "most_sentiment_aligned_index": most_idx,
            "most_sentiment_aligned_abs_corr_rating": abs_corr_rating[most_idx] if most_idx is not None else float("nan"),
            "most_sentiment_aligned_l_star": passing[most_idx]["l_star"] if most_idx is not None else None,
            "identification_matches_most_sentiment_aligned": (
                (best_j == most_idx) if (best_j is not None and most_idx is not None) else None),
            "excess_count": len(excess_j), "n_passing": len(passing),
            "excess_axis_l_stars": [passing[j]["l_star"] for j in excess_j],
            "passing_axis_l_stars": [c["l_star"] for c in passing],
            "passing_axis_abs_corr_rating": abs_corr_rating,
            "passing_axis_cos_to_cb": cos_to_cb,
        }
        per_topic[int(t)] = entry

        if not np.isnan(cb_corr_rating) and abs(cb_corr_rating) < sentiment_thr:
            print(f"  WARNING topic {t}: v_CB^t's own correlation with the continuous rating is "
                  f"weak (|r|={abs(cb_corr_rating):.3f} < {sentiment_thr}) -- the human-coded CB "
                  f"axis may not itself track sentiment well in this topic.")
        identification_abs_corr = (abs_corr_rating[best_j] if best_j is not None else float("nan"))
        if (best_j is not None and most_idx is not None and best_j != most_idx
                and not np.isnan(identification_abs_corr)
                and abs_corr_rating[most_idx] - identification_abs_corr > 0.1):
            print(f"  NOTE topic {t}: the axis most correlated with rating (contrast "
                  f"{passing[most_idx]['topic_local_index']}, |r|={abs_corr_rating[most_idx]:.3f}) "
                  f"is NOT the axis identified by cosine similarity to v_CB^t (contrast "
                  f"{passing[best_j]['topic_local_index']}, |r|={identification_abs_corr:.3f}).")

        if texts is not None:
            texts_topic = [texts[i] for i in doc_idx]
            cb_reports[int(t)] = TR.cb_axis_report(t, cb_profile, cb_corr_rating, texts_topic,
                                                    doc_idx, acts_by_layer, n_extreme=n_extreme)

    store.save_json("topol_baseline_comparison", per_topic)
    if cb_reports:
        store.save_json("topol_cb_axis_report", cb_reports)
    return per_topic
