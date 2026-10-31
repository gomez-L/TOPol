"""Assembles the final per-axis report (spec step 9).

For every gate-passing axis: topic, l*, the layers that found it, its effect
size at l*, its split-half |cos|, the three nuisance diagnostics, and the 10
highest- and 10 lowest-projecting review texts on each pole. The text extremes
are the V1 interpretation route -- no SAE, no LLM narration, just the
documents: if an axis is real and semantic, reading twenty reviews shows it.

Also logs every REJECTED split-half contrast with the value that rejected it
(`gate_topic`'s `rejected` list), so the stability threshold can be revisited
against the full empirical distribution rather than just the pass/fail count.
"""
from __future__ import annotations

import numpy as np


def _projection_extremes(proj, doc_idx_topic, texts_topic, n_extreme: int = 10) -> dict:
    """The `n_extreme` highest- and lowest-projecting (global_index, projection,
    text) records for one already-computed document projection -- shared by
    `axis_report` (a discovered axis) and `cb_axis_report` (the CB axis)."""
    order = np.argsort(proj)
    lowest = order[:n_extreme]
    highest = order[::-1][:n_extreme]

    def _side(idxs):
        return [{"global_index": int(doc_idx_topic[i]), "projection": float(proj[i]),
                "text": texts_topic[i]} for i in idxs]

    return {"pole_high_texts": _side(highest), "pole_low_texts": _side(lowest)}


def axis_report(topic_id, contrast_index: int, contrast: dict, texts_topic: list[str],
                doc_idx_topic, acts_by_layer: dict, n_extreme: int = 10) -> dict:
    """One reporting record for a single gate-passing axis.

    texts_topic   : this topic's document texts, in the SAME local order as
                    `doc_idx_topic` and `acts_by_layer`.
    doc_idx_topic : (n_t,) GLOBAL document indices for that same local order.
    acts_by_layer : {layer: (n_t, d)} RAW activations for this topic.
    """
    l = contrast["l_star"]
    X = np.asarray(acts_by_layer[l], dtype=np.float64)
    vhat = contrast["v_at_lstar"]
    vhat = vhat / (np.linalg.norm(vhat) + 1e-12)
    proj = (X - X.mean(0)) @ vhat
    extremes = _projection_extremes(proj, doc_idx_topic, texts_topic, n_extreme)

    return {
        "topic": int(topic_id), "contrast_index": int(contrast_index),
        "l_star": int(l), "layers_found": contrast["layers_found"],
        "found_layer": contrast["found_layer"], "found_depth": contrast["found_depth"],
        "effect_size_at_lstar": contrast["effect_size_at_lstar"],
        "split_half_cos": contrast["split_half_cos"],
        "corr_length": contrast.get("corr_length"),
        "corr_rating": contrast.get("corr_rating"),
        "r2_product_id": contrast.get("r2_product_id"),
        "r2_product_id_note": contrast.get("r2_product_id_note"),
        **extremes,
    }


def cb_axis_report(topic_id, cb_profile: dict, corr_rating: float, texts_topic: list[str],
                   doc_idx_topic, acts_by_layer: dict, n_extreme: int = 10) -> dict:
    """The `axis_report` analogue for v_CB^t itself: its own depth-profile l*,
    its correlation with the continuous rating, and its highest/lowest
    -projecting reviews -- the qualitative half of "is the human-coded CB
    axis actually sentiment-aligned" (`pipeline.step_topol_baseline_compare`'s
    quantitative half is `corr_rating` alone; reading the extremes is the same
    check this module's docstring already argues for: "if an axis is real and
    semantic, reading twenty reviews should show it").

    `cb_profile` : `topol_axes.depth_profile`'s return, run on the CB regime
    labels (see `pipeline.step_topol_baseline_compare`).
    """
    l = cb_profile["l_star"]
    X = np.asarray(acts_by_layer[l], dtype=np.float64)
    vhat = cb_profile["v_at_lstar"]
    vhat = vhat / (np.linalg.norm(vhat) + 1e-12)
    proj = (X - X.mean(0)) @ vhat
    extremes = _projection_extremes(proj, doc_idx_topic, texts_topic, n_extreme)

    return {
        "topic": int(topic_id), "l_star": int(l),
        "effect_size_at_lstar": cb_profile["d_at_lstar"],
        "corr_rating": corr_rating,
        **extremes,
    }


def build_report(gate_results: dict, texts: list[str], cluster_labels, cache: dict,
                 n_extreme: int = 10) -> dict:
    """Full report across all topics: every gate-passing contrast gets an
    `axis_report`; every rejected split-half contrast is logged as-is."""
    layers = cache["layers"]
    cluster_labels = np.asarray(cluster_labels)
    reports, rejected_log = [], []
    for t, res in gate_results.items():
        doc_idx = np.where(cluster_labels == t)[0]
        texts_topic = [texts[i] for i in doc_idx]
        acts_by_layer = {l: cache["acts"][l][doc_idx].astype(np.float64) for l in layers}
        for ci, c in enumerate(res["contrasts"]):
            if not c["passed"]:
                continue
            reports.append(axis_report(t, ci, c, texts_topic, doc_idx, acts_by_layer, n_extreme))
        for r in res["rejected"]:
            rejected_log.append({"topic": int(t), **r})
    return {"axes": reports, "rejected": rejected_log}


def save_report(store, report: dict, name: str = "topol_report") -> None:
    """Full text-bearing report as JSON, plus two flat CSVs (summary stats,
    rejected contrasts) for quick scanning without loading all the text."""
    import pandas as pd

    store.save_json(f"{name}_full", report)

    rows = [{
        "topic": a["topic"], "contrast_index": a["contrast_index"],
        "l_star": a["l_star"], "found_layer": a["found_layer"], "found_depth": a["found_depth"],
        "layers_found": ",".join(map(str, a["layers_found"])),
        "effect_size_at_lstar": a["effect_size_at_lstar"], "split_half_cos": a["split_half_cos"],
        "corr_length": a["corr_length"], "corr_rating": a["corr_rating"],
        "r2_product_id": a["r2_product_id"],
    } for a in report["axes"]]
    store.save_table(f"{name}_summary", pd.DataFrame(rows))

    rej_rows = [{"topic": r["topic"], "source": r.get("source"),
                "half1_index": r.get("half1_index"), "half2_index": r.get("half2_index"),
                "cos": r.get("cos")} for r in report["rejected"]]
    store.save_table(f"{name}_rejected", pd.DataFrame(rej_rows))
