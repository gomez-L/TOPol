"""Part 2, step 3/5: closing the loop -- does a polar vector's DIRECTION carry
sentiment information?

Two independent estimates of "how sentiment-aligned is this axis" are
compared against `abs_cos_to_cb` (how similar the axis's own direction is to
the human-coded CB vector, $v_{CB}^t$, the reference ground-truth sentiment
direction for that topic):

  1. `corr_rating` -- ALREADY computed for every gate-passing contrast by
     `topol_gate.nuisance_diagnostics` (Pearson corr of the axis's own
     projection with the continuous Amazon star rating). Reused here, not
     recomputed.
  2. `sentiment_delta` -- NEW: a Hugging Face sentiment classifier
     (`cardiffnlp/twitter-roberta-base-sentiment-latest`) scores every
     document ONCE (continuous polarity = p_pos - p_neg, softmax over the
     3-way neg/neu/pos head), and each axis's `sentiment_delta` is
     mean(polarity | pole B) - mean(polarity | pole A) for that axis's own
     split. Scoring is done once over the whole corpus and then aggregated
     per axis (not once per axis) -- the same per-topic `doc_idx` reuse
     `pole_vocab.py` relies on, since every axis in a topic shares that
     topic's document set and only differs in how it's split into A/B.

The hypothesis under test: axes whose direction is more similar to
$v_{CB}^t$ (larger `abs_cos_to_cb`) should also be more sentiment-aligned
(larger `abs_corr_rating` / `abs_sentiment_delta`) -- i.e. an axis's
direction alone carries information about what it means.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
#  HF sentiment scoring
# ------------------------------------------------------------------ #
def load_sentiment_model(model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
                         cache_dir: str | None = None, device: str | None = None):
    """Load the sentiment tokenizer/model, downloading once if not already
    cached (same `cache_dir` convention the notebook already uses for the
    gemma snapshot: `~/scratch/hf_cache` by default)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cache_dir = cache_dir or os.path.expanduser("~/scratch/hf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, cache_dir=cache_dir).to(device).eval()
    return tokenizer, model, device


def polarity_scalar(tokenizer, model, texts, device, batch_size: int = 32) -> np.ndarray:
    """Continuous polarity per document: softmax(logits) -> p_pos - p_neg.

    Class layout for `cardiffnlp/twitter-roberta-base-sentiment-latest` is
    0=negative, 1=neutral, 2=positive (the user's own formula/snippet),
    batched here for speed over a several-thousand-document corpus rather
    than one document at a time.
    """
    import torch

    scores = np.empty(len(texts), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = [str(t) for t in texts[i:i + batch_size]]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=128).to(device)
            logits = model(**enc).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            scores[i:i + batch_size] = (probs[:, 2] - probs[:, 0]).cpu().numpy()
    return scores


# ------------------------------------------------------------------ #
#  Per-axis aggregation
# ------------------------------------------------------------------ #
def axis_sentiment_delta(doc_sentiment: np.ndarray, cluster_labels, gate_results: dict,
                         axis_index: pd.DataFrame) -> pd.DataFrame:
    """Per axis: mean(sentiment | pole B) - mean(sentiment | pole A), reusing
    the same per-topic `doc_idx` + `contrast["labels"]` alignment as
    `pole_vocab.build_pv_tfidf_table` (see that module's docstring for why
    this reuse is valid: every axis of a topic shares the same `doc_idx`)."""
    cluster_labels = np.asarray(cluster_labels)
    rows = []
    for topic in sorted(gate_results.keys()):
        doc_idx = np.where(cluster_labels == topic)[0]
        topic_axes = axis_index[axis_index["topic"] == topic]
        if len(doc_idx) == 0 or len(topic_axes) == 0:
            continue
        sent_topic = doc_sentiment[doc_idx]
        res = gate_results[topic]
        for _, row in topic_axes.iterrows():
            axis_id, ci = int(row["axis_id"]), int(row["contrast_index"])
            pole = np.asarray(res["contrasts"][ci]["labels"])
            maskA, maskB = pole == 0, pole == 1
            if maskA.sum() == 0 or maskB.sum() == 0:
                continue
            delta = float(sent_topic[maskB].mean() - sent_topic[maskA].mean())
            rows.append({"axis_id": axis_id, "topic": topic, "sentiment_delta": delta})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
#  Direction-vs-CB alignment
# ------------------------------------------------------------------ #
def cb_direction_alignment(gate_results: dict, cache: dict, cluster_labels, cb_labels,
                           ref_layer: int, axis_index: pd.DataFrame) -> pd.DataFrame:
    """Per axis: |cos(axis vector at ref_layer, v_CB^t at ref_layer)|, plus
    each axis's own `corr_rating` (already attached by
    `topol_gate.nuisance_diagnostics`). $v_{CB}^t$ is recomputed exactly as
    `pipeline.step_topol_baseline_compare` does it, via
    `topol_axes.depth_profile` on the CB regime labels -- reused, not
    reinvented, so the comparison is apples-to-apples at the same layer.
    """
    from . import topol_axes as TAX

    cluster_labels = np.asarray(cluster_labels)
    cb_labels = np.asarray(cb_labels)
    layers = cache["layers"]

    rows = []
    for topic in sorted(gate_results.keys()):
        doc_idx = np.where(cluster_labels == topic)[0]
        cb_local = cb_labels[doc_idx]
        if len(np.unique(cb_local)) < 2:
            continue
        acts_by_layer = {l: np.asarray(cache["acts"][l], dtype=np.float64)[doc_idx] for l in layers}
        cb_profile = TAX.depth_profile(cb_local.astype(int), acts_by_layer, layers)
        v_cb_ref = cb_profile["profile"][ref_layer]["v"]
        v_cb_ref_hat = v_cb_ref / (np.linalg.norm(v_cb_ref) + 1e-12)

        res = gate_results[topic]
        topic_axes = axis_index[axis_index["topic"] == topic]
        for _, row in topic_axes.iterrows():
            axis_id, ci = int(row["axis_id"]), int(row["contrast_index"])
            contrast = res["contrasts"][ci]
            v = contrast["profile"][ref_layer]["v"]
            cos_to_cb = float((v / (np.linalg.norm(v) + 1e-12)) @ v_cb_ref_hat)
            corr_rating = contrast.get("corr_rating")
            rows.append({
                "axis_id": axis_id, "topic": topic, "contrast_index": ci,
                "cos_to_cb": cos_to_cb, "abs_cos_to_cb": abs(cos_to_cb),
                "corr_rating": corr_rating,
                "abs_corr_rating": abs(corr_rating) if corr_rating is not None
                and not np.isnan(corr_rating) else np.nan,
            })
    return pd.DataFrame(rows)


def build_alignment_table(gate_results, cache, cluster_labels, cb_labels, ref_layer,
                          axis_index, doc_sentiment) -> pd.DataFrame:
    """Full per-axis table: cos-to-CB, corr_rating (existing), sentiment_delta
    (new) -- one row per gate-passing axis, ready for
    `tables/topol_pv_sentiment_alignment.csv` and `test_alignment_hypothesis`."""
    align_df = cb_direction_alignment(gate_results, cache, cluster_labels, cb_labels,
                                      ref_layer, axis_index)
    delta_df = axis_sentiment_delta(doc_sentiment, cluster_labels, gate_results, axis_index)
    out = align_df.merge(delta_df[["axis_id", "sentiment_delta"]], on="axis_id", how="left")
    out["abs_sentiment_delta"] = out["sentiment_delta"].abs()
    return out.merge(
        axis_index[["axis_id", "l_star", "layer_id", "effect_size_at_lstar", "n_docs"]],
        on="axis_id", how="left",
    ).sort_values(["topic", "axis_id"]).reset_index(drop=True)


# ------------------------------------------------------------------ #
#  The hypothesis test itself
# ------------------------------------------------------------------ #
def test_alignment_hypothesis(df: pd.DataFrame) -> dict:
    """Pearson + Spearman (r, p) between `abs_cos_to_cb` and each
    sentiment-alignment proxy, pooled across every gate-passing axis. This is
    the direct, quantitative test of "an axis's direction similarity to
    v_CB^t predicts how sentiment-aligned it is" -- reported honestly either
    way, not just when it confirms the hypothesis.
    """
    from scipy.stats import pearsonr, spearmanr

    out = {}
    for metric in ("abs_corr_rating", "abs_sentiment_delta"):
        sub = df.dropna(subset=["abs_cos_to_cb", metric])
        if len(sub) < 3:
            out[metric] = {"n": len(sub), "pearson_r": np.nan, "pearson_p": np.nan,
                           "spearman_r": np.nan, "spearman_p": np.nan}
            continue
        pr, pp = pearsonr(sub["abs_cos_to_cb"], sub[metric])
        sr, sp = spearmanr(sub["abs_cos_to_cb"], sub[metric])
        out[metric] = {"n": len(sub), "pearson_r": float(pr), "pearson_p": float(pp),
                       "spearman_r": float(sr), "spearman_p": float(sp)}
    return out
