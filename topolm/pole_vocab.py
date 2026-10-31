"""Part 2, step 0-1: what vocabulary actually separates pole A from pole B.

Two things live here, both keyed off `nla.prepare_axis_dataset`'s `data`/
`axis_index` (543 axes across ~14 topics, each axis a gate-passing (topic,
contrast) pair with its OWN A/B partition of that topic's documents):

* `build_pv_full_text_table` -- materialize EVERY document behind every axis
  as its own annotated CSV (`topol_pv_full_texts`). `topol_nla_pole_prototypes`
  only keeps 3 nearest-neighbour snippets per (axis, pole); this keeps all of
  them, for manual reading/auditing independent of anything downstream.

* `build_pv_tfidf_table` -- per TOPIC (not per axis, not over the whole
  corpus), fit ONE `TfidfVectorizer` on that topic's own documents, then reuse
  it for every gate-passing axis belonging to that topic, each time only
  re-splitting rows by THAT axis's own pole labels. This works because
  `prepare_axis_dataset` always visits a topic's documents in the same
  `np.where(cluster_labels == t)[0]` order for every axis in that topic -- only
  `pole_labels` differs axis to axis, so the same fitted TF-IDF matrix can be
  re-sliced per axis instead of being refit 543 times (or once globally, which
  would let a topic's own generic vocabulary --the topic's frequent words--
  swamp what's actually distinctive between poles A and B).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
#  Step 0: full per-axis, per-document text export
# ------------------------------------------------------------------ #
def build_pv_full_text_table(data: dict, axis_index: pd.DataFrame) -> pd.DataFrame:
    """One row per (axis, document): every text behind every polar vector,
    on whichever pole it fell on for THAT axis's own discovered split.

    `data` is `nla.prepare_axis_dataset`'s first return value (`texts`,
    `topic_ids` == axis_id, `pole_labels`); `axis_index` is its second
    return value, the axis_id -> (topic, contrast_index, l_star, layer_id)
    lookup.
    """
    axis_ids = np.asarray(data["topic_ids"])
    pole_labels = np.asarray(data["pole_labels"])
    texts = data["texts"]

    df = pd.DataFrame({
        "axis_id": axis_ids,
        "pole": np.where(pole_labels == 0, "A", "B"),
        "text": texts,
        "n_words": [len(str(t).split()) for t in texts],
    })
    # doc_index_in_topic: this row's position among its OWN axis's rows,
    # i.e. the document's index within that topic's doc_idx array -- stable
    # per axis since every axis of a topic iterates the same doc_idx order.
    df["doc_index_in_topic"] = df.groupby("axis_id").cumcount()

    out = df.merge(
        axis_index[["axis_id", "topic", "contrast_index", "l_star", "layer_id"]],
        on="axis_id", how="left",
    )
    cols = ["axis_id", "topic", "contrast_index", "l_star", "layer_id",
            "pole", "doc_index_in_topic", "text", "n_words"]
    return out[cols].sort_values(["axis_id", "pole", "doc_index_in_topic"]).reset_index(drop=True)


# ------------------------------------------------------------------ #
#  Step 1: per-topic differential TF-IDF vocabulary
# ------------------------------------------------------------------ #
def fit_topic_tfidf(topic_texts, max_features: int = 3000, ngram_range=(1, 2)):
    """Fit one TfidfVectorizer on a single topic's own documents.

    English stopwords removed and a floor of `min_df=2` -- a topic's
    documents are the whole corpus this vectorizer will ever see, so a
    hapax legomenon gets IDF-inflated to "maximally distinctive" for no
    real reason; `min_df=2` keeps only words that recur within the topic.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range,
                          stop_words="english", min_df=2, lowercase=True)
    X = vec.fit_transform(topic_texts)
    return vec, X


def axis_differential_terms(X, terms, pole_labels_for_topic_docs, top_n: int = 20):
    """Given a topic's fitted TF-IDF matrix `X` (rows aligned with
    `pole_labels_for_topic_docs`, one axis's own A/B split of those SAME
    rows), rank terms by mean-TF-IDF difference between the two poles.

    Returns (terms_distinctive_of_A, terms_distinctive_of_B), each a list of
    `top_n` (term, score) pairs, score = the positive side of (meanA - meanB)
    or (meanB - meanA) respectively -- so a term only appears on the side it
    is actually MORE characteristic of, not just "present".
    """
    pole = np.asarray(pole_labels_for_topic_docs)
    maskA, maskB = pole == 0, pole == 1
    if maskA.sum() == 0 or maskB.sum() == 0:
        return [], []
    meanA = np.asarray(X[maskA].mean(axis=0)).ravel()
    meanB = np.asarray(X[maskB].mean(axis=0)).ravel()
    diff = meanB - meanA   # positive => distinctive of B, negative => distinctive of A

    order_B = np.argsort(-diff)[:top_n]
    order_A = np.argsort(diff)[:top_n]
    terms_B = [(terms[i], float(diff[i])) for i in order_B if diff[i] > 0]
    terms_A = [(terms[i], float(-diff[i])) for i in order_A if diff[i] < 0]
    return terms_A, terms_B


def build_pv_tfidf_table(gate_results: dict, texts, cluster_labels, axis_index: pd.DataFrame,
                         top_n: int = 20, max_features: int = 3000):
    """Loop topics once (fit), then every gate-passing axis in that topic
    (reuse fit). Returns (long_df, summary_df):

      long_df    one row per (axis_id, pole, rank, term, score) -- for
                 wordclouds and any further programmatic use.
      summary_df one row per axis_id with `top_terms_A`/`top_terms_B` as
                 comma-joined strings -- a quick-scan table.
    """
    cluster_labels = np.asarray(cluster_labels)
    texts = np.asarray(texts, dtype=object)

    long_rows, summary_rows = [], []
    for topic in sorted(gate_results.keys()):
        doc_idx = np.where(cluster_labels == topic)[0]
        topic_axes = axis_index[axis_index["topic"] == topic]
        if len(doc_idx) == 0 or len(topic_axes) == 0:
            continue

        vec, X = fit_topic_tfidf(texts[doc_idx].tolist(), max_features=max_features)
        terms = np.asarray(vec.get_feature_names_out())

        res = gate_results[topic]
        for _, row in topic_axes.iterrows():
            axis_id, ci = int(row["axis_id"]), int(row["contrast_index"])
            contrast = res["contrasts"][ci]
            pole_labels = np.asarray(contrast["labels"])   # aligned with doc_idx, same order
            terms_A, terms_B = axis_differential_terms(X, terms, pole_labels, top_n=top_n)

            for rank, (term, score) in enumerate(terms_A):
                long_rows.append({"axis_id": axis_id, "topic": topic, "pole": "A",
                                  "rank": rank, "term": term, "score": score})
            for rank, (term, score) in enumerate(terms_B):
                long_rows.append({"axis_id": axis_id, "topic": topic, "pole": "B",
                                  "rank": rank, "term": term, "score": score})
            summary_rows.append({
                "axis_id": axis_id, "topic": topic, "contrast_index": ci,
                "top_terms_A": ", ".join(t for t, _ in terms_A),
                "top_terms_B": ", ".join(t for t, _ in terms_B),
            })

    long_df = pd.DataFrame(long_rows)
    summary_df = pd.DataFrame(summary_rows).merge(
        axis_index[["axis_id", "l_star", "layer_id", "effect_size_at_lstar", "n_docs"]],
        on="axis_id", how="left",
    ).sort_values(["topic", "axis_id"]).reset_index(drop=True)
    return long_df, summary_df


def term_frequencies_for_wordcloud(long_df: pd.DataFrame, axis_id: int, pole: str) -> dict:
    """{term: score} for one (axis_id, pole), ready for
    `WordCloud.generate_from_frequencies` / `viz.plot_pole_wordclouds`."""
    sub = long_df[(long_df["axis_id"] == axis_id) & (long_df["pole"] == pole)]
    return dict(zip(sub["term"], sub["score"]))
