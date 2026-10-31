"""Semantic (meaning-based) family clustering: does axis i "mean" something
similar to axis j?

`field.axis_family_clustering` answers a geometric question -- do two axes'
raw activation-space directions point the same way? `depth_profile_family_
clustering` (removed; see `field.py`'s module docstring) tried to answer a
depth-behaviour question and produced a degenerate, uninformative result
(543 singleton families, chance-level agreement with the direction-based
clustering) because a layer-by-layer effect-size CURVE is a poor proxy for
what a direction is actually about.

This module answers a third, different question directly: do two axes
contrast SIMILAR VOCABULARY, regardless of whether their raw directions are
geometrically aligned? Two axes can point in unrelated directions in the
2,560-dimensional residual stream yet both be "government/politics vs.
religion" splits (in different topics) -- or point in similar directions
for reasons that have nothing to do with sharing a topic-level contrast
(e.g. both happen to load on generic sentiment intensifiers). Clustering by
shared distinctive vocabulary is a direct, human-checkable operationalisation
of "these axes mean the same thing" that does not require trusting a
generative language model's output (contrast `pole_llm.py`, whose base-model
completions are frequently generic -- see that module's docstring) and does
not depend on which reference layer the raw direction happens to be compared
at (contrast the pooled, fixed-layer participation-ratio view in `field.py`).

Method: every axis's differential TF-IDF vocabulary (`pole_vocab.py`, already
computed for all gate-passing axes) is turned into one SIGNED vector over the
corpus-wide term vocabulary -- positive weight for a term distinctive of that
axis's pole B, negative for pole A, magnitude the TF-IDF mean-difference score
`pole_vocab.axis_differential_terms` already ranks by. Two axes are compared
by the cosine similarity of these signed vectors; because which pole is
labelled "A" and which "B" is an arbitrary bookkeeping choice (see
`topol_axes.fix_sign`), the comparison -- like every other sign-sensitive
comparison in this pipeline -- uses the ABSOLUTE cosine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Corpus-preprocessing placeholders (numbers/dates redacted to NUM/DATE
# tokens upstream), not semantic content -- excluded so two axes that both
# merely contain digits/dates don't register as "similar meaning".
DEFAULT_STOPTERMS = {"num", "num num", "date"}


def build_term_vectors(tfidf_long: pd.DataFrame, stopterms=None):
    """One signed, L2-normalised vector per axis over the shared term
    vocabulary (`pole_vocab.build_pv_tfidf_table`'s long-format output).

    Returns (M, axis_ids, terms): M is (n_axes, n_terms), row i is axis
    `axis_ids[i]`'s signed term vector (already L2-normalised, so a plain
    dot product below IS the cosine similarity).
    """
    stop = DEFAULT_STOPTERMS if stopterms is None else set(stopterms)
    df = tfidf_long[~tfidf_long["term"].isin(stop)]

    terms = sorted(df["term"].unique())
    term_idx = {w: i for i, w in enumerate(terms)}
    axis_ids = sorted(df["axis_id"].unique())
    axis_pos = {a: i for i, a in enumerate(axis_ids)}

    M = np.zeros((len(axis_ids), len(terms)), dtype=np.float64)
    for row in df.itertuples(index=False):
        sign = 1.0 if row.pole == "B" else -1.0
        M[axis_pos[row.axis_id], term_idx[row.term]] += sign * row.score

    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    M = M / norms
    return M, np.asarray(axis_ids), terms


def semantic_similarity_matrix(M: np.ndarray) -> np.ndarray:
    """(n_axes, n_axes) absolute cosine similarity -- sign-invariant, since
    pole A/B labelling is an arbitrary bookkeeping choice (see module
    docstring)."""
    S = np.abs(M @ M.T)
    np.fill_diagonal(S, 1.0)
    return S


def semantic_family_clustering(M: np.ndarray, merge_threshold: float = 0.3) -> np.ndarray:
    """Average-linkage clustering on 1 - |cos| distance, same machinery as
    `field.axis_family_clustering` (so the two are directly comparable), but
    over MEANING vectors instead of raw activation-space directions.

    `merge_threshold` defaults to 0.3, chosen from the empirical distribution
    of pairwise |cos| over this run's 543 axes (median 0.06, 95th percentile
    0.29, 99th percentile 0.50): 0.3 sits at roughly the 95th-96th percentile,
    i.e. a family merge requires vocabulary overlap achieved by at most ~1
    axis pair in 20 by chance under this corpus's own term-score distribution
    -- selective, not a threshold copied from the (very different, dense,
    2560-dimensional) geometric clustering.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    n = M.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int)
    S = np.abs(M @ M.T)
    D = np.clip(1.0 - S, 0.0, None)
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    return fcluster(Z, t=(1.0 - merge_threshold), criterion="distance")


def label_family(M: np.ndarray, terms: list[str], member_rows: np.ndarray, top_n: int = 8):
    """Auto-generate a human-readable (pole-A-terms, pole-B-terms) label for
    a family: average its members' signed term vectors and take the top-`n`
    most positive (pole-B-leaning) and most negative (pole-A-leaning) terms.
    Sign here is arbitrary per family (inherited from whichever axis anchors
    the average), so the two term lists should be read as "the family's two
    poles", not as a claim about which one is really "A"."""
    if len(member_rows) == 0:
        return [], []
    mean_vec = M[member_rows].mean(axis=0)
    order = np.argsort(mean_vec)
    neg_terms = [terms[i] for i in order[:top_n] if mean_vec[i] < 0]
    pos_terms = [terms[i] for i in order[::-1][:top_n] if mean_vec[i] > 0]
    return neg_terms, pos_terms


def semantic_families_table(M, axis_ids, terms, axis_index: pd.DataFrame,
                            merge_threshold: float = 0.3, top_n: int = 8) -> pd.DataFrame:
    """One row per semantic family: size, member axes, topics spanned, and
    an auto-generated two-pole vocabulary label."""
    labels = semantic_family_clustering(M, merge_threshold=merge_threshold)
    idx_lookup = axis_index.set_index("axis_id")

    rows = []
    for fid in np.unique(labels):
        member_rows = np.where(labels == fid)[0]
        member_axis_ids = axis_ids[member_rows]
        topics = sorted({int(idx_lookup.loc[a, "topic"]) for a in member_axis_ids
                         if a in idx_lookup.index})
        neg_terms, pos_terms = label_family(M, terms, member_rows, top_n=top_n)
        rows.append({
            "family_id": int(fid), "n_axes": len(member_axis_ids),
            "n_topics": len(topics), "topics": ",".join(map(str, topics)),
            "axis_ids": ",".join(map(str, member_axis_ids.tolist())),
            "pole_neg_terms": ", ".join(neg_terms), "pole_pos_terms": ", ".join(pos_terms),
        })
    return pd.DataFrame(rows).sort_values(["n_topics", "n_axes"], ascending=False).reset_index(drop=True)


def agreement_with_geometric(semantic_labels, geometric_labels) -> float:
    """Adjusted Rand index between this semantic (vocabulary) clustering and
    the existing geometric (raw activation-direction cosine) clustering from
    `field.axis_family_clustering` -- are "similar meaning" and "similar
    direction" the same partition of the 543 axes, a different one, or
    somewhere in between?"""
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(semantic_labels, geometric_labels))
