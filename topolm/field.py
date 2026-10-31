"""Polarity field reconstruction (in the FULL representation space).

Preserves the original TOPol estimator - per-topic centroid difference
    v_i = mu_i^B - mu_i^A
computed on the full activation vectors (never on 2-D UMAP coords) - and adds a
regularised logistic-probe direction per topic as a principled alternative
(diff-in-means is robust to spurious features; the probe is optimal but can overfit;
we report both and their agreement).

Also preserved: the drift statistics (mean magnitude m_bar, mean pairwise cosine
s_bar) and the random-CB label-permutation test on magnitude - the *existence*
null (structure beyond label shuffling).  What that null does and does NOT license
is made explicit in `existence_test`'s docstring.
"""
from __future__ import annotations

import numpy as np
from itertools import combinations


# ------------------------------------------------------------------ #
#  Centroids and polarity vectors
# ------------------------------------------------------------------ #
def _centroids(embeddings, cluster_labels, cb_labels):
    """Return per-cluster centroids for regimes A(0) and B(1) plus bookkeeping."""
    embeddings = np.asarray(embeddings)
    clusters = np.unique(cluster_labels)
    muA, muB, kept, nA, nB = {}, {}, [], {}, {}
    for c in clusters:
        inA = (cluster_labels == c) & (cb_labels == 0)
        inB = (cluster_labels == c) & (cb_labels == 1)
        if inA.sum() == 0 or inB.sum() == 0:
            continue          # a topic must be populated on both sides of the CB
        muA[c] = embeddings[inA].mean(0)
        muB[c] = embeddings[inB].mean(0)
        nA[c], nB[c] = int(inA.sum()), int(inB.sum())
        kept.append(c)
    return muA, muB, kept, nA, nB


def reconstruct_field(embeddings, cluster_labels, cb_labels):
    """Build the polarity field.  Returns a dict of arrays keyed by cluster order.

    keys:
      clusters  : (k,) cluster ids kept
      V         : (k, d) polarity vectors v_i = mu_B - mu_A  (difference-in-means)
      muA, muB  : (k, d) centroids
      magnitudes: (k,)  ||v_i||
      nA, nB    : (k,)  support sizes
    """
    muA, muB, kept, nA, nB = _centroids(embeddings, cluster_labels, cb_labels)
    kept = sorted(kept)
    MA = np.stack([muA[c] for c in kept]) if kept else np.zeros((0, embeddings.shape[1]))
    MB = np.stack([muB[c] for c in kept]) if kept else np.zeros((0, embeddings.shape[1]))
    V = MB - MA
    return {
        "clusters": np.array(kept),
        "V": V,
        "muA": MA, "muB": MB,
        "magnitudes": np.linalg.norm(V, axis=1),
        "nA": np.array([nA[c] for c in kept]),
        "nB": np.array([nB[c] for c in kept]),
    }


def probe_directions(embeddings, cluster_labels, cb_labels, clusters, C=1.0, random_state=42):
    """Per-topic L2-logistic-probe polarity direction (unit-norm weight vector).

    Returned in the same cluster order as `reconstruct_field`.  Also returns the
    cosine between each probe direction and the difference-in-means vector, a
    diagnostic of how much the two estimators agree.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    embeddings = np.asarray(embeddings)
    W = []
    for c in clusters:
        mask = cluster_labels == c
        X, y = embeddings[mask], cb_labels[mask].astype(int)
        if len(np.unique(y)) < 2:
            W.append(np.zeros(embeddings.shape[1]))
            continue
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(X), y)
        # map weights back to raw space (divide by per-feature std)
        w = clf.coef_.ravel() / sc.scale_
        W.append(w / (np.linalg.norm(w) + 1e-12))
    W = np.stack(W) if W else np.zeros((0, embeddings.shape[1]))
    return W


# ------------------------------------------------------------------ #
#  Field statistics (as in the original repo, in full space)
# ------------------------------------------------------------------ #
def field_stats(V):
    """Mean drift magnitude m_bar and mean pairwise cosine similarity s_bar."""
    V = np.asarray(V)
    if V.shape[0] < 2:
        return {"m_bar": float(np.linalg.norm(V, axis=1).mean()) if len(V) else 0.0,
                "s_bar": np.nan, "pairwise_cos": np.array([])}
    mags = np.linalg.norm(V, axis=1)
    m_bar = float(mags.mean())
    Vn = V / (mags[:, None] + 1e-12)
    cos = []
    for i, j in combinations(range(len(V)), 2):
        cos.append(float(Vn[i] @ Vn[j]))
    cos = np.array(cos)
    return {"m_bar": m_bar, "s_bar": float(cos.mean()), "pairwise_cos": cos}


# ------------------------------------------------------------------ #
#  Existence test (preserved)
# ------------------------------------------------------------------ #
def existence_test(embeddings, cluster_labels, cb_labels, n_permutations=1000, random_state=42):
    """Random-CB permutation test on mean drift magnitude.

    What it licenses:
        Rejecting this null means the observed centroid separation is larger than
        under random regime assignment that PRESERVES topic composition -> the
        field reflects real structure tied to the boundary, not sampling noise.

    What it does NOT license:
        It says nothing about whether that structure is the *semantic* dimension
        of interest rather than a nuisance covariate (length / author / time).
        That question is answered by the causal steering test (`steering.py`), not
        here.  It is also NOT the dimensionality test - see `dimensionality.py`.

    Returns observed m_bar/s_bar, the null distributions, and the magnitude p-value
    p = (1 + #{m_perm >= m_obs}) / (N + 1).
    """
    rng = np.random.default_rng(random_state)
    obs = field_stats(reconstruct_field(embeddings, cluster_labels, cb_labels)["V"])
    m_obs, s_obs = obs["m_bar"], obs["s_bar"]

    m_null, s_null = [], []
    y = np.asarray(cb_labels)
    for _ in range(n_permutations):
        yp = rng.permutation(y)
        f = reconstruct_field(embeddings, cluster_labels, yp)
        st = field_stats(f["V"])
        m_null.append(st["m_bar"])
        s_null.append(st["s_bar"])
    m_null = np.array(m_null)
    s_null = np.array(s_null, dtype=float)
    p_mag = float((1 + np.sum(m_null >= m_obs)) / (n_permutations + 1))
    return {
        "m_obs": m_obs, "s_obs": s_obs,
        "m_null": m_null, "s_null": s_null,
        "p_magnitude": p_mag,
        "m_null_mean": float(m_null.mean()),
        "s_null_mean": float(np.nanmean(s_null)),
    }


# ==================================================================== #
#  V1 axis-discovery field assembly (topol_axes.py / topol_gate.py)
#
#  Additive to everything above -- the CB estimator (`reconstruct_field`,
#  `probe_directions`, `field_stats`, `existence_test`) is untouched by this
#  section and remains bit-identical to prior runs.
#
#  Two DIFFERENT uncentred-second-moment computations coexist in this
#  codebase and must not be confused:
#    * `dimensionality.participation_ratio` on the CB ensemble (rows = one
#      vector PER TOPIC) is uncentred because a 1-D field is exactly the case
#      where every topic shifts the SAME way -- centring would erase that
#      shared direction.
#    * `pooled_field_assembly` below pools DISCOVERED AXES across topics (rows
#      = one vector per gate-passing axis, several axes may come from the
#      same topic) and is ALSO uncentred, but for the reason stated in the
#      spec this implements: PR is meaningful here because axes from
#      different topics are not constructed orthogonal to one another (unlike
#      `topol_axes.recursive_bipartition`'s within-topic deflation, which
#      forces orthogonality by construction -- see `topic_rank_estimate`).
#  Same convention (`center=False`), different justification each time; do
#  not let one invariant silently stand in for the other.
# ==================================================================== #
def topic_rank_estimate(gate_result: dict) -> dict:
    """r_t = number of gate-passing contrasts for one topic.

    NOT a participation ratio: `topol_axes.recursive_bipartition`'s deflation
    sequence makes a topic's discovered axes orthogonal to each other by
    construction, so a PR computed within one topic would just recover this
    same count with extra arithmetic. Report r_t directly.
    """
    r_t = sum(1 for c in gate_result["contrasts"] if c["passed"])
    return {
        "r_t": r_t,
        "n_candidates": len(gate_result["contrasts"]),
        "note": "rank estimate, not PR -- axes are orthogonal by construction within a topic",
    }


def pooled_field_assembly(all_topic_results: dict, ref_layer: int, layers: list[int]) -> dict:
    """Pool every gate-PASSING axis, evaluated at the fixed reference layer
    `ref_layer`, across ALL topics into one ensemble, and compute its
    uncentred second moment (participation ratio + spectrum).

    `all_topic_results` : {topic_id: gate_result} from `topol_gate.gate_topic`.
    `ref_layer` should default to the layer `layer_select.py` chose, so PR here
    is comparable to prior (CB-only) runs' operating layer.

    Also carries each pooled axis's full depth profile (`d(l)` at every
    cached layer) for `axis_families`' depth-profile clustering below.
    """
    from .dimensionality import participation_ratio, spectrum   # local: avoids a
                                                                 # field<->dimensionality
                                                                 # import cycle

    rows, d_rows, topic_ids, contrast_ids = [], [], [], []
    for tid, res in all_topic_results.items():
        for ci, c in enumerate(res["contrasts"]):
            if not c["passed"]:
                continue
            rows.append(c["profile"][ref_layer]["v"])
            d_rows.append([c["profile"][l]["d"] for l in layers])
            topic_ids.append(tid)
            contrast_ids.append(ci)

    d = rows[0].shape[0] if rows else 0
    V = np.stack(rows) if rows else np.zeros((0, d))
    D = np.stack(d_rows) if d_rows else np.zeros((0, len(layers)))
    pr = participation_ratio(V, center=False) if len(rows) else float("nan")
    spec = spectrum(V, center=False) if len(rows) else np.array([])
    return {
        "V": V, "depth_profiles": D, "layers": list(layers),
        "topic_ids": topic_ids, "contrast_ids": contrast_ids,
        "pr": pr, "spectrum": spec, "ref_layer": int(ref_layer), "k": len(rows),
    }


def axis_family_clustering(V, merge_threshold: float = 0.7):
    """Cluster pooled axes into families by |cos| (average linkage).

    Distance = 1 - |cos|; families merge while their average-linkage |cos| is
    >= `merge_threshold`. Sign-invariant by construction (|cos|), consistent
    with the rest of this path never depending on a discovered axis's sign.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    V = np.asarray(V, dtype=float)
    k = V.shape[0]
    if k < 2:
        return np.zeros(k, dtype=int)
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    D = 1.0 - np.abs(Vn @ Vn.T)
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, None)
    Z = linkage(squareform(D, checks=False), method="average")
    return fcluster(Z, t=(1.0 - merge_threshold), criterion="distance")


    # -- Removed: depth-profile-correlation family clustering. -----------
    # An earlier version of this module clustered pooled axes a second way,
    # by correlation between their depth profiles d(l) rather than by
    # direction, as a hoped-for "geometry-independent" corroboration of
    # `axis_family_clustering`. Empirically it clustered nothing: every one
    # of the 543 axes in the Amazon-reviews run formed its own singleton
    # family (agreement_ari = 0.0 against the cosine-based clustering by
    # construction, not because the two methods substantively disagreed).
    # A layer-by-layer effect-size CURVE is a poor proxy for what a
    # direction is about -- two axes can look nothing alike as curves while
    # meaning the same thing, or vice versa -- so this was a degenerate
    # method, not an informative negative result, and answering "do these
    # axes mean the same thing" is better served by `axis_semantics.py`'s
    # vocabulary-based semantic family clustering (uses each axis's own
    # differential TF-IDF terms, not its depth profile). See that module's
    # docstring for the replacement.


def axis_families(pooled: dict, cos_merge: float = 0.7) -> dict:
    """Geometric (|cos|-direction) family clustering of a pooled ensemble
    (`pooled_field_assembly`), plus each family's topic membership.

    This is the RAW-DIRECTION view of family structure: two axes are the
    same family here if their activation-space vectors point (up to sign)
    the same way, regardless of what vocabulary distinguishes their poles.
    For the complementary MEANING view -- do two axes contrast similar
    vocabulary, regardless of geometric direction -- see
    `axis_semantics.semantic_families_table`; the two are expected to (and,
    empirically, do) capture different structure, not the same structure
    twice, so neither supersedes the other.
    """
    V = pooled["V"]
    k = V.shape[0]
    if k < 2:
        return {"cos_family_labels": [], "families_by_cos": {}}

    labels_cos = axis_family_clustering(V, merge_threshold=cos_merge)

    def _membership(labels):
        fam = {}
        for i, lab in enumerate(labels):
            fam.setdefault(int(lab), []).append(pooled["topic_ids"][i])
        return {fid: sorted(set(map(str, ts))) for fid, ts in fam.items()}

    return {
        "cos_family_labels": labels_cos.tolist(),
        "families_by_cos": _membership(labels_cos),
    }
