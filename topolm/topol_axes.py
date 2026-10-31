"""Per-topic axis discovery: recursive bipartition over the cached residual
stack (`topol_activations.py`), independent of the CB diff-in-means field
(`field.py`) which this module never reads or writes.

Topic assignment itself is unchanged -- it still comes from the existing
EmbeddingGemma + UMAP + Leiden path (`manifold.py`). What is new is what
happens INSIDE one topic, at one cached layer:

    1. centre the topic's documents on the topic centroid at that layer
       (`recursive_bipartition`'s `X.mean(0)`);
    2. 2-means (cosine) splits the centred cloud in two;
    3. the between-means direction is projected out of the (still centred)
       cloud, and step 2 repeats on the residual -- a deflation sequence, up
       to `max_depth` times, stopping early if a split's smaller side falls
       below `min_cluster_size`.

Because each split is found AFTER the previous directions have been removed,
the directions recorded for one (topic, layer) are orthogonal to each other by
construction -- this is what licenses the "PR is not meaningful within a
topic" note in `field.py`'s cross-topic assembly.

ON CENTRING -- READ THIS BEFORE TOUCHING `dimensionality.py`
--------------------------------------------------------------
`dimensionality.py` analyses the CROSS-TOPIC ensemble of per-topic CB vectors
(rows = topics) UNCENTRED, on purpose: a purely 1-D field is exactly the case
where every topic shifts the same way, and mean-centring across topics would
erase that shared direction. That rule is about a DIFFERENT object (the
per-topic vectors, stacked across topics) and must not be applied here.

Here the object is one topic's DOCUMENTS at one layer, and centring on the
topic centroid is required: cosine 2-means only finds a sensible bipolar split
when the cloud is centred on its own mean (otherwise "cosine similarity" is
dominated by whatever arbitrary offset the topic anchor happens to have, not
by the direction that actually splits the cloud in two). The two centring
rules do not contradict each other; they are answers to different questions.
Do not let one leak into the other's implementation.

Effect size is centring-invariant: mu_B - mu_A and the pooled within-group
scatter are both unchanged by subtracting a constant from every document, so
`effect_size` below is computed directly from the (uncentred) per-layer
activations -- centring only matters for the clustering step.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ #
#  Cosine 2-means
# ------------------------------------------------------------------ #
def kmeans2_cosine(X, rng, n_init: int = 10, max_iter: int = 100, eps: float = 1e-12):
    """Spherical (cosine) 2-means: normalise rows, assign by cosine similarity
    to the nearest of 2 unit centroids, update centroids as the mean of their
    assigned rows renormalised to unit length. Returns (labels in {0,1}, C).
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.clip(norms, eps, None)

    best = None
    for _ in range(n_init):
        seed = rng.choice(n, size=2, replace=False)
        C = Xn[seed].copy()
        labels = np.full(n, -1)
        for it in range(max_iter):
            sims = Xn @ C.T                       # (n, 2)
            new_labels = np.argmax(sims, axis=1)
            if it > 0 and np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
            for k in (0, 1):
                mask = labels == k
                if not mask.any():
                    far = int(np.argmin(sims[:, 1 - k]))
                    labels[far] = k
                    mask = labels == k
                c = Xn[mask].mean(0)
                cn = np.linalg.norm(c)
                C[k] = c / cn if cn > eps else Xn[int(rng.integers(n))]
        sims = Xn @ C.T
        obj = float(sims[np.arange(n), labels].sum())     # total cosine sim, maximise
        if best is None or obj > best[0]:
            best = (obj, labels.copy(), C.copy())
    return best[1], best[2]


# ------------------------------------------------------------------ #
#  Recursive bipartition (one topic, one layer)
# ------------------------------------------------------------------ #
def recursive_bipartition(X_layer, max_depth: int, min_cluster_size: int, rng,
                          eps: float = 1e-12):
    """X_layer: (n_t, d) RAW mean-pooled activations for one topic's documents
    at one cached layer (same document order every call for a given topic).

    Returns a list of candidates (possibly empty), each:
        depth, labels (n_t,) in {0,1}, effect_size (d(l) at THIS layer),
        n_a, n_b.
    """
    X_layer = np.asarray(X_layer, dtype=np.float64)
    n = X_layer.shape[0]
    if n < 2 * min_cluster_size:
        return []

    mean = X_layer.mean(0)
    Xr = X_layer - mean            # centred cloud, required (see module docstring)
    candidates = []
    for depth in range(1, max_depth + 1):
        labels, _ = kmeans2_cosine(Xr, rng)
        n_a = int((labels == 0).sum())
        n_b = int((labels == 1).sum())
        if min(n_a, n_b) < min_cluster_size:
            break

        muA = X_layer[labels == 0].mean(0)
        muB = X_layer[labels == 1].mean(0)
        v = muB - muA
        pooled = np.concatenate([X_layer[labels == 0] - muA, X_layer[labels == 1] - muB])
        sigma = float(np.sqrt((pooled ** 2).sum(1).mean()))
        effect_size = float(np.linalg.norm(v) / (sigma + eps))

        candidates.append({
            "depth": depth, "labels": labels.copy(),
            "effect_size": effect_size, "n_a": n_a, "n_b": n_b,
        })

        # project the between-means direction out of the (still centred) cloud
        # and recurse -- v computed in Xr-space equals v computed in X_layer-space
        # exactly, since the shared mean cancels in a difference of means.
        u = v / (np.linalg.norm(v) + eps)
        Xr = Xr - np.outer(Xr @ u, u)
    return candidates


def discover_topic_axes(acts_by_layer: dict, layers: list[int], max_depth: int = 4,
                        min_cluster_size: int = 30, random_state: int = 42):
    """Run `recursive_bipartition` at every cached layer for one topic.

    acts_by_layer: {layer: (n_t, d) RAW activations}, one consistent document
    order shared across all layers (so partitions are comparable cross-layer).
    Returns a flat list of candidates tagged with `layer`.
    """
    rng = np.random.default_rng(random_state)
    candidates = []
    for l in layers:
        for c in recursive_bipartition(acts_by_layer[l], max_depth, min_cluster_size, rng):
            c["layer"] = int(l)
            candidates.append(c)
    return candidates


# ------------------------------------------------------------------ #
#  Cross-layer deduplication
# ------------------------------------------------------------------ #
def dedup_candidates(candidates: list[dict], ari_merge: float = 0.7):
    """Cluster candidate partitions (within one topic, across all layers/depths)
    by adjusted Rand index, average linkage; candidates whose average-linkage
    ARI similarity is >= `ari_merge` are merged into one contrast.

    Each contrast is represented by the member with the highest local
    `effect_size`. `layers_found` is the sorted set of layers among its
    members -- itself a reported result (point 3 of the spec this implements).
    """
    from sklearn.metrics import adjusted_rand_score
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    m = len(candidates)
    if m == 0:
        return []
    if m == 1:
        return [{"members": [0], "representative": 0,
                "layers_found": [candidates[0]["layer"]]}]

    D = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            ari = adjusted_rand_score(candidates[i]["labels"], candidates[j]["labels"])
            dist = max(0.0, 1.0 - ari)
            D[i, j] = D[j, i] = dist
    Z = linkage(squareform(D, checks=False), method="average")
    flat = fcluster(Z, t=(1.0 - ari_merge), criterion="distance")

    contrasts = []
    for cid in np.unique(flat):
        members = [i for i in range(m) if flat[i] == cid]
        rep = max(members, key=lambda i: candidates[i]["effect_size"])
        layers_found = sorted({candidates[i]["layer"] for i in members})
        contrasts.append({"members": members, "representative": rep,
                          "layers_found": layers_found})
    return contrasts


# ------------------------------------------------------------------ #
#  Depth profile and peak layer
# ------------------------------------------------------------------ #
def depth_profile(labels, acts_by_layer: dict, layers: list[int], eps: float = 1e-12):
    """v(l) = mu_B(l) - mu_A(l), d(l) = ||v(l)|| / sigma_pooled(l), for a FIXED
    document partition (`labels`) evaluated at every cached layer.

    `labels` need only be an assignment over one shared document index set (the
    partition is layer-agnostic); `acts_by_layer` must use that same order.
    """
    labels = np.asarray(labels)
    profile = {}
    for l in layers:
        X = np.asarray(acts_by_layer[l], dtype=np.float64)
        muA = X[labels == 0].mean(0)
        muB = X[labels == 1].mean(0)
        v = muB - muA
        pooled = np.concatenate([X[labels == 0] - muA, X[labels == 1] - muB])
        sigma = float(np.sqrt((pooled ** 2).sum(1).mean()))
        d = float(np.linalg.norm(v) / (sigma + eps))
        profile[int(l)] = {"v": v, "d": d, "sigma": sigma}
    l_star = max(profile, key=lambda l: profile[l]["d"])
    return {"profile": profile, "l_star": int(l_star),
            "d_at_lstar": profile[l_star]["d"], "v_at_lstar": profile[l_star]["v"]}


# ------------------------------------------------------------------ #
#  Sign (cosmetic only -- see field.py; never used for comparisons)
# ------------------------------------------------------------------ #
def fix_sign(v):
    """Deterministic sign convention for FILE OUTPUT ONLY: force the
    largest-magnitude coordinate positive. Signs are meaningless for discovered
    axes -- no statistic in this path may depend on one. Comparisons use
    |cos|; the field is assembled from outer products v v^T (sign-invariant by
    construction, see `field.py`). This function exists purely so a saved
    vector looks the same across re-runs."""
    v = np.asarray(v, dtype=float)
    j = int(np.argmax(np.abs(v)))
    return v if v[j] >= 0 else -v


# ------------------------------------------------------------------ #
#  Glue: discover -> dedup -> depth profile, for one topic
# ------------------------------------------------------------------ #
def topic_axes_pipeline(acts_by_layer: dict, layers: list[int], max_depth: int = 4,
                        min_cluster_size: int = 30, ari_merge: float = 0.7,
                        random_state: int = 42):
    """Full per-topic discovery: candidates at every layer -> cross-layer dedup
    -> depth profile + peak layer for each surviving contrast.

    Returns a list of contrasts, each:
        labels        the representative candidate's document partition
        found_layer   the layer that candidate was recorded at
        found_depth   its recursion depth
        layers_found  sorted layers among all members of its dedup cluster
        n_members     size of that dedup cluster
        profile, l_star, d_at_lstar, v_at_lstar   full cross-layer depth profile
    """
    candidates = discover_topic_axes(acts_by_layer, layers, max_depth=max_depth,
                                     min_cluster_size=min_cluster_size,
                                     random_state=random_state)
    contrasts_raw = dedup_candidates(candidates, ari_merge=ari_merge)

    contrasts = []
    for c in contrasts_raw:
        rep = candidates[c["representative"]]
        dp = depth_profile(rep["labels"], acts_by_layer, layers)
        contrasts.append({
            "labels": rep["labels"], "found_layer": rep["layer"],
            "found_depth": rep["depth"], "layers_found": c["layers_found"],
            "n_members": len(c["members"]),
            **dp,
        })
    return contrasts
