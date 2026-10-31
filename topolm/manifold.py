"""Topic segmentation via UMAP + Leiden.

Preserved verbatim in spirit from the original TOPol:
    * a `transform_mode='graph'` UMAP builds the kNN graph that Leiden partitions,
    * a separate 2-D UMAP embedding is kept ONLY for visualisation.

CRITICAL: neither the cluster centroids nor the polarity vectors are computed
here.  Clustering assigns each document a topic id; the polarity field is built
in `field.py` in the FULL representation space.  UMAP coordinates never enter the
vector arithmetic - they are display coordinates only.
"""
from __future__ import annotations

import numpy as np


def cluster_documents(embeddings, cfg, random_state=None):
    """Return dict with cluster labels and a 2-D layout for plotting.

    embeddings : (N, d_model) full-space representations (the SAME array used for the field).
    """
    from umap import UMAP
    random_state = cfg.random_state if random_state is None else random_state
    mc = cfg.manifold

    umap_graph = UMAP(
        n_components=mc.n_components, n_neighbors=mc.n_neighbors, min_dist=mc.min_dist,
        metric=mc.metric, transform_mode="graph", random_state=random_state,
    )
    umap_graph.fit(embeddings)

    umap_2d = UMAP(
        n_components=2, n_neighbors=mc.n_neighbors, min_dist=mc.min_dist,
        metric=mc.metric, transform_mode="embedding", random_state=random_state,
    )
    coords_2d = umap_2d.fit_transform(embeddings)

    labels = _leiden_from_umap_graph(umap_graph, mc.leiden_resolution, random_state)
    return {
        "labels": np.asarray(labels),
        "coords_2d": np.asarray(coords_2d),
        "n_clusters": int(len(np.unique(labels))),
        "umap_graph": umap_graph,
    }


def _leiden_from_umap_graph(umap_graph, resolution, random_state):
    """Leiden partition of the UMAP fuzzy-simplicial graph.

    Uses scikit-network if available (matches the original repo), otherwise
    falls back to python-igraph + leidenalg.
    """
    from scipy.sparse import csr_matrix
    A = csr_matrix(umap_graph.graph_)
    try:
        from sknetwork.clustering import Leiden
        model = Leiden(resolution=resolution, random_state=random_state)
        return model.fit_predict(A)
    except Exception:
        import igraph as ig
        import leidenalg
        A = A.tocoo()
        edges = list(zip(A.row.tolist(), A.col.tolist()))
        g = ig.Graph(n=A.shape[0], edges=edges, edge_attrs={"weight": A.data.tolist()})
        part = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            weights="weight", resolution_parameter=resolution, seed=random_state,
        )
        labels = np.zeros(A.shape[0], dtype=int)
        for cid, comm in enumerate(part):
            labels[comm] = cid
        return labels


def cluster_boundary_independent(embeddings, cb_labels, cfg, random_state=None):
    """Fit the partition on regime A only, then assign regime B by nearest centroid.

    WHY THIS EXISTS
    ---------------
    Clustering the pooled corpus makes the partition depend on the contextual
    boundary. When a topic's regime displacement approaches its own internal
    scatter, Leiden resolves the A-side and B-side as two separate communities, the
    topic is populated on only one side, and it contributes no polarity vector.

    The failure is silent AND selective: topics are split in order of displacement
    magnitude, so the ones that moved MOST are dropped FIRST. Mean drift is biased
    downward, and the bias worsens as the true effect grows (Proposition 3 in the
    paper; field guide Part X).

    Fitting on regime A alone makes topic identity independent of the boundary by
    construction, so every topic spans both regimes and the selection effect
    disappears. The cost is an assumption -- that topic structure is stable across
    the boundary -- which is itself testable by clustering B alone and comparing
    partitions (`partition_stability` below).

    Use when `spanning_fraction` on the pooled partition falls below 1.
    """
    embeddings = np.asarray(embeddings)
    cb = np.asarray(cb_labels)
    idx_A = np.where(cb == 0)[0]
    if len(idx_A) < cfg.manifold.n_neighbors + 1:
        raise ValueError("too few regime-A documents to fit a partition on")

    res_A = cluster_documents(embeddings[idx_A], cfg, random_state=random_state)
    labels_A = res_A["labels"]

    # centroids of the A-side partition, in the SAME space the partition was fit in
    cents, ids = [], []
    for c in np.unique(labels_A):
        cents.append(embeddings[idx_A][labels_A == c].mean(0))
        ids.append(c)
    C = np.stack(cents)
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)

    # assign every document (A and B) to its nearest A-centroid by cosine
    E = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    assign = np.asarray(ids)[np.argmax(E @ Cn.T, axis=1)]

    # 2-D layout for display only, computed on the full corpus
    coords = cluster_documents(embeddings, cfg, random_state=random_state)["coords_2d"]
    return {
        "labels": assign,
        "coords_2d": coords,
        "n_clusters": int(len(np.unique(assign))),
        "fitted_on": "regime_A",
        "n_fit_docs": int(len(idx_A)),
    }


def spanning_fraction(cluster_labels, cb_labels):
    """Share of communities populated on BOTH sides of the boundary.

    The headline diagnostic for Proposition 3. A value below 1 means topics are
    being dropped, the retained set is biased toward small displacements, and
    `bar m` is an underestimate. Report it next to every drift magnitude.
    """
    cl = np.asarray(cluster_labels); cb = np.asarray(cb_labels)
    clusters = np.unique(cl)
    both = sum(1 for c in clusters
               if ((cl == c) & (cb == 0)).any() and ((cl == c) & (cb == 1)).any())
    return {"spanning_fraction": both / max(1, len(clusters)),
            "n_spanning": int(both), "n_clusters": int(len(clusters)),
            "dropped": int(len(clusters) - both)}


def shift_to_scatter(embeddings, cluster_labels, cb_labels):
    """Per-topic ||v_i|| / sigma_i -- the quantity the operating window bounds.

    Values approaching 1 are the warning zone: displacement comparable to
    within-topic scatter is where boundary-dependent segmentation starts splitting
    topics. Report alongside the spanning fraction.
    """
    E = np.asarray(embeddings, dtype=float)
    cl = np.asarray(cluster_labels); cb = np.asarray(cb_labels)
    out = {}
    for c in np.unique(cl):
        mA = (cl == c) & (cb == 0)
        mB = (cl == c) & (cb == 1)
        if mA.sum() < 2 or mB.sum() < 2:
            continue
        v = E[mB].mean(0) - E[mA].mean(0)
        pooled = np.concatenate([E[mA] - E[mA].mean(0), E[mB] - E[mB].mean(0)])
        sigma = float(np.sqrt((pooled ** 2).sum(1).mean()))
        out[int(c)] = {"shift": float(np.linalg.norm(v)), "scatter": sigma,
                       "ratio": float(np.linalg.norm(v) / (sigma + 1e-12)),
                       "nA": int(mA.sum()), "nB": int(mB.sum())}
    return out


def partition_stability(embeddings, cb_labels, cfg, random_state=None):
    """Cluster A alone and B alone; compare partitions on the shared assignment.

    Tests the assumption the boundary-independent segmentation makes. High
    agreement means topic structure is stable across the boundary and fitting on A
    is safe. Low agreement means the regimes genuinely talk about different things,
    and a displacement field may be the wrong object for that comparison.
    """
    from sklearn.metrics import adjusted_rand_score
    E = np.asarray(embeddings); cb = np.asarray(cb_labels)
    iA, iB = np.where(cb == 0)[0], np.where(cb == 1)[0]
    lA = cluster_documents(E[iA], cfg, random_state=random_state)["labels"]
    lB = cluster_documents(E[iB], cfg, random_state=random_state)["labels"]

    def _assign(fit_idx, fit_lab):
        cents = np.stack([E[fit_idx][fit_lab == c].mean(0) for c in np.unique(fit_lab)])
        Cn = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-12)
        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        return np.unique(fit_lab)[np.argmax(En @ Cn.T, axis=1)]

    return {"ari_A_vs_B_partition": float(adjusted_rand_score(_assign(iA, lA),
                                                             _assign(iB, lB))),
            "k_from_A": int(len(np.unique(lA))), "k_from_B": int(len(np.unique(lB)))}
