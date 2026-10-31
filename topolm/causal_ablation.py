"""Causal ablation: remove the sentiment-related directions from a topic's
own activations and measure what changes -- the necessity-half of a question
every earlier step (`field.py`, `topol_axes.py`, `pole_sentiment.py`) only
answers correlationally ("this direction correlates with regime/rating/
RoBERTa polarity"). `data.py`'s own docstring names this as goal (4) of a
valid pipeline: "collapse regime separability under ablation but not under a
random control."

Two families of direction are identified per topic and removed together:

  1. `v_CB^t` -- the supervised, rating-derived polarity vector (`field.py`),
     ground truth for every other step in this pipeline.
  2. Gate-passing DISCOVERED axes (`topol_axes.py`/`topol_gate.py`) flagged
     as sentiment-related by an INDEPENDENT signal: a pretrained RoBERTa
     sentiment classifier's per-document polarity score
     (`pole_sentiment.polarity_scalar`), not this pipeline's own
     `corr_rating`/`cos_to_cb` -- so the criterion for "this axis is about
     sentiment" is not measured by the same instrument being tested for
     necessity. `pole_sentiment.axis_sentiment_delta`'s `sentiment_delta`
     (mean RoBERTa polarity, pole B minus pole A) is the selection signal;
     `AblationConfig.sentiment_delta_threshold` is the cutoff.

Ablation is orthogonal projection, done TOPIC-LOCALLY: each topic's own
identified directions are removed only from that topic's own documents (a
topic's v_CB^t/discovered axes are not meaningful directions for another
topic's cloud). Everything here operates on raw `(N, d_model)` arrays and
plain per-topic dicts, so it is reusable at any single cached layer without
re-running discovery -- re-running the expensive per-topic recursive
bipartition + split-half gate on ablated activations (would new axes emerge
once sentiment is gone?) is explicitly OUT of scope here; the module only
tests necessity of what discovery already found, not re-discovery.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import topol_axes as TAX


# ------------------------------------------------------------------ #
#  Direction selection
# ------------------------------------------------------------------ #
def cb_directions_at_layer(gate_results: dict, cache: dict, cluster_labels, cb_labels,
                           layer: int) -> dict:
    """Per topic: v_CB^t at `layer`, recomputed exactly as
    `pipeline.step_topol_baseline_compare`/`pole_sentiment.cb_direction_alignment`
    both already do (via `topol_axes.depth_profile` on the CB regime labels) --
    reused, not reinvented, so this is the SAME vector those steps compare
    against, at the SAME layer. Topics with fewer than 2 CB classes present
    (no v_CB^t defined) are skipped.
    """
    cluster_labels = np.asarray(cluster_labels)
    cb_labels = np.asarray(cb_labels)
    out = {}
    for topic in sorted(gate_results.keys()):
        doc_idx = np.where(cluster_labels == topic)[0]
        cb_local = cb_labels[doc_idx]
        if len(np.unique(cb_local)) < 2:
            continue
        X = np.asarray(cache["acts"][layer], dtype=np.float64)[doc_idx]
        profile = TAX.depth_profile(cb_local.astype(int), {layer: X}, [layer])
        v = profile["profile"][layer]["v"]
        out[int(topic)] = v
    return out


def select_sentiment_axes(align_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Gate-passing discovered axes whose RoBERTa-based `|sentiment_delta|`
    clears `threshold` -- the independent (non-`corr_rating`, non-`cos_to_cb`)
    sentiment criterion. `align_df` is `pole_sentiment.build_alignment_table`'s
    output (`tables/topol_pv_sentiment_alignment.csv`)."""
    sub = align_df.dropna(subset=["sentiment_delta"])
    return sub[sub["sentiment_delta"].abs() >= threshold].reset_index(drop=True)


def discovered_directions_at_layer(gate_results: dict, selected_axes: pd.DataFrame,
                                   layer: int) -> dict:
    """Per topic: list of (axis_id, raw vector at `layer`) for every axis
    `select_sentiment_axes` selected in that topic. Falls back to skipping an
    axis if `layer` is missing from its depth profile (should not happen for
    any layer in the run's own activation cache, since every gate-passing
    contrast's profile spans every cached layer -- guarded anyway)."""
    out: dict[int, list] = {}
    for _, row in selected_axes.iterrows():
        topic, ci, axis_id = int(row["topic"]), int(row["contrast_index"]), int(row["axis_id"])
        contrast = gate_results[topic]["contrasts"][ci]
        if layer not in contrast["profile"]:
            continue
        v = contrast["profile"][layer]["v"]
        out.setdefault(topic, []).append((axis_id, v))
    return out


# ------------------------------------------------------------------ #
#  Orthogonal projection / ablation
# ------------------------------------------------------------------ #
def orthonormal_basis(vectors: list, tol: float = 1e-8) -> np.ndarray:
    """QR-orthonormalise a list of d-dim vectors into a `(d, r)` basis,
    r <= len(vectors) -- near-parallel/duplicate directions collapse to one
    basis column instead of double-counting, and a zero vector contributes
    none. Returns a `(d, 0)` array if every input vector was negligible."""
    if not vectors:
        d = 0
        return np.zeros((d, 0))
    V = np.stack([np.asarray(v, dtype=np.float64) for v in vectors], axis=1)  # (d, n)
    Q, R = np.linalg.qr(V)
    keep = np.abs(np.diag(R)) > tol
    return Q[:, keep]


def ablate_matrix(X: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Remove the subspace spanned by `basis` (an orthonormal `(d, r)`
    matrix, e.g. from `orthonormal_basis`) from every row of `X` (N, d)."""
    if basis.shape[1] == 0:
        return X.copy()
    return X - (X @ basis) @ basis.T


def random_control_basis(d: int, r: int, rng: np.random.Generator) -> np.ndarray:
    """A random orthonormal `(d, r)` basis of the SAME rank as a real
    ablation target -- the matched control `data.py` asks for: does removing
    ANY `r`-dimensional subspace collapse regime separability (a rank/norm
    artefact), or only the specific one identified as sentiment-related?"""
    if r == 0:
        return np.zeros((d, 0))
    Q, _ = np.linalg.qr(rng.standard_normal((d, r)))
    return Q


# ------------------------------------------------------------------ #
#  Assembling the per-topic ablation plan
# ------------------------------------------------------------------ #
def build_ablation_plan(gate_results: dict, cache: dict, cluster_labels, cb_labels,
                        align_df: pd.DataFrame, layer: int, sentiment_delta_threshold: float,
                        include_cb: bool = True, include_discovered_axes: bool = True) -> dict:
    """Per topic: the orthonormal basis of directions to ablate, plus a
    human-readable record of what went into it. Returns
    `{"plan": {topic: {"basis": (d,r) array, "sources": [str, ...]}},
      "summary": DataFrame, "selected_axes": DataFrame}` -- the last is the
    RoBERTa-selected discovered-axis rows (empty if
    `include_discovered_axes=False`), returned for reuse by
    `excess_axis_effect_size_before_after` instead of recomputing.
    """
    cb_dirs = cb_directions_at_layer(gate_results, cache, cluster_labels, cb_labels, layer) \
        if include_cb else {}
    selected = select_sentiment_axes(align_df, sentiment_delta_threshold) \
        if include_discovered_axes else align_df.iloc[0:0]
    disc_dirs = discovered_directions_at_layer(gate_results, selected, layer) \
        if include_discovered_axes else {}

    topics = sorted(set(gate_results.keys()) | set(cb_dirs) | set(disc_dirs))
    plan, rows = {}, []
    for topic in topics:
        vectors, sources = [], []
        if topic in cb_dirs:
            vectors.append(cb_dirs[topic]); sources.append("v_CB")
        for axis_id, v in disc_dirs.get(topic, []):
            vectors.append(v); sources.append(f"axis{axis_id}")
        basis = orthonormal_basis(vectors)
        plan[topic] = {"basis": basis, "sources": sources}
        rows.append({"topic": topic, "n_directions_input": len(vectors),
                    "ablated_rank": basis.shape[1], "sources": ", ".join(sources) or "(none)"})
    summary = pd.DataFrame(rows).sort_values("topic").reset_index(drop=True)
    return {"plan": plan, "summary": summary, "selected_axes": selected}


def ablate_embedding(X: np.ndarray, cluster_labels, plan: dict,
                     random_control: bool = False, rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply `build_ablation_plan`'s per-topic basis to a full `(N, d)`
    embedding matrix, topic by topic (each document only ever has ITS OWN
    topic's directions removed). `random_control=True` swaps every topic's
    real basis for a random one of the same rank, drawn from `rng`."""
    cluster_labels = np.asarray(cluster_labels)
    rng = rng or np.random.default_rng(0)
    out = X.copy()
    for topic, entry in plan.items():
        idx = np.where(cluster_labels == topic)[0]
        if len(idx) == 0:
            continue
        basis = entry["basis"]
        if random_control:
            basis = random_control_basis(X.shape[1], basis.shape[1], rng)
        out[idx] = ablate_matrix(out[idx], basis)
    return out


# ------------------------------------------------------------------ #
#  Necessity test: does a linear probe survive ablation?
# ------------------------------------------------------------------ #
def probe_balanced_accuracy(X: np.ndarray, y: np.ndarray, n_splits: int = 5,
                            C: float = 1.0, random_state: int = 42) -> float:
    """Cross-validated balanced accuracy of an L2 logistic probe -- the
    single-layer body of `layer_select.probe_separability_by_layer`'s inner
    loop, factored out so it can be called on a before/after pair directly."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y).astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(sc.transform(X[tr]), y[tr])
        pred = clf.predict(sc.transform(X[te]))
        accs.append(balanced_accuracy_score(y[te], pred))
    return float(np.mean(accs))


def necessity_test(X_full: np.ndarray, cluster_labels, cb_labels, plan: dict,
                   n_splits: int = 5, C: float = 1.0, random_state: int = 42) -> dict:
    """The `res` dict `viz.plot_ablation` already expects
    (`acc_full`/`acc_ablate_direction`/`acc_ablate_random`/`necessity_drop`/
    `control_drop`): a pooled A/B probe's balanced accuracy on the untouched
    activations, on the real per-topic ablation, and on a matched-rank random
    control ablation. A necessity_drop >> control_drop is the direct evidence
    that what was removed was load-bearing for the regime split, not just
    "some subspace of that rank.\""""
    rng = np.random.default_rng(random_state)
    X_ablated = ablate_embedding(X_full, cluster_labels, plan, random_control=False)
    X_random = ablate_embedding(X_full, cluster_labels, plan, random_control=True, rng=rng)

    acc_full = probe_balanced_accuracy(X_full, cb_labels, n_splits, C, random_state)
    acc_direction = probe_balanced_accuracy(X_ablated, cb_labels, n_splits, C, random_state)
    acc_random = probe_balanced_accuracy(X_random, cb_labels, n_splits, C, random_state)
    return {
        "acc_full": acc_full, "acc_ablate_direction": acc_direction, "acc_ablate_random": acc_random,
        "necessity_drop": acc_full - acc_direction, "control_drop": acc_full - acc_random,
    }


# ------------------------------------------------------------------ #
#  Field geometry before / after
# ------------------------------------------------------------------ #
def field_geometry_before_after(X_before: np.ndarray, X_after: np.ndarray, cluster_labels, cb_labels) -> dict:
    """Reconstructs `v_CB^t` before and after ablation (`field.reconstruct_field`,
    reused verbatim) and reports the direct sanity check: for topics where
    v_CB^t was itself ablated, `||v_CB^t||` after should collapse towards 0;
    for topics ablated only via discovered axes (or not ablated at all),
    it need not."""
    from . import field as F

    fld_before = F.reconstruct_field(X_before, cluster_labels, cb_labels)
    fld_after = F.reconstruct_field(X_after, cluster_labels, cb_labels)
    return {
        "clusters": fld_before["clusters"],
        "magnitude_before": fld_before["magnitudes"], "magnitude_after": fld_after["magnitudes"],
        "V_before": fld_before["V"], "V_after": fld_after["V"],
    }


def excess_axis_effect_size_before_after(gate_results: dict, selected_axes: pd.DataFrame,
                                         cluster_labels, X_before: np.ndarray, X_after: np.ndarray,
                                         layer: int) -> pd.DataFrame:
    """Per EXCESS axis (gate-passing, NOT selected as sentiment-related): its
    own discriminative effect size `d = ||mu_B - mu_A|| / sigma_pooled`
    (`topol_axes.depth_profile`'s definition) at `layer`, evaluated on the
    topic's documents BEFORE ablation (== the axis's own already-stored
    `profile[layer]["d"]`, not recomputed) vs. AFTER -- an axis's discovered
    DIRECTION is a fixed vector either way, so what can actually change is
    whether that fixed direction still discriminates its own pole partition
    once the sentiment directions are projected out of the activations it is
    measured against. `d_after` close to `d_before` is the evidence an
    axis's independence from sentiment (already implied by its low
    `cos_to_cb`) is real, not just a low-cosine coincidence; a large drop
    means it wasn't as independent as the cosine threshold suggested.

    `X_before`/`X_after` : (N, d) arrays at `layer`, same document order as
    `cluster_labels` (typically `cache["acts"][layer]` and
    `ablate_embedding(...)`'s output).
    """
    cluster_labels = np.asarray(cluster_labels)
    selected_ids = set(zip(selected_axes["topic"].astype(int), selected_axes["contrast_index"].astype(int)))
    rows = []
    for topic, res in gate_results.items():
        doc_idx = np.where(cluster_labels == int(topic))[0]
        if len(doc_idx) == 0:
            continue
        X_after_topic = X_after[doc_idx]
        for ci, c in enumerate(res["contrasts"]):
            if not c["passed"] or (int(topic), ci) in selected_ids or layer not in c["profile"]:
                continue
            d_before = c["profile"][layer]["d"]
            after_profile = TAX.depth_profile(np.asarray(c["labels"]), {layer: X_after_topic}, [layer])
            d_after = after_profile["profile"][layer]["d"]
            rows.append({"topic": int(topic), "contrast_index": ci,
                        "d_before": d_before, "d_after": d_after,
                        "d_retained_frac": d_after / d_before if d_before > 1e-12 else float("nan")})
    return pd.DataFrame(rows).sort_values(["topic", "contrast_index"]).reset_index(drop=True)
