"""The gate: the ONE test a discovered contrast (`topol_axes.py`) must pass to
be reported as real. Everything else discovery produces is recorded, never
filtered on.

WHAT THE GATE DOES
-------------------
    1. Split the topic's documents into two disjoint halves.
    2. Run `topol_axes.topic_axes_pipeline` independently on each half AND
       once on the full topic.
    3. Match HALF-1 contrasts against HALF-2 contrasts greedily by |cos|
       (`match_contrasts`) -- this is the literal split-half stability test.
    4. A half1/half2 match with |cos| >= `stability_threshold` is a validated,
       reproducible axis. Map it onto the closest FULL-topic contrast (by
       |cos| at the half-1 member's own peak layer) so the reported vector,
       depth profile and effect size come from the full document set, which
       has more statistical power than either half alone.

Why match half-vs-half rather than full-vs-half: the spec this implements
says "match contrasts ACROSS HALVES", i.e. half1 vs half2 -- the full-topic
run does not enter the stability test itself, only the final reporting. This
mapping step is a deliberate design choice to bridge "which half-pair is
stable" back to "which reported axis it validates"; it is logged explicitly
(`split_half_cos` per contrast) so it can be audited or replaced.

Full-topic contrasts that no matched half-pair maps onto are recorded as
NOT gated (passed=False) -- they are not deleted, per "everything else is
recorded, not gated" for the nuisance diagnostics, and for symmetry we extend
the same non-destructive stance to ungated contrasts: they stay in the output,
just flagged.

NUISANCE DIAGNOSTICS (never gated on)
--------------------------------------
Correlation of each axis's document projection with review length and star
rating, and a one-way ANOVA R^2 against product id. These are recorded for
every full-topic contrast (gated or not) as diagnostics for the next
iteration -- see `nuisance_diagnostics`. `product_ids=None` (the current
Amazon corpus prep exposes no product identifier) reports that R^2 as NaN
with an explicit note, rather than silently omitting the field.
"""
from __future__ import annotations

import numpy as np

from . import topol_axes as AX


# ------------------------------------------------------------------ #
#  Split
# ------------------------------------------------------------------ #
def split_half(n: int, random_state: int = 42):
    """Disjoint ~50/50 split of `n` LOCAL document indices (0..n-1)."""
    rng = np.random.default_rng(random_state)
    idx = np.arange(n)
    rng.shuffle(idx)
    half = n // 2
    return np.sort(idx[:half]), np.sort(idx[half:])


# ------------------------------------------------------------------ #
#  Greedy cross-half matching
# ------------------------------------------------------------------ #
def _unit(x, eps=1e-12):
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x)
    return x / n if n > eps else x


def match_contrasts(contrasts_1: list[dict], contrasts_2: list[dict]):
    """Greedy |cos| matching between two independently-discovered contrast
    sets, each carrying a full per-layer profile from `topol_axes.depth_profile`.

    Reference layer for a pair is EACH side's own l*: half 1 and half 2 need
    not agree on where a shared axis peaks, so the pair score is the better of
    "compare at half 1's peak" and "compare at half 2's peak" (both directions
    are always computable since both contrasts carry the full cached-layer
    profile, not just their own peak).

    Returns (matches, unmatched_1, unmatched_2). `matches` is EVERY greedily
    assigned pair with its |cos| (not just the ones above threshold) so the
    full distribution can be logged and the threshold revisited empirically.
    """
    def _pair_cos(c1, c2):
        l1, l2 = c1["l_star"], c2["l_star"]
        cos_l1 = abs(float(_unit(c1["profile"][l1]["v"]) @ _unit(c2["profile"][l1]["v"])))
        cos_l2 = abs(float(_unit(c1["profile"][l2]["v"]) @ _unit(c2["profile"][l2]["v"])))
        return max(cos_l1, cos_l2)

    pairs = [(_pair_cos(c1, c2), i, j)
             for i, c1 in enumerate(contrasts_1) for j, c2 in enumerate(contrasts_2)]
    pairs.sort(key=lambda t: -t[0])

    used1, used2, matches = set(), set(), []
    for cos, i, j in pairs:
        if i in used1 or j in used2:
            continue
        used1.add(i); used2.add(j)
        matches.append({"i": i, "j": j, "cos": cos})

    unmatched1 = [i for i in range(len(contrasts_1)) if i not in used1]
    unmatched2 = [j for j in range(len(contrasts_2)) if j not in used2]
    return matches, unmatched1, unmatched2


# ------------------------------------------------------------------ #
#  Nuisance diagnostics -- recorded, never gated on
# ------------------------------------------------------------------ #
def pearsonr(x, y):
    """Pearson correlation, NaN on degenerate input (too few points, or
    either side constant) -- public because `pipeline.step_topol_baseline_compare`
    reuses this to correlate v_CB^t's own projection against the continuous
    rating, the same way `nuisance_diagnostics` below uses it per axis."""
    if x is None or y is None:
        return float("nan")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _anova_r2(values, groups):
    """One-way ANOVA R^2 = SS_between / SS_total."""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    if len(np.unique(groups)) < 2:
        return float("nan")
    grand = values.mean()
    ss_tot = float(((values - grand) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    ss_between = 0.0
    for g in np.unique(groups):
        vg = values[groups == g]
        ss_between += len(vg) * (vg.mean() - grand) ** 2
    return float(ss_between / ss_tot)


def nuisance_diagnostics(projection, lengths=None, ratings=None, product_ids=None):
    """Correlation with review length and star rating; one-way-ANOVA R^2 with
    product id. Diagnostics only -- never used to gate a contrast."""
    out = {
        "corr_length": pearsonr(projection, lengths),
        "corr_rating": pearsonr(projection, ratings),
    }
    if product_ids is not None:
        out["r2_product_id"] = _anova_r2(projection, product_ids)
    else:
        out["r2_product_id"] = float("nan")
        out["r2_product_id_note"] = "product id not available in this corpus prep"
    return out


def _project_axis(contrast, acts_by_layer):
    """Document projection onto a contrast's vector at its own l*, centred on
    the topic mean at that layer (matches how `topol_axes` found the split)."""
    l = contrast["l_star"]
    X = np.asarray(acts_by_layer[l], dtype=np.float64)
    vhat = _unit(contrast["v_at_lstar"])
    Xc = X - X.mean(0)
    return Xc @ vhat


# ------------------------------------------------------------------ #
#  Full per-topic gate
# ------------------------------------------------------------------ #
def gate_topic(acts_by_layer: dict, layers: list[int], discovery_cfg, random_state: int = 42,
              lengths=None, ratings=None, product_ids=None):
    """Discover on the full topic and independently on two halves; stability-
    gate every full-topic contrast; attach (never gate on) nuisance
    diagnostics. See module docstring for the half-vs-half -> full-contrast
    mapping this implements.
    """
    n = next(iter(acts_by_layer.values())).shape[0]
    full_contrasts = AX.topic_axes_pipeline(
        acts_by_layer, layers, max_depth=discovery_cfg.max_depth,
        min_cluster_size=discovery_cfg.min_cluster_size,
        ari_merge=discovery_cfg.ari_merge, random_state=random_state)

    idx1, idx2 = split_half(n, random_state=random_state)
    acts1 = {l: acts_by_layer[l][idx1] for l in layers}
    acts2 = {l: acts_by_layer[l][idx2] for l in layers}
    contrasts1 = AX.topic_axes_pipeline(
        acts1, layers, max_depth=discovery_cfg.max_depth,
        min_cluster_size=discovery_cfg.min_cluster_size,
        ari_merge=discovery_cfg.ari_merge, random_state=random_state + 1)
    contrasts2 = AX.topic_axes_pipeline(
        acts2, layers, max_depth=discovery_cfg.max_depth,
        min_cluster_size=discovery_cfg.min_cluster_size,
        ari_merge=discovery_cfg.ari_merge, random_state=random_state + 2)

    matches, unmatched1, unmatched2 = match_contrasts(contrasts1, contrasts2)
    threshold = discovery_cfg.stability_threshold

    # map each passing half1/half2 match onto its closest full-topic contrast
    full_best = {}   # full_idx -> {"cos": ..., "half1_i": ..., "half2_j": ...}
    for m in matches:
        if m["cos"] < threshold:
            continue
        c1 = contrasts1[m["i"]]
        l_ref = c1["l_star"]
        v1 = _unit(c1["profile"][l_ref]["v"])
        best_fi, best_cos = None, -1.0
        for fi, fc in enumerate(full_contrasts):
            cos = abs(float(v1 @ _unit(fc["profile"][l_ref]["v"])))
            if cos > best_cos:
                best_cos, best_fi = cos, fi
        if best_fi is not None and (best_fi not in full_best or m["cos"] > full_best[best_fi]["cos"]):
            full_best[best_fi] = {"cos": m["cos"], "half1_i": m["i"], "half2_j": m["j"]}

    results = []
    for fi, fc in enumerate(full_contrasts):
        entry = full_best.get(fi)
        proj = _project_axis(fc, acts_by_layer)
        diag = nuisance_diagnostics(proj, lengths, ratings, product_ids)
        results.append({
            "topic_local_index": fi,
            "found_layer": fc["found_layer"], "found_depth": fc["found_depth"],
            "layers_found": fc["layers_found"],
            "l_star": fc["l_star"], "effect_size_at_lstar": fc["d_at_lstar"],
            "v_at_lstar": AX.fix_sign(fc["v_at_lstar"]),
            "profile": fc["profile"], "labels": fc["labels"],
            "passed": entry is not None,
            "split_half_cos": entry["cos"] if entry else None,
            **diag,
        })

    rejected = []
    for m in matches:
        if m["cos"] < threshold:
            rejected.append({"source": "half_pair", "half1_index": m["i"],
                             "half2_index": m["j"], "cos": m["cos"]})
    for i in unmatched1:
        rejected.append({"source": "half1_unmatched", "half1_index": i, "cos": None})
    for j in unmatched2:
        rejected.append({"source": "half2_unmatched", "half2_index": j, "cos": None})

    return {
        "contrasts": results,
        "n_full_contrasts": len(full_contrasts),
        "n_half1_contrasts": len(contrasts1), "n_half2_contrasts": len(contrasts2),
        "match_cos_distribution": [m["cos"] for m in matches],
        "rejected": rejected,
        "stability_threshold": threshold,
    }
