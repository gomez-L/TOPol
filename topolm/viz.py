"""Plotting.  Every function returns a matplotlib Figure so the RunStore can save
it as a vectorised PDF (editable text) plus a PNG thumbnail.

Colour/marker conventions match the original TOPol figures where sensible:
regime A in blues, regime B in reds, centroids as white squares, drift as arrows.
The 2-D field plot draws arrows between 2-D centroids for DISPLAY ONLY - the axis
title states that the quantitative field lives in the full space.
"""
from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def _apply_style():
    matplotlib.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.dpi": 110, "savefig.dpi": 200,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    })


# ------------------------------------------------------------------ #
def plot_layer_selection(result, figsize=(7, 3.5)):
    _apply_style()
    per = result["per_layer"]
    layers = sorted(per)
    mean = [per[l]["acc_mean"] for l in layers]
    std = [per[l]["acc_std"] for l in layers]
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.errorbar(layers, mean, yerr=std, marker="o", ms=4, lw=1.5, capsize=2, color="#2b6cb0")
    b = result["best_layer"]
    ax.axvline(b, ls="--", color="#e53e3e", lw=1)
    ax.scatter([b], [per[b]["acc_mean"]], color="#e53e3e", zorder=5,
               label=f"chosen layer {b} (acc={result['best_acc']:.3f})")
    ax.axhline(0.5, ls=":", color="grey", lw=1, label="chance")
    ax.set_xlabel("residual layer (hidden_states index)")
    ax.set_ylabel("A/B probe balanced acc.")
    ax.set_title("Layer selection by A/B separability")
    ax.legend(fontsize=8, frameon=False)
    return fig


def plot_field_2d(coords_2d, cluster_labels, cb_labels, figsize=(7.5, 7)):
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    clusters = np.unique(cluster_labels)
    A = cb_labels == 0
    B = cb_labels == 1
    ax.scatter(coords_2d[A, 0], coords_2d[A, 1], c=cluster_labels[A], cmap="Blues",
               s=14, alpha=0.6, linewidths=0)
    ax.scatter(coords_2d[B, 0], coords_2d[B, 1], c=cluster_labels[B], cmap="Reds",
               s=14, alpha=0.6, linewidths=0)
    for c in clusters:
        ma = A & (cluster_labels == c); mb = B & (cluster_labels == c)
        if ma.sum() == 0 or mb.sum() == 0:
            continue
        ca = coords_2d[ma].mean(0); cb = coords_2d[mb].mean(0)
        ax.annotate("", xy=cb, xytext=ca,
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
        for pt, lab in [(ca, f"A{c}"), (cb, f"B{c}")]:
            ax.text(pt[0], pt[1], lab, fontsize=6, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", ec="black", fc="white", lw=0.6))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Topic clusters & drift (2-D UMAP display; field computed in full space)")
    return fig


def plot_existence(res, figsize=(8, 3.5)):
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize, layout="constrained")
    axes[0].hist(res["m_null"], bins=30, color="#a0aec0", alpha=0.9)
    axes[0].axvline(res["m_obs"], color="#e53e3e", lw=2,
                    label=f"observed (p={res['p_magnitude']:.4f})")
    axes[0].set_title("Drift magnitude vs random-CB null")
    axes[0].set_xlabel(r"$\bar m$"); axes[0].legend(fontsize=8, frameon=False)
    s_null = res["s_null"][~np.isnan(res["s_null"])]
    axes[1].hist(s_null, bins=30, color="#a0aec0", alpha=0.9)
    if not np.isnan(res["s_obs"]):
        axes[1].axvline(res["s_obs"], color="#3182ce", lw=2, label="observed")
    axes[1].set_title("Pairwise cosine vs null")
    axes[1].set_xlabel(r"$\bar s$"); axes[1].legend(fontsize=8, frameon=False)
    return fig


def plot_dimensionality(res, figsize=(8, 3.5)):
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize, layout="constrained")
    # PR observed vs null
    axes[0].hist(res["pr_null"], bins=30, color="#a0aec0", alpha=0.9, label="null")
    axes[0].axvline(res["pr_obs"], color="#e53e3e", lw=2,
                    label=f"observed PR={res['pr_obs']:.2f}\n(p={res['p_multidimensional']:.4f})")
    axes[0].axvspan(*res["pr_ci"], color="#cbd5e0", alpha=0.5)
    axes[0].set_title("Participation ratio vs null")
    axes[0].set_xlabel("effective #polarity dimensions")
    axes[0].legend(fontsize=8, frameon=False)
    # spectrum with null band
    spec = res["spectrum_obs"]; x = np.arange(1, len(spec) + 1)
    axes[1].plot(x, spec, "o-", color="#2b6cb0", ms=4, label="observed")
    lo, hi = res["spectrum_null_lo"], res["spectrum_null_hi"]
    xm = np.arange(1, len(res["spectrum_null_mean"]) + 1)
    axes[1].fill_between(xm, lo[:len(xm)], hi[:len(xm)], color="#cbd5e0", alpha=0.6,
                         label="null 95%")
    axes[1].set_title("Field spectrum (variance explained)")
    axes[1].set_xlabel("component"); axes[1].set_ylabel("fraction")
    axes[1].legend(fontsize=8, frameon=False)
    return fig


def plot_vector_relations(V, cluster_ids, ax=None, title=None, figsize=(6, 5.5)):
    """Pairwise cosine similarity between per-topic polarity vectors: how aligned,
    anti-aligned or orthogonal the topics' A->B drifts are to each other. This is
    the direct, quantitative view of "the field of vectors" that `s_bar` (a single
    mean) collapses away."""
    _apply_style()
    fig = ax.figure if ax is not None else plt.subplots(figsize=figsize, layout="constrained")[0]
    ax = ax if ax is not None else fig.axes[0]
    V = np.asarray(V, dtype=float)
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    S = Vn @ Vn.T
    im = ax.imshow(S, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(cluster_ids))); ax.set_xticklabels(cluster_ids, fontsize=7, rotation=90)
    ax.set_yticks(range(len(cluster_ids))); ax.set_yticklabels(cluster_ids, fontsize=7)
    ax.set_title(title or "Polarity-vector cosine similarity (topic x topic)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="cosine")
    return fig


def plot_vector_compass(V, cluster_ids, axes2d, magnitudes=None, ax=None, title=None, figsize=(6, 5.5)):
    """Polarity vectors as arrows from a common origin, projected onto their own
    top-2 principal axes (`axes2d`, UNcentered -- same convention as the
    participation-ratio test in dimensionality.py: a field where every topic
    shares one direction is the 1-D case, so centring would hide exactly that).
    Angle between arrows = how the topics' drifts relate; length = magnitude of
    that drift in the 2-D projection (a lower bound on the true full-space norm,
    shown alongside the exact value in `magnitudes`).
    """
    _apply_style()
    fig = ax.figure if ax is not None else plt.subplots(figsize=figsize, layout="constrained")[0]
    ax = ax if ax is not None else fig.axes[0]
    V = np.asarray(V, dtype=float)
    axes2d = np.asarray(axes2d, dtype=float)[:2]
    proj = V @ axes2d.T                                    # (k, 2)
    mags = np.asarray(magnitudes) if magnitudes is not None else np.linalg.norm(V, axis=1)
    rng = (mags.max() - mags.min()) or 1.0
    norm_mag = (mags - mags.min()) / rng
    cmap = plt.get_cmap("viridis")
    for i, c in enumerate(cluster_ids):
        ax.annotate("", xy=proj[i], xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=cmap(norm_mag[i]), lw=1.8))
        ax.text(proj[i, 0], proj[i, 1], str(c), fontsize=7, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", ec="none", fc="white", alpha=0.75))
    lim = (np.abs(proj).max() * 1.25) if len(proj) else 1.0
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("principal polarity axis 0"); ax.set_ylabel("principal polarity axis 1")
    ax.set_title(title or "Polarity-vector field (2-D projection; colour = magnitude)")
    return fig


def plot_pole_rmse_contrast(df, ax=None, title=None, figsize=(7, 4.5)):
    """AR (activation-reconstruction) fidelity, regime A vs regime B, per
    topic. `df` is `nla.pole_contrast_table`'s output. A topic where the two
    bars differ a lot means the NLA/AR bottleneck represents one regime's
    documents more faithfully than the other's -- itself a finding, not just
    a fidelity caveat."""
    _apply_style()
    fig = ax.figure if ax is not None else plt.subplots(figsize=figsize, layout="constrained")[0]
    ax = ax if ax is not None else fig.axes[0]
    x = np.arange(len(df))
    width = 0.35
    if "rmse_A" in df:
        ax.bar(x - width / 2, df["rmse_A"], width, label="regime A", color="#3182ce")
    if "rmse_B" in df:
        ax.bar(x + width / 2, df["rmse_B"], width, label="regime B", color="#e53e3e")
    ax.set_xticks(x); ax.set_xticklabels(df["cluster"], fontsize=8)
    ax.set_xlabel("topic"); ax.set_ylabel("AR reconstruction RMSE")
    ax.set_title(title or "AR fidelity by regime, per topic")
    ax.legend(fontsize=8, frameon=False)
    return fig


def plot_pole_direction_summary(df, ax=None, title=None, max_chars=55, figsize=(14, None)):
    """One row per topic: pole-A representative words, three decoded sweep
    snapshots (just beyond A, midpoint, just beyond B) showing how the
    description drifts along the pole-A -> pole-B direction, pole-B
    representative words. `df` is `nla.pole_direction_summary`'s output.

    This is text, not a numeric plot -- rendered as a table (matplotlib's
    `ax.table`) so it saves through the same `store.save_fig` path as every
    other figure in this pipeline, rather than being a print statement no one
    can drop into a paper/slide.
    """
    _apply_style()

    def _trunc(s):
        s = str(s)
        return s if len(s) <= max_chars else s[: max_chars - 1] + "…"

    cols = ["cluster", "pole_A_words", "beyond_A", "midpoint", "beyond_B", "pole_B_words"]
    headers = ["topic", "pole A words", "← beyond A", "midpoint A↔B", "beyond B →", "pole B words"]
    cell_text = [[_trunc(row[c]) for c in cols] for _, row in df.iterrows()]

    fig_h = figsize[1] if figsize[1] is not None else 0.65 + 0.45 * max(len(df), 1)
    fig = ax.figure if ax is not None else plt.subplots(figsize=(figsize[0], fig_h), layout="constrained")[0]
    ax = ax if ax is not None else fig.axes[0]
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=headers, loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.6)
    for (row, _col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#e2e8f0")
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#edf2f7")
    ax.set_title(title or "Pole A -> B semantic drift, per topic", fontsize=11, pad=10)
    return fig


def plot_sae_decomposition(decomp, title="Polarity vector decomposition", figsize=(7.5, 4.5)):
    _apply_style()
    feats = decomp["features"]
    labels = [f"{f['feature']}: {str(f['label'])[:46]}" for f in feats]
    g = np.array([f["g"] if "g" in f else f.get("alignment", 0.0) for f in feats])
    order = np.argsort(g)
    g, labels = g[order], [labels[i] for i in order]
    colors = ["#3182ce" if v < 0 else "#e53e3e" for v in g]  # A pole blue, B pole red
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.barh(range(len(g)), g, color=colors)
    ax.set_yticks(range(len(g))); ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="black", lw=0.8)
    sub = ""
    if "coverage" in decomp:
        sub = f"  |  coverage={decomp['coverage']:.2f}, SAE fidelity cos={decomp.get('fidelity_cos', float('nan')):.2f}"
    ax.set_title(title + sub)
    ax.set_xlabel(r"alignment-weighted activation change $g_f$  (<0: A pole, >0: B pole)")
    return fig


def plot_steering_curve(curve, figsize=(7, 4), keys=("prob_B", "proj_target", "proj_w", "valence")):
    _apply_style()
    rows = curve["rows"]; alphas = curve["alphas"]
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    avail = [k for k in keys if k in rows[0]]
    for k in avail:
        y = [r[k] for r in rows]
        # scale proj/valence to [0,1]-ish for co-plotting readability
        y = np.array(y, dtype=float)
        if k != "prob_B":
            rng = (y.max() - y.min()) or 1.0
            y = (y - y.min()) / rng
        ax.plot(alphas, y, "o-", ms=4, label=k)
    ax.axvline(0, color="grey", ls=":", lw=1)
    slope = curve.get("prob_B_slope")
    ttl = "Steering curve (sufficiency)"
    if slope is not None:
        ttl += f"  |  prob_B slope={slope:+.3f}, align(v,w)={curve['align_v_w']:+.2f}"
    ax.set_title(ttl)
    ax.set_xlabel(r"steering strength $\alpha$")
    ax.set_ylabel("panel score (normalised)")
    ax.legend(fontsize=8, frameon=False)
    return fig


def plot_ablation(res, figsize=(5.5, 3.5)):
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    names = ["full", "ablate v", "ablate random"]
    vals = [res["acc_full"], res["acc_ablate_direction"], res["acc_ablate_random"]]
    ax.bar(names, vals, color=["#2b6cb0", "#e53e3e", "#a0aec0"])
    ax.axhline(0.5, ls=":", color="grey", label="chance")
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("A/B probe balanced acc.")
    ax.set_title(f"Ablation (necessity): drop={res['necessity_drop']:.3f} vs control {res['control_drop']:.3f}")
    ax.legend(fontsize=8, frameon=False)
    return fig


def plot_predictive_rank_curve(curve, null_perm=None, spherical=None,
                               r_star=None, figsize=(7, 4.2)):
    """Leave-one-topic-out subspace-capture curve S(r).

    Read it against BOTH nulls. The closed-form random-subspace null sqrt(r/m) is
    near zero for large m, so beating it is necessary but weak; the permutation
    null is the real bar. A curve that saturates early = few shared axes. A curve
    that hugs the null at every rank = spectral spread with no shared structure,
    i.e. per-topic noise -- the case the participation ratio cannot detect.
    """
    _apply_style()
    r = np.asarray(curve["ranks"])
    S = np.asarray(curve["S"])
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    ax.errorbar(r, S, yerr=curve.get("S_sem"), marker="o", ms=5, lw=1.8,
                capsize=2, color="#2b6cb0", label="observed (Euclidean)")
    if spherical is not None:
        ax.plot(np.asarray(spherical["ranks"]), np.asarray(spherical["S"]),
                "s--", ms=4, lw=1.4, color="#805ad5",
                label="canonicalised tangent (RISE)")
    ax.plot(r, curve["S_null"], ":", lw=1.4, color="#718096",
            label=r"random-subspace null $\sqrt{r/m}$")
    if null_perm is not None and null_perm.get("S_perm_mean") is not None:
        mp = np.asarray(null_perm["S_perm_mean"])[:len(r)]
        lo = np.asarray(null_perm["S_perm_lo"])[:len(r)]
        hi = np.asarray(null_perm["S_perm_hi"])[:len(r)]
        ax.plot(r[:len(mp)], mp, "-", lw=1.2, color="#e53e3e", label="permutation null")
        ax.fill_between(r[:len(lo)], lo, hi, color="#feb2b2", alpha=0.45)
    if r_star is not None and not np.isnan(r_star):
        ax.axvline(r_star, ls="--", lw=1, color="#38a169",
                   label=f"$r^*$ = {int(r_star)}")

    ax.set_xlabel("rank $r$ of training subspace (fit on all other topics)")
    ax.set_ylabel(r"held-out capture $\|W_r^\top v_i\|/\|v_i\|$")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(r)
    ax.set_title("Leave-one-topic-out predictive rank curve")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    return fig


def plot_route_agreement(df, figsize=(7, 4)):
    """Per-topic agreement between the SAE and concept-projection routes."""
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    x = np.arange(len(df))
    ax.bar(x - 0.27, df["agreement_cos"], width=0.27, label="SAE vs concept", color="#2b6cb0")
    ax.bar(x, df["sae_vs_v"], width=0.27, label="SAE vs $v_i$", color="#e53e3e")
    ax.bar(x + 0.27, df["concept_vs_v"], width=0.27, label="concept vs $v_i$", color="#38a169")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(df["cluster"], fontsize=8)
    ax.set_xlabel("topic"); ax.set_ylabel("cosine")
    ax.set_title("Interpretation-route agreement (geometric, label-free)")
    ax.legend(fontsize=8, frameon=False)
    return fig


def plot_affective_residual(res, sweep=None, ranks=None, figsize=(9, 3.8)):
    """Affective residual test: what survives after removing emotion?

    Left panel: energy accounted for by the affective subspace vs a rank-matched
    random subspace. The gap is the specificity of the affective basis.
    Right panel: participation ratio of the full field, of the residual, of the
    random-control residual, and the permutation null band for the residual.
    """
    _apply_style()
    ncol = 3 if sweep else 2
    fig, axes = plt.subplots(1, ncol, figsize=(figsize[0] * ncol / 2, figsize[1]),
                             layout="constrained")

    # --- energy ------------------------------------------------------- #
    ax = axes[0]
    ax.bar(["affective", "random\n(rank-matched)"],
           [res["affective_fraction"], res["control_affective_fraction_mean"]],
           yerr=[0, res["control_affective_fraction_sd"]],
           color=["#805ad5", "#a0aec0"], capsize=3)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of field energy removed")
    ax.set_title(f"Energy in the affective subspace (rank {res['rank_affective']})\n"
                 f"specificity = {res['specificity']:+.3f}")

    # --- dimensionality ----------------------------------------------- #
    ax = axes[1]
    names = ["full\nfield", "residual\n(affective out)", "residual\n(random out)"]
    vals = [res["pr_full"], res["pr_residual"], res["control_pr_residual_mean"]]
    ax.bar(names, vals, color=["#2b6cb0", "#e53e3e", "#a0aec0"],
           yerr=[0, 0, res["control_pr_residual_sd"]], capsize=3)
    lo, hi = res.get("pr_residual_null_lo"), res.get("pr_residual_null_hi")
    if lo is not None and not np.isnan(lo):
        ax.axhspan(lo, hi, color="#fed7d7", alpha=0.55, zorder=0,
                   label="residual permutation null 95%")
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.axhline(1.0, ls=":", color="grey", lw=1)
    ax.set_ylabel("participation ratio")
    ax.set_title(f"Dimensionality after projection\n"
                 f"p(residual structure) = {res['p_structure_residual']:.4f}")

    # --- rank sweep ---------------------------------------------------- #
    if sweep:
        ax = axes[2]
        r = ranks if ranks is not None else list(range(1, len(sweep) + 1))
        ax.plot(r, [s["pr_residual"] for s in sweep], "o-", color="#e53e3e",
                label="residual PR (affective out)")
        ax.plot(r, [s["control_pr_residual_mean"] for s in sweep], "s--",
                color="#a0aec0", label="residual PR (random out)")
        ax.axhline(1.0, ls=":", color="grey", lw=1)
        ax.set_xlabel("rank of affective subspace")
        ax.set_ylabel("participation ratio")
        ax.set_title("Rank sweep\n(claim must survive at small rank)")
        ax.legend(fontsize=7, frameon=False)
    return fig


# ==================================================================== #
#  V1 axis discovery: activation caching / discovery / gate / field /
#  baseline comparison (topol_activations.py, topol_axes.py, topol_gate.py,
#  field.py's pooled_field_assembly). Additive -- every plot above is
#  unchanged.
# ==================================================================== #
def plot_activation_exclusion(per_doc_excluded, figsize=(6, 3.5)):
    """Histogram of how many pooled-token positions were excluded (BOS/
    special/chat-template) per document when caching activations
    (`topol_activations.cache_activations`). Mostly a sanity check: for the
    -pt field model on plain review text this should be a tight spike at 1
    (just the BOS token) -- a long tail would mean something unexpected
    (e.g. truncation padding) is being swept into "excluded" too."""
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    vals = np.asarray(per_doc_excluded)
    bins = max(1, int(vals.max()) + 1) if len(vals) else 1
    ax.hist(vals, bins=np.arange(bins + 1) - 0.5, color="#2b6cb0", alpha=0.85)
    ax.set_xlabel("excluded (BOS/special/chat-template) tokens per document")
    ax.set_ylabel("# documents")
    ax.set_title("Activation caching: pooled-token exclusion per document")
    return fig


def plot_depth_profiles(profiles, title=None, figsize=(7, 4.5)):
    """d(l) vs layer for one or more discovered contrasts (`topol_axes
    .depth_profile`'s output), each marked at its own l*.

    `profiles` : list of {"label": str, "layers": [...], "d": [...],
    "l_star": int} -- build one entry per axis you want to compare on the
    same plot (e.g. every gate-passing axis in one topic, or a topic's axes
    next to its CB baseline profile).
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    cmap = plt.get_cmap("tab10")
    for i, p in enumerate(profiles):
        color = cmap(i % 10)
        ax.plot(p["layers"], p["d"], "o-", color=color, ms=4, lw=1.5, label=p["label"])
        if p.get("l_star") is not None and p["l_star"] in p["layers"]:
            j = p["layers"].index(p["l_star"])
            ax.scatter([p["l_star"]], [p["d"][j]], color=color, s=90, zorder=5,
                      edgecolors="black", linewidths=0.8)
    ax.set_xlabel("cached residual layer")
    ax.set_ylabel(r"effect size  $d(\ell) = \|v(\ell)\| / \sigma_{pooled}(\ell)$")
    ax.set_title(title or "Depth profile (marker = l*)")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    return fig


def plot_gate_stability(match_cos_distribution, threshold, figsize=(6, 3.5)):
    """Distribution of split-half |cos| matches (`topol_gate.gate_topic`'s
    `match_cos_distribution`), with `stability_threshold` marked -- exactly
    the distribution the spec asks to log "so the threshold can be
    revisited empirically" rather than trusting the default 0.6 blindly."""
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    vals = np.asarray(match_cos_distribution, dtype=float)
    ax.hist(vals, bins=20, range=(0, 1), color="#a0aec0", alpha=0.9)
    ax.axvline(threshold, color="#e53e3e", lw=2,
              label=f"stability_threshold={threshold:.2f}")
    n_pass = int((vals >= threshold).sum())
    ax.set_xlabel("split-half |cos| match")
    ax.set_ylabel("# matched contrast pairs")
    ax.set_title(f"Gate stability distribution ({n_pass}/{len(vals)} pairs pass)")
    ax.legend(fontsize=8, frameon=False)
    return fig


def plot_baseline_layer_agreement(baseline_df, figsize=(7, 4.5)):
    """Per topic: CB-matched axis's l* vs every excess axis's l* (spec step 8's
    "layer agreement" -- the direct test of whether excess structure lives at
    a different depth than the CB/sentiment axis). `baseline_df` is one row
    per topic with columns `topic`, `cb_l_star`, `identification_l_star`,
    `excess_axis_l_stars` (list-like), as built by
    `pipeline.step_topol_baseline_compare`'s per-topic dict."""
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    for _, row in baseline_df.iterrows():
        t = row["topic"]
        ax.scatter([t], [row["cb_l_star"]], color="#2b6cb0", marker="s", s=70, zorder=5,
                  label="CB l*" if t == baseline_df["topic"].iloc[0] else None)
        ax.scatter([t], [row["identification_l_star"]], color="#38a169", marker="o", s=70,
                  zorder=5, label="identification l*" if t == baseline_df["topic"].iloc[0] else None)
        excess = row["excess_axis_l_stars"]
        if len(excess):
            ax.scatter([t] * len(excess), excess, color="#e53e3e", marker="x", s=60,
                      zorder=4, label="excess axis l*" if t == baseline_df["topic"].iloc[0] else None)
    ax.set_xlabel("topic")
    ax.set_ylabel("layer")
    ax.set_title("Layer agreement: CB axis vs identification vs excess axes")
    ax.legend(fontsize=8, frameon=False)
    return fig


def plot_sentiment_alignment(baseline_df, sentiment_corr_threshold=None,
                             excess_cos_threshold=None, figsize=(7, 4.5)):
    """Per topic: |corr(projection, continuous rating)| for v_CB^t itself,
    every gate-passing discovered axis, and which of those is the
    cosine-identification match -- the direct, measured replacement for the
    old "identification should land at recursion depth 1" assumption
    (`pipeline.step_topol_baseline_compare`'s module-level docstring explains
    why that assumption was dropped).

    `baseline_df` : one row per topic, as built by
    `step_topol_baseline_compare`'s per-topic dict -- needs `topic`,
    `cb_corr_rating`, `identification_cos`, `identification_corr_rating`,
    `passing_axis_abs_corr_rating` (list-like) and
    `identification_matches_most_sentiment_aligned`.

    FALLBACK: `corr_rating` is only ever real when a continuous rating was
    recovered for the run (`data.recover_ratings`) -- a non-Amazon dataset,
    or an Amazon prep file that never carried a `rating` column (only a
    precomputed binary `label`), leaves `cb_corr_rating`/`identification_
    corr_rating`/every `passing_axis_abs_corr_rating` entry NaN for every
    topic, which used to render this figure as an empty axes box (nothing
    ever satisfied `not np.isnan(...)`). When that's the case for the WHOLE
    `baseline_df`, this now plots `|cos(axis, v_CB)|` instead (`identification
    _cos` / `passing_axis_cos_to_cb`, always populated once discovery has run
    -- direction agreement with v_CB, not sentiment correlation) so the
    figure still shows a real v_CB-vs-identification-vs-passing-axes
    comparison rather than nothing.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    if not len(baseline_df):
        ax.set_title("Sentiment alignment: v_CB vs identification vs all passing axes")
        return fig

    has_rating_data = (
        baseline_df["cb_corr_rating"].apply(lambda v: v is not None and not np.isnan(v)).any()
        or baseline_df["identification_corr_rating"].apply(
            lambda v: v is not None and not np.isnan(v)).any()
        or baseline_df["passing_axis_abs_corr_rating"].apply(
            lambda lst: any(not np.isnan(c) for c in lst)).any()
    )

    first = baseline_df["topic"].iloc[0]
    if has_rating_data:
        for _, row in baseline_df.iterrows():
            t = row["topic"]
            others = [c for c in row["passing_axis_abs_corr_rating"] if not np.isnan(c)]
            if others:
                ax.scatter([t] * len(others), others, color="#a0aec0", marker=".", s=40,
                          zorder=3, label="passing axis |r|" if t == first else None)
            cb_r = row["cb_corr_rating"]
            if not np.isnan(cb_r):
                ax.scatter([t], [abs(cb_r)], color="#2b6cb0", marker="s", s=70, zorder=5,
                          label="v_CB |r|" if t == first else None)
            ident_r = row.get("identification_corr_rating")
            if ident_r is not None and not np.isnan(ident_r):
                match = row.get("identification_matches_most_sentiment_aligned")
                color = "#38a169" if match else "#e53e3e"
                ax.scatter([t], [abs(ident_r)], color=color, marker="o", s=70, zorder=4,
                          label=None)
        if sentiment_corr_threshold is not None:
            ax.axhline(sentiment_corr_threshold, ls="--", color="grey", lw=1,
                      label=f"sentiment_corr_threshold={sentiment_corr_threshold}")
        ax.scatter([], [], color="#38a169", marker="o", s=70,
                  label="identification axis |r| (matches most sentiment-aligned)")
        ax.scatter([], [], color="#e53e3e", marker="o", s=70,
                  label="identification axis |r| (does NOT match)")
        ax.set_ylabel("|corr(projection, rating)|")
        ax.set_title("Sentiment alignment: v_CB vs identification vs all passing axes")
    else:
        for _, row in baseline_df.iterrows():
            t = row["topic"]
            others = [c for c in row.get("passing_axis_cos_to_cb", []) if not np.isnan(c)]
            if others:
                ax.scatter([t] * len(others), others, color="#a0aec0", marker=".", s=40,
                          zorder=3, label="passing axis |cos to v_CB|" if t == first else None)
            ident_cos = row.get("identification_cos")
            if ident_cos is not None and not np.isnan(ident_cos):
                ax.scatter([t], [ident_cos], color="#2b6cb0", marker="o", s=70, zorder=5,
                          label="identification |cos to v_CB|" if t == first else None)
        if excess_cos_threshold is not None:
            ax.axhline(excess_cos_threshold, ls="--", color="grey", lw=1,
                      label=f"excess_cos_threshold={excess_cos_threshold}")
        ax.set_ylabel("|cos(axis, v_CB)|")
        ax.set_title("Direction alignment: identification vs all passing axes vs v_CB\n"
                    "(no continuous rating recovered this run -- showing cosine, not "
                    "sentiment correlation)")
    ax.set_xlabel("topic")
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return fig


def plot_pooled_spectrum(spectrum, pr, figsize=(6, 3.8)):
    """Eigen-energy spectrum of the pooled, cross-topic gate-passing-axis
    ensemble (`field.pooled_field_assembly`), uncentred second moment --
    same convention as `plot_dimensionality`'s right panel, but for a
    DIFFERENT ensemble (axes from different topics, not per-topic CB
    vectors) where PR is meaningful precisely because those axes are not
    constructed orthogonal to one another (see `field.topic_rank_estimate`
    for the within-topic case, where PR would NOT be meaningful)."""
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    spec = np.asarray(spectrum, dtype=float)
    x = np.arange(1, len(spec) + 1)
    ax.bar(x, spec, color="#805ad5", alpha=0.85)
    ax.set_xlabel("component"); ax.set_ylabel("fraction of variance")
    ax.set_title(f"Pooled cross-topic axis field spectrum  (PR={pr:.2f}, k={len(spec)})")
    return fig


# ==================================================================== #
#  Part 2 additions: pole vocabulary + sentiment-direction alignment
# ==================================================================== #
def plot_pole_wordclouds(freqs_A: dict, freqs_B: dict, title=None, figsize=(9, 4.2)):
    """Two-panel word cloud, pole A (blues) vs pole B (reds) -- same
    colour convention as `plot_field_2d`/`plot_pole_rmse_contrast`. `freqs_A`/
    `freqs_B` are {term: score} dicts from
    `pole_vocab.term_frequencies_for_wordcloud` (score = mean-TF-IDF
    difference in favour of that pole)."""
    from wordcloud import WordCloud

    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize, layout="constrained")
    for ax, freqs, pole, cmap in ((axes[0], freqs_A, "A", "Blues"), (axes[1], freqs_B, "B", "Reds")):
        ax.set_title(f"pole {pole}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if not freqs:
            ax.text(0.5, 0.5, "(no distinctive terms)", ha="center", va="center", fontsize=8)
            continue
        wc = WordCloud(width=500, height=400, background_color="white",
                       colormap=cmap, prefer_horizontal=0.9).generate_from_frequencies(freqs)
        ax.imshow(wc, interpolation="bilinear")
    fig.suptitle(title or "Pole-distinctive vocabulary (TF-IDF)", fontsize=11)
    return fig


def plot_direction_sentiment_alignment(df, metric_col: str, metric_label: str,
                                       r=None, p=None, figsize=(6.5, 5)):
    """Scatter: |cos(axis, v_CB^t)| (x) vs |sentiment-alignment metric| (y),
    one point per gate-passing axis, coloured by topic -- the direct,
    quantitative view of the hypothesis "an axis's direction similarity to
    v_CB^t predicts how sentiment-aligned it is". `metric_col` is an
    `abs_...` column of `pole_sentiment.build_alignment_table`'s output
    (`abs_corr_rating` or `abs_sentiment_delta`); `r`/`p` (from
    `pole_sentiment.test_alignment_hypothesis`) are annotated if given."""
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    sub = df.dropna(subset=["abs_cos_to_cb", metric_col])
    topics = sorted(sub["topic"].unique())
    cmap = plt.get_cmap("tab20")
    for i, t in enumerate(topics):
        s = sub[sub["topic"] == t]
        ax.scatter(s["abs_cos_to_cb"], s[metric_col], color=cmap(i % 20), s=22,
                  alpha=0.8, label=f"topic {t}")
    if len(sub) >= 2:
        m, b = np.polyfit(sub["abs_cos_to_cb"], sub[metric_col], 1)
        xs = np.linspace(sub["abs_cos_to_cb"].min(), sub["abs_cos_to_cb"].max(), 50)
        ax.plot(xs, m * xs + b, color="black", lw=1.5, ls="--", zorder=5)
    if r is not None and p is not None:
        ax.text(0.03, 0.97, f"pearson r={r:.3f}, p={p:.4g}\nn={len(sub)}",
               transform=ax.transAxes, ha="left", va="top", fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", ec="grey", fc="white", alpha=0.85))
    ax.set_xlabel("|cos(axis, v_CB^t)|  --  direction similarity to CB")
    ax.set_ylabel(f"|{metric_label}|  --  sentiment alignment")
    ax.set_title("Direction similarity to v_CB^t vs. sentiment alignment (all gate-passing axes)")
    ax.legend(fontsize=6, frameon=False, ncol=2, loc="lower right")
    return fig


# ==================================================================== #
#  causal.ipynb -- ablation before/after
# ==================================================================== #
def plot_umap_before_after(coords_before, coords_after, color_labels, label_name="topic",
                           title=None, figsize=(11, 5)):
    """Side-by-side 2-D UMAP layout, same colouring both panels, before vs.
    after ablation -- `coords_before`/`coords_after` should come from
    re-running `manifold.cluster_documents` on the pre-/post-ablation
    embedding (same `cfg.manifold` settings both times); `color_labels`
    (e.g. the ORIGINAL topic ids, held fixed) makes it a direct visual read
    of how much the manifold reorganises once the identified directions are
    gone, not just a color-shuffle."""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize, layout="constrained")
    color_labels = np.asarray(color_labels)
    cmap = plt.get_cmap("tab20")
    norm = matplotlib.colors.Normalize(vmin=color_labels.min(), vmax=color_labels.max())
    for ax, coords, sub in ((axes[0], coords_before, "before ablation"),
                            (axes[1], coords_after, "after ablation")):
        ax.scatter(coords[:, 0], coords[:, 1], c=color_labels, cmap=cmap, norm=norm,
                  s=10, alpha=0.6, linewidths=0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(sub)
    fig.suptitle(title or f"UMAP projection, coloured by {label_name} (fixed across both panels)")
    return fig


def plot_cluster_agreement(labels_before, labels_after, ari=None, figsize=(6, 5.5)):
    """Contingency heatmap: original topic (rows) vs. topic recovered after
    ablation + re-clustering (columns), row-normalised -- a diagonal-heavy
    matrix means topic identity survives ablation (topic structure was not
    riding mainly on the ablated directions); a smeared matrix means much of
    it was. `ari` (adjusted Rand index, `sklearn.metrics.adjusted_rand_score`)
    is the single-number summary, annotated in the title if given."""
    _apply_style()
    labels_before = np.asarray(labels_before)
    labels_after = np.asarray(labels_after)
    rows = np.unique(labels_before)
    cols = np.unique(labels_after)
    M = np.zeros((len(rows), len(cols)))
    for i, r in enumerate(rows):
        mask = labels_before == r
        n = mask.sum()
        for j, c in enumerate(cols):
            M[i, j] = (labels_after[mask] == c).sum() / n if n else 0.0
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=7)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=7)
    ax.set_xlabel("topic after ablation + re-clustering"); ax.set_ylabel("original topic")
    ttl = "Topic agreement, before vs. after ablation"
    if ari is not None:
        ttl += f"  (ARI={ari:.3f})"
    ax.set_title(ttl)
    fig.colorbar(im, ax=ax, shrink=0.8, label="row fraction")
    return fig


def plot_cb_magnitude_before_after(topics, magnitude_before, magnitude_after,
                                   ablated_topics=None, figsize=(8, 4)):
    """Per topic: ||v_CB^t|| before vs. after ablation -- the direct sanity
    check for topics where v_CB^t was itself one of the ablated directions
    (bar should collapse towards 0 there; `ablated_topics` marks which ones
    to expect that for for)."""
    _apply_style()
    topics = np.asarray(topics)
    x = np.arange(len(topics))
    w = 0.38
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.bar(x - w / 2, magnitude_before, width=w, color="#2b6cb0", label="before ablation")
    ax.bar(x + w / 2, magnitude_after, width=w, color="#e53e3e", label="after ablation")
    if ablated_topics is not None:
        ablated = set(int(t) for t in ablated_topics)
        for i, t in enumerate(topics):
            if int(t) in ablated:
                ax.text(i, max(magnitude_before[i], magnitude_after[i]) * 1.02, "*",
                       ha="center", va="bottom", fontsize=12, color="black")
    ax.set_xticks(x); ax.set_xticklabels(topics, fontsize=7)
    ax.set_xlabel("topic"); ax.set_ylabel(r"$||v_{CB}^t||$")
    ax.set_title(r"$v_{CB}^t$ magnitude before vs. after ablation  (* = v_CB itself was ablated)")
    ax.legend(fontsize=8, frameon=False)
    return fig


def plot_excess_axis_retention(df, figsize=(6.5, 4.5)):
    """Per EXCESS (non-ablated) gate-passing axis: `d_retained_frac` --
    `causal_ablation.excess_axis_effect_size_before_after`'s output. Points
    near 1.0 = that axis's own discriminative effect size survived ablation
    intact (real independence from the removed directions); points near 0 =
    it collapsed too (the low `cos_to_cb` that spared it from selection
    understated how entangled it actually was)."""
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    sub = df.dropna(subset=["d_retained_frac"])
    ax.scatter(sub["d_before"], sub["d_retained_frac"], color="#805ad5", s=24, alpha=0.75)
    ax.axhline(1.0, ls=":", color="grey", lw=1, label="fully retained")
    ax.axhline(0.0, ls=":", color="#e53e3e", lw=1, label="fully collapsed")
    ax.set_xlabel("effect size d, before ablation (own l*-independent, at ablation layer)")
    ax.set_ylabel("d_after / d_before")
    ax.set_title(f"Excess-axis effect size retained after ablation (n={len(sub)})")
    ax.legend(fontsize=8, frameon=False)
    return fig
