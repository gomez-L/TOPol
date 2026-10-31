"""Intrinsic dimensionality of the polarity field (the rigorous replacement for
the "s_bar = 0.12 vs 0.66" argument).

Motivation
----------
Average pairwise cosine `s_bar` conflates alignment strength with dimensionality
and reports no effective count with uncertainty.  Instead we analyse the spectrum
of the stacked polarity vectors:

    V  in  R^{k x d}          (rows = per-topic polarity vectors, full space)
    singular values  sigma_1 >= ... >= sigma_r,   eigen-energy  lambda_i = sigma_i^2

and summarise with the participation ratio

    PR(V) = (sum_i lambda_i)^2 / sum_i lambda_i^2   in  [1, k],

the effective number of independent polarity directions.  A permutation null
(random CB, topic composition preserved) gives a reference band, so we can state
"the field has PR = x (95% CI ...) vs null PR = y", turning a qualitative claim
into a tested, quantitative one.  For the sentiment-dominated case we expect
PR ~ 1; for genuinely multidimensional discourse, PR clearly above the null.

IMPORTANT - do NOT mean-centre.  The polarity vectors are anchored at a common
origin to form the field, and a purely 1-dimensional field is exactly the case
where every topic shifts the SAME way, i.e. the shared (mean) direction carries
all the signal.  Mean-centring subtracts that shared direction and would make a
1-D field look high-dimensional.  We therefore analyse the uncentered second-moment
spectrum (`center=False`), so PR counts distinct polarity directions and a shared
shift correctly yields PR ~ 1.
"""
from __future__ import annotations

import numpy as np

from .field import reconstruct_field


def participation_ratio(V, center=False):
    """Effective dimensionality of the row-vectors of V via the participation ratio."""
    V = np.asarray(V, dtype=float)
    if V.shape[0] < 2:
        return 1.0
    X = V - V.mean(0, keepdims=True) if center else V
    # eigen-energy = squared singular values of X
    s = np.linalg.svd(X, compute_uv=False)
    lam = s ** 2
    if lam.sum() <= 0:
        return 1.0
    return float((lam.sum() ** 2) / (np.sum(lam ** 2)))


def spectrum(V, center=False):
    """Return normalised eigen-energy (variance explained) per component."""
    V = np.asarray(V, dtype=float)
    X = V - V.mean(0, keepdims=True) if center else V
    s = np.linalg.svd(X, compute_uv=False)
    lam = s ** 2
    return lam / lam.sum() if lam.sum() > 0 else lam



def bootstrap_pr(embeddings, cluster_labels, cb_labels, n_boot=1000, random_state=42):
    """Document-level bootstrap CI for the participation ratio.

    Resamples DOCUMENTS with replacement inside each (topic, regime) cell and
    recomputes the centroids, so the interval reflects the real source of
    uncertainty: finitely many documents per cell.

    Do NOT bootstrap by resampling the polarity VECTORS (topic-level): duplicated
    vectors are perfectly collinear, which concentrates the spectrum and biases PR
    DOWNWARD.  The document-level bootstrap has the opposite, milder bias - resampling
    injects extra independent centroid noise, which disperses the vectors and biases
    PR slightly UPWARD.  Report the direction of the bias, and when a claim is
    one-sided prefer the bootstrap whose bias runs AGAINST the claim.
    """
    rng = np.random.default_rng(random_state)
    cluster_labels = np.asarray(cluster_labels)
    cb_labels = np.asarray(cb_labels)
    E = np.asarray(embeddings)
    cells = {}
    for c in np.unique(cluster_labels):
        for r in (0, 1):
            cells[(c, r)] = np.where((cluster_labels == c) & (cb_labels == r))[0]
    boots = []
    for _ in range(n_boot):
        rows = []
        for c in np.unique(cluster_labels):
            a, b = cells[(c, 0)], cells[(c, 1)]
            if len(a) == 0 or len(b) == 0:
                continue
            mA = E[rng.choice(a, len(a), replace=True)].mean(0)
            mB = E[rng.choice(b, len(b), replace=True)].mean(0)
            rows.append(mB - mA)
        if len(rows) >= 2:
            boots.append(participation_ratio(np.stack(rows)))
    boots = np.array(boots)
    return {
        "pr_boot_mean": float(boots.mean()) if len(boots) else float("nan"),
        "pr_boot_lo": float(np.percentile(boots, 2.5)) if len(boots) else float("nan"),
        "pr_boot_hi": float(np.percentile(boots, 97.5)) if len(boots) else float("nan"),
        "pr_boot": boots,
    }


def dimensionality_test(embeddings, cluster_labels, cb_labels,
                        n_permutations=1000, center=False, random_state=42):
    """Observed PR + spectrum vs a random-CB null band, with a p-value.

    Reporting protocol (three DISTINCT questions - do not conflate them):
      1. Does structure exist?   -> `p_structure` (lower tail vs the random-CB null).
      2. How many dimensions?    -> `pr_obs` itself, with a bootstrap CI (see
                                    `bootstrap_pr`); compare against 1, NOT against
                                    the null, since the null sits at the MAXIMUM (~k).
      3. Is corpus X richer than corpus Y? -> compare the two bootstrap distributions.
    """
    rng = np.random.default_rng(random_state)
    V_obs = reconstruct_field(embeddings, cluster_labels, cb_labels)["V"]
    pr_obs = participation_ratio(V_obs, center=center)
    spec_obs = spectrum(V_obs, center=center)

    pr_null, spec_null = [], []
    y = np.asarray(cb_labels)
    for _ in range(n_permutations):
        yp = rng.permutation(y)
        Vp = reconstruct_field(embeddings, cluster_labels, yp)["V"]
        pr_null.append(participation_ratio(Vp, center=center))
        s = spectrum(Vp, center=center)
        spec_null.append(s)
    pr_null = np.array(pr_null)

    # pad spectra to common length for a null band
    L = max([len(spec_obs)] + [len(s) for s in spec_null])
    def pad(s): 
        out = np.zeros(L); out[:len(s)] = s; return out
    spec_null_mat = np.stack([pad(s) for s in spec_null])
    # TWO one-sided tests.  Under the random-CB null each v_i is a difference of two
    # subsample means of the SAME topic distribution, i.e. noise; in d >> k dimensions
    # such vectors are near-orthogonal, so the null spectrum is FLAT and null PR ~ k
    # (its MAXIMUM).  Real structure therefore CONCENTRATES the spectrum and pushes PR
    # DOWN.  The informative test is the LOWER tail.
    p_structure = float((1 + np.sum(pr_null <= pr_obs)) / (n_permutations + 1))
    p_dispersion = float((1 + np.sum(pr_null >= pr_obs)) / (n_permutations + 1))

    return {
        "pr_obs": pr_obs,
        "pr_null": pr_null,
        "pr_null_mean": float(pr_null.mean()),
        "pr_ci": (float(np.percentile(pr_null, 2.5)), float(np.percentile(pr_null, 97.5))),
        "p_structure": p_structure,        # lower tail: field more concentrated than chance
        "p_dispersion": p_dispersion,      # upper tail: ~1 by construction, kept for completeness
        "p_multidimensional": p_structure, # backwards-compatible alias (now the CORRECT tail)
        "spectrum_obs": pad(spec_obs),
        "spectrum_null_mean": spec_null_mat.mean(0),
        "spectrum_null_hi": np.percentile(spec_null_mat, 97.5, axis=0),
        "spectrum_null_lo": np.percentile(spec_null_mat, 2.5, axis=0),
        "k": int(V_obs.shape[0]),
    }


def principal_axes(V, n_axes=None, center=False):
    """Right singular vectors of the (centred) field = the principal polarity axes.

    These are the directions to name (via SAE decomposition) and causally validate
    (via steering), replacing per-topic labelling with a compact field-level basis.
    Returns (axes (n_axes, d), energy (n_axes,)).
    """
    V = np.asarray(V, dtype=float)
    mean = V.mean(0, keepdims=True) if center else np.zeros((1, V.shape[1]))
    X = V - mean
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    lam = s ** 2
    energy = lam / lam.sum() if lam.sum() > 0 else lam
    if n_axes is not None:
        Vt, energy = Vt[:n_axes], energy[:n_axes]
    return Vt, energy
