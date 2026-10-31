"""Layer selection.

New step (not in the original pipeline): choose the residual layer on which the
polarity field is built by which layer's activations best separate regimes A and B
under a cross-validated linear probe.  This grounds the otherwise arbitrary choice
of "which embedding" and is a standard mechanistic-interpretability move.
"""
from __future__ import annotations

import numpy as np


def probe_separability_by_layer(acts_all_layers, labels, n_splits=5, C=1.0, random_state=42):
    """Balanced-accuracy of an L2 logistic probe for A/B at each layer.

    acts_all_layers : (N, L, d_model)
    labels          : (N,) in {0,1}
    Returns dict{layer_index: {"acc_mean","acc_std"}} and the argmax layer.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(labels).astype(int)
    N, L, d = acts_all_layers.shape
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    per_layer = {}
    for l in range(L):
        X = acts_all_layers[:, l, :]
        accs = []
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=C, max_iter=2000)
            clf.fit(sc.transform(X[tr]), y[tr])
            pred = clf.predict(sc.transform(X[te]))
            accs.append(balanced_accuracy_score(y[te], pred))
        per_layer[l] = {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs))}
    best = max(per_layer, key=lambda l: per_layer[l]["acc_mean"])
    return {"per_layer": per_layer, "best_layer": int(best),
            "best_acc": per_layer[best]["acc_mean"]}
