"""Corpus loading and contextual-boundary construction.

FOUR DATASETS, TWO OF THEM REAL
-------------------------------
    us_speeches      US central bank speeches (BIS repository), CB at a
                     macroeconomic breakpoint.  Non-affective discourse: the
                     DISCOVERY condition.
    amazon_reviews   Product reviews, CB at a rating threshold.  Sentiment-
                     saturated: the KNOWN-ANSWER condition, used to calibrate.
    synthetic_1d     Planted one-dimensional field.  Ground truth PR = 1.
    synthetic_kd     Planted three-dimensional field.  Ground truth PR ~ 3.

The synthetic corpora are not a fallback -- they are how the estimators get
validated against an answer known by construction, and they caught four bugs that
produced entirely plausible output (field guide, Part IX).  Run them first, then
run the real ones.  But the real corpora are the point, and this module makes them
first-class rather than an afterthought.

THE CONTEXTUAL BOUNDARY IS A RESEARCH DECISION
----------------------------------------------
Every loader returns `labels` in {0, 1} = regime {A, B}, and every loader makes the
rule that produced them explicit and reportable.  We do not search over boundaries
to maximise the field: that would be circular (you would find the split that best
separates the corpus, then announce that the corpus separates there).  The helpers
below construct a CB from a stated rule -- a date, a rating threshold, a metadata
column -- and record it in the returned `meta` so it lands in the run manifest.

If you DO iterate on the boundary by inspecting the resulting field, the
permutation p-value is optimistic.  Report how many boundaries you examined, and
check final claims on held-out documents.
"""
from __future__ import annotations

import os
import pickle
import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
#  Generic IO
# ------------------------------------------------------------------ #
def load_pickle_df(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        df = pickle.load(f)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame in {path}, got {type(df)}")
    return df.reset_index(drop=True)


def _read_any(path: str) -> pd.DataFrame:
    """Load a DataFrame from .pickle/.pkl, .parquet, .csv, .tsv or .jsonl."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pickle", ".pkl"):
        return load_pickle_df(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".tsv", ".tab"):
        return pd.read_csv(path, sep="\t")
    if ext in (".jsonl", ".json"):
        return pd.read_json(path, lines=(ext == ".jsonl"))
    raise ValueError(f"unsupported extension {ext!r} for {path}")


# ------------------------------------------------------------------ #
#  Contextual-boundary constructors
# ------------------------------------------------------------------ #
def cb_from_date(dates, breakpoint, window_years=None):
    """CB from a temporal breakpoint.  Regime A = before, B = on/after.

    window_years : optional symmetric window around the breakpoint; rows outside
        it get NaN and are dropped.  A window guards against the boundary quietly
        becoming "old vs recent language" rather than "before vs after the event".
    """
    d = pd.to_datetime(pd.Series(list(dates)))
    bp = pd.Timestamp(breakpoint)
    lab = (d >= bp).astype(float)
    if window_years is not None:
        lo = bp - pd.DateOffset(years=window_years)
        hi = bp + pd.DateOffset(years=window_years)
        lab = lab.where((d >= lo) & (d <= hi), other=np.nan)
    return lab.to_numpy(), {"rule": "date", "breakpoint": str(bp.date()),
                            "window_years": window_years}


def cb_from_rating(ratings, negative_max=2, positive_min=4):
    """CB from a rating threshold.  A = low, B = high, middle band dropped.

    Dropping the middle is deliberate: it makes the boundary a contrast between
    clear poles rather than a cut through a continuum, which is what makes this
    corpus a usable KNOWN-ANSWER calibration case.
    """
    r = pd.Series(list(ratings)).astype(float)
    lab = pd.Series(np.nan, index=r.index)
    lab[r <= negative_max] = 0.0
    lab[r >= positive_min] = 1.0
    return lab.to_numpy(), {"rule": "rating", "negative_max": negative_max,
                            "positive_min": positive_min}


def cb_from_column(values, a_values, b_values):
    """CB from categorical metadata: speaker, institution, venue, era tag."""
    v = pd.Series(list(values)).astype(str)
    lab = pd.Series(np.nan, index=v.index)
    lab[v.isin([str(x) for x in a_values])] = 0.0
    lab[v.isin([str(x) for x in b_values])] = 1.0
    return lab.to_numpy(), {"rule": "column", "A": list(a_values), "B": list(b_values)}


def balance_regimes(df, label_col="label", n_per_class=None, random_state=42):
    """Equalise regime sizes.  Unbalanced cells bias centroid precision, and the
    permutation null conditions on cell sizes, so balancing keeps the observed
    field and its null on the same footing."""
    rng = np.random.default_rng(random_state)
    parts = []
    counts = df[label_col].value_counts()
    n = int(n_per_class or counts.min())
    for lab in sorted(df[label_col].dropna().unique()):
        sub = df[df[label_col] == lab]
        take = min(n, len(sub))
        idx = rng.choice(sub.index.to_numpy(), size=take, replace=False)
        parts.append(df.loc[np.sort(idx)])
    return (pd.concat(parts)
              .sample(frac=1.0, random_state=random_state)
              .reset_index(drop=True))


# ------------------------------------------------------------------ #
#  Real corpus 1: US central bank speeches  (DISCOVERY condition)
# ------------------------------------------------------------------ #
def load_us_speeches(path, breakpoint="2007-05-01", window_years=None,
                     min_chars=500, max_docs=None, balance=True, random_state=42):
    """US central bank speeches with a temporal contextual boundary.

    Expected columns: `text`, plus either `label` (already 0/1) or `date`.
    The reference corpus is a DataFrame with [date (datetime), text (str),
    label (float 0/1)].

    The default breakpoint (May 2007) precedes the Global Financial Crisis. In the
    reference work it was selected from a PCA-constructed macroeconomic regime
    index rather than by eye.  If you re-derive it, record how: the boundary is the
    one place a researcher's judgement enters the pipeline and it must be auditable.

    THIS IS THE DISCOVERY CONDITION.  Do not interpret its field until the Amazon
    calibration gate has passed.
    """
    df = _read_any(path)
    cols = {c.lower(): c for c in df.columns}
    tcol = cols.get("text")
    if tcol is None:
        raise ValueError(f"no `text` column in {path}; found {list(df.columns)}")
    df = df.rename(columns={tcol: "text"})

    meta = {"source": os.path.basename(path), "corpus": "us_speeches"}
    if "label" in cols and df[cols["label"]].notna().any():
        df["label"] = df[cols["label"]].astype(float)
        meta["rule"] = "precomputed label column"
    elif "date" in cols:
        lab, m = cb_from_date(df[cols["date"]], breakpoint, window_years)
        df["label"] = lab
        df["date"] = pd.to_datetime(df[cols["date"]])
        meta.update(m)
    else:
        raise ValueError("need either a `label` or a `date` column to define the CB")

    df = df[df["text"].astype(str).str.len() >= min_chars]
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    if balance:
        df = balance_regimes(df, n_per_class=(max_docs // 2 if max_docs else None),
                             random_state=random_state)
    elif max_docs:
        df = df.head(max_docs)

    meta.update({"n_docs": len(df), "min_chars": min_chars,
                 "n_A": int((df["label"] == 0).sum()),
                 "n_B": int((df["label"] == 1).sum())})
    if "date" in df.columns:
        meta["date_range"] = [str(df["date"].min()), str(df["date"].max())]
    return df["text"].tolist(), df["label"].to_numpy(float), None, meta


# ------------------------------------------------------------------ #
#  Real corpus 2: product reviews  (CALIBRATION condition)
# ------------------------------------------------------------------ #
def load_amazon_reviews(path, n_per_class=5000, negative_max=2, positive_min=4,
                        min_chars=100, random_state=42):
    """Product reviews with a rating-threshold contextual boundary.

    Expected columns: `text` (or `review_body`/`content`), plus either `label`
    (0/1) or a rating column (`rating`/`stars`/`overall`/`star_rating`).

    THIS IS THE CALIBRATION CONDITION.  The boundary IS the rating, so evaluative
    valence is ground truth by construction.  A valid pipeline must:
        (1) return PR with an interval close to 1;
        (2) decompose the leading axis onto valence-like dictionary features;
        (3) move generated sentiment monotonically under steering;
        (4) collapse regime separability under ablation but not under a random
            control;
        (5) show NO non-affective residual in the affective test (step 13).
    Passing this is what licenses believing the speech-corpus result.
    """
    df = _read_any(path)
    cols = {c.lower(): c for c in df.columns}
    tcol = next((cols[c] for c in ("text", "review_body", "content", "review", "body")
                 if c in cols), None)
    if tcol is None:
        raise ValueError(f"no text column in {path}; found {list(df.columns)}")
    df = df.rename(columns={tcol: "text"})

    meta = {"source": os.path.basename(path), "corpus": "amazon_reviews"}
    if "label" in cols and df[cols["label"]].notna().any():
        df["label"] = df[cols["label"]].astype(float)
        meta["rule"] = "precomputed label column"
    else:
        rcol = next((cols[c] for c in ("rating", "stars", "overall", "star_rating")
                     if c in cols), None)
        if rcol is None:
            raise ValueError("need either a `label` or a rating column")
        lab, m = cb_from_rating(df[rcol], negative_max, positive_min)
        df["label"] = lab
        meta.update(m)

    df = df[df["text"].astype(str).str.len() >= min_chars]
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df = balance_regimes(df, n_per_class=n_per_class, random_state=random_state)
    meta.update({"n_docs": len(df), "min_chars": min_chars,
                 "n_A": int((df["label"] == 0).sum()),
                 "n_B": int((df["label"] == 1).sum())})
    return df["text"].tolist(), df["label"].to_numpy(float), None, meta


# ------------------------------------------------------------------ #
#  Continuous rating recovery (for diagnostics ONLY -- see below)
# ------------------------------------------------------------------ #
def recover_ratings(cfg, texts, labels, data_dir: str = "data"):
    """Best-effort recovery of the per-review CONTINUOUS star rating, for
    diagnostics that need more than the two-class CB label `cb_from_rating`
    produced (that label drops the middle band and collapses everything
    else to 0/1 -- correlating an axis against the raw rating instead tests
    whether it tracks sentiment as a continuum, e.g. distinguishing a 1-star
    review from a 2-star one, not just "which side of the threshold").

    Purely additive: `load_amazon_reviews`/`load_dataset` above are
    untouched, so the CB estimator's own path stays bit-identical. Only
    `amazon_reviews` carries a rating; every other dataset returns None so
    callers can request this unconditionally and degrade gracefully (the
    dependent diagnostic reports NaN with a note, same as `r2_product_id`
    when `product_ids` is None -- see `topol_gate.nuisance_diagnostics`).

    Mirrors `load_amazon_reviews`'s exact filter + balance sequence, using
    the SAME public helpers with the SAME arguments, to reconstruct a
    dataframe in the same row order -- then verifies that order against the
    `labels` this run actually used before trusting it. Returns None (with a
    printed note) if anything does not line up, rather than risk a silently
    misaligned rating array.
    """
    if cfg.dataset != "amazon_reviews":
        return None
    dc = cfg.corpus
    path = dc.amazon_path or os.path.join(data_dir, "amazon_reviews_full.pkl")
    if not os.path.exists(path):
        print(f"  rating recovery: {path} not found, skipping (corr_rating will be NaN)")
        return None
    try:
        df = _read_any(path)
        cols = {c.lower(): c for c in df.columns}
        tcol = next((cols[c] for c in ("text", "review_body", "content", "review", "body")
                    if c in cols), None)
        df = df.rename(columns={tcol: "text"})
        if "label" in cols and df[cols["label"]].notna().any():
            df["label"] = df[cols["label"]].astype(float)
        else:
            rcol = next((cols[c] for c in ("rating", "stars", "overall", "star_rating")
                        if c in cols), None)
            lab, _ = cb_from_rating(df[rcol], dc.negative_max, dc.positive_min)
            df["label"] = lab
        rcol = next((cols[c] for c in ("rating", "stars", "overall", "star_rating") if c in cols), None)
        if rcol is None:
            print("  rating recovery: no rating column found, skipping")
            return None
        df["rating"] = pd.to_numeric(df[rcol], errors="coerce")
        df = df[df["text"].astype(str).str.len() >= dc.review_min_chars]
        df = df.dropna(subset=["label"]).reset_index(drop=True)
        df = balance_regimes(df, n_per_class=dc.n_per_class, random_state=cfg.random_state)

        if len(df) != len(labels) or not np.allclose(df["label"].to_numpy(float), labels):
            print("  rating recovery: reconstructed row order does not match this run's "
                  "labels -- skipping (corr_rating will be NaN)")
            return None
        return df["rating"].to_numpy(float)
    except Exception as e:
        print(f"  rating recovery failed ({e}), skipping (corr_rating will be NaN)")
        return None


# ------------------------------------------------------------------ #
#  Dispatcher
# ------------------------------------------------------------------ #
def load_dataset(cfg, data_dir: str = "data", backend=None):
    """Return (texts, labels, topics, meta).

    `topics` is ground truth and exists only for synthetic corpora, where it scores
    the recovered partition (adjusted Rand index).  It is None for real corpora --
    which is the point: there, the topics are discovered.
    """
    ds = cfg.dataset
    dc = getattr(cfg, "corpus", None)

    def _get(attr, default):
        return getattr(dc, attr, default) if dc is not None else default

    if ds == "us_speeches":
        path = _get("speeches_path", None) or os.path.join(data_dir, "us_speeches_full.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run:\n"
                f"  python scripts/00_prepare_data.py --dataset us_speeches")
        return load_us_speeches(
            path, breakpoint=_get("breakpoint", "2007-05-01"),
            window_years=_get("window_years", None),
            min_chars=_get("speech_min_chars", 500),
            max_docs=_get("max_docs", None), random_state=cfg.random_state)

    if ds == "amazon_reviews":
        path = _get("amazon_path", None) or os.path.join(data_dir, "amazon_reviews_full.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run:\n"
                f"  python scripts/00_prepare_data.py --dataset amazon_reviews")
        return load_amazon_reviews(
            path, n_per_class=_get("n_per_class", 5000),
            negative_max=_get("negative_max", 2),
            positive_min=_get("positive_min", 4),
            min_chars=_get("review_min_chars", 100),
            random_state=cfg.random_state)

    if ds in ("synthetic_1d", "synthetic_kd"):
        assert backend is not None, "synthetic datasets need the SyntheticBackend"
        texts, labels, topics = backend.make_dataset(n_docs=2000)
        return texts, labels, topics, {"corpus": ds, "planted": True,
                                       "rule": "planted regime assignment"}

    raise ValueError(f"unknown dataset {ds!r}")
