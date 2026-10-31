#!/usr/bin/env python3
"""Step 00 - build the two REAL corpora and their contextual boundaries.

Run this once per corpus before step 01. Synthetic datasets need no preparation.

    # US central bank speeches (DISCOVERY condition)
    python scripts/00_prepare_data.py --dataset us_speeches
    # (uses data/us_speeches_full.pkl by default)

    # Product reviews (CALIBRATION condition)
    python scripts/00_prepare_data.py --dataset amazon_reviews
    # (uses data/amazon_reviews_full.pkl by default)

    # Inspect an existing prepared corpus without rebuilding
    python scripts/00_prepare_data.py --dataset us_speeches --inspect

WHAT THIS SCRIPT IS FOR
-----------------------
The contextual boundary is the one place a researcher's judgement enters the
pipeline. Everything downstream is mechanical; the CB is a claim about what
comparison is worth making. This script makes that claim explicit, records the
rule in the output file, and prints the diagnostics you need to decide whether
the boundary is doing what you think.

WHERE THE DATA COMES FROM
-------------------------
Prepared corpora are included in the data directory:

  US central bank speeches: data/us_speeches_full.pkl
  Product reviews: data/amazon_reviews_full.pkl

To prepare from raw data instead, use --raw with a file containing:
  - Speeches: a table with text and date columns (e.g., BIS central bankers' speeches)
  - Reviews: a table with text and rating columns

CHOOSING THE BREAKPOINT HONESTLY
--------------------------------
The reference breakpoint was selected from a PCA-constructed macroeconomic regime
indexes dataset. `--macro-index` accepts a CSV of macro series and allows reproducing
that construction: standardise the series, take PC1, and offer the largest
structural breaks as candidates. It is possible to pick a date by hand instead.
"""
import warnings; warnings.filterwarnings("ignore")
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from topolm.data import (_read_any, cb_from_date, cb_from_rating, cb_from_column,  # noqa: E402
                         balance_regimes, load_us_speeches, load_amazon_reviews)


# ------------------------------------------------------------------ #
def macro_breakpoint_candidates(macro_csv, date_col="date", n_candidates=5):
    """Reproduce the PCA macroeconomic regime index and rank candidate breakpoints.

    Standardise every numeric series, take the first principal component as a
    single regime index, and score each date by the absolute difference in mean
    index level before and after it. This is a transparent stand-in for a formal
    structural-break test (Bai-Perron, CUSUM); use one of those if the breakpoint
    is load-bearing for the argument.
    """
    df = _read_any(macro_csv)
    dcol = next((c for c in df.columns if c.lower() == date_col), df.columns[0])
    dates = pd.to_datetime(df[dcol])
    X = df.drop(columns=[dcol]).select_dtypes(include=[np.number]).to_numpy(float)
    X = np.nan_to_num(X, nan=np.nanmean(X))
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    pc1 = U[:, 0] * S[0]

    scores = []
    for i in range(len(pc1) // 10, len(pc1) - len(pc1) // 10):
        scores.append((abs(pc1[:i].mean() - pc1[i:].mean()), dates.iloc[i]))
    scores.sort(reverse=True)
    return [{"date": str(d.date()), "separation": float(s)}
            for s, d in scores[:n_candidates]], pc1, dates


# ------------------------------------------------------------------ #
def prepare_speeches(args, out_path):
    df = _read_any(args.raw)
    cols = {c.lower(): c for c in df.columns}
    tcol = next((cols[c] for c in ("text", "speech", "content", "body") if c in cols), None)
    dcol = next((cols[c] for c in ("date", "speech_date", "published") if c in cols), None)
    if tcol is None:
        raise SystemExit(f"no text column found; columns are {list(df.columns)}")
    df = df.rename(columns={tcol: "text"})
    if dcol:
        df["date"] = pd.to_datetime(df[dcol], errors="coerce")

    # optional country/institution filter -- the reference corpus is US only
    if args.country_filter:
        ccol = next((cols[c] for c in ("country", "institution", "central_bank")
                     if c in cols), None)
        if ccol:
            before = len(df)
            pat = args.country_filter
            df = df[df[ccol].astype(str).str.contains(pat, case=False, na=False)]
            print(f"  country filter {pat!r}: {before} -> {len(df)} rows")
        else:
            print(f"  WARNING: --country-filter given but no country column found")

    # --- construct the contextual boundary --------------------------- #
    if args.cb_column:
        a = args.cb_a.split(",") if args.cb_a else []
        b = args.cb_b.split(",") if args.cb_b else []
        lab, meta = cb_from_column(df[cols[args.cb_column.lower()]], a, b)
    else:
        if "date" not in df.columns:
            raise SystemExit("need a date column for a temporal CB, or use --cb-column")
        lab, meta = cb_from_date(df["date"], args.breakpoint, args.window_years)
    df["label"] = lab
    df = df[df["text"].astype(str).str.len() >= args.min_chars]
    df = df.dropna(subset=["label"]).reset_index(drop=True)

    print(f"  CB rule: {meta}")
    print(f"  after filtering: {len(df)} docs "
          f"(A={int((df['label']==0).sum())}, B={int((df['label']==1).sum())})")

    df = balance_regimes(df, n_per_class=(args.max_docs // 2 if args.max_docs else None),
                         random_state=args.seed)
    keep = ["text", "label"] + (["date"] if "date" in df.columns else [])
    df = df[keep]
    df.to_pickle(out_path)
    meta.update({"n_docs": len(df), "min_chars": args.min_chars,
                 "n_A": int((df["label"] == 0).sum()),
                 "n_B": int((df["label"] == 1).sum()),
                 "balanced": True, "seed": args.seed, "raw": args.raw})
    return df, meta


def prepare_reviews(args, out_path):
    df = _read_any(args.raw)
    cols = {c.lower(): c for c in df.columns}
    tcol = next((cols[c] for c in ("text", "review_body", "content", "review", "body")
                 if c in cols), None)
    rcol = next((cols[c] for c in ("rating", "stars", "overall", "star_rating")
                 if c in cols), None)
    if tcol is None or rcol is None:
        raise SystemExit(f"need text and rating columns; found {list(df.columns)}")
    df = df.rename(columns={tcol: "text"})

    lab, meta = cb_from_rating(df[rcol], args.negative_max, args.positive_min)
    df["label"] = lab
    df["rating"] = pd.to_numeric(df[rcol], errors="coerce")
    df = df[df["text"].astype(str).str.len() >= args.min_chars]
    df = df.dropna(subset=["label"]).reset_index(drop=True)

    print(f"  CB rule: {meta}")
    print(f"  dropped middle band: ratings in "
          f"({args.negative_max}, {args.positive_min}) excluded")
    print(f"  after filtering: {len(df)} docs "
          f"(A={int((df['label']==0).sum())}, B={int((df['label']==1).sum())})")

    df = balance_regimes(df, n_per_class=args.n_per_class, random_state=args.seed)
    df = df[["text", "label", "rating"]]
    df.to_pickle(out_path)
    meta.update({"n_docs": len(df), "min_chars": args.min_chars,
                 "n_A": int((df["label"] == 0).sum()),
                 "n_B": int((df["label"] == 1).sum()),
                 "balanced": True, "seed": args.seed, "raw": args.raw})
    return df, meta


# ------------------------------------------------------------------ #
def inspect(path, dataset):
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found -- prepare it first")
    if dataset == "us_speeches":
        texts, labels, _, meta = load_us_speeches(path)
    else:
        texts, labels, _, meta = load_amazon_reviews(path)
    lens = np.array([len(t) for t in texts])
    print(f"[00] {path}")
    print(f"     meta: {json.dumps(meta, default=str)}")
    print(f"     doc chars: median {int(np.median(lens))}, "
          f"p10 {int(np.percentile(lens,10))}, p90 {int(np.percentile(lens,90))}")
    a = lens[labels == 0]; b = lens[labels == 1]
    print(f"     length by regime: A median {int(np.median(a))}, "
          f"B median {int(np.median(b))}")
    ratio = np.median(b) / max(1, np.median(a))
    if ratio > 1.25 or ratio < 0.8:
        print(f"     WARNING: median length differs by {ratio:.2f}x across regimes. "
              f"Document length is a CONFOUND for this boundary -- include a length "
              f"scorer in the steering panel (step 08) so a length effect cannot "
              f"masquerade as a semantic one.")
    else:
        print(f"     length balance across regimes looks fine ({ratio:.2f}x)")


# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser(description="Prepare a real corpus and its CB")
    ap.add_argument("--dataset", required=True,
                    choices=["us_speeches", "amazon_reviews"])
    ap.add_argument("--raw", default=None,
                    help="raw corpus file (.csv/.tsv/.jsonl/.parquet/.pickle); "
                         "defaults to data/us_speeches_full.pkl or data/amazon_reviews_full.pkl")
    ap.add_argument("--out", default=None, help="output pickle path")
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(REPO_ROOT), "data"))
    ap.add_argument("--inspect", action="store_true",
                    help="describe an already-prepared corpus and exit")
    ap.add_argument("--seed", type=int, default=42)
    # speeches
    ap.add_argument("--breakpoint", default="2007-05-01")
    ap.add_argument("--window-years", type=int, default=None)
    ap.add_argument("--max-docs", type=int, default=600)
    ap.add_argument("--country-filter", default=None,
                    help="regex on a country/institution column, e.g. 'United States'")
    ap.add_argument("--macro-index", default=None,
                    help="CSV of macro series; prints PCA breakpoint candidates")
    ap.add_argument("--cb-column", default=None,
                    help="use a categorical column for the CB instead of a date")
    ap.add_argument("--cb-a", default=None, help="comma-separated values for regime A")
    ap.add_argument("--cb-b", default=None, help="comma-separated values for regime B")
    # reviews
    ap.add_argument("--n-per-class", type=int, default=5000)
    ap.add_argument("--negative-max", type=int, default=2)
    ap.add_argument("--positive-min", type=int, default=4)
    # shared
    ap.add_argument("--min-chars", type=int, default=None)
    args = ap.parse_args()

    if args.min_chars is None:
        args.min_chars = 500 if args.dataset == "us_speeches" else 100

    # Set default raw file if not provided
    if args.raw is None:
        default_raw = ("us_speeches_full.pkl" if args.dataset == "us_speeches"
                       else "amazon_reviews_full.pkl")
        args.raw = os.path.join(args.data_dir, default_raw)

    default_name = ("us_speeches_full.pkl" if args.dataset == "us_speeches"
                    else "amazon_reviews_full.pkl")
    out_path = args.out or os.path.join(args.data_dir, default_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if args.inspect:
        inspect(out_path, args.dataset)
        return

    if args.macro_index:
        cands, pc1, dates = macro_breakpoint_candidates(args.macro_index)
        print("[00] PCA macro-index breakpoint candidates (largest regime separation):")
        for c in cands:
            print(f"       {c['date']}   separation {c['separation']:.3f}")
        print("[00] pass one of these to --breakpoint, and state in the paper that "
              "the boundary came from the index rather than from inspection")
        if not args.raw:
            return

    if not os.path.exists(args.raw):
        raise SystemExit(f"raw file not found: {args.raw}")

    print(f"[00] preparing {args.dataset} from {args.raw}")
    if args.dataset == "us_speeches":
        df, meta = prepare_speeches(args, out_path)
    else:
        df, meta = prepare_reviews(args, out_path)

    with open(out_path.replace(".pickle", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[00] wrote {out_path}  ({len(df)} docs)")
    print(f"[00] wrote {out_path.replace('.pickle', '_meta.json')}")
    print(f"[00] next:  python scripts/01_extract_representations.py --dataset {args.dataset} "
          f"--backend gemma  (then 02..08 in order, or open notebooks/pipeline_2.ipynb)")
    inspect(out_path, args.dataset)


if __name__ == "__main__":
    main()
