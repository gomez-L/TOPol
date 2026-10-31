#!/usr/bin/env python3
"""Step 04 - reconstruct the polarity field in full space (diff-in-means + probe dirs)."""
import warnings; warnings.filterwarnings("ignore")
from _common import base_parser, open_existing_store, load_config_for, apply_overrides
from topolm import pipeline as P
import numpy as np

def main():
    args = base_parser("Reconstruct field").parse_args()
    store = open_existing_store(args)
    cfg = apply_overrides(load_config_for(store), args)
    r = P.step_field(store, cfg)
    ag = np.mean(np.abs(r["agreement"])) if r["agreement"] else float("nan")
    print(f"[04] topics={len(r['field']['clusters'])}  m_bar={r['stats']['m_bar']:.3f}  "
          f"s_bar={r['stats']['s_bar']:.3f}  mean|cos(diffmean,probe)|={ag:.3f}")

if __name__ == "__main__":
    main()
