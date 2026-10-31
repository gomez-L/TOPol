#!/usr/bin/env python3
"""Step 03 - UMAP + Leiden topic segmentation (clustering + 2-D display only)."""
import warnings; warnings.filterwarnings("ignore")
from _common import base_parser, open_existing_store, load_config_for, apply_overrides
from topolm import pipeline as P

def main():
    args = base_parser("Cluster manifold").parse_args()
    store = open_existing_store(args)
    cfg = apply_overrides(load_config_for(store), args)
    r = P.step_cluster(store, cfg)
    print(f"[03] n_clusters={r['cluster']['n_clusters']}  (figure: 02_field_2d.pdf)")

if __name__ == "__main__":
    main()
