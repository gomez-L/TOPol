#!/usr/bin/env python3
"""Step 02 - choose the residual layer by A/B probe separability, embed all docs there."""
import warnings; warnings.filterwarnings("ignore")
from _common import base_parser, open_existing_store, load_config_for, apply_overrides, make_backend
from topolm import pipeline as P

def main():
    args = base_parser("Select residual layer").parse_args()
    store = open_existing_store(args)
    cfg = apply_overrides(load_config_for(store), args)
    backend = make_backend(cfg)
    r = P.step_select_layer(store, cfg, backend)
    acc = r["selection"].get("best_acc")
    print(f"[02] chosen layer={r['layer']}  probe_acc={acc}  embeddings={r['embeddings'].shape}")

if __name__ == "__main__":
    main()
