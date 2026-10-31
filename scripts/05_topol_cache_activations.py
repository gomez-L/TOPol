#!/usr/bin/env python3
"""Step 05 - cache mean-pooled residual activations at every layer (subject to
`discovery.layer_stride`), excluding BOS/special/chat-template tokens.

This is additive to step 01's `acts_all_layers` (which the CB field/layer
selection use and which this script never touches): `topol_axes.py`'s
recursive-bipartition discovery needs the WHOLE cached residual stack, not
just the one layer `02_select_layer.py` picks, and needs BOS excluded from the
pool so a large-norm special-token position cannot dominate a topic centroid.

Run after 01_extract_representations.py (needs `texts` from the same run).
"""
import warnings; warnings.filterwarnings("ignore")
from _common import base_parser, open_existing_store, load_config_for, apply_overrides, make_backend
from topolm import pipeline as P

def main():
    ap = base_parser("Cache per-layer activations for axis discovery")
    ap.add_argument("--layer-stride", type=int, default=None,
                    help="override discovery.layer_stride")
    args = ap.parse_args()
    store = open_existing_store(args)
    cfg = apply_overrides(load_config_for(store), args)
    if args.layer_stride is not None:
        cfg.discovery.layer_stride = args.layer_stride
    backend = make_backend(cfg)

    cache = P.step_topol_cache_activations(store, cfg, backend)
    print(f"[05] cached layers={cache['layers']} "
          f"excluded={cache['excluded']['total_excluded_tokens']}/"
          f"{cache['excluded']['total_tokens']} pooled-token positions "
          f"({cache['excluded']['fraction_excluded']:.2%})")

if __name__ == "__main__":
    main()
