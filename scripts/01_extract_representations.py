#!/usr/bin/env python3
"""Step 01 - load the corpus, embed it twice, and extract all-layer activations.

Two representations are produced, for two different jobs:

  SEGMENTATION  EmbeddingGemma-300M  -> the vectors UMAP and Leiden operate on.
                Built from the Gemma 3 architecture and trained for sentence
                similarity, which is the geometry clustering needs. Produces
                LABELS ONLY; never enters vector arithmetic.

  FIELD         Gemma 3 12B residual  -> the space polarity vectors live in, and
                the only space where Gemma Scope 2 dictionaries are defined and
                steering hooks can be attached.

Datasets: us_speeches | amazon_reviews | synthetic_1d | synthetic_kd
Run scripts/00_prepare_data.py first for the two real corpora.

Creates a new run directory that later steps resume.
"""
import warnings; warnings.filterwarnings("ignore")
from _common import base_parser, build_config, open_new_store, make_backend, make_seg_backend
from topolm import pipeline as P

def main():
    args = base_parser("Extract all-layer representations").parse_args()
    cfg = build_config(args)
    store = open_new_store(cfg, args)
    backend = make_backend(cfg)
    seg_backend = make_seg_backend(cfg, backend)
    r = P.step_extract(store, cfg, backend, seg_backend=seg_backend,
                       data_dir=args.data_dir)
    print(f"[01] run={store.dir}")
    print(f"[01] docs={len(r['texts'])}  d_model={backend.d_model}  "
          f"layers={backend.n_layers}  acts_all={r['acts_all_layers'].shape}")

if __name__ == "__main__":
    main()
