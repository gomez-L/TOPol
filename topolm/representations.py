"""Representation extraction.

Thin wrapper over the backend so the rest of the pipeline is agnostic to whether
activations come from Gemma 3 12B or from the synthetic model.

This module produces the FIELD representations -- the residual stream in which
polarity vectors, their decomposition and every causal test live.  It is NOT the
module that feeds UMAP: topic segmentation runs on EmbeddingGemma-300M via
`backends/segmentation.py`, because clustering needs similarity geometry while the
field needs intervenability.
"""
from __future__ import annotations

import numpy as np


def extract_layer(backend, texts, layer, pooling="mean"):
    """(N, d_model) pooled residual-stream activations at a single layer."""
    return backend.embed(list(texts), layer=layer, pooling=pooling)


def extract_all_layers(backend, texts, pooling="mean", subsample=None, random_state=42):
    """(N, L, d_model) activations across layers, for layer selection.

    `subsample` (int) limits the number of documents for the (cheaper) layer sweep.
    Returns (acts, idx) where idx are the row indices used.
    """
    n = len(texts)
    if subsample is not None and subsample < n:
        rng = np.random.default_rng(random_state)
        idx = np.sort(rng.choice(n, size=subsample, replace=False))
        sel = [texts[i] for i in idx]
    else:
        idx = np.arange(n)
        sel = list(texts)
    acts = backend.embed_all_layers(sel, pooling=pooling)
    return acts, idx
