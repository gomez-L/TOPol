"""Layer-wise activation caching for the axis-discovery path (`topol_axes.py`).

This is additive to, and separate from, the existing single-layer field
extraction in `representations.py` / `pipeline.step_select_layer`:

  * the CB field (`field.py`) needs ONE layer, chosen by probe separability,
    and pools over the FULL attention mask (`backends/*._pool`).
  * discovery needs EVERY cached layer (subject to `layer_stride`, so memory
    stays bounded), and must NOT let BOS / special / chat-template positions
    -- which carry disproportionately large activation norms (see
    `config.py`'s note on this) -- dominate a mean-pooled document vector.

Those are different requirements, so this module calls a different backend
method (`embed_all_layers_masked`) and writes its own cache; it never touches
`acts_all_layers`, `embeddings`, or any other object the CB path reads or
writes. The CB baseline is therefore untouched by this module's existence.

Cache is stored as float16, one array per cached layer (see `store.py`'s
`save_array_f16`) -- the dominant memory cost here is N docs x L layers x
d_model, and cosine/distance geometry does not need float32 precision.
"""
from __future__ import annotations

import numpy as np


def cached_layers(n_layers: int, layer_stride: int = 2, extra_layers=None) -> list[int]:
    """Layer indices to cache: every `layer_stride`-th layer, always including
    layer 0 (embedding output), `n_layers` (the final residual layer) and any
    `extra_layers` requested -- notably the layer `layer_select.py` chose,
    which need not fall on the stride grid, but which `discovery.ref_layer`
    defaults to for comparability with prior (CB-only) runs. Without this, a
    reference layer chosen off-grid would raise a KeyError the first time
    `field.pooled_field_assembly` looked it up."""
    stride = max(1, int(layer_stride))
    layers = set(range(0, n_layers + 1, stride))
    layers.add(n_layers)
    for l in (extra_layers or []):
        if 0 <= l <= n_layers:
            layers.add(int(l))
    return sorted(layers)


def cache_activations(backend, texts, layer_stride: int = 2, extra_layers=None) -> dict:
    """Mean-pool every cached layer, excluding BOS/special/chat-template tokens.

    Returns:
      acts     : {layer: (N, d_model) float16 array}
      layers   : sorted list of cached layer indices
      excluded : {"total_tokens", "total_excluded_tokens", "per_doc_excluded",
                  "fraction_excluded"} -- how much of the pooled-token budget
                 was dropped as BOS/special/chat-template positions.
    """
    if not hasattr(backend, "embed_all_layers_masked"):
        raise AttributeError(
            f"{type(backend).__name__} does not implement embed_all_layers_masked; "
            "activation caching needs BOS/special-token exclusion at pooling time, "
            "which the CB path's embed()/embed_all_layers() deliberately do not do.")

    layers = cached_layers(backend.n_layers, layer_stride, extra_layers=extra_layers)
    result = backend.embed_all_layers_masked(list(texts), layers)
    acts = {l: np.asarray(result["acts"][l], dtype=np.float16) for l in layers}

    excl = dict(result["excluded"])
    total = excl.get("total_tokens", 0)
    dropped = excl.get("total_excluded_tokens", 0)
    excl["fraction_excluded"] = float(dropped / total) if total else 0.0

    print(f"[topol_activations] cached {len(layers)} layers (stride={layer_stride}) "
          f"for {len(texts)} docs | excluded {dropped}/{total} "
          f"({excl['fraction_excluded']:.2%}) pooled-token positions as "
          f"BOS/special/chat-template")
    return {"acts": acts, "layers": layers, "excluded": excl}


def save_activation_cache(store, cache: dict, name_prefix: str = "topol_acts") -> None:
    """Persist a `cache_activations` result: one float16 array per layer plus a
    small JSON manifest (layers cached, exclusion counts)."""
    for l in cache["layers"]:
        store.save_array_f16(f"{name_prefix}_L{l}", cache["acts"][l], meta={"layer": int(l)})
    excl = dict(cache["excluded"])
    if isinstance(excl.get("per_doc_excluded"), np.ndarray):
        excl["per_doc_excluded"] = excl["per_doc_excluded"].tolist()
    store.save_json(f"{name_prefix}_meta", {"layers": cache["layers"], "excluded": excl})


def load_activation_cache(store, layers=None, name_prefix: str = "topol_acts") -> dict:
    """Reload a cache written by `save_activation_cache`."""
    meta = store.load_json(f"{name_prefix}_meta")
    use_layers = layers if layers is not None else meta["layers"]
    acts = {l: store.load_array(f"{name_prefix}_L{l}") for l in use_layers}
    return {"acts": acts, "layers": list(use_layers), "excluded": meta["excluded"]}
