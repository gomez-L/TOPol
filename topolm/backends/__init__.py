from .base import Backend
from .synthetic import SyntheticBackend


def make_backend(cfg):
    """The FIELD backend: Gemma 3 12B (or the synthetic stand-in).

    Supplies the residual-stream representations in which polarity vectors live,
    the generation used for causal tests and narration, and the Gemma Scope 2
    dictionary. This is NOT the model that feeds UMAP -- see
    `make_segmentation_backend`.
    """
    if cfg.backend == "synthetic":
        regime = "1d" if cfg.dataset in ("synthetic_1d", "amazon_reviews") else "kd"
        return SyntheticBackend(regime=regime, random_state=cfg.random_state)
    elif cfg.backend == "gemma":
        from .gemma import GemmaBackend
        return GemmaBackend(cfg.model, cfg.sae)
    raise ValueError(f"unknown backend {cfg.backend!r}")


def make_segmentation_backend(cfg, field_backend=None):
    """The SEGMENTATION backend: EmbeddingGemma-300M.

    Supplies the vectors UMAP and Leiden operate on. EmbeddingGemma is built from
    the Gemma 3 architecture, so segmentation and field come from one model family,
    and it is trained for sentence similarity -- the geometry clustering needs and
    the one thing a base decoder's mean-pooled stream is weakest at.

    It produces LABELS ONLY. Cluster assignments travel into the field computation;
    these embeddings never enter vector arithmetic.

    Returns None when `cfg.segmentation.source == "field"`, in which case the
    caller clusters the field representations instead -- a literally single-model
    pipeline, kept as an ablation so the paper can report what decoupling bought.
    """
    sc = getattr(cfg, "segmentation", None)
    if sc is not None and getattr(sc, "source", "embedding") == "field":
        return None
    if cfg.backend == "synthetic":
        # The synthetic backend already provides similarity geometry directly.
        return field_backend
    from .segmentation import EmbeddingGemmaBackend
    return EmbeddingGemmaBackend(
        name=sc.name if sc else "google/embeddinggemma-300m",
        prompt_name=sc.prompt_name if sc else "Clustering",
        batch_size=sc.batch_size if sc else 64,
        normalize=sc.normalize if sc else True,
    )


__all__ = ["Backend", "SyntheticBackend", "make_backend", "make_segmentation_backend"]
