"""Topic segmentation representations -- deliberately a DIFFERENT model from the field.

WHY SEGMENTATION AND THE FIELD USE DIFFERENT REPRESENTATIONS
------------------------------------------------------------
The pipeline asks two things of a representation, and they have incompatible
requirements. Conflating them was a design error in the first version of this
codebase; see the field guide, Part XIV.

  JOB A -- topic segmentation.
      Needs good *similarity* geometry: nearest neighbours, clustering, a usable
      kNN graph. This is exactly what contrastively-trained retrieval embedders
      are optimised for, and exactly where naive mean-pooling of a base
      decoder's residual stream is weakest: causal attention means each token
      only sees leftward context, so the pooled vector is not a balanced summary.

  JOB B -- the polarity field, its decomposition, and the causal tests.
      Needs the representation to be (i) linearly probeable for the regime
      distinction and (ii) INTERVENABLE: you must be able to add a vector to it
      mid-forward-pass and read features off a dictionary trained on that exact
      space. This forces the generation model's residual stream. You cannot steer
      an embedding API, and you cannot decompose an EmbeddingGemma vector onto
      Gemma Scope features -- different space, different basis, no correspondence.

Job B's requirement is absolute; Job A's is not. A topic assignment is just a
partition of documents, and a partition can be computed anywhere. So we compute
it where it is best computed, and carry only the LABELS into the field space.

Crucially, the weakness of raw decoder pooling does not harm Job B at all.
Probing and steering do not need retrieval-quality geometry -- the entire
steering and SAE literature operates on unadapted base-model residual streams,
and linear probes decode a wide range of properties from them. For the field,
the unadapted residual stream is not a compromise; it is the correct object.

SIDE BENEFIT: decoupling removes a circularity. When clustering and the field
come from one space, a clustering artefact of that space can shape the field.
With two models, topic identity is fixed independently of the space whose
internals we probe -- the same logic as defining experimental conditions by
external criteria before measuring responses.

WHAT IT DOES NOT FIX: Proposition 3 (boundary-dependent segmentation). The A->B
shift is in the text, so any semantic embedder sees it and can still split a
topic in two. The boundary-independent segmentation (fit on regime A, assign B)
remains necessary. The two fixes are complementary.

COST TO MEASURE: a topic coherent in EmbeddingGemma space may be less coherent
in Gemma-3 residual space. `cross_space_coherence` quantifies that, and it should
be reported.
"""
from __future__ import annotations

import numpy as np


class EmbeddingGemmaBackend:
    """EmbeddingGemma-300m for topic segmentation only.

    A 0.3B purpose-built sentence-similarity model (released Sep 2025). Small
    enough to be free relative to the generator, and trained for exactly the
    geometry clustering needs.

    EmbeddingGemma expects task-specific prompt prefixes; sentence-transformers
    exposes these via `prompt_name`. Use the clustering prompt for segmentation,
    NOT the retrieval/query prompt -- they induce different geometries.
    """

    def __init__(self, name: str = "google/embeddinggemma-300m",
                 device: str | None = None, prompt_name: str | None = "Clustering",
                 batch_size: int = 64, normalize: bool = True):
        from sentence_transformers import SentenceTransformer
        self.name = name
        self.prompt_name = prompt_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.model = SentenceTransformer(name, device=device)

    def embed(self, texts, show_progress: bool = False):
        """(N, d) sentence embeddings for segmentation.

        Note these are L2-normalised by default, so they lie on a hypersphere.
        That matters for `predictive.py`'s spherical variant, and it is a reason
        to check `base_point_spread` if you ever compute a field in THIS space
        (we do not -- the field lives in the residual stream, which is not
        norm-constrained, so Euclidean geometry is correct there).
        """
        kw = {"batch_size": self.batch_size,
              "normalize_embeddings": self.normalize,
              "show_progress_bar": show_progress}
        if self.prompt_name:
            try:
                return np.asarray(self.model.encode(list(texts), prompt_name=self.prompt_name, **kw))
            except Exception:
                pass  # prompt name unsupported in this version; fall through
        return np.asarray(self.model.encode(list(texts), **kw))


def cross_space_coherence(seg_embeddings, field_embeddings, cluster_labels):
    """Do the topics defined in segmentation space stay coherent in field space?

    Reports mean silhouette in both spaces plus the ratio of within- to
    between-topic distance in the field space. A large drop means the partition
    is not meaningful where the field is computed, which would undermine the
    per-topic centroids -- report it, and consider clustering in field space as a
    robustness check.
    """
    from sklearn.metrics import silhouette_score

    def _stats(X):
        X = np.asarray(X, dtype=float)
        labs = np.asarray(cluster_labels)
        if len(np.unique(labs)) < 2:
            return {"silhouette": np.nan, "within": np.nan, "between": np.nan, "ratio": np.nan}
        sil = float(silhouette_score(X, labs, metric="cosine"))
        cents, withins = [], []
        for c in np.unique(labs):
            P = X[labs == c]
            mu = P.mean(0)
            cents.append(mu)
            withins.append(float(np.mean(np.linalg.norm(P - mu, axis=1))))
        C = np.stack(cents)
        d = [np.linalg.norm(C[i] - C[j])
             for i in range(len(C)) for j in range(i + 1, len(C))]
        within, between = float(np.mean(withins)), float(np.mean(d)) if d else np.nan
        return {"silhouette": sil, "within": within, "between": between,
                "ratio": within / between if between else np.nan}

    return {"segmentation_space": _stats(seg_embeddings),
            "field_space": _stats(field_embeddings)}
