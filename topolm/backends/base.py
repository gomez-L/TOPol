"""The `Backend` contract.

A backend supplies everything model-dependent in one activation space:
    * embeddings (single layer and all-layers, for layer selection),
    * text generation with optional residual-stream steering (causal test),
    * SAE encode/decode and decoder directions (the interpretable basis),
    * human-readable feature labels (hypotheses only).

Two implementations are provided:
    * `GemmaBackend`     - Gemma 3 12B + Gemma Scope 2 (needs GPU + weights).
    * `SyntheticBackend` - planted low/high-dimensional polarity structure for
                           offline development and tests.

Keeping this seam means the analysis modules (field, dimensionality, sae, steering)
never import torch and are unit-testable without a GPU.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class Backend(ABC):
    d_model: int
    n_layers: int
    d_sae: int

    # ------------------------------ embeddings ------------------------------ #
    @abstractmethod
    def embed(self, texts: list[str], layer: int, pooling: str = "mean") -> np.ndarray:
        """Return (N, d_model) pooled residual-stream activations at `layer`."""

    @abstractmethod
    def embed_all_layers(self, texts: list[str], pooling: str = "mean") -> np.ndarray:
        """Return (N, n_layers, d_model) pooled activations for layer selection."""

    @abstractmethod
    def embed_all_layers_masked(self, texts: list[str], layers: list[int]) -> dict:
        """Mean-pool ONLY the requested `layers`, excluding BOS/special/chat-template
        token positions from the pool (the CB field's `embed`/`embed_all_layers`
        intentionally pool over the FULL attention mask instead -- this is a
        separate path so that pooling convention never changes).

        Used by `topol_activations.py` to cache the full residual stack (subject
        to a layer stride) for the axis-discovery path. Returns:
            {"acts": {layer: (N, d_model) float32 array},
             "excluded": {"total_tokens": int, "total_excluded_tokens": int,
                          "per_doc_excluded": (N,) int array}}
        """

    # ------------------------------ generation ------------------------------ #
    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        steering_vector: np.ndarray | None = None,
        layer: int | None = None,
        alpha: float = 0.0,
        apply_to: str = "all",
        max_new_tokens: int = 60,
    ) -> list[str]:
        """Generate continuations, optionally adding `alpha * steering_vector`
        to the residual stream at `layer` during the forward pass."""

    # --------------------------------- SAE ---------------------------------- #
    @abstractmethod
    def sae_encode(self, acts: np.ndarray, layer: int) -> tuple[np.ndarray, np.ndarray]:
        """Encode (N, d_model) activations -> (features (N, d_sae), reconstruction (N, d_model))."""

    @abstractmethod
    def sae_decoder_directions(self, layer: int) -> np.ndarray:
        """Return (d_sae, d_model) unit-norm feature (decoder) directions."""

    @abstractmethod
    def feature_labels(self, layer: int, indices: list[int]) -> list[str]:
        """Return human-readable labels for feature indices (hypotheses only)."""

    # ------------------------------ token-level ------------------------------ #
    @abstractmethod
    def token_activations(self, texts: list[str], layer: int) -> tuple[list[str], np.ndarray]:
        """Return (tokens, acts): per-token decoded strings and (N_tokens, d_model)
        residual-stream activations at `layer`, flattened across `texts`
        (special/padding positions dropped).

        Grounds SAE feature labels in real corpus words instead of a bare feature
        index (see `sae.top_activating_tokens`): running these activations through
        the SAME SAE used for the polarity-vector decomposition and ranking tokens
        by a feature's activation gives concrete examples like 'inflation',
        'tighten' rather than only "[gemma-scope L12 feature 4821]"."""
