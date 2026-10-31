"""A backend with *planted* polarity structure.

It fabricates a dictionary of orthonormal "semantic axes" and generates document
activations as:  topic_anchor  +  regime_shift(along a subset of axes)  +  noise.

Two regimes exist:
    * "1d"  -> the A->B shift is concentrated on ONE axis with a consistent sign
               across topics  =>  high pairwise cosine, low participation ratio
               (the Amazon-like / sentiment-dominated case).
    * "kd"  -> the shift is spread over several axes with topic-varying signs
               =>  low pairwise cosine, higher participation ratio
               (the central-bank-speeches-like / multidimensional case).

Layer profile.  Signal (anchor + shift) is modulated by a smooth MID-peaking gain
across layers while noise stays flat, so the A/B separability (hence the chosen
layer) peaks in the middle of the network - mimicking the empirical finding that
mid-layer residual streams are the most linearly probe-able.  This lets the
layer-selection step demonstrate sensibly offline.

SAE.  The dictionary contains the true axes plus random atoms, so `sae_encode`
recovers which axes moved and `feature_labels` returns their names.  Generation is
a faithful encode/decode of the (steered) hidden state, which lets the causal
(steering / ablation) plumbing run end-to-end offline.  None of this is a claim
about real language - it exists to exercise and test the analysis code.

NOTE: one backend instance holds the per-document latent cache produced by
`make_dataset`; reuse the SAME instance throughout a run so `embed` can reconstruct
document activations at any layer.
"""
from __future__ import annotations

import base64
import numpy as np

from .base import Backend

_MARK_DOC = "\u2063D:"     # invisible marker: this text is document <tag>
_MARK_VEC = "\u2063V:"     # invisible marker: this text encodes a raw activation vector


class SyntheticBackend(Backend):
    def __init__(
        self,
        regime: str = "kd",
        d_model: int = 128,
        n_layers: int = 12,
        n_topics: int = 8,
        n_axes: int = 6,
        n_active_axes: int = 3,
        shift_scale: float = 7.0,
        noise_scale: float = 2.5,
        anchor_scale: float = 6.0,
        random_state: int = 42,
    ):
        self.regime = regime
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_topics = n_topics
        self.n_axes = n_axes
        self.n_active_axes = n_active_axes
        self.shift_scale = shift_scale
        self.noise_scale = noise_scale
        self.anchor_scale = anchor_scale
        self.rng = np.random.default_rng(random_state)

        # Orthonormal semantic axes (the ground-truth polarity directions).
        Q, _ = np.linalg.qr(self.rng.standard_normal((d_model, d_model)))
        self.axes = Q[:, :n_axes]                       # (d_model, n_axes)
        self.axis_names = [f"axis::{n}" for n in _AXIS_WORDS[:n_axes]]

        # Well-separated topic anchors.
        self.topic_anchors = self.rng.standard_normal((n_topics, d_model)) * anchor_scale

        # Per-topic regime-shift coefficients over the axes.
        if regime == "1d":
            coeff = np.zeros((n_topics, n_axes))
            coeff[:, 0] = shift_scale                    # same axis, same sign everywhere
        else:
            coeff = np.zeros((n_topics, n_axes))
            active = np.arange(min(n_active_axes, n_axes))
            for t in range(n_topics):
                signs = self.rng.choice([-1.0, 1.0], size=len(active))
                mag = shift_scale * (0.6 + 0.8 * self.rng.random(len(active)))
                coeff[t, active] = signs * mag
        self.topic_shift = coeff @ self.axes.T           # (n_topics, d_model)

        # SAE dictionary: true axes first, then random atoms.
        n_random = 90
        R = self.rng.standard_normal((d_model, n_random))
        R /= np.linalg.norm(R, axis=0, keepdims=True)
        D = np.concatenate([self.axes, R], axis=1)       # (d_model, d_sae)
        self.decoder = (D / np.linalg.norm(D, axis=0, keepdims=True)).T   # (d_sae, d_model)
        self.d_sae = self.decoder.shape[0]
        self._sae_labels = list(self.axis_names) + [f"feature::{i}" for i in range(n_random)]
        self._seed = random_state

    # ----------------------------- layer profile --------------------------- #
    def _sig_gain(self, layer: int) -> float:
        """Smooth mid-peaking signal gain across layers (SNR peaks mid-network)."""
        mid = 0.6 * (self.n_layers - 1)
        sigma = 0.28 * (self.n_layers - 1)
        return 0.25 + 1.0 * float(np.exp(-((layer - mid) ** 2) / (2 * sigma ** 2)))

    # ----------------------------- data synthesis -------------------------- #
    def make_dataset(self, n_docs: int = 2000):
        """Create synthetic (texts, labels, topic_ids) with planted structure.

        The document text encodes (topic, label, per-doc seed) so activations can be
        reconstructed deterministically at any layer with no in-memory state - this
        keeps multi-process script pipelines reproducible.
        """
        topics = self.rng.integers(0, self.n_topics, size=n_docs)
        labels = self.rng.integers(0, 2, size=n_docs).astype(float)   # 0=A, 1=B
        texts = [f"doc{i} {_MARK_DOC}{int(topics[i])},{int(labels[i])},{i}" for i in range(n_docs)]
        return texts, labels, topics

    def _doc_signal_noise(self, topic: int, label: int, seed: int):
        signal = self.topic_anchors[topic].copy()
        if label >= 1:
            signal = signal + self.topic_shift[topic]
        noise = np.random.default_rng((self._seed, int(seed))).standard_normal(self.d_model) * self.noise_scale
        return signal, noise

    # ------------------------------ text codec ----------------------------- #
    def _encode_vec(self, vec: np.ndarray, tag: str = "gen") -> str:
        b = base64.b64encode(vec.astype(np.float32).tobytes()).decode()
        return f"{tag} {_MARK_VEC}{b}"

    def _decode_vec(self, text: str):
        if _MARK_VEC not in text:
            return None
        b = text.split(_MARK_VEC, 1)[1].strip().split()[0]
        try:
            return np.frombuffer(base64.b64decode(b), dtype=np.float32).astype(np.float64)
        except Exception:
            return None

    def _vec_at_layer(self, text: str, layer: int) -> np.ndarray:
        if _MARK_DOC in text:
            payload = text.split(_MARK_DOC, 1)[1].strip().split()[0]
            topic, label, seed = payload.split(",")
            signal, noise = self._doc_signal_noise(int(topic), int(label), int(seed))
            return signal * self._sig_gain(layer) + noise        # noise flat across layers
        v = self._decode_vec(text)
        if v is not None and v.shape[0] == self.d_model:
            return v                                             # already at its layer scale
        h = abs(hash(text)) % (2**32)                            # last-resort fallback
        return np.random.default_rng(h).standard_normal(self.d_model)

    # ------------------------------ embeddings ----------------------------- #
    def embed(self, texts, layer, pooling="mean"):
        return np.stack([self._vec_at_layer(t, layer) for t in texts])

    def embed_all_layers(self, texts, pooling="mean"):
        return np.stack([self.embed(texts, l) for l in range(self.n_layers + 1)], axis=1)

    def embed_all_layers_masked(self, texts, layers):
        """Synthetic documents have no real tokens (see `make_dataset`'s marker
        strings), so there is nothing to exclude -- one pseudo-token per
        document, zero excluded. Exists so the discovery/gate path is
        unit-testable offline, matching every other backend method here."""
        acts = {l: self.embed(texts, l) for l in layers}
        n = len(texts)
        return {
            "acts": acts,
            "excluded": {
                "total_tokens": n,
                "total_excluded_tokens": 0,
                "per_doc_excluded": np.zeros(n, dtype=int),
            },
        }

    # ------------------------------ generation ----------------------------- #
    def generate(self, prompts, steering_vector=None, layer=None, alpha=0.0,
                 apply_to="all", max_new_tokens=60):
        layer = 0 if layer is None else layer
        out = []
        for p in prompts:
            h = self._vec_at_layer(p, layer)
            if steering_vector is not None and alpha != 0.0:
                sv = np.asarray(steering_vector, dtype=np.float64)
                sv = sv / (np.linalg.norm(sv) + 1e-12)
                h = h + alpha * sv
            out.append(self._encode_vec(h))
        return out

    # --------------------------------- SAE --------------------------------- #
    def sae_encode(self, acts, layer):
        # Linear dictionary projection.  (The real GemmaBackend uses a true JumpReLU
        # SAE; here a linear code is deliberate - it lets the difference a_B - a_A
        # cancel the large arbitrary topic anchor exactly, so the decomposition test
        # recovers the planted axes rather than anchor-driven ReLU artefacts.)
        A = np.asarray(acts, dtype=np.float64)
        feats = A @ self.decoder.T                   # (N, d_sae)
        recon = feats @ self.decoder
        return feats, recon

    def sae_decoder_directions(self, layer):
        return self.decoder.copy()

    def feature_labels(self, layer, indices):
        return [self._sae_labels[i] for i in indices]

    def token_activations(self, texts, layer):
        """Synthetic documents have no real sub-word structure (they encode
        topic/label/seed in a marker string), so each whitespace-separated
        'word' is tagged with its document's activation vector. This exists
        only to exercise `sae.top_activating_tokens` offline; it is not a claim
        about real tokenization."""
        tokens, acts = [], []
        for t in texts:
            vec = self._vec_at_layer(t, layer)
            for w in t.split():
                tokens.append(w)
                acts.append(vec)
        return tokens, (np.stack(acts, 0) if acts else np.zeros((0, self.d_model)))

    # --------- convenience for demos: ground-truth valence direction ------- #
    def valence_direction(self):
        """Unit direction of the planted 'valence' axis (axis 0)."""
        return self.axes[:, 0].copy()


_AXIS_WORDS = [
    "valence", "confidence_doubt", "transparency_obfuscation", "abstraction",
    "temporal_orientation", "normative_framing", "formality", "certainty",
]
