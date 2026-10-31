"""Central, serialisable configuration for the TOPol-Mech pipeline.

MODEL STRATEGY: ONE OPEN ANALYSIS MODEL
---------------------------------------
Everything that touches meaning runs on ONE open model: **Gemma 3 4B**.

  field + SAE + steering + narration :  Gemma 3 4B    (google/gemma-3-4b-*)
  topic segmentation (labels only)   :  EmbeddingGemma-300m

No proprietary APIs anywhere in the pipeline. This is a methodological
requirement, not a preference. The analysis model must be the SAME model whose
activations carry the field; otherwise the narration describes a different system
from the one being measured, and nothing it says can be checked against the
geometry. A closed model cannot be probed, steered, ablated or decomposed.

The one exception is topic segmentation, which produces LABELS ONLY. A partition
is just an assignment of documents to groups; it can be computed anywhere and
carried anywhere. We compute it with a small open sentence encoder because that
is what similarity geometry needs, and because it removes a circularity: topic
identity is then fixed independently of the space whose internals we probe. Set
`segmentation.source = "field"` to use Gemma 3 4B for this too and drop to a
literally single-model pipeline.

PT OR IT?  -- we default the FIELD to **-pt** (verified 2026-07-23)
-------------------------------------------------------------------
Gemma Scope 2 covers BOTH pretrained and instruction-tuned models at every layer,
so this is a methodological choice, not an availability constraint. We choose -pt
for the field because:

  1. The field measures the CORPUS, not the assistant. Instruction tuning writes
     persona and format structure into the same residual stream the polarity
     vectors are computed from.
  2. Chat-template contamination. IT models expect a template, and template/prefix
     tokens carry much larger activation norms than content tokens, so they
     dominate a mean-pooled document vector. The Gemma Scope authors excluded
     exactly those positions when training IT SAEs.
  3. Base models are enough for the generation TOPol needs. Steering continues a
     regime-A document -- pure language modelling, which a base model does more
     naturally than an assistant-tuned one.

Only the contrastive narration route genuinely wants instruction following, so it
gets its own model field (`narration.model`, default -it). Base->IT SAE transfer is
documented to be good, so the mixed setup is a mild, quantified compromise rather
than an error.

WHY GEMMA 3 4B SPECIFICALLY (checked 2026-07-22)
------------------------------------------------
Gemma Scope 2 -- the released sparse-autoencoder suite -- covers the Gemma 3
family only, at both -pt and -it (270m/1b/4b/12b/27b). Gemma 4 shipped
2026-07-21 with no released dictionaries. Faithful decomposition is a core
contribution, so dictionary availability is the binding constraint, not raw
capability. 4B is the size we run everything on -- narrator, field and
activation-patching subject alike. We deliberately do NOT report other sizes:
a single fixed model keeps every number in the paper commensurable.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal
import json


@dataclass
class SegmentationConfig:
    """Representation used ONLY to build the topic partition (labels only)."""
    name: str = "google/embeddinggemma-300m"
    prompt_name: str | None = "Clustering"   # EmbeddingGemma task prefix
    batch_size: int = 64
    normalize: bool = True
    # "embedding" = dedicated embedder (recommended); "field" = reuse the
    # generator's residual stream (the old single-model behaviour, kept for
    # ablation so the paper can report what decoupling bought).
    source: Literal["embedding", "field"] = "embedding"
    # Fit the partition on regime A only and assign B by nearest centroid,
    # which makes topic identity independent of the contextual boundary
    # (removes the Proposition 3 selection bias).
    boundary_independent: bool = True


@dataclass
class ModelConfig:
    """THE analysis model. Its residual stream IS the analysis space, and it is
    also the narrator. One model, one space, everything checkable against
    everything else."""
    name: str = "google/gemma-3-4b-pt"       # -pt: the field measures text, not the assistant
    layer: int | None = None                 # None -> chosen by layer_select
    pooling: Literal["mean", "last"] = "mean"
    dtype: str = "bfloat16"
    device: str = "auto"
    max_length: int = 2048
    batch_size: int = 8
    attn_implementation: str | None = None   # e.g. "eager" for stable hooks


@dataclass
class NarrationConfig:
    """Route 3: contrastive labelling.

    Runs on the instruction-tuned sibling of the field model, because this is the
    one step that needs reliable instruction following (JSON-formatted output).
    Set `model = None` to use the field model itself and keep the pipeline on a
    single set of weights -- a base model can do this with few-shot prompting,
    less reliably. Either way the narrator is open and probeable, which is the
    property that matters: its hypotheses can be checked against the field.
    """
    enabled: bool = True
    model: str | None = "google/gemma-3-4b-pt"
    n_exemplars: int = 12                    # per pole
    max_dims: int = 5
    min_evidence: int = 2
    chars_per_side: int = 12000              # fits an open context window
    max_new_tokens: int = 900


@dataclass
class AffectiveConfig:
    """The affective residual test: is the field reducible to emotion?"""
    enabled: bool = True
    # "vad"      -> compact bipolar valence/arousal/dominance basis (rank <= 3)
    # "emotions" -> mean-difference vectors from an emotion-labelled prompt set
    source: Literal["vad", "emotions"] = "vad"
    rank: int | None = None                  # None -> chosen by energy
    energy: float = 0.95
    rank_sweep: tuple[int, ...] = (1, 2, 3, 5, 8)
    n_random_controls: int = 20
    emotion_prompts_json: str | None = None  # {emotion: [prompts]} for "emotions"


@dataclass
class SAEConfig:
    """Gemma Scope 2 -- the sparse autoencoder suite for Gemma 3."""
    enabled: bool = True
    size: str = "4b"                         # fixed: matches Gemma 3 4B
    variant: Literal["pt", "it"] = "pt"      # should match the FIELD model suffix

    # Folder inside the repo. The "_all" folders cover EVERY layer with a narrower
    # width/L0 range; the plain folders cover only 4 depths (25/50/65/85%) but with
    # the full width range up to 1m. We default to _all so layer selection stays
    # free; see `constrain_layer_selection` for the alternative.
    site: str = "resid_post_all"
    constrain_layer_selection: bool = False  # True -> restrict layers to the 4 wide depths

    # Gemma Scope 2 naming: layer_{L}_width_{W}_l0_{small|medium|large}
    width: str = "64k"                       # card recommends 64k or 256k
    l0: str = "medium"                       # card recommends medium (L0 ~30-60)
    use_transcoder: bool = False             # site="transcoder_all"; reportedly
                                             # more interpretable than SAEs
    top_k_features: int = 12
    coverage_k: int = 20
    neuronpedia_model: str = "gemma-3-4b"    # confirm on neuronpedia.org/gemma-scope-2
    # NOTE: an -it model with -pt dictionaries is a SILENT error: it degrades
    # coverage without raising. Keep `variant` equal to the model suffix.
    use_saelens: bool = False                # True -> load via SAELens instead

    @property
    def repo(self) -> str:
        return f"google/gemma-scope-2-{self.size}-{self.variant}"

    @property
    def release(self) -> str:
        """SAELens release string."""
        site = "transcoder_all" if self.use_transcoder else self.site
        return f"gemma-scope-2-{self.size}-{self.variant}-{site}"


@dataclass
class ConceptConfig:
    """Route 2: supervised projection onto externally named bipolar axes."""
    enabled: bool = True
    dimensions_pickle: str = "data/narrative_dimensions.pickle"
    center: bool = True                      # anisotropy correction
    top_k: int = 12
    n_null: int = 2000


@dataclass
class CorpusConfig:
    """Real-corpus paths and contextual-boundary rules.

    The CB rule is a research decision, so it lives in the config, is serialised
    with the run, and lands in the manifest. Changing it is changing the question.
    """
    # --- US central bank speeches (DISCOVERY condition) ---
    speeches_path: str | None = None          # default: data/us_fed_speeches.pickle
    breakpoint: str = "2007-05-01"            # pre-GFC macroeconomic regime break
    window_years: int | None = None           # symmetric window; None = whole corpus
    speech_min_chars: int = 500
    max_docs: int | None = 600                # reference corpus size after filtering

    # --- product reviews (CALIBRATION condition) ---
    amazon_path: str | None = None            # default: data/amazon_reviews.pickle
    n_per_class: int = 5000                   # 10k balanced total
    negative_max: int = 2                     # A pole: ratings <= 2
    positive_min: int = 4                     # B pole: ratings >= 4 (3 dropped)
    review_min_chars: int = 100


@dataclass
class ManifoldConfig:
    n_neighbors: int = 15
    min_dist: float = 0.0
    n_components: int = 5
    metric: str = "cosine"
    leiden_resolution: float = 1.0


@dataclass
class StatsConfig:
    n_permutations: int = 1000
    n_bootstrap: int = 1000
    alpha: float = 0.05


@dataclass
class PredictiveConfig:
    """Leave-one-topic-out predictive rank curve (RISE-inspired)."""
    enabled: bool = True
    ranks: tuple[int, ...] | None = None      # None -> 1..k-1
    n_permutations: int = 200
    include_spherical: bool = True            # canonicalised-tangent variant
    saturation_frac: float = 0.90             # defines r*


@dataclass
class DiscoveryConfig:
    """Recursive-bipartition discovery + the split-half gate (the V1 axis-discovery
    path, distinct from and additive to the CB diff-in-means field in field.py).

    layer_stride     cache every `layer_stride`-th residual layer (plus layer 0 and
                     the last layer always) to bound activation-cache memory.
    max_depth        recursion depth of the per-topic, per-layer deflation sequence.
    min_cluster_size stop recursing once either side of a 2-means split falls below
                     this many documents.
    ari_merge        cross-layer dedup: candidate partitions with average-linkage
                     ARI similarity >= this are merged into one contrast.
    stability_threshold  split-half gate: a contrast passes if its matched
                     half-vs-half |cos| at l* is >= this.
    excess_cos_threshold the baseline-comparison headline count: a gate-passing
                     axis counts as "excess structure" if |cos| to v_CB^t at
                     l_ref is below this.
    ref_layer        l_ref for pooled cross-topic field assembly and baseline
                     comparison. None -> the layer step_select_layer chose, so
                     results stay comparable to prior (CB-only) runs.
    sentiment_corr_threshold  baseline comparison: below this |corr(v_CB^t's own
                     projection, continuous rating)|, v_CB^t itself is flagged as
                     weakly sentiment-aligned in that topic -- a check on the
                     human-coded CB, not an assumption about WHERE discovery
                     should find a match (see `pipeline.step_topol_baseline_compare`).
    """
    layer_stride: int = 2
    max_depth: int = 4
    min_cluster_size: int = 30
    ari_merge: float = 0.7
    stability_threshold: float = 0.6
    excess_cos_threshold: float = 0.3
    ref_layer: int | None = None
    sentiment_corr_threshold: float = 0.2


@dataclass
class AblationConfig:
    """Causal ablation study (`causal.ipynb` / `causal_ablation.py`): remove the
    sentiment-related directions from a topic's own activations and see what
    survives -- the necessity-half of the "is this direction load-bearing for
    the regime split" question the field/discovery steps only answer
    correlationally.

    Two families of direction are identified per topic and (optionally)
    removed together:
      1. `v_CB^t` itself (`include_cb`) -- the supervised, rating-derived
         polarity vector every other step treats as ground truth.
      2. Gate-passing DISCOVERED axes flagged as sentiment-related by an
         independent signal -- a pretrained RoBERTa sentiment classifier's
         per-document polarity score (`pole_sentiment.py`), not the topol
         pipeline's own `corr_rating`/`cos_to_cb`, so the criterion for "this
         axis is about sentiment" does not come from the same measurement
         being tested. An axis qualifies if `|sentiment_delta|` (mean RoBERTa
         polarity, pole B minus pole A) is >= `sentiment_delta_threshold`.

    Both families are ablated at ONE shared layer (`layer`, default: the run's
    `topol_field_summary["ref_layer"]`) by orthogonal projection, per topic,
    so ablated documents are only ever compared against directions discovered
    in their OWN topic.
    """
    layer: int | None = None                  # None -> this run's ref_layer
    include_cb: bool = True
    include_discovered_axes: bool = True
    sentiment_delta_threshold: float = 0.3    # ~80th pct of |sentiment_delta| in the pilot run
    n_random_controls: int = 20               # matched-rank random-subspace controls per topic
    probe_n_splits: int = 5
    probe_C: float = 1.0


@dataclass
class SteeringConfig:
    alphas: tuple[float, ...] = (-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0)
    apply_to: Literal["all", "last"] = "all"
    max_new_tokens: int = 60
    n_prompts: int = 24
    panel: tuple[str, ...] = ("probe_projection", "ab_classifier", "valence")
    norm_relative: bool = True                # scale alpha by residual norm


@dataclass
class Config:
    dataset: Literal["us_speeches", "amazon_reviews", "synthetic_1d", "synthetic_kd"] = "synthetic_kd"
    backend: Literal["gemma", "synthetic"] = "synthetic"
    random_state: int = 42
    runs_dir: str = "runs"
    model: ModelConfig = field(default_factory=ModelConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    narration: NarrationConfig = field(default_factory=NarrationConfig)
    affective: AffectiveConfig = field(default_factory=AffectiveConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    concept: ConceptConfig = field(default_factory=ConceptConfig)
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    manifold: ManifoldConfig = field(default_factory=ManifoldConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    predictive: PredictiveConfig = field(default_factory=PredictiveConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        def _t(sub):
            return {k: (tuple(v) if isinstance(v, list) else v) for k, v in sub.items()}
        return cls(
            dataset=d.get("dataset", "synthetic_kd"),
            backend=d.get("backend", "synthetic"),
            random_state=d.get("random_state", 42),
            runs_dir=d.get("runs_dir", "runs"),
            model=ModelConfig(**d.get("model", {})),
            segmentation=SegmentationConfig(**d.get("segmentation", {})),
            narration=NarrationConfig(**d.get("narration", {})),
            affective=AffectiveConfig(**_t(d.get("affective", {}))),
            sae=SAEConfig(**d.get("sae", {})),
            concept=ConceptConfig(**d.get("concept", {})),
            corpus=CorpusConfig(**d.get("corpus", {})),
            manifold=ManifoldConfig(**d.get("manifold", {})),
            stats=StatsConfig(**d.get("stats", {})),
            predictive=PredictiveConfig(**_t(d.get("predictive", {}))),
            steering=SteeringConfig(**_t(d.get("steering", {}))),
            discovery=DiscoveryConfig(**d.get("discovery", {})),
            ablation=AblationConfig(**d.get("ablation", {})),
        )


def default_config(**overrides) -> Config:
    cfg = Config()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
