"""
TOPol-Mech: a mechanistic reimplementation of the TOPol semantic-polarity-field
framework, pruned to exactly the modules `notebooks/pipeline_2.ipynb` and its
`scripts/00-08_*.py` equivalents use.

THE CB BASELINE (unchanged from the original TOPol pipeline):
    * The DataFrame-centric flow  text + label  ->  embedding  ->  cluster  ->  drift.
    * Topic segmentation via a neighbour-tuned UMAP graph + Leiden partitioning
      (used ONLY for clustering and 2-D visualisation).
    * Polarity vectors as per-topic centroid differences computed in the FULL
      representation space (never in the 2-D UMAP space): `field.reconstruct_field`.
    * `layer_select`     - pick the residual layer by A/B linear-probe separability.

AXIS DISCOVERY (the pipeline_2.ipynb addition, on top of the CB baseline):
    * `topol_activations` - cache mean-pooled activations at every layer, BOS/
                           special/chat-template tokens excluded from the pool.
    * `topol_axes`       - per-topic, per-layer recursive bipartition (centred
                           cosine 2-means, deflation, cross-layer dedup, depth
                           profile).
    * `topol_gate`       - the split-half stability gate + nuisance diagnostics.
    * `topol_report`     - the final per-axis report (text extremes) + rejected log.
    * `field`            - also carries the CROSS-TOPIC pooling of discovered axes
                           (`pooled_field_assembly`, `axis_families`,
                           `topic_rank_estimate`), alongside the unchanged CB estimator.
    * `nla`              - per-axis, per-discovered-layer NLA/AR interrogation
                           (`prepare_axis_dataset`) of what a surviving axis means.

All model-dependent work goes through a `Backend` (`GemmaBackend` for real runs,
`SyntheticBackend` for offline development/testing).

MODEL: google/gemma-3-4b-pt. `backends/gemma.py` still carries Gemma-Scope-2 SAE
loading code (`gemmascope2.py`) for API compatibility, though pipeline_2.ipynb
never calls it -- axis discovery decomposes nothing onto an SAE dictionary. No
proprietary APIs anywhere in the pipeline.
"""

from .config import Config, default_config
from .store import RunStore

__all__ = ["Config", "default_config", "RunStore"]
__version__ = "0.1.0"
