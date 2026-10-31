#!/usr/bin/env python3
"""Part 2, step 11 - the polar field, one topic at a time.

`07_topol_field_assembly.py` already pools every gate-passing axis across
ALL topics into one ensemble (`field.pooled_field_assembly`) -- useful for a
cross-topic view, but with ~40 axes per topic and 14-15 topics pooled
together, that single view is too dense to read any one topic's own
structure from. This subsets that SAME pooled ensemble by topic and reuses
the EXISTING `viz.plot_vector_relations`/`plot_vector_compass` (which only
need a `(k, d)` matrix + ids, nothing topic-specific) -- one relations +
compass pair per topic, so each topic's own field can be read on its own
terms as well as pooled.

Needs `06_topol_discovery.py`/`07_topol_field_assembly.py`'s gate results and
cache from the same run.
"""
import warnings; warnings.filterwarnings("ignore")
from _common import base_parser, open_existing_store, load_config_for, apply_overrides
from topolm import field as F
from topolm import dimensionality as DIM
from topolm import topol_activations as TA
from topolm import viz as V
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = base_parser("Per-topic polar field views")
    args = ap.parse_args()
    store = open_existing_store(args)
    apply_overrides(load_config_for(store), args)

    gate_results = store.load_object("topol_gate_results")
    cache = TA.load_activation_cache(store)
    field_summary = store.load_json("topol_field_summary")
    ref_layer = field_summary["ref_layer"]

    pooled = F.pooled_field_assembly(gate_results, ref_layer, cache["layers"])
    topic_ids = np.asarray(pooled["topic_ids"])
    contrast_ids = np.asarray(pooled["contrast_ids"])
    V_all = pooled["V"]

    n_done = 0
    for t in sorted(set(topic_ids.tolist())):
        mask = topic_ids == t
        if mask.sum() < 2:
            print(f"[11] topic {t}: only {mask.sum()} gate-passing axis(es) -- need >= 2 to plot, skipping")
            continue
        Vt = V_all[mask]
        ids_t = [f"c{c}" for c in contrast_ids[mask]]
        axes2d, _ = DIM.principal_axes(Vt, n_axes=2, center=False)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
        V.plot_vector_relations(Vt, ids_t, ax=axs[0], title=f"Topic {t} -- axis cosine similarity")
        V.plot_vector_compass(Vt, ids_t, axes2d, ax=axs[1], title=f"Topic {t} -- 2-D projection")
        store.save_fig(f"pv_topic_field_{t}", fig)
        plt.close(fig)
        n_done += 1
        print(f"[11] topic {t}: {mask.sum()} axes plotted -> figures/pv_topic_field_{t}")

    print(f"[11] {n_done}/{len(set(topic_ids.tolist()))} topics plotted "
          f"(pooled, all-topics view already exists as figures/19_pooled_field_of_axes)")


if __name__ == "__main__":
    main()
