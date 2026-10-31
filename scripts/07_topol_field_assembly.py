#!/usr/bin/env python3
"""Step 07 - cross-topic field assembly at a fixed reference layer.

Per topic: r_t = number of gate-passing contrasts (a rank ESTIMATE, not a
participation ratio -- recursive projection makes a topic's own axes
orthogonal by construction, see `field.topic_rank_estimate`).

Across topics: pool every gate-passing axis at `discovery.ref_layer` (default:
the layer 02_select_layer.py chose) into one ensemble and compute its
UNCENTRED second moment (participation ratio + spectrum) -- here PR IS
meaningful, since axes from different topics are not constructed orthogonal.
Also clusters the pooled axes into families by |cos| (raw activation-space
direction). For the complementary MEANING-based family clustering (does axis
i contrast similar vocabulary to axis j, regardless of raw direction), see
`14_semantic_family_clustering.py` / `topolm/axis_semantics.py` -- an earlier
depth-profile-correlation clustering was tried here and removed after it
produced a degenerate result (543 singleton families); see `field.py`'s
module docstring for why.

Needs step 06's gate results and step 05's activation cache from the same run.
"""
import warnings; warnings.filterwarnings("ignore")
from _common import base_parser, open_existing_store, load_config_for, apply_overrides
from topolm import pipeline as P

def main():
    ap = base_parser("Cross-topic field assembly")
    ap.add_argument("--ref-layer", type=int, default=None,
                    help="override discovery.ref_layer")
    args = ap.parse_args()
    store = open_existing_store(args)
    cfg = apply_overrides(load_config_for(store), args)
    if args.ref_layer is not None:
        cfg.discovery.ref_layer = args.ref_layer

    r = P.step_topol_field(store, cfg)
    fam = r["families"]
    print(f"[07] k={r['pooled']['k']}  PR={r['pooled']['pr']:.3f} @ layer {r['pooled']['ref_layer']}")
    print(f"[07] families: {len(fam.get('families_by_cos', {}))} by |cos| direction "
          f"(run 14_semantic_family_clustering.py for the meaning-based view)")

if __name__ == "__main__":
    main()
