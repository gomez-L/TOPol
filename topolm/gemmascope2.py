"""Gemma Scope 2 -- sparse autoencoders and transcoders for the Gemma 3 family.

VERIFIED AGAINST THE MODEL CARD 2026-07-23. Earlier versions of this file assumed
the Gemma Scope 1 layout and were wrong in four ways; all four are fixed here and
called out below so the mistakes are not repeated.

WHAT IS ACTUALLY IN A GEMMA SCOPE 2 REPO
-----------------------------------------
Repos are `google/gemma-scope-2-{270m,1b,4b,12b,27b}-{pt,it}`. BOTH pretrained and
instruction-tuned models are covered, at every layer. Each repo has folders:

  Single-layer, 4 layers only (25%, 50%, 65%, 85% depth), WIDE choice of width/L0:
      resid_post, attn_out, mlp_out, transcoder
  Single-layer, EVERY layer, narrower choice of width/L0:
      resid_post_all, attn_out_all, mlp_out_all, transcoder_all
  Multi-layer:
      crosscoder   (4 concatenated residual layers; base models only)
      clt          (cross-layer transcoders; 270M and 1B only)

  SAE id format:  layer_{L}_width_{W}_l0_{small|medium|large}
  Widths:         16k, 64k, 256k, 1m      -- card recommends 64k or 256k
  L0 targets:     small (10-20), medium (30-60), large (60-150)
                                          -- card recommends medium

FOUR THINGS THE PREVIOUS VERSION GOT WRONG
-------------------------------------------
 1. Path layout. Gemma Scope 1 used `layer_{L}/width_{W}/average_l0_{K}/params.npz`.
    Gemma Scope 2 uses `{folder}/layer_{L}_width_{W}_l0_{size}`.
 2. L0 naming. It is a SIZE WORD (small/medium/large), not an integer. The old
    `_default_l0` returned 71, which cannot match anything here.
 3. Width default. We defaulted to 16k; the card recommends 64k or 256k.
 4. Layer coverage. `resid_post` has only FOUR layers. A pipeline that selects a
    layer by probe accuracy will usually land somewhere else -- see below.

THE LAYER-COVERAGE COLLISION, AND HOW WE HANDLE IT
---------------------------------------------------
TOPol selects the analysis layer by cross-validated regime separability, which can
return any layer. But the wide dictionaries live at only 4 depths. Two resolutions,
both supported:

  site = "resid_post_all"  (DEFAULT)  every layer is covered, narrower width/L0
                                      choice. Layer selection stays free.
  site = "resid_post" + constrain_layer_selection = True
                                      restrict layer selection to the 4 subset
                                      depths, buying the wider dictionaries.

Use the default unless you specifically need 256k/1m width, in which case
constrain the layer selection and say so -- it is a real restriction on the
analysis, not a detail.

PT vs IT -- WHICH VARIANT SHOULD THE FIELD USE?
------------------------------------------------
Both exist, so this is a methodological choice rather than an availability
constraint, and we default to **pt** for three reasons:

 1. The field is supposed to measure the CORPUS, not the assistant. Instruction
    tuning adds persona and format structure that has nothing to do with the
    discourse under study, and it lands in the same residual stream the polarity
    vectors are computed from.
 2. Chat-template contamination of pooling. IT models expect a template, and
    template/prefix tokens carry much larger activation norms than content tokens.
    Mean-pooling a document on an IT model therefore weights the template heavily.
    This is not speculation: the Gemma Scope authors excluded exactly those
    positions when training IT SAEs, for exactly this reason.
 3. Base models are sufficient for the generation TOPol actually needs. Steering
    continues a regime-A document; that is pure language modelling, and a base
    model does it more naturally than an assistant-tuned one. Only the contrastive
    narration route genuinely wants instruction following, and that runs as a
    separate, clearly-labelled step.

PT->IT TRANSFER IS DOCUMENTED TO BE GOOD. The Gemma Scope technical report found
that SAEs trained on the base model transfer very well to the IT model, with IT
SAEs only marginally better, and recommends PT SAEs for the equivalent IT model in
many cases. So a pt-dictionary/it-model pairing is a mild, quantified degradation
-- NOT the silent catastrophe an earlier version of these docs claimed.
`check_variant_consistency` therefore warns and explains rather than failing.
"""
from __future__ import annotations

import re
import numpy as np

GEMMA3_SIZES = ("270m", "1b", "4b", "12b", "27b")
SITES_SUBSET = ("resid_post", "attn_out", "mlp_out", "transcoder")
SITES_ALL = ("resid_post_all", "attn_out_all", "mlp_out_all", "transcoder_all")
SITES_MULTI = ("crosscoder", "clt")
WIDTHS = ("16k", "64k", "256k", "1m")
L0_SIZES = ("small", "medium", "large")

# The 4 depths the subset folders are trained at, as fractions of model depth.
SUBSET_DEPTHS = (0.25, 0.50, 0.65, 0.85)


def scope2_repo(size: str = "4b", variant: str = "pt") -> str:
    size = size.lower()
    if size not in GEMMA3_SIZES:
        raise ValueError(f"Gemma Scope 2 covers {GEMMA3_SIZES}; got {size!r}. "
                         "Gemma 4 has no released dictionaries.")
    if variant not in ("pt", "it"):
        raise ValueError("variant must be 'pt' or 'it'")
    return f"google/gemma-scope-2-{size}-{variant}"


def subset_layers(n_layers: int) -> list[int]:
    """The 4 block indices the wide (`resid_post` etc.) dictionaries cover.

    Gemma 3 12B has 48 blocks, so these are roughly 12, 24, 31, 41 -- consistent
    with the model card's own example, `layer_12_width_16k_l0_small`.
    """
    return sorted({int(round(d * n_layers)) for d in SUBSET_DEPTHS})


def check_variant_consistency(model_name: str, sae_variant: str, strict: bool = False):
    """Compare the model suffix with the dictionary variant.

    Mismatch is a documented, mild degradation rather than an error: base-model
    SAEs transfer well to the IT model. We surface it so the choice is deliberate
    and reportable, not accidental.
    """
    m = "it" if model_name.rstrip("/").endswith("-it") else (
        "pt" if model_name.rstrip("/").endswith("-pt") else "unknown")
    ok = (m == sae_variant) or m == "unknown"
    msg = None
    if not ok:
        msg = (f"variant mismatch: model is '-{m}' but dictionaries are "
               f"'-{sae_variant}'. This is tolerable -- base-model SAEs are "
               f"documented to transfer well to IT models, with IT SAEs only "
               f"marginally better -- but it should be a deliberate, reported "
               f"choice. Set sae.variant='{m}' to match.")
        if strict:
            raise ValueError(msg)
    return {"model_variant": m, "sae_variant": sae_variant, "match": ok, "message": msg}


def pooling_advice(model_name: str) -> str | None:
    """Warn about template-token contamination when pooling on an IT model."""
    if model_name.rstrip("/").endswith("-it"):
        return ("pooling on an -it model: chat-template and prefix tokens carry much "
                "larger activation norms than content tokens and will dominate a "
                "mean-pooled document vector. Either feed raw text with no template "
                "and accept a mild distribution shift, or mask template positions "
                "before pooling. The Gemma Scope authors excluded these positions "
                "when training IT SAEs for this reason. Using a -pt model avoids the "
                "issue entirely.")
    return None


# ------------------------------------------------------------------ #
#  Repository introspection
# ------------------------------------------------------------------ #
_PAT = re.compile(r"layer_(\d+)_width_([0-9]+[kKmM])_l0_(small|medium|large)",
                  re.IGNORECASE)


def discover_layout(repo_id: str):
    """List the repo and parse the real Gemma Scope 2 naming convention.

    Returns folders present, and per-folder the available (layer, width, l0) sets.
    Run this once before committing hyperparameters to a paper.
    """
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo_id)

    folders, recs = set(), []
    for f in files:
        parts = f.split("/")
        if len(parts) >= 2:
            folders.add(parts[0])
        m = _PAT.search(f)
        if m:
            recs.append({"file": f, "folder": parts[0],
                         "layer": int(m.group(1)),
                         "width": m.group(2).lower(),
                         "l0": m.group(3).lower()})
    by_folder = {}
    for r in recs:
        d = by_folder.setdefault(r["folder"], {"layers": set(), "widths": set(), "l0s": set()})
        d["layers"].add(r["layer"]); d["widths"].add(r["width"]); d["l0s"].add(r["l0"])
    for d in by_folder.values():
        d["layers"] = sorted(d["layers"]); d["widths"] = sorted(d["widths"])
        d["l0s"] = sorted(d["l0s"])
    return {"repo": repo_id, "n_files": len(files), "folders": sorted(folders),
            "by_folder": by_folder, "records": recs}


def resolve_sae_id(layer: int, width: str = "64k", l0: str = "medium") -> str:
    """Build the SAELens `sae_id` string."""
    if width.lower() not in WIDTHS:
        raise ValueError(f"width must be one of {WIDTHS}; got {width!r}")
    if l0.lower() not in L0_SIZES:
        raise ValueError(f"l0 must be one of {L0_SIZES}; got {l0!r}")
    return f"layer_{int(layer)}_width_{width.lower()}_l0_{l0.lower()}"


def release_id(size: str, variant: str, site: str) -> str:
    """SAELens release string, e.g. 'gemma-scope-2-4b-pt-resid_post_all'."""
    return f"gemma-scope-2-{size.lower()}-{variant}-{site}"


# ------------------------------------------------------------------ #
#  Loading
# ------------------------------------------------------------------ #
def load_via_saelens(size="4b", variant="pt", site="resid_post_all",
                     layer=12, width="64k", l0="medium"):
    """Preferred path. The repos declare `library_name: saelens`.

        sae, cfg_dict, sparsity = load_via_saelens(...)
    """
    from sae_lens import SAE
    return SAE.from_pretrained(release=release_id(size, variant, site),
                               sae_id=resolve_sae_id(layer, width, l0))


def _pick(mapping, *names):
    """Case-insensitive lookup across known Gemma-Scope-2 key spellings.

    `params.safetensors` uses lowercase keys (`w_enc`, `w_dec`, ...); some
    other layouts/tools use `W_enc`/`encoder.weight`. Works for both an
    `np.load` NpzFile and a plain dict (`safetensors.numpy.load_file`).
    """
    keys = list(mapping.keys()) if hasattr(mapping, "keys") else list(mapping.files)
    lower = {k.lower(): k for k in keys}
    for n in names:
        if n in keys:
            return mapping[n]
        if n.lower() in lower:
            return mapping[lower[n.lower()]]
    raise KeyError(f"none of {names} found; available keys: {keys}")


class GemmaScope2SAE:
    """Direct reader, independent of SAELens, exposing the interface topolm needs.

    JumpReLU:  a_f = (w_f . x + b_f) * 1[w_f . x + b_f > theta_f]
    Below threshold exactly zero; above it the value passes through UNSHRUNK,
    which is what avoids the magnitude shrinkage an L1 penalty induces.
    """

    def __init__(self, W_enc, W_dec, b_enc, b_dec, threshold, meta=None):
        self.W_enc = np.asarray(W_enc)          # (d_model, d_sae)
        self.W_dec = np.asarray(W_dec)          # (d_sae, d_model)
        self.b_enc = np.asarray(b_enc)
        self.b_dec = np.asarray(b_dec)
        self.threshold = np.asarray(threshold)
        self.meta = meta or {}

    @classmethod
    def from_saelens(cls, size="4b", variant="pt", site="resid_post_all",
                     layer=12, width="64k", l0="medium"):
        sae, cfg_dict, _ = load_via_saelens(size, variant, site, layer, width, l0)
        g = lambda *names: next(getattr(sae, n) for n in names if hasattr(sae, n))
        to_np = lambda t: t.detach().float().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
        return cls(to_np(g("W_enc")), to_np(g("W_dec")), to_np(g("b_enc")),
                   to_np(g("b_dec")),
                   to_np(g("threshold")) if hasattr(sae, "threshold") else np.zeros(1),
                   meta={"release": release_id(size, variant, site),
                         "sae_id": resolve_sae_id(layer, width, l0),
                         "cfg": dict(cfg_dict) if cfg_dict else {}})

    @classmethod
    def from_hub(cls, repo_id, layer, site="resid_post_all", width="64k", l0="medium"):
        """Fallback: fetch parameter files directly from the Hub."""
        from huggingface_hub import hf_hub_download
        want = resolve_sae_id(layer, width, l0)
        layout = discover_layout(repo_id)
        # An SAE's folder holds several files (weights + a config.json); matching
        # on the id substring alone picks up all of them, and whichever sorts
        # first from the Hub listing might be config.json, not the weights.
        # Keep only the actual parameter-file formats this class can load.
        cands = [r for r in layout["records"]
                 if r["folder"] == site and want in r["file"]
                 and r["file"].endswith((".npz", ".safetensors"))]
        if not cands:
            avail = layout["by_folder"].get(site, {})
            raise FileNotFoundError(
                f"no SAE {want!r} in {repo_id}/{site}. Available there: "
                f"layers={avail.get('layers')}, widths={avail.get('widths')}, "
                f"l0s={avail.get('l0s')}. Folders in repo: {layout['folders']}")
        # An SAE's folder also ships companion files alongside the weights
        # (e.g. "examples.safetensors" = per-feature max-activating examples,
        # not weights). Prefer the real parameter file when one is present.
        preferred = [r for r in cands if r["file"].rsplit("/", 1)[-1].lower().startswith("params")]
        chosen = preferred[0] if preferred else cands[0]
        path = hf_hub_download(repo_id=repo_id, filename=chosen["file"])
        if path.endswith(".npz"):
            p = np.load(path)
            return cls(_pick(p, "W_enc", "w_enc", "encoder.weight"),
                       _pick(p, "W_dec", "w_dec", "decoder.weight"),
                       _pick(p, "b_enc", "b_enc", "encoder.bias"),
                       _pick(p, "b_dec", "b_dec", "decoder.bias"),
                       _pick(p, "threshold", "log_threshold"),
                       meta={**chosen, "repo": repo_id})
        if path.endswith(".safetensors"):
            from safetensors.numpy import load_file
            t = load_file(path)
            return cls(_pick(t, "W_enc", "w_enc"), _pick(t, "W_dec", "w_dec"),
                       _pick(t, "b_enc", "b_enc"), _pick(t, "b_dec", "b_dec"),
                       _pick(t, "threshold", "log_threshold") if any(
                           k.lower() in ("threshold", "log_threshold") for k in t.keys())
                       else np.zeros(1),
                       meta={**chosen, "repo": repo_id})
        raise ValueError(f"unsupported parameter file: {path}")

    # ------------------------------------------------------------- #
    def encode(self, acts):
        # SAELens' reference JumpReLUSAE centers the input on `b_dec` before the
        # encoder matmul (`process_sae_in`, `apply_b_dec_to_input` defaults to
        # True) -- decode() already adds b_dec back, so skipping this here was
        # asymmetric. b_dec reaches into the tens of thousands on some
        # dimensions (Gemma's residual stream has large, roughly per-dimension-
        # constant outlier activations), so without centering, `pre` was
        # dominated by the raw offset and essentially never crossed threshold.
        x = np.asarray(acts, dtype=np.float32) - self.b_dec
        pre = x @ self.W_enc + self.b_enc
        return pre * (pre > self.threshold)          # JumpReLU

    def decode(self, feats):
        return feats @ self.W_dec + self.b_dec

    def encode_decode(self, acts):
        f = self.encode(acts)
        return f, self.decode(f)

    def decoder_directions(self):
        W = np.asarray(self.W_dec, dtype=np.float32)
        return W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)

    @property
    def d_sae(self):
        return int(self.W_dec.shape[0])

    @property
    def d_model(self):
        return int(self.W_dec.shape[1])


# ------------------------------------------------------------------ #
def neuronpedia_label(model_id, source_id, index, timeout=10):
    """Best-effort auto-interp label. TREAT AS A HYPOTHESIS, NOT A FINDING.

    These are produced by prompting a model with max-activating examples --
    epistemically the same move as the contrastive narration this pipeline
    replaces, just applied to a fixed reusable basis. What makes the dictionary
    route an improvement is not label trustworthiness but that (a) the
    decomposition is a geometric fact independent of any label, (b) coverage is
    measured, (c) the pole sign is intrinsic, and (d) every named feature is a
    steerable direction, so the label becomes falsifiable.
    """
    try:
        import requests
        url = f"https://www.neuronpedia.org/api/feature/{model_id}/{source_id}/{index}"
        r = requests.get(url, timeout=timeout)
        if r.ok:
            expl = (r.json().get("explanations") or [{}])[0].get("description")
            if expl:
                return f"[{source_id}:{index}] {expl}"
    except Exception:
        pass
    return f"[{source_id} feature {index}] (unlabelled; check Neuronpedia)"
