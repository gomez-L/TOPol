"""Real backend: Gemma 3 4B for field representations, generation and narration;
Gemma Scope 2 for the sparse dictionary.

Runs on the user's machine (GPU + model weights + internet for first download).
This backend does NOT feed UMAP -- see backends/segmentation.py for that.
`torch`/`transformers`/`huggingface_hub` are imported lazily inside methods so the
rest of the package stays importable and testable without them.

Layer-index convention (READ THIS)
----------------------------------
We index residual layers by Hugging Face `hidden_states` position:
    hidden_states[0]   = embedding output
    hidden_states[L]   = residual stream AFTER transformer block L-1  (== resid_post of block L-1)
So the field / SAE at `layer = L` live on `hidden_states[L]`, and the matching
Gemma-Scope SAE is the one trained on block `L-1` (`resid_post`).  We therefore
load Gemma-Scope with `gemma_scope_layer = layer - 1`, and steering hooks the
module whose OUTPUT is `hidden_states[L]`, i.e. `model.model.layers[L-1]`.
"""
from __future__ import annotations

import os
import numpy as np

from .base import Backend
from ..config import ModelConfig, SAEConfig
from .. import gemmascope2 as GS2


class GemmaBackend(Backend):
    def __init__(self, model_cfg: ModelConfig, sae_cfg: SAEConfig, sae_l0: int | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = model_cfg
        self.sae_cfg = sae_cfg
        self.sae_l0 = sae_l0
        self._torch = torch

        device = model_cfg.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        dtype = getattr(torch, model_cfg.dtype, torch.float32)

        # Handle cache_dir for gated models
        cache_dir = getattr(model_cfg, "cache_dir", None)
        if cache_dir:
            os.environ["HF_HOME"] = cache_dir

        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg.name, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_cfg.name, dtype=dtype, cache_dir=cache_dir
        ).to(device).eval()

        # Gemma 3 is multimodal; text config is nested under text_config
        config = self.model.config
        text_config = getattr(config, "text_config", config)
        
        # Try common attribute names for hidden size and layer count
        self.d_model = (getattr(text_config, "hidden_size", None) or 
                        getattr(text_config, "hidden_dim", None) or 
                        getattr(text_config, "d_model", None))
        self.n_layers = (getattr(text_config, "num_hidden_layers", None) or 
                         getattr(text_config, "num_layers", None))
        
        if self.d_model is None:
            raise AttributeError(f"Cannot determine hidden size from config. Available attributes: {dir(text_config)}")
        if self.n_layers is None:
            raise AttributeError(f"Cannot determine number of layers from config. Available attributes: {dir(text_config)}")
        self._sae_cache: dict[int, GS2.GemmaScope2SAE] = {}
        self.d_sae = None  # set on first SAE load

    # ------------------------------ embeddings ----------------------------- #
    def _pool(self, hidden, mask, pooling):
        # hidden: (B, T, d) ; mask: (B, T)
        torch = self._torch
        m = mask.unsqueeze(-1).to(hidden.dtype)
        if pooling == "mean":
            return (hidden * m).sum(1) / m.sum(1).clamp_min(1e-9)
        # last non-pad token
        idx = mask.sum(1) - 1
        return hidden[torch.arange(hidden.size(0)), idx]

    def _forward_hidden(self, texts, layers, pooling):
        """Return dict{layer: (N, d)} pooled activations, batched."""
        torch = self._torch
        want = list(layers)
        out = {l: [] for l in want}
        # Requesting every layer keeps the full hidden_states tuple (one (B,T,d)
        # tensor per layer) resident on the GPU at once, so a batch size tuned
        # for single-layer embedding is far too large here -- cap it hard.
        bs = self.cfg.batch_size if len(want) <= 1 else min(self.cfg.batch_size, 2)
        with torch.inference_mode():
            for i in range(0, len(texts), bs):
                batch = texts[i:i + bs]
                enc = self.tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True,
                    max_length=self.cfg.max_length,
                ).to(self.device)
                hs = self.model(**enc, output_hidden_states=True).hidden_states  # tuple len n_layers+1
                for l in want:
                    pooled = self._pool(hs[l], enc["attention_mask"], pooling)
                    out[l].append(pooled.float().cpu().numpy())
                del hs, enc
                if self.device == "cuda":
                    torch.cuda.empty_cache()
        return {l: np.concatenate(v, 0) for l, v in out.items()}

    def embed(self, texts, layer, pooling="mean"):
        return self._forward_hidden(texts, [layer], pooling)[layer]

    def embed_all_layers(self, texts, pooling="mean"):
        layers = list(range(self.n_layers + 1))
        d = self._forward_hidden(texts, layers, pooling)
        return np.stack([d[l] for l in layers], axis=1)          # (N, L+1, d)

    def embed_all_layers_masked(self, texts, layers):
        """Same batching as `_forward_hidden`, but the pooling mask additionally
        drops BOS / special / chat-template positions (kept = attention_mask AND
        NOT a special-token id), so the discovery path's document centroids are
        not dominated by the large-norm BOS position. `_pool`/`_forward_hidden`
        (the CB path) deliberately keep pooling over the full attention mask --
        this is a separate method for exactly that reason, so the CB baseline
        stays bit-identical no matter what this one does.
        """
        torch = self._torch
        special_ids = set(self.tokenizer.all_special_ids)
        want = list(layers)
        out = {l: [] for l in want}
        total_tokens = 0
        total_excluded = 0
        per_doc_excluded: list[int] = []
        bs = self.cfg.batch_size if len(want) <= 1 else min(self.cfg.batch_size, 2)
        with torch.inference_mode():
            for i in range(0, len(texts), bs):
                batch = texts[i:i + bs]
                enc = self.tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True,
                    max_length=self.cfg.max_length,
                ).to(self.device)
                attn = enc["attention_mask"]
                ids = enc["input_ids"]
                special_mask = torch.zeros_like(attn, dtype=torch.bool)
                for sid in special_ids:
                    special_mask |= (ids == sid)
                keep = attn.bool() & (~special_mask)                    # (B, T)
                n_attended = attn.sum(1)
                n_kept = keep.sum(1)
                total_tokens += int(n_attended.sum().item())
                total_excluded += int((n_attended - n_kept).sum().item())
                per_doc_excluded.extend((n_attended - n_kept).cpu().numpy().tolist())

                hs = self.model(**enc, output_hidden_states=True).hidden_states
                keep_f = keep.unsqueeze(-1).to(hs[0].dtype)
                denom = keep_f.sum(1).clamp_min(1e-9)
                for l in want:
                    pooled = (hs[l] * keep_f).sum(1) / denom
                    out[l].append(pooled.float().cpu().numpy())
                del hs, enc
                if self.device == "cuda":
                    torch.cuda.empty_cache()
        acts = {l: np.concatenate(v, 0) for l, v in out.items()}
        return {
            "acts": acts,
            "excluded": {
                "total_tokens": total_tokens,
                "total_excluded_tokens": total_excluded,
                "per_doc_excluded": np.array(per_doc_excluded, dtype=int),
            },
        }

    # ------------------------------ token-level ----------------------------- #
    def token_activations(self, texts, layer):
        """Per-token (not pooled) activations at `layer`, with decoded token
        strings, for corpus-grounded SAE feature labels. Same batching/hook
        conventions as `_forward_hidden`, but keeps every position instead of
        pooling, so the batch size is capped the same way.

        Transfers the whole batch tensor off the GPU ONCE (`hs.float().cpu()
        .numpy()`), not once per token: doing it per-token (the original
        version of this method) issues a separate tiny GPU sync for every
        single token, which for hundreds of documents is hundreds of
        thousands of transfers - bit-identical result, but needlessly slow
        and needlessly hard on host-RAM bookkeeping. Filtering and decoding
        happen afterwards on the already-host-resident numpy arrays.
        """
        torch = self._torch
        special_ids = set(self.tokenizer.all_special_ids)
        bs = min(self.cfg.batch_size, 2)
        all_tokens: list[str] = []
        all_acts: list[np.ndarray] = []
        with torch.inference_mode():
            for i in range(0, len(texts), bs):
                batch = texts[i:i + bs]
                enc = self.tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True,
                    max_length=self.cfg.max_length,
                ).to(self.device)
                hs = self.model(**enc, output_hidden_states=True).hidden_states[layer]  # (B, T, d)
                hs_np = hs.float().cpu().numpy()                       # one bulk transfer per batch
                mask_np = enc["attention_mask"].cpu().numpy()
                ids_np = enc["input_ids"].cpu().numpy()
                for b in range(hs_np.shape[0]):
                    for t in range(hs_np.shape[1]):
                        tid = int(ids_np[b, t])
                        if not bool(mask_np[b, t]) or tid in special_ids:
                            continue
                        piece = self.tokenizer.convert_ids_to_tokens(tid)
                        word = self.tokenizer.convert_tokens_to_string([piece]).strip()
                        all_tokens.append(word or piece)
                        all_acts.append(hs_np[b, t])
                del hs, hs_np, enc
                if self.device == "cuda":
                    torch.cuda.empty_cache()
        acts = np.stack(all_acts, 0) if all_acts else np.zeros((0, self.d_model))
        return all_tokens, acts

    # ------------------------------ generation ----------------------------- #
    def generate(self, prompts, steering_vector=None, layer=None, alpha=0.0,
                 apply_to="all", max_new_tokens=60):
        torch = self._torch
        handle = None
        if steering_vector is not None and alpha != 0.0 and layer is not None:
            sv = torch.tensor(np.asarray(steering_vector), dtype=self.model.dtype, device=self.device)
            sv = sv / (sv.norm() + 1e-9)
            block = self.model.model.layers[layer - 1]           # output == hidden_states[layer]

            def hook(_module, _inp, output):
                hs = output[0] if isinstance(output, tuple) else output
                if apply_to == "last":
                    hs[:, -1, :] = hs[:, -1, :] + alpha * sv
                else:
                    hs = hs + alpha * sv
                if isinstance(output, tuple):
                    return (hs,) + tuple(output[1:])
                return hs

            handle = block.register_forward_hook(hook)

        try:
            texts = []
            bs = self.cfg.batch_size
            with torch.no_grad():
                for i in range(0, len(prompts), bs):
                    batch = prompts[i:i + bs]
                    enc = self.tokenizer(batch, return_tensors="pt", padding=True,
                                         truncation=True, max_length=self.cfg.max_length).to(self.device)
                    gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
                    dec = self.tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                                      skip_special_tokens=True)
                    texts.extend(dec)
            return texts
        finally:
            if handle is not None:
                handle.remove()

    # --------------------------------- SAE --------------------------------- #
    def _get_sae(self, layer):
        if layer in self._sae_cache:
            return self._sae_cache[layer]
        site = "transcoder_all" if self.sae_cfg.use_transcoder else self.sae_cfg.site
        sae = GS2.GemmaScope2SAE.from_hub(
            self.sae_cfg.repo, layer=layer - 1, site=site,
            width=self.sae_cfg.width, l0=self.sae_l0 or self.sae_cfg.l0,
        )
        self.d_sae = sae.W_dec.shape[0]
        self._sae_cache[layer] = sae
        return sae

    def sae_encode(self, acts, layer):
        sae = self._get_sae(layer)
        return sae.encode_decode(np.asarray(acts))

    def sae_decoder_directions(self, layer):
        return self._get_sae(layer).decoder_directions()          # (d_sae, d_model), unit rows

    def feature_labels(self, layer, indices):
        # Best-effort Neuronpedia lookup; falls back to a locator string (still a hypothesis).
        labels = []
        for idx in indices:
            labels.append(self._neuronpedia_label(layer - 1, idx))
        return labels

    def _neuronpedia_label(self, gs_layer, idx):
        try:
            import requests
            src = f"{gs_layer}-gemmascope-res-{self.sae_cfg.width}"
            url = f"https://www.neuronpedia.org/api/feature/{self.sae_cfg.neuronpedia_model}/{src}/{idx}"
            r = requests.get(url, timeout=10)
            if r.ok:
                j = r.json()
                expl = (j.get("explanations") or [{}])[0].get("description")
                if expl:
                    return f"[{gs_layer}:{idx}] {expl}"
        except Exception:
            pass
        return f"[gemma-scope L{gs_layer} feature {idx}] (look up on Neuronpedia)"
