"""Natural Language Autoencoder (NLA) + Activation Reconstruction (AR).

A small, self-contained companion analysis, downstream of the main TOPol-Mech
pipeline and deliberately decoupled from it: everything here trains on the
POOLED per-document activations and texts the main pipeline already saved to
disk (`embeddings.npz`, `texts.pkl`, `cluster_labels.npz`, `labels.npz` in a
`RunStore` run directory) -- it never touches Gemma or the GPU model. That is
a deliberate simplification, not an oversight:

  * per-TOKEN activation sequences would need the same heavy token-level
    forward pass that caused today's OOM crashes; pooled per-document vectors
    are already computed, already on disk, and are enough to ask "how well
    does a small bottleneck reconstruct a document's activation, and does
    that differ between regime A and regime B".
  * the "description" a document's activation is trained to reconstruct is
    the document's OWN (truncated) text -- a bootstrapped label that needs no
    external annotation step, using a small self-contained whitespace
    vocabulary built from the corpus itself rather than the model's own
    (256k-token) tokenizer.

Stages implemented here (of the user's larger 8-stage spec), scoped down
deliberately for a first, simple pass:

  STAGE 1  data prep         -> `prepare_dataset`
  STAGE 2  NLA training       -> `Vocab`, `NLAEncoder`, `NLADecoder`, `train_nla`
                                 (VAE-style: CE reconstruction loss + KL
                                 regularisation toward a standard normal prior)
  STAGE 3  activation          -> `ARHead`, `train_ar_cv`
           reconstruction (AR)   (a SEPARATE MLP trained on the frozen NLA
                                 encoder's embeddings, evaluated by 5-fold CV
                                 so every document gets an out-of-fold RMSE)
  (contrast)                  -> `pole_contrast_table`
                                 per-cluster comparison of AR reconstruction
                                 RMSE between regime A's documents and regime
                                 B's documents -- does the autoencoder
                                 represent one regime more faithfully than the
                                 other, per topic?

Stages 4-8 (behavioural correlation, coherence validation, intervention
tests, summary scoring, narrative export) are NOT implemented -- out of scope
for this first pass.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"


# ------------------------------------------------------------------ #
#  STAGE 1 - data prep: a tiny self-contained vocabulary + tokenizer
# ------------------------------------------------------------------ #
def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+|[.,!?;:]", text.lower())


class Vocab:
    """Whitespace/regex word-level vocabulary built from the corpus itself --
    not the model's own tokenizer, so this module needs no Gemma/GPU access."""

    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    def encode(self, text: str, max_len: int = 48) -> list[int]:
        toks = _tokenize(text)[: max_len - 2]
        unk = self.stoi[UNK]
        return [self.stoi[BOS]] + [self.stoi.get(t, unk) for t in toks] + [self.stoi[EOS]]

    def decode(self, ids) -> str:
        words = []
        for i in ids:
            i = int(i)
            w = self.itos[i] if 0 <= i < len(self.itos) else UNK
            if w == EOS:
                break
            if w in (PAD, BOS):
                continue
            words.append(w)
        return " ".join(words)


def build_vocab(texts, max_size: int = 4000) -> Vocab:
    counter = Counter()
    for t in texts:
        counter.update(_tokenize(t))
    specials = [PAD, BOS, EOS, UNK]
    words = [w for w, _ in counter.most_common(max_size) if w not in specials]
    return Vocab(specials + words)


def _standardize(acts_raw: np.ndarray):
    """Per-dimension z-score, fit on this dataset's own activations. Different
    datasets (and different chosen layers) have wildly different natural
    activation scale -- e.g. US speeches (layer 34) has max |value| ~46,
    Amazon reviews (layer 16) has max |value| ~36,000 (Gemma's residual
    stream carries large, roughly per-dimension-constant outlier values;
    see the `b_dec` centering fix in gemmascope2.py for the same
    phenomenon). Feeding raw, un-normalised activations into a freshly
    initialised Linear layer overflows (`exp(logvar)` -> inf -> NaN loss)
    for whichever dataset happens to have the larger natural scale.

    Factored out of `prepare_dataset` so `prepare_axis_dataset` (which
    standardises a pooled, multi-layer dataset) uses the exact same rule.
    """
    act_mean = acts_raw.mean(axis=0)
    act_std = acts_raw.std(axis=0) + 1e-6
    return (acts_raw - act_mean) / act_std, act_mean, act_std


def _tokenize_batch(texts, vocab: Vocab, max_len: int) -> np.ndarray:
    token_ids = np.full((len(texts), max_len), vocab.pad_id, dtype=np.int64)
    for i, t in enumerate(texts):
        ids = vocab.encode(t, max_len=max_len)
        token_ids[i, : len(ids)] = ids
    return token_ids


def prepare_dataset(texts, activations, cluster_labels, vocab, max_len=48,
                    layer_id: int = 0):
    """STAGE 1. Pair each document's (already extracted) pooled activation
    with its own (truncated) text, tokenised against `vocab`, plus its topic
    id -- ready for `train_nla`/`train_ar_cv`.

    Returns a dict of numpy/plain-python arrays, all length N (one row per
    document): activations (N, d_model), token_ids (N, max_len) int64
    (pad-collated), topic_ids (N,) int64, layer_ids (N,) int64 (constant here
    -- one dataset is analysed at one fixed layer, chosen by probe
    separability upstream; kept as a real input so a future joint-dataset
    run has somewhere to put a non-constant layer id -- see
    `prepare_axis_dataset` below for exactly that joint-dataset run).
    """
    acts_raw = np.asarray(activations, dtype=np.float32)
    cluster_labels = np.asarray(cluster_labels)
    topic_ids = np.searchsorted(np.unique(cluster_labels), cluster_labels).astype(np.int64)
    token_ids = _tokenize_batch(texts, vocab, max_len)
    acts, act_mean, act_std = _standardize(acts_raw)

    return {
        "activations": acts,
        "activations_raw": acts_raw,
        "act_mean": act_mean,
        "act_std": act_std,
        "token_ids": token_ids,
        "topic_ids": topic_ids,
        "layer_ids": np.full(len(texts), int(layer_id), dtype=np.int64),
        "n_topics": int(topic_ids.max()) + 1 if len(topic_ids) else 1,
    }


# ------------------------------------------------------------------ #
#  The "future joint-dataset run" prepare_dataset's non-constant
#  layer_ids field was left in place for: one axis per (topic, gate-
#  passing contrast) from `topol_gate.gate_topic`, EACH read off at ITS
#  OWN discovered peak layer l*, pooled into one dataset and trained as
#  one NLA/AR model -- so NLA interrogation of a discovered polarity
#  vector always uses the layer that vector was actually found at,
#  never one fixed dataset-wide layer.
# ------------------------------------------------------------------ #
def prepare_axis_dataset(gate_results: dict, texts, cluster_labels, cache: dict,
                         vocab: Vocab, max_len: int = 48):
    """Pool every gate-PASSING discovered axis (topic, contrast) into one
    long-format NLA dataset: one row per document-within-that-axis's-topic,
    with the activation taken from `cache["acts"][l*]` -- THAT axis's own
    peak layer, not a single dataset-wide layer.

    Each axis becomes its own `topic_id` for NLA's conditioning embeddings
    (an "axis id"), and each row's `layer_id` indexes into the sorted set of
    distinct l* values actually used across all surviving axes -- reusing
    `NLAEncoder`/`NLADecoder`'s existing (topic, layer) conditioning exactly
    as designed, with no changes to the model classes themselves. The pole
    label for each row is the AXIS's OWN discovered partition (`contrast
    ["labels"]`), not the corpus's CB regime label -- this dataset is for
    interrogating discovered axes, not re-deriving the CB pole.

    A document that participates in several gate-passing axes within the
    same topic appears once per axis (with a different layer/pole each
    time) -- this is intentional: each axis is a distinct semantic contrast,
    and NLA should see it in its own right rather than only once per corpus.

    Returns (data, axis_index):
      data       : dict ready for `train_nla`/`train_ar_cv`, exactly like
                   `prepare_dataset`'s return shape, PLUS `texts` (the
                   per-row text list, same order/length as every array) and
                   `pole_labels` (N,) -- each row's axis-local A/B label.
      axis_index : DataFrame, one row per axis id, with topic/contrast_index/
                   l_star/effect_size/split_half_cos/n_docs -- the lookup
                   table for turning an `axis_id` back into what it means.
    """
    cluster_labels = np.asarray(cluster_labels)
    layers_used = sorted({c["l_star"] for res in gate_results.values()
                          for c in res["contrasts"] if c["passed"]})
    layer_to_idx = {l: i for i, l in enumerate(layers_used)}

    row_texts, row_acts, row_axis_id, row_layer_id, row_pole = [], [], [], [], []
    axis_rows = []
    axis_id = 0
    for t, res in gate_results.items():
        doc_idx = np.where(cluster_labels == t)[0]
        for ci, c in enumerate(res["contrasts"]):
            if not c["passed"]:
                continue
            l = c["l_star"]
            acts_l = np.asarray(cache["acts"][l], dtype=np.float32)[doc_idx]
            labels = np.asarray(c["labels"])
            for k in range(len(doc_idx)):
                row_texts.append(texts[doc_idx[k]])
                row_acts.append(acts_l[k])
                row_axis_id.append(axis_id)
                row_layer_id.append(layer_to_idx[l])
                row_pole.append(int(labels[k]))
            axis_rows.append({
                "axis_id": axis_id, "topic": int(t), "contrast_index": int(ci),
                "l_star": int(l), "layer_id": int(layer_to_idx[l]),
                "effect_size_at_lstar": c["effect_size_at_lstar"],
                "split_half_cos": c["split_half_cos"], "n_docs": int(len(doc_idx)),
            })
            axis_id += 1

    if not row_texts:
        raise ValueError("no gate-passing axes to build an NLA dataset from -- "
                         "run topol_gate.gate_topic first, or relax discovery.stability_threshold")

    acts_raw = np.stack(row_acts).astype(np.float32)
    acts, act_mean, act_std = _standardize(acts_raw)
    token_ids = _tokenize_batch(row_texts, vocab, max_len)
    n_axes = axis_id

    data = {
        "activations": acts, "activations_raw": acts_raw,
        "act_mean": act_mean, "act_std": act_std, "token_ids": token_ids,
        "topic_ids": np.asarray(row_axis_id, dtype=np.int64),
        "layer_ids": np.asarray(row_layer_id, dtype=np.int64),
        "n_topics": n_axes,
        "texts": row_texts, "pole_labels": np.asarray(row_pole, dtype=np.int64),
        "layers_used": layers_used,
    }
    axis_index = pd.DataFrame(axis_rows)
    return data, axis_index


def subsample_axis_dataset(data: dict, max_per_pole: int = 25, random_state: int = 42) -> dict:
    """Cap each (axis, pole) cell of a `prepare_axis_dataset` dataset at
    `max_per_pole` documents, sampled without replacement.

    `prepare_axis_dataset` pools every gate-passing axis's FULL topic every
    time (one row per document per axis it belongs to), so a corpus with
    ~40 axes/topic produces a dataset ~40x the corpus size (e.g. 9,971 docs
    -> ~393k rows for the Amazon run). Training NLA's GRU decoder on that
    many rows is a CPU-hours job; capping each (axis, pole) cell at a fixed
    size keeps every axis represented (so every axis still gets a trained
    prototype/sweep) while keeping wall-clock training tractable on CPU.
    This only shrinks the TRAINING set -- `pole_prototypes_table`/
    `direction_sweep_table` are meant to be called afterwards against the
    FULL (unsampled) dataset, so centroids and nearest-real-doc grounding
    still draw on every real document, not just the training subsample.
    """
    rng = np.random.default_rng(random_state)
    axis_ids = np.asarray(data["topic_ids"])
    poles = np.asarray(data["pole_labels"])
    keep = []
    for aid in np.unique(axis_ids):
        for p in (0, 1):
            idx = np.where((axis_ids == aid) & (poles == p))[0]
            if len(idx) > max_per_pole:
                idx = rng.choice(idx, size=max_per_pole, replace=False)
            keep.append(idx)
    keep = np.sort(np.concatenate(keep)) if keep else np.array([], dtype=int)

    out = {}
    n = len(axis_ids)
    for k, v in data.items():
        if isinstance(v, np.ndarray) and len(v) == n:
            out[k] = v[keep]
        elif isinstance(v, list) and len(v) == n:
            out[k] = [v[i] for i in keep]
        else:
            out[k] = v
    return out


def axis_layer_id_map(axis_index) -> dict:
    """{axis_id: layer_id} lookup for `pole_prototypes_table`/`direction_sweep_table`'s
    `layer_id_map`, straight from `prepare_axis_dataset`'s `axis_index`."""
    return dict(zip(axis_index["axis_id"].astype(int), axis_index["layer_id"].astype(int)))


# ------------------------------------------------------------------ #
#  STAGE 2 - NLA: activation <-> 256-d embedding <-> description tokens
# ------------------------------------------------------------------ #
class NLAEncoder(nn.Module):
    """activation (+ topic, layer) -> (mu, logvar) of a latent embedding.

    VAE-style stochastic bottleneck: reparameterised sampling at train time,
    KL-regularised toward a standard normal prior (see `train_nla`), so the
    latent space is smooth/interpolatable rather than an arbitrary bottleneck.
    """

    def __init__(self, d_model: int, n_topics: int, n_layers: int,
                latent_dim: int = 256, hidden_dim: int = 512, cond_dim: int = 32):
        super().__init__()
        self.topic_emb = nn.Embedding(max(n_topics, 1), cond_dim)
        self.layer_emb = nn.Embedding(max(n_layers, 1), cond_dim)
        self.net = nn.Sequential(
            nn.Linear(d_model + 2 * cond_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, acts, topic_ids, layer_ids):
        x = torch.cat([acts, self.topic_emb(topic_ids), self.layer_emb(layer_ids)], dim=-1)
        h = self.net(x)
        # Clamp logvar: exp(logvar) appears directly in both the reparameterised
        # std and the KL term, so an unconstrained logvar can run away
        # (positive feedback: a slightly-too-large logvar -> slightly-too-large
        # gradient pushing it larger) until exp() overflows to inf, which then
        # poisons every downstream computation into NaN. +-10 keeps exp(logvar)
        # within [~4.5e-5, ~22026] -- large enough to never meaningfully
        # constrain a well-behaved training run, small enough to never overflow.
        logvar = torch.clamp(self.logvar(h), min=-10.0, max=10.0)
        return self.mu(h), logvar


class NLADecoder(nn.Module):
    """latent embedding (+ topic, layer) -> description token logits.

    A small GRU decoder conditioned on the latent as its initial hidden
    state (classic seq2seq/image-captioning style) -- not a transformer,
    deliberately, to keep this fast to train on a few hundred documents.
    """

    def __init__(self, vocab_size: int, latent_dim: int = 256, n_topics: int = 1,
                n_layers: int = 1, emb_dim: int = 128, hidden_dim: int = 256,
                cond_dim: int = 32):
        super().__init__()
        self.topic_emb = nn.Embedding(max(n_topics, 1), cond_dim)
        self.layer_emb = nn.Embedding(max(n_layers, 1), cond_dim)
        self.init_hidden = nn.Linear(latent_dim + 2 * cond_dim, hidden_dim)
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, z, topic_ids, layer_ids, input_ids):
        """Teacher-forced forward: `input_ids` are the target tokens shifted
        right (i.e. BOS-prefixed); returns logits (B, T, vocab_size)."""
        cond = torch.cat([z, self.topic_emb(topic_ids), self.layer_emb(layer_ids)], dim=-1)
        h0 = torch.tanh(self.init_hidden(cond)).unsqueeze(0)          # (1, B, hidden)
        emb = self.token_emb(input_ids)
        out, _ = self.gru(emb, h0)
        return self.out(out)

    @torch.no_grad()
    def generate(self, z, topic_ids, layer_ids, vocab: Vocab, max_len: int = 48):
        cond = torch.cat([z, self.topic_emb(topic_ids), self.layer_emb(layer_ids)], dim=-1)
        h = torch.tanh(self.init_hidden(cond)).unsqueeze(0)
        tok = torch.full((z.shape[0], 1), vocab.stoi[BOS], dtype=torch.long, device=z.device)
        out_ids = []
        for _ in range(max_len):
            emb = self.token_emb(tok)
            o, h = self.gru(emb, h)
            tok = self.out(o).argmax(-1)
            out_ids.append(tok)
        return torch.cat(out_ids, dim=1)


def train_nla(data: dict, vocab: Vocab, latent_dim=256, hidden_dim=512, epochs=40,
             lr=1e-3, batch_size=32, beta_kl=1e-3, device="cpu", random_state=42,
             verbose=True):
    """STAGE 2. Jointly train `NLAEncoder` + `NLADecoder`: CE reconstruction
    loss (predict each document's own tokens from its activation) + KL(q(z|x)
    || N(0, I)) regularisation, weighted by `beta_kl`."""
    torch.manual_seed(random_state)
    d_model = data["activations"].shape[1]
    n_topics = data["n_topics"]
    n_layers = int(data["layer_ids"].max()) + 1

    encoder = NLAEncoder(d_model, n_topics, n_layers, latent_dim, hidden_dim).to(device)
    decoder = NLADecoder(len(vocab), latent_dim, n_topics, n_layers).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)

    acts = torch.tensor(data["activations"], device=device)
    toks = torch.tensor(data["token_ids"], device=device)
    topics = torch.tensor(data["topic_ids"], device=device)
    layers = torch.tensor(data["layer_ids"], device=device)
    n = acts.shape[0]
    rng = np.random.default_rng(random_state)

    history = []
    for epoch in range(epochs):
        perm = rng.permutation(n)
        tot_ce = tot_kl = 0.0
        for start in range(0, n, batch_size):
            idx = torch.tensor(perm[start:start + batch_size], device=device)
            a, t, tp, ly = acts[idx], toks[idx], topics[idx], layers[idx]

            mu, logvar = encoder(a, tp, ly)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)

            logits = decoder(z, tp, ly, t[:, :-1])                    # predict t[:, 1:]
            ce = Fnn.cross_entropy(logits.reshape(-1, logits.shape[-1]), t[:, 1:].reshape(-1),
                                   ignore_index=vocab.pad_id)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = ce + beta_kl * kl

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()), max_norm=5.0)
            opt.step()
            tot_ce += float(ce) * len(idx)
            tot_kl += float(kl) * len(idx)
        history.append({"epoch": epoch, "ce": tot_ce / n, "kl": tot_kl / n})
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  CE={tot_ce / n:.3f}  KL={tot_kl / n:.3f}")

    return {"encoder": encoder, "decoder": decoder, "history": history,
            "n_topics": n_topics, "n_layers": n_layers, "latent_dim": latent_dim}


# ------------------------------------------------------------------ #
#  STAGE 3 - AR: NLA embedding -> reconstructed activation
# ------------------------------------------------------------------ #
class ARHead(nn.Module):
    """embedding -> reconstructed activation. A plain MLP regression head,
    trained SEPARATELY from (on top of, frozen) the NLA encoder -- this is
    the fidelity check: how much of the activation survives being routed
    through the compressed, description-oriented bottleneck?"""

    def __init__(self, latent_dim: int, d_model: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, z):
        return self.net(z)


def _embed(encoder: NLAEncoder, acts, topics, layers, device):
    encoder.eval()
    with torch.no_grad():
        mu, _ = encoder(torch.tensor(acts, device=device), torch.tensor(topics, device=device),
                        torch.tensor(layers, device=device))
    return mu


def train_ar_cv(nla_result: dict, data: dict, n_splits=5, epochs=60, lr=1e-3,
                device=None, random_state=42, verbose=True):
    """STAGE 3. Train `ARHead` (embedding -> activation, MSE) under K-fold
    cross-validation so every document gets an OUT-OF-FOLD reconstruction and
    RMSE -- not a single 80/20 split, because several topic/pole cells here
    have too few documents for a held-out slice to cover them all (e.g. one
    US-speeches topic has only 1 regime-A document). Plain KFold (not
    stratified by topic/pole -- some cells are too small to stratify).

    `device` defaults to wherever `nla_result["encoder"]` already lives (so
    this always matches whatever device `train_nla` was run on) rather than
    defaulting to CPU independently -- passing a mismatched device here is
    exactly what raises "Expected all tensors to be on the same device".

    Returns per-document RMSE (N,), overall RMSE/R^2, and the embeddings used.
    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score

    encoder = nla_result["encoder"]
    if device is None:
        device = next(encoder.parameters()).device
    acts = data["activations"]                 # standardized -- what the encoder/AR train on
    acts_raw = data["activations_raw"]          # original scale -- what RMSE is reported in
    act_mean, act_std = data["act_mean"], data["act_std"]
    z_all = _embed(encoder, acts, data["topic_ids"], data["layer_ids"], device).cpu().numpy()
    n, d_model = acts.shape

    rmse_per_doc = np.full(n, np.nan, dtype=np.float64)
    recon_all_raw = np.zeros_like(acts_raw)
    kf = KFold(n_splits=min(n_splits, n), shuffle=True, random_state=random_state)

    for fold, (tr, te) in enumerate(kf.split(z_all)):
        torch.manual_seed(random_state + fold)
        ar = ARHead(nla_result["latent_dim"], d_model).to(device)
        opt = torch.optim.Adam(ar.parameters(), lr=lr)
        z_tr = torch.tensor(z_all[tr], device=device)
        a_tr = torch.tensor(acts[tr], device=device)          # train against standardized target
        for _ in range(epochs):
            opt.zero_grad()
            loss = Fnn.mse_loss(ar(z_tr), a_tr)
            loss.backward()
            opt.step()
        ar.eval()
        with torch.no_grad():
            pred_te = ar(torch.tensor(z_all[te], device=device)).cpu().numpy()
        pred_te_raw = pred_te * act_std + act_mean              # back to real activation units
        recon_all_raw[te] = pred_te_raw
        rmse_per_doc[te] = np.sqrt(((pred_te_raw - acts_raw[te]) ** 2).mean(axis=1))
        if verbose:
            print(f"  fold {fold + 1}/{kf.n_splits}: held-out RMSE = {rmse_per_doc[te].mean():.4f}")

    overall_rmse = float(np.sqrt(np.mean((recon_all_raw - acts_raw) ** 2)))
    overall_r2 = float(r2_score(acts_raw.ravel(), recon_all_raw.ravel()))
    return {"rmse_per_doc": rmse_per_doc, "reconstructed": recon_all_raw, "embeddings": z_all,
            "overall_rmse": overall_rmse, "overall_r2": overall_r2}


# ------------------------------------------------------------------ #
#  Deciphering the 256-d embeddings: what does a pole, or the A->B
#  direction itself, actually mean?
# ------------------------------------------------------------------ #
def decode_embedding(nla_result, vocab: Vocab, z, topic_id: int, layer_id: int = 0,
                     max_len: int = 24, device=None) -> str:
    """Decode a single 256-d embedding (a pole centroid, or a point along a
    pole-A -> pole-B sweep -- doesn't have to be a real document's own
    embedding) into text via the trained NLA decoder."""
    if device is None:
        device = next(nla_result["decoder"].parameters()).device
    z_t = torch.tensor(np.asarray(z, dtype=np.float32)[None, :], device=device)
    topic_t = torch.tensor([int(topic_id)], device=device)
    layer_t = torch.tensor([int(layer_id)], device=device)
    gen = nla_result["decoder"].generate(z_t, topic_t, layer_t, vocab, max_len=max_len)
    return vocab.decode(gen[0].tolist())


def _cluster_pole_embeddings(z_all, cluster_labels, cb_labels):
    """Per (cluster, pole) mean embedding -- the NLA-space analogue of the
    main pipeline's raw-activation-space regime centroids (`field.py`'s
    muA/muB)."""
    cluster_labels = np.asarray(cluster_labels)
    cb_labels = np.asarray(cb_labels)
    out = {}
    for c in np.unique(cluster_labels):
        for pole, cb in (("A", 0), ("B", 1)):
            mask = (cluster_labels == c) & (cb_labels == cb)
            if mask.sum() == 0:
                continue
            out[(int(c), pole)] = z_all[mask].mean(axis=0)
    return out


def pole_prototypes_table(nla_result, data, vocab, texts, cluster_labels, cb_labels,
                          n_neighbors=3, max_len=24, device=None, layer_id_map=None):
    """Per (cluster, pole): decode the MEAN embedding into a prototype
    description, plus the `n_neighbors` REAL documents nearest that centroid
    in embedding space. The nearest-real-docs column is there deliberately --
    a small, lightly-trained decoder's generated text can be fluent but wrong
    (hallucinated), so it's grounded against real corpus documents the same
    way SAE features are grounded in real corpus tokens elsewhere in this
    pipeline rather than trusted from a generated label alone.

    `layer_id_map` : optional {cluster_id: layer_id} override for the decoder's
    layer conditioning. Defaults to None, which decodes every cluster at
    layer_id=0 -- correct when `data` was built by `prepare_dataset` (one
    dataset, one constant layer, always 0). A dataset built by
    `prepare_axis_dataset` instead uses a DIFFERENT layer_id per cluster
    (=axis id), so pass `dict(zip(axis_index.axis_id, axis_index.layer_id))`
    there -- decoding every axis's centroid at layer_id=0 regardless of which
    layer it actually came from would condition the decoder on the wrong
    layer embedding for every axis except the one that happens to map to 0.
    """
    if device is None:
        device = next(nla_result["encoder"].parameters()).device
    z_all = _embed(nla_result["encoder"], data["activations"], data["topic_ids"],
                   data["layer_ids"], device).cpu().numpy()
    cluster_labels = np.asarray(cluster_labels)
    cb_labels = np.asarray(cb_labels)
    unique_clusters = np.unique(cluster_labels)
    centroids = _cluster_pole_embeddings(z_all, cluster_labels, cb_labels)

    rows = []
    for (c, pole), z_centroid in centroids.items():
        mask = (cluster_labels == c) & (cb_labels == (0 if pole == "A" else 1))
        idx = np.where(mask)[0]
        d = np.linalg.norm(z_all[idx] - z_centroid, axis=1)
        nearest = idx[np.argsort(d)[:n_neighbors]]
        topic_id = int(np.searchsorted(unique_clusters, c))
        layer_id = int(layer_id_map[c]) if layer_id_map is not None else 0
        rows.append({
            "cluster": c, "pole": pole, "n": int(mask.sum()),
            "decoded_prototype": decode_embedding(nla_result, vocab, z_centroid, topic_id,
                                                  layer_id=layer_id, max_len=max_len, device=device),
            "nearest_real_docs": " || ".join(" ".join(texts[j].split()[:15]) for j in nearest),
        })
    return pd.DataFrame(rows).sort_values(["cluster", "pole"]).reset_index(drop=True)


def direction_sweep_table(nla_result, data, vocab, cluster_labels, cb_labels,
                          n_steps=5, extrapolate=0.3, max_len=24, device=None, layer_id_map=None):
    """Per cluster: decode a sweep of points from the pole-A centroid to the
    pole-B centroid (and a bit beyond both ends, via `extrapolate`) in
    embedding space -- shows how the decoded description drifts along the
    pole-A -> pole-B direction itself, the NLA-space analogue of a steering
    sweep (`steering.steering_curve` in the main pipeline). Interpolated
    points aren't any real document's embedding, so treat this as a
    qualitative probe of the direction's meaning, not a citation -- cross-
    check against `pole_prototypes_table`'s grounded nearest-real-docs.

    `layer_id_map` : see `pole_prototypes_table` -- same override, same reason.
    """
    if device is None:
        device = next(nla_result["encoder"].parameters()).device
    z_all = _embed(nla_result["encoder"], data["activations"], data["topic_ids"],
                   data["layer_ids"], device).cpu().numpy()
    cluster_labels = np.asarray(cluster_labels)
    cb_labels = np.asarray(cb_labels)
    unique_clusters = np.unique(cluster_labels)
    centroids = _cluster_pole_embeddings(z_all, cluster_labels, cb_labels)

    rows = []
    alphas = np.linspace(-extrapolate, 1 + extrapolate, n_steps)
    for c in unique_clusters:
        if (int(c), "A") not in centroids or (int(c), "B") not in centroids:
            continue
        zA, zB = centroids[(int(c), "A")], centroids[(int(c), "B")]
        topic_id = int(np.searchsorted(unique_clusters, c))
        layer_id = int(layer_id_map[c]) if layer_id_map is not None else 0
        for alpha in alphas:
            z = zA + alpha * (zB - zA)
            side = "A" if alpha <= 0 else ("B" if alpha >= 1 else "between")
            rows.append({
                "cluster": int(c), "alpha": round(float(alpha), 2), "pole_side": side,
                "decoded": decode_embedding(nla_result, vocab, z, topic_id,
                                            layer_id=layer_id, max_len=max_len, device=device),
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
#  Pole contrast: does AR reconstruct regime A as faithfully as regime B?
# ------------------------------------------------------------------ #
def pole_contrast_table(rmse_per_doc, cluster_labels, cb_labels):
    """Per topic, compare AR reconstruction-fidelity RMSE between regime A's
    documents and regime B's documents. One row per (cluster, pole); a
    `rmse_gap` column (B - A) makes asymmetric fidelity easy to scan."""
    df = pd.DataFrame({
        "cluster": np.asarray(cluster_labels),
        "pole": np.where(np.asarray(cb_labels) == 0, "A", "B"),
        "rmse": np.asarray(rmse_per_doc),
    }).dropna(subset=["rmse"])

    grouped = (df.groupby(["cluster", "pole"])["rmse"]
                 .agg(["mean", "std", "count"]).reset_index())
    wide = grouped.pivot(index="cluster", columns="pole", values="mean")
    out = wide.reset_index().rename(columns={"A": "rmse_A", "B": "rmse_B"})
    if "rmse_A" in out and "rmse_B" in out:
        out["rmse_gap_B_minus_A"] = out["rmse_B"] - out["rmse_A"]
    n_wide = grouped.pivot(index="cluster", columns="pole", values="count").reset_index()
    out = out.merge(n_wide.rename(columns={"A": "nA", "B": "nB"}), on="cluster", how="left")
    return out.sort_values("cluster").reset_index(drop=True)


# ------------------------------------------------------------------ #
#  Summarizing the A->B meaning: representative words + sweep snapshot,
#  one row per cluster, ready to visualize (see viz.plot_pole_direction_summary)
# ------------------------------------------------------------------ #
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for", "with", "by",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those",
    "it", "its", "as", "from", "i", "you", "he", "she", "they", "we", "my", "your", "his",
    "her", "their", "our", "not", "no", "so", "if", "then", "than", "just", "very", "also",
    "have", "has", "had", "do", "does", "did", "will", "would", "can", "could", "should",
    "there", "here", "which", "who", "what", "when", "where", "how", "num",
}


def representative_words(text: str, max_words: int = 8, stopwords=None) -> list[str]:
    """Content words from a decoded description: strip stopwords/specials,
    dedupe preserving order. Deliberately simple (no corpus-frequency
    re-weighting) -- transparent about what it's doing: "the non-generic
    words the decoder produced", nothing cleverer."""
    sw = stopwords if stopwords is not None else _STOPWORDS
    seen, out = set(), []
    for tok in str(text).split():
        t = tok.strip(".,!?;:'\"")
        if not t or t in sw or t == UNK or len(t) < 2:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_words:
            break
    return out


def pole_direction_summary(proto_df, sweep_df, max_words: int = 8) -> pd.DataFrame:
    """One row per cluster, combining `pole_prototypes_table` and
    `direction_sweep_table`: each pole's representative words (from its
    centroid's decoded prototype), plus three sweep snapshots -- just beyond
    the A end, the midpoint of the A->B walk, and just beyond the B end --
    picked by nearest `alpha` so this works regardless of `n_steps`/
    `extrapolate`. The near-A/near-B sweep points are near-duplicates of the
    prototypes by construction (same centroid, same decode call), so this
    intentionally shows the EXTRAPOLATED/midpoint points instead -- the part
    the prototype table alone doesn't cover.
    """
    rows = []
    for c in sorted(proto_df["cluster"].unique()):
        a_row = proto_df[(proto_df["cluster"] == c) & (proto_df["pole"] == "A")]
        b_row = proto_df[(proto_df["cluster"] == c) & (proto_df["pole"] == "B")]
        sweep_c = sweep_df[sweep_df["cluster"] == c].sort_values("alpha")

        a_words = representative_words(a_row["decoded_prototype"].iloc[0], max_words) if len(a_row) else []
        b_words = representative_words(b_row["decoded_prototype"].iloc[0], max_words) if len(b_row) else []

        def _closest(target_alpha):
            if len(sweep_c) == 0:
                return ""
            i = (sweep_c["alpha"] - target_alpha).abs().idxmin()
            return sweep_c.loc[i, "decoded"]

        alpha_lo = float(sweep_c["alpha"].min()) if len(sweep_c) else 0.0
        alpha_hi = float(sweep_c["alpha"].max()) if len(sweep_c) else 1.0
        rows.append({
            "cluster": c,
            "pole_A_words": ", ".join(a_words),
            "beyond_A": _closest(alpha_lo),
            "midpoint": _closest(0.5),
            "beyond_B": _closest(alpha_hi),
            "pole_B_words": ", ".join(b_words),
        })
    return pd.DataFrame(rows)
