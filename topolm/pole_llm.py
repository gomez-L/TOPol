"""Part 2, step 3: an LLM-generated natural-language gloss for each polar
vector, using the SAME `google/gemma-3-4b-pt` checkpoint already loaded as
this pipeline's `backend` (no new model download).

`gemma-3-4b-pt` is the PRETRAINED (base) checkpoint, not an instruction-tuned
one -- it was chosen throughout this pipeline because it's the model whose
internal activations are the object of study, not because it's good at
following instructions. Asking it "explain the difference between these two
groups" the way you would prompt a chat model is unreliable: a base LM's
strong prior is to CONTINUE text, not obey it. So the prompt built here is a
completion, not an instruction: a few real excerpts from pole A, a few from
pole B, then a sentence left hanging half-way through for the model to
finish. This also keeps it grounded in real corpus text (same rationale as
`nla.pole_prototypes_table`'s `nearest_real_docs`), rather than asking the
model to free-associate from nothing.

CROSS-CHECK AGAINST THE INSTRUCTION-TUNED SIBLING (`explain_axes_it` below).
Reading `topol_pv_llm_explanations` (this module's -pt output) against
`topol_pv_tfidf_summary` shows a real limitation: 84.9% of the 543 -pt
glosses reduce to a generic "Group A/B reviews are more positive/negative"
regardless of whether the axis's own TF-IDF vocabulary is about sentiment at
all -- consistent with a completion-only base model defaulting to the most
probable generic review-language continuation rather than actually reporting
the contrast it was shown. `explain_axes_it` runs the SAME grounded-evidence
prompt as an INSTRUCTION against `google/gemma-3-4b-it` (the instruction-
tuned sibling checkpoint, same architecture/training data lineage, so this
is the most controlled comparison available without changing model family)
and asks directly for the contrast, rather than relying on completion. The
two routes' outputs should be read side by side, together with TF-IDF (the
most reliable of the three, since it involves no generation at all) -- not
as "-it is right, -pt is wrong": an instruction-tuned model can also
confabulate a plausible-sounding but wrong contrast, so agreement across
TF-IDF and BOTH LLM routes is the trustworthy signal, not any one route
alone.
"""
from __future__ import annotations

import re
import numpy as np


def sample_pole_snippets(data: dict, axis_id: int, pole: str, n: int = 4,
                         max_words: int = 40, random_state: int = 42) -> list[str]:
    """`n` real documents for one (axis_id, pole), each truncated to
    `max_words` words -- short enough that several fit in one prompt."""
    axis_ids = np.asarray(data["topic_ids"])
    pole_labels = np.asarray(data["pole_labels"])
    want = 0 if pole == "A" else 1
    idx = np.where((axis_ids == axis_id) & (pole_labels == want))[0]
    if len(idx) == 0:
        return []
    rng = np.random.default_rng(random_state + int(axis_id))
    chosen = rng.choice(idx, size=min(n, len(idx)), replace=False)
    texts = [data["texts"][i] for i in chosen]
    return [" ".join(str(t).split()[:max_words]) for t in texts]


def build_pole_prompt(snippets_A: list[str], snippets_B: list[str],
                      terms_A: list[str] | None = None, terms_B: list[str] | None = None) -> str:
    """Completion-style prompt: real excerpts from each pole (+ their top
    distinctive TF-IDF words, if given, as an extra anchor), ending mid-
    sentence for the base model to complete."""
    lines = ["Below are short excerpts from two groups of the same kind of "
             "product review, Group A and Group B.", ""]
    lines.append("Group A:")
    for s in snippets_A:
        lines.append(f'- "{s}"')
    if terms_A:
        lines.append(f"Group A's characteristic words: {', '.join(terms_A[:10])}")
    lines.append("")
    lines.append("Group B:")
    for s in snippets_B:
        lines.append(f'- "{s}"')
    if terms_B:
        lines.append(f"Group B's characteristic words: {', '.join(terms_B[:10])}")
    lines.append("")
    lines.append("In one or two sentences, the main difference between what Group A "
                "reviews say and what Group B reviews say is that")
    return "\n".join(lines)


def _clean_completion(text: str, max_sentences: int = 2) -> str:
    """Trim a generated continuation to its first `max_sentences` sentences
    and drop anything after a blank line (base models often trail off into
    a new, unrelated paragraph)."""
    text = text.split("\n\n")[0].strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(sentences[:max_sentences]).strip()


def _generate_with_backoff(backend, prompts: list[str], max_new_tokens: int, batch_size: int) -> list[str]:
    """`backend.generate` over `prompts` in chunks of `batch_size`, halving the
    chunk on a CUDA OOM and retrying (down to 1) instead of dying -- the 4B
    model's own weights are a fixed ~9 GB, so a batch of a few short prompts
    should comfortably fit on any modern GPU; an OOM here is almost always
    the GPU being SHARED with another process (e.g. another notebook kernel
    that also loaded a model), not this workload genuinely needing more VRAM.
    `torch.cuda.empty_cache()` between chunks releases this process's own
    fragmentation so a smaller retry has the best chance of succeeding."""
    import torch

    out = []
    i, bs = 0, max(1, batch_size)
    while i < len(prompts):
        chunk = prompts[i:i + bs]
        try:
            out.extend(backend.generate(chunk, max_new_tokens=max_new_tokens))
            i += bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise RuntimeError(
                    "CUDA OOM generating a single prompt at a time -- this is not a batch-size "
                    "problem. Check `nvidia-smi` for OTHER processes on this GPU (a common cause: "
                    "another notebook kernel that also loaded a model and never released it) and "
                    "free them, or request a GPU with more free memory."
                ) from None
            bs = max(1, bs // 2)
            print(f"  [pole_llm] CUDA OOM at batch_size={bs * 2} -- retrying this chunk at batch_size={bs}")
    return out


def explain_axes(backend, data: dict, axis_index, tfidf_summary=None,
                 n_snippets: int = 4, max_new_tokens: int = 50, batch_size: int = 4,
                 random_state: int = 42) -> "pd.DataFrame":
    """Build one grounded completion prompt per gate-passing axis and generate
    its gloss via the EXISTING `backend.generate(prompts, ...)` (already
    batches internally, see `backends/gemma.py`). `tfidf_summary`, if given
    (`pole_vocab.build_pv_tfidf_table`'s summary_df), supplies each axis's
    top distinctive words as an extra prompt anchor.

    Generation goes through `_generate_with_backoff`, which halves
    `batch_size` on a CUDA OOM and retries rather than aborting the whole
    run -- useful on a GPU shared with another process (see that function's
    docstring). `batch_size=4` by default: these are short, single-forward-
    pass prompts (a handful of truncated review snippets), so this is about
    being considerate of a shared GPU, not a hard memory ceiling for this
    workload.
    """
    import pandas as pd

    terms_by_axis = {}
    if tfidf_summary is not None:
        for _, row in tfidf_summary.iterrows():
            terms_by_axis[int(row["axis_id"])] = (
                [t.strip() for t in str(row["top_terms_A"]).split(",") if t.strip()],
                [t.strip() for t in str(row["top_terms_B"]).split(",") if t.strip()],
            )

    axis_ids = axis_index["axis_id"].astype(int).tolist()
    prompts, rows = [], []
    for axis_id in axis_ids:
        snippets_A = sample_pole_snippets(data, axis_id, "A", n=n_snippets, random_state=random_state)
        snippets_B = sample_pole_snippets(data, axis_id, "B", n=n_snippets, random_state=random_state)
        if not snippets_A or not snippets_B:
            continue
        terms_A, terms_B = terms_by_axis.get(axis_id, (None, None))
        prompts.append(build_pole_prompt(snippets_A, snippets_B, terms_A, terms_B))
        rows.append(axis_id)

    raw = _generate_with_backoff(backend, prompts, max_new_tokens=max_new_tokens, batch_size=batch_size)
    explanations = [_clean_completion(g) for g in raw]

    df = pd.DataFrame({"axis_id": rows, "explanation": explanations})
    idx_lookup = axis_index.set_index("axis_id")
    df["topic"] = df["axis_id"].map(idx_lookup["topic"])
    df["contrast_index"] = df["axis_id"].map(idx_lookup["contrast_index"])
    return df[["axis_id", "topic", "contrast_index", "explanation"]].sort_values(
        ["topic", "axis_id"]).reset_index(drop=True)


# ------------------------------------------------------------------ #
#  Instruction-tuned comparison route (google/gemma-3-4b-it)
# ------------------------------------------------------------------ #
def build_pole_prompt_it(snippets_A: list[str], snippets_B: list[str],
                         terms_A: list[str] | None = None, terms_B: list[str] | None = None) -> str:
    """Instruction-style prompt for the SAME evidence `build_pole_prompt`
    uses, worded as a direct request rather than a hanging completion --
    appropriate for an instruction-tuned model, which (unlike the base `-pt`
    checkpoint) is trained to follow a directive rather than merely continue
    text. Caller is responsible for wrapping this in the `-it` tokenizer's
    own chat template (`tokenizer.apply_chat_template`) before generation --
    kept a plain string here so it stays backend-agnostic.
    """
    lines = ["Below are short excerpts from two groups of the same kind of "
             "product review, Group A and Group B.", ""]
    lines.append("Group A:")
    for s in snippets_A:
        lines.append(f'- "{s}"')
    if terms_A:
        lines.append(f"Group A's characteristic words: {', '.join(terms_A[:10])}")
    lines.append("")
    lines.append("Group B:")
    for s in snippets_B:
        lines.append(f'- "{s}"')
    if terms_B:
        lines.append(f"Group B's characteristic words: {', '.join(terms_B[:10])}")
    lines.append("")
    lines.append(
        "In one sentence, state the main topical or semantic contrast between what Group A "
        "reviews discuss and what Group B reviews discuss. Do not just say one group is more "
        "positive or negative -- if sentiment is genuinely the only difference, say so "
        "explicitly, but first check whether the two groups are actually about different "
        "subject matter, product features, or concerns.")
    return "\n".join(lines)


def explain_axes_it(backend_it, data: dict, axis_index, tfidf_summary=None,
                    n_snippets: int = 4, max_new_tokens: int = 80, batch_size: int = 4,
                    random_state: int = 42) -> "pd.DataFrame":
    """Same evidence and axis loop as `explain_axes`, but against a SEPARATE
    `google/gemma-3-4b-it` backend, using the instruction-style prompt
    (`build_pole_prompt_it`) wrapped in that model's own chat template.

    `backend_it` must be a `GemmaBackend` (or compatible) instance already
    pointed at `google/gemma-3-4b-it` -- loading it is the caller's
    responsibility (a SECOND ~4B-parameter model resident on the GPU
    alongside the `-pt` field model, so this is meaningfully more VRAM than
    `explain_axes` alone; load it lazily, and consider freeing the `-pt`
    backend first if VRAM is tight).
    """
    import pandas as pd

    terms_by_axis = {}
    if tfidf_summary is not None:
        for _, row in tfidf_summary.iterrows():
            terms_by_axis[int(row["axis_id"])] = (
                [t.strip() for t in str(row["top_terms_A"]).split(",") if t.strip()],
                [t.strip() for t in str(row["top_terms_B"]).split(",") if t.strip()],
            )

    axis_ids = axis_index["axis_id"].astype(int).tolist()
    prompts, rows = [], []
    for axis_id in axis_ids:
        snippets_A = sample_pole_snippets(data, axis_id, "A", n=n_snippets, random_state=random_state)
        snippets_B = sample_pole_snippets(data, axis_id, "B", n=n_snippets, random_state=random_state)
        if not snippets_A or not snippets_B:
            continue
        terms_A, terms_B = terms_by_axis.get(axis_id, (None, None))
        user_msg = build_pole_prompt_it(snippets_A, snippets_B, terms_A, terms_B)
        chat_prompt = backend_it.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}], tokenize=False, add_generation_prompt=True)
        prompts.append(chat_prompt)
        rows.append(axis_id)

    raw = _generate_with_backoff(backend_it, prompts, max_new_tokens=max_new_tokens, batch_size=batch_size)
    # instruction responses are already on-topic (no completion drift to trim), but
    # still cap to a couple of sentences for a comparable table cell to explain_axes'.
    explanations = [_clean_completion(g.strip(), max_sentences=3) for g in raw]

    df = pd.DataFrame({"axis_id": rows, "explanation_it": explanations})
    idx_lookup = axis_index.set_index("axis_id")
    df["topic"] = df["axis_id"].map(idx_lookup["topic"])
    df["contrast_index"] = df["axis_id"].map(idx_lookup["contrast_index"])
    return df[["axis_id", "topic", "contrast_index", "explanation_it"]].sort_values(
        ["topic", "axis_id"]).reset_index(drop=True)
