"""General post-hoc topic-model diagnostics.

Interpretation, labeling, comparison, and visualization helpers that operate on
any fitted model's topic-word (φ) and document-topic (θ) arrays — independent of
how the model was fit (LDA, DMR, CTM, STM, HDP, …). The structural / covariate
pieces (``estimate_effect``, ``posterior_theta_samples``, ``spline``,
``interaction``) live in :mod:`topica.stm`; coherence, diversity,
exclusivity, and the intrusion tests live in :mod:`topica.coherence`.

- :func:`frex` / :func:`label_topics` — prob / FREX / lift / score topic words
  (≈ ``stm::labelTopics``).
- :func:`topic_correlation` — topic-correlation network (≈ ``stm::topicCorr``).
- :func:`find_thoughts` — representative documents per topic (≈ ``stm::findThoughts``).
- :func:`search_k` — fit across topic counts, report quality (≈ ``stm::searchK``).
- :func:`relevance` / :func:`prepare_pyldavis` — LDAvis relevance + export.
- :func:`check_residuals` — Taddy (2012) residual-dispersion test for K.
- :func:`align_topics` / :func:`topic_stability` — match and score topics across fits.
"""

from __future__ import annotations

import html as _html
import inspect
import re
import warnings
from dataclasses import dataclass, field

import numpy as np

from .coherence import (
    _as_topic_word, _as_doc_topic, _vocabulary_of,
    coherence as _coherence, exclusivity as _exclusivity,
)


def _ref_corpus(texts):
    """Normalize a coherence reference to ``list[list[str]]``: a Corpus, raw
    strings (split on whitespace), or token lists all work."""
    if hasattr(texts, "documents"):
        return texts.documents()
    if len(texts) and isinstance(texts[0], str):
        return [t.split() for t in texts]
    return [list(t) for t in texts]


def diagnostics(model, texts=None, *, n=10, coherence_type=None, stability=False,
                n_boot=20, model_factory=None, seed=0):
    """One per-topic diagnostics table for a fitted model.

    Consolidates the quality numbers people otherwise gather one function at a
    time — coherence, exclusivity, FREX words, size, prevalence, top words, and
    (optionally) bootstrap stability — into a single row-per-topic table. It reads
    a model's analysis surface, so it works for every model and you never pass a
    raw matrix where a model is wanted, or vice versa.

    Parameters
    ----------
    model : a fitted topica model.
    texts : the reference corpus for windowed coherence (a ``Corpus``, raw
        strings, or token lists). Without it, coherence falls back to the model's
        own UMass score. Required when ``stability=True``.
    n : top-word count used for coherence, exclusivity, FREX, and the word lists.
    coherence_type : override the coherence metric (``"c_v"`` default when
        ``texts`` is given, ``"u_mass"`` otherwise).
    stability : also report per-topic bootstrap stability (mean top-word Jaccard
        over ``n_boot`` refits, matched back to this model). Off by default since
        it refits the model; needs ``texts`` (the documents) to resample.
    model_factory : ``callable(seed) -> unfitted model`` for the stability refits;
        defaults to rebuilding the model's own type as ``type(model)(num_topics=K,
        seed=seed)``. Pass your own for models whose constructor needs more.

    Returns
    -------
    A pandas ``DataFrame`` indexed by topic (columns: ``label``, ``size``,
    ``prevalence``, ``coherence``, ``exclusivity``, ``stability``, ``top_words``,
    ``frex``), or a list of row dicts when pandas is not installed.
    """
    from .coherence import coherence as _coherence, exclusivity as _exclusivity
    from .analysis import topic_labels as _topic_labels, topic_sizes as _topic_sizes

    phi = _as_topic_word(model)
    k = phi.shape[0]
    if k == 0:
        raise ValueError(
            "the model has no topics (empty topic_word). For BERTopic/Top2Vec this "
            "means clustering found no clusters — lower min_cluster_size or add data."
        )
    theta = _as_doc_topic(model)
    prevalence = theta.mean(axis=0)
    names = _topic_labels(model)
    sizes = _topic_sizes(model)["size"]

    ref = _ref_corpus(texts) if texts is not None else None
    ct = coherence_type or ("c_v" if ref is not None else "u_mass")
    if ref is not None:
        coh = np.asarray(_coherence(model, ref, coherence_type=ct, topn=n), dtype=np.float64)
    elif hasattr(model, "coherence"):
        coh = np.asarray(model.coherence(n), dtype=np.float64)
    else:
        coh = np.full(k, np.nan)

    excl = np.asarray(_exclusivity(model, n=n), dtype=np.float64)
    frex_words = frex(model, n=n)
    vocab = list(model.vocabulary)
    top_method = getattr(model, "top_words", None)

    stab = np.full(k, np.nan)
    if stability:
        if texts is None:
            raise ValueError("stability=True needs texts (the documents) to resample")
        factory = model_factory or (lambda s: type(model)(num_topics=k, seed=s))
        bs = bootstrap_stability(ref, reference=model, n_boot=n_boot, topn=n,
                                 seed=seed, model_factory=factory)
        stab = np.asarray(bs["stability"], dtype=np.float64)

    def words_for(t):
        if callable(top_method):
            try:
                return [w for w, _ in top_method(n, topic=t)]
            except Exception as exc:
                warnings.warn(
                    f"{type(model).__name__}.top_words failed ({type(exc).__name__}: "
                    f"{exc}); falling back to raw topic-word rows, which drops any "
                    "custom weighting (e.g. FREX) that top_words applies.",
                    stacklevel=2,
                )
        return [vocab[i] for i in np.argsort(phi[t])[::-1][:n]]

    rows = []
    for t in range(k):
        rows.append({
            "topic": t,
            "label": names[t] if t < len(names) else f"topic_{t}",
            "size": int(sizes[t]) if t < len(sizes) else 0,
            "prevalence": float(prevalence[t]),
            "coherence": float(coh[t]) if t < len(coh) else float("nan"),
            "exclusivity": float(excl[t]) if t < len(excl) else float("nan"),
            "stability": float(stab[t]),
            "top_words": " ".join(words_for(t)),
            "frex": " ".join(w for w, _ in frex_words[t]),
        })
    try:
        import pandas as pd

        return pd.DataFrame(rows).set_index("topic")
    except ImportError:
        return rows


# ---------------------------------------------------------------------------
# make_heldout / eval_heldout: R stm-style within-corpus word-heldout
# ---------------------------------------------------------------------------

@dataclass
class Heldout:
    """Result of :func:`make_heldout`: the training corpus and the withheld words.

    We sample a fraction of documents and remove a fraction of their tokens to
    create a within-corpus heldout set. The caller fits a model on
    ``heldout.documents`` (the reduced corpus), then scores the withheld words
    with :func:`eval_heldout`.

    Workflow::

        h = make_heldout(corpus)
        model.fit(h.documents)
        result = eval_heldout(model, h)

    Attributes
    ----------
    documents : the full corpus as token lists (length D, same order as the
        input), with held-out tokens removed from the sampled documents.
        Unsampled documents are unchanged. Fit your model on this.
    missing : list of ``(doc_index, held_out_tokens)`` for each sampled
        document. ``doc_index`` is the original position; ``held_out_tokens``
        is the list of token strings that were removed.
    doc_indices : the sorted array of document indices that were sampled.
    """

    documents: list
    missing: list
    doc_indices: np.ndarray


@dataclass
class HeldoutResult:
    """Result of :func:`eval_heldout`: per-document and aggregate held-out log-likelihoods.

    Higher (less negative) values indicate better model fit on the withheld
    words. The headline is ``mean_per_doc_loglik``.

    Attributes
    ----------
    mean_per_doc_loglik : mean over scored documents of the per-document
        held-out log-likelihood. Higher is better.
    total_loglik : sum of per-document log-likelihoods over all scored docs.
    n_docs : number of documents that had at least one held-out token in the
        model vocabulary (documents with no in-vocab held-out tokens are
        skipped).
    n_tokens : total number of scored tokens.
    per_doc_loglik : array of per-document log-likelihoods (length n_docs).
    """

    mean_per_doc_loglik: float
    total_loglik: float
    n_docs: int
    n_tokens: int
    per_doc_loglik: np.ndarray


def make_heldout(corpus, *, prop_docs=0.5, prop_words=0.5, seed=0):
    """Build a within-corpus word-heldout set (R stm's ``make.heldout``).

    We sample ``floor(prop_docs * D)`` documents and remove
    ``floor(prop_words * len(doc))`` randomly chosen token positions from each.
    The remaining tokens stay in the corpus; the removed tokens form the heldout
    set. Fit a model on ``.documents`` and score it with :func:`eval_heldout`.

    Documents too short to split (fewer than 2 tokens, or those for which the
    split would leave 0 retained or 0 held-out tokens) are silently skipped
    rather than raising an error; the sampled set may therefore be slightly
    smaller than ``floor(prop_docs * D)``.

    Parameters
    ----------
    corpus : a ``Corpus`` (its ``.documents()`` method is called), a list of
        raw strings (split on whitespace), or a list of token lists.
    prop_docs : fraction of documents to sample; default 0.5.
    prop_words : fraction of tokens to hold out per sampled document; default 0.5.
    seed : numpy Generator seed for reproducibility.

    Returns
    -------
    A :class:`Heldout` dataclass. Pass ``.documents`` to ``model.fit`` and
    the whole object to :func:`eval_heldout`.
    """
    if not 0.0 < prop_docs < 1.0:
        raise ValueError(f"prop_docs must be in (0, 1), got {prop_docs!r}")
    if not 0.0 < prop_words < 1.0:
        raise ValueError(f"prop_words must be in (0, 1), got {prop_words!r}")

    # Normalize input
    if hasattr(corpus, "documents"):
        raw = corpus.documents()
    elif len(corpus) and isinstance(corpus[0], str):
        raw = [t.split() for t in corpus]
    else:
        raw = [list(d) for d in corpus]

    D = len(raw)
    rng = np.random.default_rng(seed)

    n_sample = int(np.floor(prop_docs * D))
    candidate_idx = rng.choice(D, size=n_sample, replace=False)
    candidate_idx.sort()

    # Build the training corpus (copy of raw, some docs shortened)
    documents = [list(d) for d in raw]
    missing = []
    sampled_indices = []

    for doc_idx in candidate_idx:
        doc = raw[doc_idx]
        n_tokens = len(doc)
        n_hold = int(np.floor(prop_words * n_tokens))
        n_keep = n_tokens - n_hold
        # Must retain at least 1 and hold out at least 1
        if n_keep < 1 or n_hold < 1:
            continue
        hold_positions = rng.choice(n_tokens, size=n_hold, replace=False)
        hold_set = set(hold_positions.tolist())
        retained = [tok for pos, tok in enumerate(doc) if pos not in hold_set]
        held_out_tokens = [doc[pos] for pos in sorted(hold_positions)]
        documents[doc_idx] = retained
        missing.append((int(doc_idx), held_out_tokens))
        sampled_indices.append(int(doc_idx))

    return Heldout(
        documents=documents,
        missing=missing,
        doc_indices=np.array(sampled_indices, dtype=np.intp),
    )


def eval_heldout(model, heldout, *, seed=0):
    """Score held-out words from :func:`make_heldout` under a fitted model (R stm's ``eval.heldout``).

    We infer each sampled document's topic mixture from its retained tokens
    (``heldout.documents[doc_index]``) via the model's ``transform``, then score
    the withheld tokens under ``p(w) = sum_k theta_k * phi[k, w]``.

    Requires that ``model`` was fit on ``heldout.documents`` (the training corpus
    returned by :func:`make_heldout`). Works for any generative model that
    exposes ``transform`` and ``topic_word``: LDA, DMR, CTM, STM, HDP,
    LabeledLDA, and SupervisedLDA. The keyword/anchored Gibbs models (keyATM,
    SeededLDA, SAGE, PA, PT) do not expose ``transform`` and so fall outside this
    diagnostic, and the embedding-cluster models (BERTopic, Top2Vec) define no
    document likelihood; both raise a clear error.

    Parameters
    ----------
    model : a fitted generative model (must have been fit on ``heldout.documents``).
    heldout : a :class:`Heldout` returned by :func:`make_heldout`.
    seed : RNG seed for the Gibbs ``transform`` (variational models ignore it).

    Returns
    -------
    A :class:`HeldoutResult` dataclass. The headline metric is
    ``.mean_per_doc_loglik``; higher (less negative) is better.
    """
    if type(model).__name__ in ("BERTopic", "Top2Vec"):
        raise ValueError(
            f"{type(model).__name__} defines topics by class-based TF-IDF over "
            "document clusters, not a generative word distribution, so it has no "
            "held-out log-likelihood. Compare clustering models with coherence or "
            "topic_diversity instead."
        )
    phi = _as_topic_word(model)
    if phi.shape[0] == 0:
        raise ValueError("the model has no topics (empty topic_word)")
    vocab = {w: i for i, w in enumerate(model.vocabulary)}

    # Batch all retained docs through transform in one call
    retained_docs = [list(heldout.documents[doc_idx]) for doc_idx, _ in heldout.missing]
    if not retained_docs:
        raise ValueError("heldout.missing is empty; nothing to score")

    theta = _transform_theta(model, retained_docs, seed)  # (n_sampled, K)

    per_doc_ll = []
    total_loglik = 0.0
    total_tokens = 0

    for i, (doc_idx, held_tokens) in enumerate(heldout.missing):
        ids = [vocab[w] for w in held_tokens if w in vocab]
        if not ids:
            continue
        pw = np.clip(theta[i] @ phi[:, ids], 1e-12, None)
        doc_ll = float(np.log(pw).sum())
        per_doc_ll.append(doc_ll)
        total_loglik += doc_ll
        total_tokens += len(ids)

    if not per_doc_ll:
        raise ValueError(
            "none of the held-out tokens appeared in the model vocabulary; "
            "check that the model was fit on heldout.documents"
        )

    n_docs = len(per_doc_ll)
    return HeldoutResult(
        mean_per_doc_loglik=total_loglik / n_docs,
        total_loglik=total_loglik,
        n_docs=n_docs,
        n_tokens=total_tokens,
        per_doc_loglik=np.array(per_doc_ll, dtype=np.float64),
    )


def _accepts_kwarg(fn, name):
    """Whether ``fn`` accepts the keyword argument ``name``. PyO3 methods expose a
    text signature, so this works on the Rust models; if a callable has no
    introspectable signature we assume it does not take the kwarg (the caller then
    uses the plain form), which is safe — a wrong guess drops an optional arg, it
    does not crash."""
    try:
        params = inspect.signature(fn).parameters
    except (ValueError, TypeError):
        return False
    if name in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def _transform_theta(model, docs, seed):
    fn = getattr(model, "transform", None)
    if not callable(fn):
        raise ValueError(
            f"{type(model).__name__} has no transform(); perplexity needs a generative "
            "model that can infer topics for held-out documents"
        )
    # Pass seed= only if transform actually accepts it, rather than calling with
    # seed= and treating any TypeError as "no seed param" — a TypeError raised
    # *inside* transform (a real bug) would otherwise be silently swallowed and
    # retried without the seed.
    accepts_seed = _accepts_kwarg(fn, "seed")
    try:
        return np.asarray(fn(docs, seed=seed) if accepts_seed else fn(docs), dtype=np.float64)
    except TypeError as exc:
        raise ValueError(
            f"{type(model).__name__}.transform needs more than documents (e.g. "
            "embeddings), so it has no document likelihood. Held-out perplexity is for "
            "the generative models (LDA, DMR, CTM, STM, HDP, ...); use coherence or "
            "topic_diversity to compare clustering / embedding models."
        ) from exc


def perplexity(model, held_out, *, seed=0):
    """Document-completion held-out perplexity for a generative model.

    For each held-out document, half its tokens (even positions) estimate the
    document's topic mixture through the model's ``transform``, and the other half
    (odd positions) are scored under that mixture, ``p(w) = sum_k theta_k *
    topic_word[k, w]``. Returns ``exp(-sum log p / N_eval)``; lower is better.

    Because the scored tokens are held out from the mixture estimate, this does not
    trivially fall as ``K`` grows the way in-sample likelihood does, so it is a fair
    quantity to compare across ``K`` when justifying a topic count. It works for any
    model with a generative ``transform(documents)`` and a ``topic_word``
    distribution (LDA, DMR, CTM, STM, HDP, keyATM, ...). The embedding-cluster
    models have no document likelihood; compare those with coherence or diversity.

    (``LDA`` additionally offers the more rigorous Wallach et al. left-to-right
    estimator as ``LDA.perplexity`` / ``LDA.evaluate``.)

    Parameters
    ----------
    model : a fitted generative model.
    held_out : documents the model was not trained on (token lists or a ``Corpus``).
    seed : RNG seed for the Gibbs ``transform`` (ignored by the variational models).
    """
    if type(model).__name__ in ("BERTopic", "Top2Vec"):
        raise ValueError(
            f"{type(model).__name__} defines topics by class-based TF-IDF over "
            "document clusters, not a generative word distribution, so it has no "
            "held-out perplexity. Compare clustering models with coherence or "
            "topic_diversity instead."
        )
    if hasattr(held_out, "documents"):
        held_out = held_out.documents()
    phi = _as_topic_word(model)
    if phi.shape[0] == 0:
        raise ValueError("the model has no topics (empty topic_word)")
    vocab = {w: i for i, w in enumerate(model.vocabulary)}

    est, ev = [], []
    for d in held_out:
        d = list(d)
        if len(d) < 2:
            continue
        est.append(d[0::2])
        ev.append(d[1::2])
    if not est:
        raise ValueError("need held-out documents with at least 2 tokens each")

    theta = _transform_theta(model, est, seed)
    logp, n = 0.0, 0
    for i, evdoc in enumerate(ev):
        ids = [vocab[w] for w in evdoc if w in vocab]
        if not ids:
            continue
        pw = np.clip(theta[i] @ phi[:, ids], 1e-12, None)
        logp += float(np.log(pw).sum())
        n += len(ids)
    if n == 0:
        raise ValueError("none of the held-out tokens were in the model vocabulary")
    return float(np.exp(-logp / n))


# ---------------------------------------------------------------------------
# labelTopics: prob / FREX / lift / score
# ---------------------------------------------------------------------------

def _counts_from(word_counts, corpus):
    """Resolve ``word_counts`` / ``corpus`` to a counts array, or ``None``.

    Pass at most one. A ``corpus`` contributes its ``word_counts`` (empirical
    corpus frequencies, aligned to the vocabulary the model was fit on).
    """
    if word_counts is not None and corpus is not None:
        raise ValueError("pass either word_counts= or corpus=, not both")
    if corpus is not None:
        wc = getattr(corpus, "word_counts", None)
        if wc is None:
            raise ValueError(
                "corpus= must be a topica.Corpus (it exposes word_counts)"
            )
        return np.asarray(wc, dtype=np.float64)
    return None if word_counts is None else np.asarray(word_counts, dtype=np.float64)


def _resolve_word_counts(word_counts, V):
    """Validate ``word_counts`` to a length-``V`` list of ints, or ``[]`` if None."""
    if word_counts is None:
        return []
    wc = np.asarray(word_counts, dtype=np.float64)
    if wc.shape != (V,):
        raise ValueError(f"word_counts must have length {V} (the vocabulary size)")
    if np.any(wc < 0) or not np.all(np.isfinite(wc)):
        raise ValueError("word_counts must be finite and non-negative")
    return [int(round(c)) for c in wc]


def frex(topic_word, vocabulary=None, *, w=0.5, n=10, word_counts=None, corpus=None):
    """FREX (FRequency–EXclusivity) top words per topic.

    For each topic, words are scored by the weighted harmonic mean of the rank of
    their probability (frequency) and the rank of their exclusivity
    ``φ_{t,v} / Σ_k φ_{k,v}`` — stm's ``calcfrex``. ``w`` weights frequency vs
    exclusivity. Returns a list (per topic) of ``(word, frex)``.

    The scores come from the single, stm-faithful implementation in topica's Rust
    core (``topica-core``'s ``inspect`` module — the same one faSTM and the Stata
    plugin use), so the FREX definition can never drift between languages.

    Pass ``word_counts`` (a length-``V`` array of corpus word frequencies) or
    ``corpus`` (a :class:`topica.Corpus`, whose word counts are read for you) to
    apply stm's James-Stein exclusivity shrinkage, which is stm's default; it damps
    the exclusivity of rare words that appear in only one topic by chance. Without
    either (the default here) no shrinkage is applied.

    `topic_word` is a fitted model (uses its ``topic_word`` and ``vocabulary``)
    or a ``(K, V)`` array, in which case pass ``vocabulary``.
    """
    if not (0.0 <= w <= 1.0):
        raise ValueError(f"w (frequency weight) must be in [0, 1], got {w!r}")
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    from ._topica import inspect_frex_scores

    vocabulary = _vocabulary_of(topic_word, vocabulary)
    phi = _as_topic_word(topic_word)
    K, V = phi.shape
    wc = _resolve_word_counts(_counts_from(word_counts, corpus), V)
    scores = np.asarray(inspect_frex_scores(phi.tolist(), wc, float(w)))

    results = []
    for t in range(K):
        idx = np.argsort(scores[t])[::-1][:n]
        results.append([(vocabulary[i], float(scores[t][i])) for i in idx])
    return results


def mmr(topic_word, word_embeddings, vocabulary=None, *, n=10, diversity=0.3, n_candidates=None):
    """Maximal-marginal-relevance top words, to cut redundant near-synonyms.

    For each topic, take the top ``n_candidates`` words by ``topic_word`` weight
    and greedily reselect ``n`` of them, each pick maximizing

        ``(1 - diversity) * relevance(word) - diversity * max_cos(word, picked)``

    where relevance is the (per-topic, max-normalized) ``topic_word`` weight and the
    redundancy term is the cosine between word embeddings. ``diversity=0`` returns
    the plain top words; higher trades relevance for variety, like BERTopic's
    ``MaximalMarginalRelevance(diversity=...)``.

    Parameters
    ----------
    topic_word : a fitted model (uses its ``topic_word`` and ``vocabulary``) or a
        ``(K, V)`` array, in which case pass ``vocabulary``.
    word_embeddings : a ``(V, E)`` matrix aligned to the vocabulary — the word
        vectors (for Top2Vec, the ones you fit with; otherwise embed the vocabulary
        with your embedding model, as BERTopic's MMR does internally).
    n : words returned per topic.
    diversity : in ``[0, 1]``; 0 is the plain top words, higher is more diverse.
    n_candidates : how many top words to rerank (default ``max(5 * n, n)``).

    Returns
    -------
    A list per topic of ``(word, topic_word_weight)`` pairs, like ``top_words``.
    """
    if not 0.0 <= diversity <= 1.0:
        raise ValueError("diversity must be in [0, 1]")
    vocabulary = _vocabulary_of(topic_word, vocabulary)
    phi = _as_topic_word(topic_word)
    k, v = phi.shape
    emb = np.asarray(word_embeddings, dtype=np.float64)
    if emb.shape[0] != v:
        raise ValueError(
            f"word_embeddings has {emb.shape[0]} rows but the vocabulary has {v}"
        )
    embn = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    n_cand = n_candidates or max(5 * n, n)

    out = []
    for t in range(k):
        cand = np.argsort(phi[t])[::-1][: min(n_cand, v)]
        rel = phi[t, cand].astype(np.float64)
        rel = rel / (rel.max() if rel.max() > 0 else 1.0)
        sims = embn[cand] @ embn[cand].T  # candidate-candidate cosine
        picked = [0]                       # seed with the most relevant word
        rest = list(range(1, len(cand)))
        while rest and len(picked) < n:
            scores = [(1.0 - diversity) * rel[r] - diversity * sims[r, picked].max()
                      for r in rest]
            best = rest[int(np.argmax(scores))]
            picked.append(best)
            rest.remove(best)
        out.append([(vocabulary[cand[i]], float(phi[t, cand[i]])) for i in picked])
    return out


def label_topics(topic_word, vocabulary=None, *, n=10, word_counts=None, corpus=None):
    """stm-style topic labels: prob, FREX, lift, and score word lists per topic.

    Returns a list (per topic) of dicts with keys ``prob``, ``frex``, ``lift``,
    ``score``, each a list of ``(word, value)`` pairs. FREX, lift, and score all
    come from the single stm-faithful implementation in topica's Rust core
    (``topica-core``'s ``inspect``), so they cannot drift from faSTM / the Stata
    plugin.

    ``lift`` is stm's lift, ``log P(w|topic) − log P(w)``, where ``P(w)`` is the
    empirical word frequency. Pass ``word_counts`` (a length-``V`` array) or
    ``corpus`` (a :class:`topica.Corpus`, whose word counts are read for you) for
    the exact value; without either, ``P(w)`` is estimated from the topic-word
    matrix's column marginal (lift depends only on relative word frequency, so the
    ranking matches). ``word_counts`` / ``corpus`` also enable stm's James-Stein
    FREX shrinkage (see :func:`frex`).

    `topic_word` is a fitted model (uses its ``topic_word`` and ``vocabulary``)
    or a ``(K, V)`` array, in which case pass ``vocabulary``.
    """
    from ._topica import inspect_lift_scores, inspect_score_scores

    vocabulary = _vocabulary_of(topic_word, vocabulary)
    phi = _as_topic_word(topic_word)
    if phi.ndim != 2 or phi.shape[0] == 0:
        raise ValueError(
            "the model has no topics (empty topic_word). For BERTopic/Top2Vec this "
            "means clustering found no clusters — lower min_cluster_size, add data, "
            "or check the scale of your embeddings."
        )
    K, V = phi.shape
    word_counts = _counts_from(word_counts, corpus)

    if word_counts is None:
        # Estimate P(w) from the column marginal. Lift depends only on count
        # ratios, so any common scale works; this yields log(beta) - log(marginal).
        marginal = phi.mean(axis=0)
        lift_counts = [max(int(round(m * 1e6)), 1) for m in marginal]
    else:
        lift_counts = _resolve_word_counts(word_counts, V)
    lift_mat = np.asarray(inspect_lift_scores(phi.tolist(), lift_counts))
    score_mat = np.asarray(inspect_score_scores(phi.tolist()))

    frex_words = frex(topic_word, vocabulary, n=n, word_counts=word_counts)
    out = []
    for t in range(K):
        prob_idx = np.argsort(phi[t])[::-1][:n]
        lift_idx = np.argsort(lift_mat[t])[::-1][:n]
        score_idx = np.argsort(score_mat[t])[::-1][:n]
        out.append({
            "prob": [(vocabulary[i], float(phi[t, i])) for i in prob_idx],
            "frex": frex_words[t],
            "lift": [(vocabulary[i], float(lift_mat[t, i])) for i in lift_idx],
            "score": [(vocabulary[i], float(score_mat[t, i])) for i in score_idx],
        })
    return out


def topic_table(model, *, n=7):
    """A publication-ready topic table: one row per topic with its prevalence and
    its top probability and FREX words.

    Returns a list of dicts with ``topic``, ``prevalence`` (mean θ), ``prob`` (the
    top-`n` highest-probability words), and ``frex`` (the top-`n` FREX words —
    usually the better label). Hand it to ``pandas.DataFrame`` for the table that
    goes in a results section.

    `model` is any fitted model exposing ``topic_word``, ``doc_topic``, and
    ``vocabulary``.
    """
    phi = _as_topic_word(model)
    prevalence = _as_doc_topic(model).mean(axis=0)
    vocab = list(model.vocabulary)
    labels = label_topics(phi, vocab, n=n)
    return [
        {
            "topic": t,
            "prevalence": float(prevalence[t]),
            "prob": [w for w, _ in labels[t]["prob"]],
            "frex": [w for w, _ in labels[t]["frex"]],
        }
        for t in range(len(labels))
    ]


def topics_for_term(topic_word, terms, vocabulary=None, *, top_n=5, per_term=False,
                    normalize=False):
    """The inverse of "top words for a topic": the top topics for a term.

    Given the topic-word matrix φ, rank topics by the weight they place on a
    queried term (or terms) — "which topics is ``'immigration'`` important in, and
    how strongly?". Where :func:`label_topics` / :func:`frex` go *topic → words*,
    this goes *word → topics*, which is handy for corpus exploration and for
    checking where a seed/anchor word actually landed after fitting.

    Parameters
    ----------
    topic_word : a fitted model (uses its ``topic_word`` and ``vocabulary``) or a
        ``(K, V)`` array, in which case pass ``vocabulary``.
    terms : str or sequence of str
        A single term, or several. Terms are matched against ``vocabulary``
        exactly — a topic model's vocabulary is its *processed* tokens (usually
        lowercased, often stemmed), so query the stored form (``"immigr"``), not
        the surface word (``"Immigration"``). A term absent from the vocabulary is
        dropped with a warning; if *every* term is absent a ``ValueError`` is
        raised. Duplicate terms are collapsed to their first occurrence.
    vocabulary : sequence of str, optional
        Required only when ``topic_word`` is a bare array with no vocabulary.
    top_n : int or None, default 5
        How many topics to return, highest-weighted first. ``None`` returns all
        topics ranked. Ties are broken by ascending topic id.
    per_term : bool, default False
        Ignored for a single term. For several terms, ``False`` (the default)
        pools the terms — topics are ranked by the **summed** weight across the
        queried terms — and returns one ranked list. ``True`` instead returns a
        ``dict`` mapping each (found) term to its own ranked list, in the order
        the terms were given.
    normalize : bool, default False
        How each term's per-topic weight is defined. ``False`` uses the raw φ
        entry, ``P(word | topic)`` — directly "what share of this topic's mass is
        this word". Because φ columns are not comparable across words (a frequent
        word carries more mass everywhere), raw weights and the raw pooled sum are
        dominated by the most frequent term. ``True`` instead column-normalizes
        each term to ``P(topic | word) = φ[:, w] / Σ_t φ[t, w]`` — "what share of
        *this word* lives in each topic" — which is comparable across terms and
        frequency-robust, so pooling weights terms equally. Assumes nonnegative φ
        (leave it off for signed-axis models like S³).

    Returns
    -------
    For a single term, or several terms with ``per_term=False``: a list of
    ``(topic_id, weight)`` pairs sorted by descending weight. With several terms
    and ``per_term=True``: a ``dict`` ``{term: [(topic_id, weight), ...]}``.

    Examples
    --------
    >>> topics_for_term(model, "immigr", top_n=5)
    [(12, 0.031), (4, 0.018), ...]
    >>> topics_for_term(model, ["immigr", "border"], per_term=True)
    {"immigr": [...], "border": [...]}
    """
    single = isinstance(terms, str)
    term_list = [terms] if single else list(terms)
    if not term_list:
        raise ValueError("terms is empty; pass a term or a non-empty list of terms")
    if not all(isinstance(t, str) for t in term_list):
        raise ValueError("terms must be a string or a sequence of strings")
    # Collapse to plain str, first occurrence wins. This keeps the pooled sum from
    # double-counting a repeated term and the per_term dict from silently dropping
    # one (dict keys are unique), so both modes see the same term set.
    term_list = list(dict.fromkeys(str(t) for t in term_list))
    # bool is an int subclass; reject it so top_n=True isn't silently read as 1.
    if top_n is not None and (
        isinstance(top_n, bool) or not isinstance(top_n, (int, np.integer)) or top_n < 1
    ):
        raise ValueError(f"top_n must be a positive integer or None, got {top_n!r}")

    vocabulary = _vocabulary_of(topic_word, vocabulary)
    phi = _as_topic_word(topic_word)
    if phi.ndim != 2 or phi.shape[0] == 0:
        raise ValueError(
            "the model has no topics (empty topic_word). For BERTopic/Top2Vec this "
            "means clustering found no clusters — lower min_cluster_size, add data, "
            "or check the scale of your embeddings."
        )
    K, V = phi.shape
    if len(vocabulary) != V:
        raise ValueError(
            f"vocabulary length ({len(vocabulary)}) does not match the number of "
            f"topic_word columns ({V}); pass the vocabulary aligned to this φ."
        )
    index = {w: i for i, w in enumerate(vocabulary)}

    found = [(t, index[t]) for t in term_list if t in index]
    missing = [t for t in term_list if t not in index]
    if missing:
        if not found:
            raise ValueError(
                f"none of the requested terms are in the vocabulary: {missing!r}. "
                "The vocabulary holds the model's processed tokens (usually "
                "lowercased, often stemmed) — query that form, not the surface word."
            )
        warnings.warn(
            f"terms not in the vocabulary, dropped: {missing!r}",
            stacklevel=2,
        )

    limit = K if top_n is None else min(int(top_n), K)

    def _col(j):
        col = phi[:, j]
        if normalize:
            total = col.sum()
            return col / total if total != 0 else np.zeros_like(col)
        return col

    def _rank(weights):
        # Descending by weight; np.argsort is stable, so equal weights keep
        # ascending topic-id order — a deterministic tie-break.
        order = np.argsort(-weights, kind="stable")[:limit]
        return [(int(t), float(weights[t])) for t in order]

    if single:
        return _rank(_col(found[0][1]))
    if per_term:
        return {t: _rank(_col(j)) for t, j in found}
    # Pool the terms: rank topics by their summed (per-term normalized, if asked)
    # weight over the queried terms.
    pooled = np.sum([_col(j) for _, j in found], axis=0)
    return _rank(pooled)


# ---------------------------------------------------------------------------
# topicCorr: topic-correlation network
# ---------------------------------------------------------------------------

@dataclass
class TopicCorrelation:
    cor: np.ndarray
    adjacency: np.ndarray
    edges: list[tuple[int, int, float]] = field(default_factory=list)


def topic_correlation(doc_topic, *, threshold=0.05):
    """Topic-correlation network (≈ stm's ``topicCorr`` "simple" method).

    Correlates topic proportions across documents; topic pairs whose correlation
    exceeds ``threshold`` become network edges. Returns a
    :class:`TopicCorrelation` with the correlation matrix, a 0/1 adjacency
    matrix (zero diagonal), and the edge list.

    This is the raw across-document theta correlation, matching ``stm``'s
    ``topicCorr`` default ("simple") method. Raw theta correlation is
    compositionally biased (the simplex constraint induces spurious negative
    correlation); for the closure-corrected alternatives use
    ``viz.topic_correlation(model, method="clr")`` (the viz layer's default) or
    ``method="partial"``/``"eta"``.

    `doc_topic` is a fitted model (uses its ``doc_topic``) or a ``(D, K)`` array.
    """
    theta = _as_doc_topic(doc_topic)
    cor = np.corrcoef(theta.T)
    cor = np.nan_to_num(cor)
    K = cor.shape[0]
    adj = (cor > threshold).astype(int)
    np.fill_diagonal(adj, 0)
    edges = [
        (i, j, float(cor[i, j]))
        for i in range(K)
        for j in range(i + 1, K)
        if cor[i, j] > threshold
    ]
    return TopicCorrelation(cor=cor, adjacency=adj, edges=edges)


# ---------------------------------------------------------------------------
# findThoughts: representative documents per topic
# ---------------------------------------------------------------------------

def find_thoughts(doc_topic, texts=None, *, topic, n=3):
    """The `n` documents most associated with `topic` (≈ stm's ``findThoughts``).

    Returns a list of ``(doc_index, proportion, text)`` sorted by descending
    topic proportion; ``text`` is ``None`` when ``texts`` is not supplied.

    `doc_topic` is a fitted model (uses its ``doc_topic``) or a ``(D, K)`` array.
    """
    theta = _as_doc_topic(doc_topic)
    if topic < 0 or topic >= theta.shape[1]:
        raise ValueError(f"topic {topic} out of range (num_topics={theta.shape[1]})")
    if texts is not None and len(texts) != theta.shape[0]:
        raise ValueError(
            f"texts has {len(texts)} entries but doc_topic has {theta.shape[0]} "
            "rows; pass texts aligned to the kept documents (corpus.kept_indices), "
            "not the original documents — pruning may have dropped some."
        )
    col = theta[:, topic]
    # argpartition for the top-n (O(D)) then sort just those n, rather than a full
    # O(D log D) argsort of every document.
    n_eff = min(n, col.shape[0])
    part = np.argpartition(col, -n_eff)[-n_eff:]
    idx = part[np.argsort(col[part])[::-1]]
    out = []
    for i in idx:
        text = texts[i] if texts is not None else None
        out.append((int(i), float(theta[i, topic]), text))
    return out


# ---------------------------------------------------------------------------
# searchK: fit across topic counts, report quality
# ---------------------------------------------------------------------------

# Whether a higher or lower value of each metric is better. Coherence here is
# mean UMass (negative; less-negative is better), so "maximize".
# Default number of seeds per K in search_k. Flip this one value to make
# confidence intervals (num_seeds>1) the default; every downstream path already
# handles both cases.
_SEARCH_K_DEFAULT_SEEDS = 1

SEARCH_K_DIRECTIONS = {
    "coherence": "maximize",
    "exclusivity": "maximize",
    "heldout_loglik": "maximize",
    "perplexity": "minimize",
    "polarization": "maximize",
    # Opt-in ldatuning-style criteria (search_k(criteria=...)).
    "deveaud": "maximize",   # mean pairwise JS divergence between topics
    "cao_juan": "minimize",  # mean pairwise cosine similarity between topics
}

# Extra K-selection criteria computable from the topic-word matrix, requested via
# search_k(criteria=...). Off by default and out of the frontier.
_SEARCH_K_CRITERIA = ("deveaud", "cao_juan")


def _argbest_k(rows, score):
    """The ``k`` at the maximum score, breaking ties toward the *smallest* ``k``.

    Parsimony tie-break: with two K values equally good, prefer the simpler model.
    This also makes the pick independent of the order ``ks`` was passed in (a plain
    ``argmax`` returns the first index, so it would depend on grid ordering)."""
    score = np.asarray(score, dtype=np.float64)
    best = np.nanmax(score)
    tied = [i for i, s in enumerate(score) if s == best]
    return int(min(rows[i]["k"] for i in tied))


def _resolve_workers(n_jobs, n_tasks):
    """Thread-worker count for a grid of ``n_tasks`` fits. ``n_jobs=1`` stays
    serial; ``n_jobs<=0`` or ``None`` uses all cores; otherwise ``min(n_jobs,
    n_tasks)``. A single-task grid never parallelizes."""
    if n_tasks <= 1:
        return 1
    # None / non-positive / non-finite (inf, nan) -> all cores.
    if n_jobs is None or not np.isfinite(n_jobs) or n_jobs <= 0:
        import os
        n_jobs = os.cpu_count() or 1
    return max(1, min(int(n_jobs), n_tasks))


def _frontier_score(coherence, exclusivity):
    """``z(coherence) + z(exclusivity)`` across a K-grid, both maximized.

    Each metric is z-scored across the scanned K values (comparable scales) and
    summed. A zero-variance metric contributes nothing; a NaN in the *other*
    metric is neutralized before scoring, but a K with a non-finite value in
    *either* metric is marked degenerate (``-inf``) so it is never recommended --
    unless every K is degenerate, in which case the all-zero score falls back to
    the smallest K. Shared by the aggregate frontier and the per-seed frontier so
    the two use identical logic."""
    score = np.zeros(len(coherence))
    finite = np.ones(len(coherence), dtype=bool)
    for v in (np.asarray(coherence, np.float64), np.asarray(exclusivity, np.float64)):
        finite &= np.isfinite(v)
        sd = np.nanstd(v)
        if sd > 0:
            score += np.nan_to_num((v - np.nanmean(v)) / sd, nan=0.0)
    if finite.any():
        score[~finite] = -np.inf
    return score


class SearchKResult(list):
    """The :func:`search_k` result: a list of per-K dict rows, with the
    optimization direction stamped in and a safe ``best_k`` selector.

    It is a ``list`` subclass, so it iterates and indexes exactly like the rows
    it always returned. The additions remove two traps. The first is sorting the
    wrong way: ``coherence`` is mean UMass (negative; less-negative is better),
    so naively taking the minimum picks the worst K. The second is subtler:
    UMass coherence is roughly *monotone-decreasing* in K, so selecting K by
    coherence alone returns the smallest K in the grid regardless of the data.
    ``best_k`` defaults to a coherence/exclusivity *frontier* (a knee, not a
    maximum) to avoid that, and to the held-out metric when one is supplied.
    """

    @property
    def directions(self) -> dict:
        """``{metric: "maximize"|"minimize"}`` for the metrics actually present."""
        present = set().union(*[r.keys() for r in self]) if self else set()
        return {m: d for m, d in SEARCH_K_DIRECTIONS.items() if m in present}

    def _frontier_k(self) -> int:
        """K that maximizes ``z(coherence) + z(exclusivity)`` across the grid.

        Each metric is z-scored across the scanned K values (so the two scales
        are comparable) and added in its own optimization direction. The pick
        is the K that is jointly high on both — the knee, not either extreme.
        A metric with zero variance across the grid contributes nothing.
        """
        for m in ("coherence", "exclusivity"):
            if m not in self[0]:
                raise ValueError(
                    f"frontier selection needs {m!r} in the results "
                    f"(present: {sorted(self[0])})"
                )
        if len(self) < 2:
            raise ValueError(
                "frontier selection needs at least two K values to z-score; "
                "scan a wider grid or pass a single metric"
            )
        # With multiple seeds, score the *same* per-seed frontier curve the 1-SE
        # rule bands around (mean of each seed's z(coherence)+z(exclusivity)), so
        # 'best' and '1se' are consistent. Single seed: frontier of the mean row.
        mean = getattr(self, "_frontier_mean", None)
        if mean is not None:
            return _argbest_k(self, mean)
        score = _frontier_score([r["coherence"] for r in self],
                                 [r["exclusivity"] for r in self])
        return _argbest_k(self, score)

    def best_k(self, metric: str | None = None, *, rule: str = "best",
               frontier_metrics=None, weights=None) -> int:
        """Return the ``k`` chosen by ``metric``.

        With ``metric=None`` (the default), selection is:

        - the held-out metric when a held-out set was supplied
          (``"heldout_loglik"`` for a :class:`Heldout`, ``"perplexity"`` for a
          legacy corpus) — the principled, non-monotone criterion;
        - otherwise the ``"frontier"`` (see below), since bare ``"coherence"``
          is roughly monotone in K and would just return the grid floor.

        ``metric`` may also be given explicitly:

        - ``"frontier"`` — the K maximizing ``z(coherence) + z(exclusivity)``,
          the knee the ``plot_search_k`` curve shows (needs at least two K).
        - any column metric (``"coherence"``, ``"exclusivity"``,
          ``"heldout_loglik"``, ``"perplexity"``), optimized in its correct
          direction. Asking for bare ``"coherence"`` on a multi-K grid warns,
          because UMass coherence is roughly monotone in K.

        ``rule`` chooses how the optimum is turned into a pick:

        - ``"best"`` (default) — the K that optimizes the metric, ties broken
          toward the smaller K.
        - ``"1se"`` — the one-standard-error rule: the smallest (simplest) K whose
          metric is within one standard error of the optimum. Needs the per-K
          standard errors from ``search_k(num_seeds>1)``; raises otherwise.
        - ``"elbow"`` — the diminishing-returns knee of a *scalar* metric's
          K-curve (Kneedle: the K of maximum distance from the endpoints chord).
          For monotone-improving metrics like ``"heldout_loglik"`` whose optimum
          is the grid edge, the elbow is the more useful pick. Needs at least
          three K values; not defined for the ``"frontier"``.

        ``frontier_metrics`` / ``weights`` (only with ``metric="frontier"``)
        customize the ``"frontier"`` composite: by default it is an equal-weight
        ``z(coherence) + z(exclusivity)``. Pass a different metric list (e.g.
        ``["coherence", "deveaud"]``) and/or non-negative per-metric ``weights`` to
        reshape the knee. A custom frontier supports ``rule="best"`` only, and it
        z-scores the across-seed *mean* rows; under ``num_seeds>1`` that can differ
        marginally from the implicit default frontier, which z-scores each seed
        first (to keep ``rule="1se"`` consistent).
        """
        if rule not in ("best", "1se", "elbow"):
            raise ValueError(f"rule must be 'best', '1se', or 'elbow', got {rule!r}")
        if not self:
            raise ValueError("search_k returned no rows")
        if metric is None:
            if "heldout_loglik" in self[0]:
                metric = "heldout_loglik"
            elif "perplexity" in self[0]:
                metric = "perplexity"
            elif len(self) >= 2 and "coherence" in self[0] and "exclusivity" in self[0]:
                metric = "frontier"
            else:
                metric = "coherence"
        if metric != "frontier" and (frontier_metrics is not None or weights is not None):
            raise ValueError(
                "frontier_metrics and weights only apply to metric='frontier'; "
                f"got metric={metric!r}")
        if metric == "frontier":
            if rule == "elbow":
                raise ValueError("rule='elbow' is not defined for the frontier; "
                                 "use it on a scalar metric like 'heldout_loglik'")
            if frontier_metrics is not None or weights is not None:
                return self._custom_frontier_k(frontier_metrics, weights, rule)
            return self._frontier_k_1se() if rule == "1se" else self._frontier_k()
        if metric not in SEARCH_K_DIRECTIONS:
            raise ValueError(
                f"unknown metric {metric!r}; choose 'frontier' or one of "
                f"{sorted(SEARCH_K_DIRECTIONS)}"
            )
        if metric not in self[0]:
            raise ValueError(
                f"metric {metric!r} not in results (present: {sorted(self[0])}); "
                f"pass held_out= to get a held-out metric"
            )
        if metric == "coherence" and len(self) >= 2:
            coh_metric = self[0].get("coherence_metric", "u_mass")
            if coh_metric == "u_mass":
                warnings.warn(
                    "best_k(metric='coherence'): mean UMass coherence is roughly "
                    "monotone-decreasing in K, so this tends to return the smallest "
                    "K in the grid. Prefer metric='frontier' (coherence/exclusivity "
                    "knee) or pass held_out= for held-out log-likelihood.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"best_k(metric='coherence'): selecting on {coh_metric!r} "
                    "coherence alone ignores exclusivity and parsimony. Prefer "
                    "metric='frontier' (coherence/exclusivity knee) or pass "
                    "held_out= for held-out log-likelihood.",
                    UserWarning,
                    stacklevel=2,
                )
        present = [r for r in self if np.isfinite(r[metric])]
        if not present:
            raise ValueError(f"metric {metric!r} has no finite value")
        maximize = SEARCH_K_DIRECTIONS[metric] == "maximize"
        if rule == "1se":
            return self._one_se_k(present, metric, maximize)
        if rule == "elbow":
            return self._elbow_k(present, metric, maximize)
        best = (max if maximize else min)(r[metric] for r in present)
        # Parsimony tie-break: smallest k achieving the best value.
        return int(min(r["k"] for r in present if r[metric] == best))

    def _one_se_k(self, present, metric, maximize):
        """One-standard-error rule on a scalar ``metric``: the smallest K whose
        mean is within one SE of the optimum (in the metric's direction)."""
        se_key = metric + "_se"
        if se_key not in present[0]:
            raise ValueError(
                f"rule='1se' needs per-K standard errors for {metric!r}; "
                "refit with search_k(num_seeds>1)"
            )
        best_row = (max if maximize else min)(present, key=lambda r: r[metric])
        se = best_row[se_key]
        if maximize:
            thresh = best_row[metric] - se
            within = [r for r in present if r[metric] >= thresh]
        else:
            thresh = best_row[metric] + se
            within = [r for r in present if r[metric] <= thresh]
        return int(min(r["k"] for r in within))  # simplest K within 1 SE

    def _elbow_k(self, present, metric, maximize):
        """Diminishing-returns knee of ``metric`` vs K (Kneedle): normalize the
        curve, orient it so higher = better, and take the K of maximum vertical
        distance above the chord joining the first and last scanned K."""
        rows = sorted(present, key=lambda r: r["k"])
        if len(rows) < 3:
            raise ValueError("rule='elbow' needs at least three K values")
        x = np.array([r["k"] for r in rows], dtype=np.float64)
        y = np.array([r[metric] for r in rows], dtype=np.float64)
        if not maximize:
            y = -y  # orient so a better metric is a higher y
        xr, yr = np.ptp(x), np.ptp(y)
        if xr == 0 or yr == 0:
            return int(x[0])  # flat curve -> simplest K
        xn = (x - x.min()) / xr
        yn = (y - y.min()) / yr
        # A diminishing-returns curve bulges above the endpoints chord; its peak
        # is the elbow.
        chord = yn[0] + (yn[-1] - yn[0]) * (xn - xn[0]) / (xn[-1] - xn[0])
        gap = yn - chord
        if np.max(gap) <= 1e-12:  # convex or straight: no diminishing-returns knee
            warnings.warn(
                f"rule='elbow': the {metric!r} curve does not bend toward "
                "diminishing returns (it is convex or straight), so the elbow is "
                "not well defined; returning the smallest K.",
                UserWarning, stacklevel=3)
            return int(x.min())
        return int(x[int(np.argmax(gap))])

    def _custom_frontier_k(self, frontier_metrics, weights, rule):
        """Frontier over a caller-chosen metric set / weights. Supports
        ``rule='best'`` only (the per-seed 1-SE band is the default frontier's)."""
        if rule != "best":
            raise ValueError(
                f"a custom frontier supports rule='best' only, got {rule!r}")
        metrics = list(frontier_metrics) if frontier_metrics is not None \
            else ["coherence", "exclusivity"]
        if len(metrics) < 1:
            raise ValueError("frontier_metrics must name at least one metric")
        for m in metrics:
            if m not in SEARCH_K_DIRECTIONS:
                raise ValueError(f"unknown frontier metric {m!r}")
            if m not in self[0]:
                raise ValueError(f"frontier metric {m!r} not in results "
                                 f"(present: {sorted(self[0])})")
        if weights is None:
            weights = [1.0] * len(metrics)
        if len(weights) != len(metrics):
            raise ValueError("weights must match frontier_metrics in length")
        if any((not np.isfinite(w)) or w < 0 for w in weights):
            raise ValueError("weights must be finite and non-negative")
        if len(self) < 2:
            raise ValueError("frontier selection needs at least two K values")
        score = np.zeros(len(self))
        finite = np.ones(len(self), dtype=bool)
        for w, m in zip(weights, metrics):
            v = np.array([r[m] for r in self], dtype=np.float64)
            if SEARCH_K_DIRECTIONS[m] == "minimize":
                v = -v
            finite &= np.isfinite(v)
            sd = np.nanstd(v)
            if sd > 0:
                score += float(w) * np.nan_to_num((v - np.nanmean(v)) / sd, nan=0.0)
        if finite.any():
            score[~finite] = -np.inf
        return _argbest_k(self, score)

    def _frontier_k_1se(self) -> int:
        """One-standard-error rule on the frontier composite: needs the per-seed
        frontier scores stored by ``search_k(num_seeds>1)``."""
        mean = getattr(self, "_frontier_mean", None)
        sem = getattr(self, "_frontier_se", None)
        if mean is None or sem is None:
            raise ValueError(
                "rule='1se' on the frontier needs per-seed scores; "
                "refit with search_k(num_seeds>1)"
            )
        finite = np.isfinite(mean)
        if not finite.any():  # every K degenerate -> smallest K, as _frontier_k does
            return int(min(r["k"] for r in self))
        best_i = int(np.nanargmax(np.where(finite, mean, np.nan)))
        thresh = mean[best_i] - sem[best_i]
        within = [i for i in range(len(self)) if finite[i] and mean[i] >= thresh]
        return int(min(self[i]["k"] for i in within))


def _aggregate_over_seeds(ks, seeds, tasks, fitted):
    """Collapse per-``(K, seed)`` fits into one row per K carrying each metric's
    mean and a ``<metric>_se`` (standard error), and stash the per-seed frontier
    scores on the result so ``best_k(rule='1se')`` works on the composite too."""
    by_k = {k: [] for k in ks}
    for (k, _fs), row in zip(tasks, fitted):
        by_k[k].append(row)
    n = len(seeds)
    keep_scalar = {"k", "coherence_metric"}
    no_se = {"dispersion_pvalue"}  # an SE on a p-value is not meaningful
    rows, coh_by_k, exc_by_k = [], [], []
    for k in ks:
        seed_rows = by_k[k]
        agg = {"k": k}
        if "coherence_metric" in seed_rows[0]:
            agg["coherence_metric"] = seed_rows[0]["coherence_metric"]
        for key in seed_rows[0]:
            if key in keep_scalar:
                continue
            vals = np.array([r[key] for r in seed_rows], dtype=np.float64)
            agg[key] = float(np.mean(vals))
            if key not in no_se:
                agg[key + "_se"] = float(np.std(vals, ddof=1) / np.sqrt(n))
        rows.append(agg)
        coh_by_k.append([r["coherence"] for r in seed_rows])
        exc_by_k.append([r["exclusivity"] for r in seed_rows])

    result = SearchKResult(rows)
    # Per-seed frontier: z-score within each seed across K (via the shared helper),
    # then mean / SE over seeds. Lets the 1-SE rule apply to the frontier composite.
    coh = np.asarray(coh_by_k, np.float64)  # (nK, n_seeds)
    exc = np.asarray(exc_by_k, np.float64)
    fs = np.vstack([_frontier_score(coh[:, s], exc[:, s]) for s in range(n)])  # (n, nK)
    with np.errstate(invalid="ignore"):
        result._frontier_mean = fs.mean(axis=0)
        result._frontier_se = fs.std(axis=0, ddof=1) / np.sqrt(n)
    return result


def search_k(
    docs,
    ks,
    *,
    model="lda",
    prevalence=None,
    content=None,
    held_out=None,
    iters=500,
    num_samples=3,
    sample_interval=10,
    seed=42,
    coherence_n=10,
    coherence_type="u_mass",
    n_jobs=1,
    num_seeds=_SEARCH_K_DEFAULT_SEEDS,
    criteria=(),
):
    """Fit a model for each K and report quality metrics (stm's ``searchK``).

    With ``model="lda"`` (default) fits an :class:`~topica.LDA` per K. With
    ``model="stm"`` fits an :class:`~topica.STM` per K — pass ``prevalence``
    (a covariate design matrix) and optional ``content`` (group labels) to
    scan K for the model you'll actually report.

    Returns a :class:`SearchKResult` (a list of per-K dicts) with ``k``,
    ``coherence`` (mean of the selected coherence type; for ``model="stm"`` with
    the default ``u_mass`` this is stm's semantic coherence, labelled
    ``coherence_metric="semcoh"``), ``exclusivity`` (mean top-word exclusivity),
    ``dispersion`` (residual dispersion, Taddy 2012 — ``>> 1`` means K is too
    small) with its ``dispersion_pvalue``, and — when ``held_out`` is supplied —
    a held-out quality metric. The result also carries ``.directions`` (whether
    higher or lower is better per metric) and a ``.best_k(metric=...)`` selector.
    ``best_k`` defaults to the held-out metric when one is supplied, otherwise to
    a coherence/exclusivity frontier (a knee), because bare UMass coherence is
    roughly monotone in K and would just return the smallest K scanned. Duplicate
    ``ks`` are dropped; ties in ``best_k`` break toward the smaller (simpler) K.

    Two held-out paths are supported, determined by the type of ``held_out``:

    - **Heldout object** (from :func:`make_heldout`): scored with
      :func:`eval_heldout`; results stored under ``"heldout_loglik"``
      (``mean_per_doc_loglik``, higher / less negative is better). Use this
      path for the standard within-corpus word-heldout diagnostic.
    - **Corpus or token lists** (legacy): scored with :func:`perplexity`;
      results stored under ``"perplexity"`` (lower is better). This is the
      document-completion perplexity on a separate held-out set.

    Parameters
    ----------
    docs : training documents (``list[list[str]]`` or a ``Corpus``).
    ks : sequence of topic counts to scan.
    model : ``"lda"`` (default) or ``"stm"``.
    prevalence : covariate design matrix for ``model="stm"``; ignored otherwise.
    content : optional content group labels (sequence of str/int) for ``model="stm"``.
    held_out : optional held-out set. Pass a :class:`Heldout` (from
        :func:`make_heldout`) or a separate corpus / token lists.
    iters : training iterations per fit.
    num_samples : Gibbs samples per fit (LDA only).
    sample_interval : iterations between Gibbs samples (LDA only).
    seed : RNG seed for every fit and transform call.
    coherence_n : top-word count used for coherence and exclusivity.
    coherence_type : one of ``"u_mass"``, ``"c_uci"``, ``"c_npmi"``, ``"c_v"`` (default ``"u_mass"``).
    n_jobs : number of worker threads for the per-fit work (default ``1``, serial).
        The fits are independent and each keeps its own fixed seed, so the results
        are identical to the serial run; only the wall-clock changes (the Rust fits
        release the GIL). ``n_jobs<=0`` (or ``None``) uses all cores, capped at the
        number of fits. Note it multiplies with any intra-fit threading
        (``num_threads=`` on the model), so ``n_jobs`` above the core count can
        oversubscribe.
    num_seeds : number of seeds fit per K (default ``1``). With ``num_seeds>1``,
        each K is refit over seeds ``seed, seed+1, ...``; every metric column then
        holds the across-seed mean and gains a ``<metric>_se`` standard-error
        column, and ``best_k(rule="1se")`` becomes available (the simplest K within
        one SE of the optimum). The per-K work parallelizes over ``(K, seed)`` via
        ``n_jobs``. A single seed carries no standard errors (backward-compatible).
    criteria : optional extra K-selection criteria to report as columns (default
        none). ``"deveaud"`` (Deveaud et al. 2014; mean pairwise Jensen-Shannon
        divergence between topics, higher = more distinct) and ``"cao_juan"``
        (Cao Juan et al. 2009; mean pairwise topic cosine, lower = less redundant).
        Opt-in and out of the frontier, but selectable via ``best_k("deveaud")`` /
        ``best_k("cao_juan")`` and carry standard errors under ``num_seeds>1`` like
        any other metric.
    """
    from . import LDA, STM  # local import to avoid a cycle at module load

    if model not in ("lda", "stm"):
        raise ValueError("model must be 'lda' or 'stm'")
    if int(num_seeds) < 1:
        raise ValueError(f"num_seeds must be >= 1, got {num_seeds!r}")
    if content is not None and model != "stm":
        raise ValueError("content covariates are only supported when model='stm'")

    criteria = tuple(criteria)
    bad = [c for c in criteria if c not in _SEARCH_K_CRITERIA]
    if bad:
        raise ValueError(
            f"unknown criteria {bad}; choose from {list(_SEARCH_K_CRITERIA)}")

    valid_coh = ("u_mass", "c_uci", "c_npmi", "c_v")
    coherence_type = coherence_type.lower()
    # A "stratified_<type>" request (content models only) scores each group's own
    # top words against its own subcorpus (topica.content.stratified_coherence).
    stratified = coherence_type.startswith("stratified_")
    base_ct = coherence_type[len("stratified_"):] if stratified else coherence_type
    if base_ct not in valid_coh:
        raise ValueError(
            f"coherence_type must be one of {valid_coh} "
            f"(optionally 'stratified_'-prefixed for content models), got {coherence_type!r}")
    if stratified and (model != "stm" or content is None):
        raise ValueError(
            "stratified coherence needs model='stm' with a content covariate")

    if model == "stm" and content is not None and not stratified:
        warnings.warn(
            "Exclusivity and coherence calculations on a model with content covariates "
            "are computed on the baseline/group-average topic-word distributions. "
            "These metrics do not capture group-specific wording variations. Pass "
            "coherence_type='stratified_c_v' (etc.) for group-stratified metrics.",
            UserWarning,
            stacklevel=2,
        )

    ks_in = [int(k) for k in ks]
    ks = list(dict.fromkeys(ks_in))  # de-duplicate, preserving order
    if len(ks) != len(ks_in):
        warnings.warn(
            "search_k: duplicate values in `ks` were dropped so each K is fit "
            "once (duplicates would also overweight that K in the frontier).",
            UserWarning,
            stacklevel=2,
        )

    ref_docs = _ref_corpus(docs)  # token lists, reused across every K

    def _fit_row(k, fit_seed):
        if model == "stm":
            m = STM(num_topics=k, seed=fit_seed)
            m.fit(docs, prevalence, content=content, iters=iters)
        else:
            m = LDA(num_topics=k, seed=fit_seed)
            m.fit(docs, iters=iters, num_samples=num_samples,
                  sample_interval=sample_interval)

        coh_label = coherence_type
        if stratified:
            from .content import (stratified_coherence as _strat,
                                  topic_polarization as _pol,
                                  group_exclusivity as _gex)
            coh_val = float(np.mean(_strat(m, docs, content, coherence_type=base_ct,
                                          n=coherence_n)))
        elif coherence_type == "u_mass" and model == "stm":
            # stm's searchK reports semantic coherence (semCoh1beta, 0.01
            # smoothing), not gensim UMass -- use the stm-faithful metric for STM.
            from .coherence import semantic_coherence
            coh_val = float(np.mean(semantic_coherence(m, ref_docs, n=coherence_n)))
            coh_label = "semcoh"
        elif coherence_type == "u_mass" and hasattr(m, "coherence"):
            coh_val = float(np.mean(m.coherence(coherence_n)))
        else:
            from .coherence import coherence as external_coherence
            # m.top_words(coherence_n) returns a list of lists of (word, weight) tuples.
            # Convert to list[list[str]]
            topics = [[w for w, _ in top_list] for top_list in m.top_words(coherence_n)]
            scores = external_coherence(topics, ref_docs, coherence_type=base_ct, topn=coherence_n)
            coh_val = float(np.mean(scores))

        row = {
            "k": k,
            "coherence": coh_val,
            "coherence_metric": coh_label,
            "exclusivity": (float(np.mean(_gex(m, n=coherence_n))) if stratified
                            else _mean_exclusivity(m.topic_word, coherence_n)),
        }
        # Residual dispersion (Taddy 2012): dispersion >> 1 is direct evidence K
        # is too small -- the non-monotone signal stm's searchK reports. Diagnostic
        # column, not a frontier metric (it keeps falling as K grows).
        rc = check_residuals(m, ref_docs)
        row["dispersion"] = float(rc.dispersion)
        row["dispersion_pvalue"] = float(rc.pvalue)
        # Opt-in ldatuning-style criteria from the topic-word matrix.
        for c in criteria:
            row[c] = _extra_criterion(c, m.topic_word)
        if stratified:
            row["polarization"] = float(np.mean(_pol(m)))
        if held_out is not None:
            if isinstance(held_out, Heldout):
                result = eval_heldout(m, held_out, seed=fit_seed)
                row["heldout_loglik"] = float(result.mean_per_doc_loglik)
            else:
                row["perplexity"] = float(perplexity(m, held_out, seed=fit_seed))
        return row

    # One task per (K, seed). Each fit is independent with its own fixed seed, so
    # results are identical to the serial path (verified). topica's Rust fits
    # release the GIL, so a thread pool parallelizes wall-clock with no pickling.
    seeds = [seed + s for s in range(int(num_seeds))]
    tasks = [(k, fs) for k in ks for fs in seeds]
    workers = _resolve_workers(n_jobs, len(tasks))
    if workers == 1:
        fitted = [_fit_row(k, fs) for (k, fs) in tasks]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fitted = list(pool.map(lambda t: _fit_row(*t), tasks))  # preserves order

    if len(seeds) == 1:
        return SearchKResult(fitted)  # one row per K, no standard errors
    return _aggregate_over_seeds(ks, seeds, tasks, fitted)


# ---------------------------------------------------------------------------
# selectModel: best-of-N runs at fixed K  (stm §3.4)
# ---------------------------------------------------------------------------

@dataclass
class SelectModelResult:
    """Result of :func:`select_model`.

    Attributes
    ----------
    models : list of N fitted models, one per run.
    coherence : array of shape ``(N,)`` — per-run mean UMass coherence.
    exclusivity : array of shape ``(N,)`` — per-run mean top-word exclusivity.
    run_seeds : array of shape ``(N,)`` — seed used for each run.
    """

    models: list
    coherence: np.ndarray
    exclusivity: np.ndarray
    run_seeds: np.ndarray


def select_model(
    docs,
    K,
    *,
    runs=20,
    model="lda",
    prevalence=None,
    word_embeddings=None,
    vocabulary=None,
    doc_embeddings=None,
    iters=500,
    num_samples=3,
    sample_interval=10,
    seed=42,
    coherence_n=10,
    fraction=None,
    burn_in_iters=None,
):
    """Run N initializations at a fixed K and return the fitted candidates (stm's ``selectModel``).

    All ``runs`` models are fit from different random seeds. With
    ``fraction`` set, the procedure uses two stages: a short burn-in
    (``burn_in_iters``, defaulting to 20% of ``iters``) followed by
    full training of the top ``ceil(fraction * runs)`` models by their
    objective (ELBO where the model has one, else log-likelihood, else
    mean coherence). This mirrors stm's "run briefly, keep the best ~20%"
    heuristic.

    This is for models whose fit depends on the random seed — the ones that
    scatter across local optima. ``ETM``, ``ProdLDA``, ``FASTopic``,
    ``CombinedTM``, and ``ZeroShotTM`` all benefit. ``STM``/``CTM`` use a
    deterministic spectral init, so every run is identical and multi-start buys
    nothing — pick one of the stochastic models instead. (``DTM`` is not selected
    here: its topics are time-varying, so coherence/exclusivity are not a single
    number; use ``DTM(init="spectral")`` for a deterministic fit.)

    Parameters
    ----------
    docs : training documents (``list[list[str]]`` or a ``Corpus``).
    K : number of topics for every run.
    runs : number of random initializations.
    model : which model to fit. One of ``"lda"`` (default), ``"stm"``,
        ``"prodlda"``, ``"etm"``, ``"fastopic"``, ``"combinedtm"``,
        ``"zeroshottm"``.
    prevalence : covariate design matrix; required when ``model="stm"``.
    word_embeddings : ``(vocab, dim)`` word-embedding matrix; required when
        ``model="etm"`` (paired with ``vocabulary``).
    vocabulary : the word list aligning ``word_embeddings`` rows; required when
        ``model="etm"``.
    doc_embeddings : ``(num_docs, dim)`` document-embedding matrix; required when
        ``model`` is ``"fastopic"``, ``"combinedtm"``, or ``"zeroshottm"``.
    iters : full-training iterations per run (or per survivor when
        ``fraction`` is used).
    num_samples : Gibbs samples per run (LDA only).
    sample_interval : iterations between Gibbs samples (LDA only).
    seed : base RNG seed; run ``r`` uses seed ``seed + r``.
    coherence_n : top-word count for coherence and exclusivity.
    fraction : if given (a float in ``(0, 1]``), keep only the top
        ``ceil(fraction * runs)`` models (by their objective) after
        ``burn_in_iters`` and run those survivors to full ``iters``.
        ``None`` (default) runs all initializations to full ``iters``.
    burn_in_iters : burn-in length used for early discard; defaults to
        ``max(1, round(0.2 * iters))`` when ``fraction`` is set.

    Returns
    -------
    A :class:`SelectModelResult` with ``models``, ``coherence``,
    ``exclusivity``, and ``run_seeds`` arrays of length equal to the
    number of survivors (all ``runs`` when ``fraction`` is ``None``).
    """
    from . import (  # local import to avoid a cycle
        LDA, STM, CombinedTM, ETM, FASTopic, ProdLDA, ZeroShotTM,
    )

    valid = ("lda", "stm", "prodlda", "etm", "fastopic", "combinedtm",
             "zeroshottm")
    if model not in valid:
        raise ValueError(f"model must be one of {valid}, got {model!r}")
    if not isinstance(runs, int) or runs < 1:
        raise ValueError(f"runs must be a positive integer, got {runs!r}")
    if fraction is not None and not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction!r}")
    # Per-model required data.
    if model == "stm" and prevalence is None:
        raise ValueError("model='stm' requires prevalence=")
    if model == "etm" and (word_embeddings is None or vocabulary is None):
        raise ValueError("model='etm' requires word_embeddings= and vocabulary=")
    if model in ("fastopic", "combinedtm", "zeroshottm") and doc_embeddings is None:
        raise ValueError(f"model={model!r} requires doc_embeddings=")

    def _make(s):
        if model == "stm":
            return STM(num_topics=K, seed=s)
        if model == "prodlda":
            return ProdLDA(num_topics=K, seed=s)
        if model == "etm":
            return ETM(num_topics=K, seed=s)
        if model == "fastopic":
            return FASTopic(num_topics=K, seed=s)
        if model == "combinedtm":
            return CombinedTM(num_topics=K, seed=s)
        if model == "zeroshottm":
            return ZeroShotTM(num_topics=K, seed=s)
        return LDA(num_topics=K, seed=s)

    def _fit(m, n_iters):
        if model == "stm":
            m.fit(docs, prevalence, iters=n_iters)
        elif model == "etm":
            m.fit(docs, word_embeddings, vocabulary, iters=n_iters)
        elif model in ("fastopic", "combinedtm", "zeroshottm"):
            m.fit(docs, doc_embeddings, iters=n_iters)
        elif model == "lda":
            m.fit(docs, iters=n_iters, num_samples=num_samples,
                  sample_interval=sample_interval)
        else:  # prodlda
            m.fit(docs, iters=n_iters)

    def _objective(m):
        """Scalar objective for early discard: higher is better. Falls back to
        mean coherence for models with no scalar bound (e.g. FASTopic)."""
        if hasattr(m, "bound"):
            b = float(m.bound)
            if not np.isnan(b):
                return b
        if hasattr(m, "log_likelihood") and callable(m.log_likelihood):
            return float(m.log_likelihood())
        return float(np.mean(m.coherence(coherence_n)))

    run_seeds = [seed + r for r in range(runs)]

    if fraction is None:
        # Simple path: run every initialization to full iters.
        fitted = []
        for s in run_seeds:
            m = _make(s)
            _fit(m, iters)
            fitted.append(m)
        survivor_seeds = run_seeds
    else:
        # Two-stage: burn-in, then re-run survivors.
        n_burn = burn_in_iters if burn_in_iters is not None else max(1, round(0.2 * iters))
        import math
        n_keep = max(1, math.ceil(fraction * runs))

        # Stage 1: burn-in for all runs.
        burn_models = []
        for s in run_seeds:
            m = _make(s)
            _fit(m, n_burn)
            burn_models.append(m)

        # Rank by objective (higher is better); keep top n_keep.
        scored = sorted(
            zip(run_seeds, burn_models),
            key=lambda pair: _objective(pair[1]),
            reverse=True,
        )
        survivors = scored[:n_keep]

        # Stage 2: run survivors to full iters.
        fitted = []
        survivor_seeds = []
        for s, _ in survivors:
            m = _make(s)
            _fit(m, iters)
            fitted.append(m)
            survivor_seeds.append(s)

    coh = np.array([float(np.mean(m.coherence(coherence_n))) for m in fitted])
    excl = np.array([_mean_exclusivity(m.topic_word, coherence_n) for m in fitted])

    return SelectModelResult(
        models=fitted,
        coherence=coh,
        exclusivity=excl,
        run_seeds=np.array(survivor_seeds, dtype=np.intp),
    )


def plot_models(result, *, ax=None, label_runs=True):
    """Coherence-vs-exclusivity scatter for :func:`select_model` candidates (stm's ``plotModels``).

    Each point is one run. The upper-right corner is the best region:
    both coherent (interpretable) and exclusive (distinctive). Use
    this plot to pick a run from :func:`select_model` before fitting
    your full analysis.

    Parameters
    ----------
    result : a :class:`SelectModelResult` returned by :func:`select_model`.
    ax : matplotlib ``Axes`` to draw on; a new figure is created if
        ``None``.
    label_runs : annotate each point with its run index; default
        ``True``.

    Returns
    -------
    The matplotlib ``Axes``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "plot_models needs matplotlib (pip install matplotlib)."
        ) from e

    coh = np.asarray(result.coherence)
    excl = np.asarray(result.exclusivity)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(coh, excl, color="C0", zorder=3)
    if label_runs:
        for i, (x, y) in enumerate(zip(coh, excl)):
            ax.annotate(str(i), (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=8)

    ax.set_xlabel("Mean semantic coherence (UMass)")
    ax.set_ylabel("Mean exclusivity")
    ax.set_title("Model candidates: coherence vs. exclusivity")
    ax.figure.tight_layout()
    return ax


def plot_search_k(rows, *, metrics=("coherence", "exclusivity"), ax=None):
    """Plot :func:`search_k` results: each metric against the number of topics.

    Researchers read this curve to choose `K`: coherence and exclusivity usually
    trade off, so the goal is a knee, not a maximum. ``rows.best_k()`` returns
    that knee directly (the ``"frontier"`` selector). Each metric gets its own
    y-axis (they live on different scales). ``rows`` is the list returned by
    :func:`search_k`; ``metrics`` selects which of its keys to draw (any of
    ``"coherence"``, ``"exclusivity"``, ``"perplexity"``, ``"heldout_loglik"``).
    Only metrics present in the rows are drawn; absent keys are silently skipped.
    Returns the primary matplotlib ``Axes``. Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - exercised via message
        raise ImportError(
            "plot_search_k needs matplotlib (pip install matplotlib)."
        ) from e

    rows = sorted(rows, key=lambda r: r["k"])
    ks = [r["k"] for r in rows]
    metrics = [m for m in metrics if any(m in r for r in rows)]
    if not metrics:
        raise ValueError("none of the requested metrics are present in rows")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    lines = []
    for i, metric in enumerate(metrics):
        a = ax if i == 0 else ax.twinx()
        if i >= 2:  # offset a third axis so it doesn't overlap the second
            a.spines["right"].set_position(("axes", 1.0 + 0.18 * (i - 1)))
        color = f"C{i}"
        vals = [r.get(metric, float("nan")) for r in rows]
        (line,) = a.plot(ks, vals, marker="o", color=color, label=metric)
        a.set_ylabel(metric, color=color)
        a.tick_params(axis="y", labelcolor=color)
        lines.append(line)

    ax.set_xlabel("number of topics (K)")
    ax.set_xticks(ks)
    ax.legend(lines, [li.get_label() for li in lines], loc="best")
    ax.figure.tight_layout()
    return ax


def plot_topic_discovery(model, *, ax=None):
    """Plot an HDP fit's topic-discovery trajectory: the inferred number of
    topics K against the Gibbs iteration, with the per-token log-likelihood on a
    twin axis. Watching K rise, fall, and settle (while the log-likelihood
    plateaus) is the nonparametric model's headline convergence check — the
    analog of reading a `search_k` curve, but learned in a single fit.

    ``model`` is a fitted :class:`~topica.HDP` (its ``topic_count_history`` and
    ``log_likelihood_history`` are read). Returns the primary matplotlib
    ``Axes``. Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - exercised via message
        raise ImportError(
            "plot_topic_discovery needs matplotlib (pip install matplotlib)."
        ) from e

    tch = list(model.topic_count_history)
    llh = list(model.log_likelihood_history)
    if not tch:
        raise ValueError(
            "no discovery trace recorded; fit with report_interval > 0 "
            "(or the default auto cadence)"
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    iters = [it for it, _ in tch]
    ks = [k for _, k in tch]
    (line_k,) = ax.plot(iters, ks, color="C0", marker="o", ms=3, label="topics (K)")
    ax.set_xlabel("Gibbs iteration")
    ax.set_ylabel("number of topics (K)", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")

    lines = [line_k]
    if llh:
        a2 = ax.twinx()
        (line_ll,) = a2.plot(
            [it for it, _ in llh], [ll for _, ll in llh],
            color="C1", marker="s", ms=2, label="log-likelihood",
        )
        a2.set_ylabel("per-token log-likelihood", color="C1")
        a2.tick_params(axis="y", labelcolor="C1")
        lines.append(line_ll)

    ax.legend(lines, [li.get_label() for li in lines], loc="best")
    ax.figure.tight_layout()
    return ax


def _mean_exclusivity(topic_word, n: int) -> float:
    from .coherence import exclusivity
    return float(np.mean(exclusivity(topic_word, n=n)))


def _cao_juan(topic_word) -> float:
    """Mean pairwise cosine similarity between topic-word distributions
    (Cao Juan et al. 2009). Lower is better -- redundant topics are similar, so
    the least-redundant K minimizes it. ``nan`` for a single topic."""
    phi = np.asarray(topic_word, dtype=np.float64)
    k = phi.shape[0]
    if k < 2:
        return float("nan")
    norm = np.linalg.norm(phi, axis=1)
    unit = phi / np.where(norm > 0, norm, 1.0)[:, None]
    sim = unit @ unit.T
    return float(sim[np.triu_indices(k, 1)].mean())


def _deveaud(topic_word) -> float:
    """Mean pairwise Jensen-Shannon divergence between topic-word distributions
    (Deveaud et al. 2014). Higher is better -- distinct topics diverge, so the
    most-distinct K maximizes it. ``nan`` for a single topic."""
    phi = np.asarray(topic_word, dtype=np.float64)
    phi = phi / np.clip(phi.sum(axis=1, keepdims=True), 1e-300, None)
    k = phi.shape[0]
    if k < 2:
        return float("nan")

    def _kl(p, q):
        # The mixture q = (p+q)/2 is strictly positive wherever p > 0, so no
        # smoothing is needed and the result is an exact Jensen-Shannon term.
        mask = p > 0
        return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

    total, pairs = 0.0, 0
    for i in range(k):
        for j in range(i + 1, k):
            mix = 0.5 * (phi[i] + phi[j])
            total += 0.5 * _kl(phi[i], mix) + 0.5 * _kl(phi[j], mix)
            pairs += 1
    return total / pairs


def _extra_criterion(name, topic_word) -> float:
    return {"deveaud": _deveaud, "cao_juan": _cao_juan}[name](topic_word)


# ---------------------------------------------------------------------------
# LDAvis relevance + pyLDAvis export
# ---------------------------------------------------------------------------

def relevance(topic_word, vocabulary=None, *, topic=None, lam=0.6, n=10, term_frequency=None):
    """LDAvis *relevance* of words to topics (Sievert & Shirley 2014):

    ``relevance(w | t) = λ·log p(w|t) + (1-λ)·log[p(w|t) / p(w)]``

    λ=1 ranks by probability; λ=0 by lift (exclusivity); the LDAvis default 0.6
    balances them. ``p(w)`` is the corpus word marginal — pass ``term_frequency``
    (word counts in `vocabulary` order) for the empirical marginal, else the
    topic-averaged φ is used. Returns ``(word, relevance)`` lists per topic, or
    for one ``topic``.

    `topic_word` is a fitted model (uses its ``topic_word`` and ``vocabulary``)
    or a ``(K, V)`` array, in which case pass ``vocabulary``.
    """
    vocabulary = _vocabulary_of(topic_word, vocabulary)
    phi = _as_topic_word(topic_word)
    k, _ = phi.shape
    if term_frequency is not None:
        tf = np.asarray(term_frequency, dtype=np.float64)
        pw = tf / tf.sum()
    else:
        pw = phi.mean(axis=0)
    pw = np.clip(pw, 1e-12, None)
    log_phi = np.log(np.clip(phi, 1e-12, None))
    rel = lam * log_phi + (1.0 - lam) * (log_phi - np.log(pw))  # (K, V)

    def top(t):
        idx = np.argsort(rel[t])[::-1][:n]
        return [(vocabulary[i], float(rel[t, i])) for i in idx]

    if topic is not None:
        if topic < 0 or topic >= k:
            raise ValueError(f"topic {topic} out of range (num_topics={k})")
        return top(topic)
    return [top(t) for t in range(k)]


@dataclass
class PyLDAvisInputs:
    """The five arrays ``pyLDAvis.prepare`` needs, for when pyLDAvis is not
    installed. ``pyLDAvis.prepare(*inputs.unpack())`` reconstructs the view."""

    topic_term_dists: np.ndarray
    doc_topic_dists: np.ndarray
    doc_lengths: np.ndarray
    vocab: list
    term_frequency: np.ndarray

    def unpack(self):
        return (self.topic_term_dists, self.doc_topic_dists, self.doc_lengths,
                self.vocab, self.term_frequency)


def prepare_pyldavis(model, docs, **kwargs):
    """Build the LDAvis intertopic-distance visualization for a fitted model.

    `docs` are the tokenized training documents (``list[list[str]]``), used for
    document lengths and term frequencies. If ``pyLDAvis`` is installed this
    returns its ``PreparedData`` (pass to ``pyLDAvis.display`` / ``save_html``);
    otherwise it returns a :class:`PyLDAvisInputs` you can feed to
    ``pyLDAvis.prepare`` later. Extra ``kwargs`` go to ``pyLDAvis.prepare``
    (e.g. ``sort_topics=False``).
    """
    # Accept a Corpus directly (recover its token lists), so a corpus built via
    # from_dataframe does not need re-tokenizing from the original text.
    if hasattr(docs, "documents") and callable(getattr(docs, "documents")):
        docs = docs.documents()
    phi = np.asarray(model.topic_word, dtype=np.float64)
    theta = np.asarray(model.doc_topic, dtype=np.float64)
    vocab = list(model.vocabulary)
    if len(docs) != theta.shape[0]:
        raise ValueError(
            f"docs has {len(docs)} entries but doc_topic has {theta.shape[0]} rows; "
            "pass the same documents used to fit the model"
        )
    vindex = {w: i for i, w in enumerate(vocab)}
    tf = np.zeros(len(vocab))
    doc_lengths = np.zeros(len(docs), dtype=np.int64)
    for d, doc in enumerate(docs):
        for w in doc:
            i = vindex.get(w)
            if i is not None:
                tf[i] += 1.0
                doc_lengths[d] += 1
    inputs = PyLDAvisInputs(phi, theta, doc_lengths, vocab, tf)
    try:
        import pyLDAvis
    except ImportError:
        return inputs
    return pyLDAvis.prepare(phi, theta, doc_lengths, vocab, tf, **kwargs)


# ---------------------------------------------------------------------------
# checkResiduals: residual-dispersion test for K selection (Taddy 2012)
# ---------------------------------------------------------------------------

def _gammq(a, x):
    """Regularized upper incomplete gamma Q(a, x) (Numerical Recipes)."""
    import math
    if x < 0 or a <= 0:
        return float("nan")
    if x == 0.0:
        return 1.0  # Q(a, 0) = 1
    if x < a + 1.0:  # series for the lower P, then complement
        ap = a
        s = 1.0 / a
        d = s
        for _ in range(500):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-14:
                break
        return 1.0 - s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    # continued fraction for the upper Q
    fpmin = 1e-300
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def _chisq_sf(x, df):
    """Upper-tail (survival) probability of a chi-square with `df` df."""
    if df <= 0:
        return float("nan")
    return _gammq(df / 2.0, x / 2.0)


@dataclass
class ResidualCheck:
    """Result of :func:`check_residuals`: multinomial residual dispersion."""

    dispersion: float
    pvalue: float
    df: float


def check_residuals(model, docs, *, tol=0.01):
    """Residual-dispersion test for whether K is too small (Taddy 2012), a faithful
    port of R ``stm``'s ``checkResiduals``.

    Under a correctly specified model the multinomial residuals have dispersion
    ``σ² = 1``. A dispersion well above 1 (small p-value) is evidence the latent
    topics cannot absorb the overdispersion — i.e. K is too low. Run it alongside
    :func:`search_k`. `docs` are the tokenized training documents aligned to
    ``model.doc_topic``'s rows.

    Returns a :class:`ResidualCheck` with ``dispersion`` (σ²), ``pvalue`` (χ²
    test of σ²=1 vs σ²>1), and ``df``.

    The dispersion, ``df``, and χ² statistic come from the shared ``topica-core``
    ``inspect::residual_dispersion`` implementation (the same port faSTM and the
    Stata plugin consume), so every host reports one stm-faithful number. Only the
    upper-tail χ² p-value is formed here.
    """
    from ._topica import inspect_residual_dispersion

    phi = np.asarray(model.topic_word, dtype=np.float64)
    theta = np.asarray(model.doc_topic, dtype=np.float64)
    vocab = list(model.vocabulary)
    n = theta.shape[0]
    if len(docs) != n:
        raise ValueError(
            f"docs has {len(docs)} entries but doc_topic has {n} rows; "
            "pass the same documents used to fit the model"
        )
    vindex = {w: i for i, w in enumerate(vocab)}
    docs_ids = [
        [vindex[w] for w in doc if w in vindex] for doc in docs
    ]

    dispersion, df, _n_params, statistic, _nhat = inspect_residual_dispersion(
        phi.tolist(), theta.tolist(), docs_ids, float(tol)
    )
    pvalue = _chisq_sf(statistic, df) if df > 0 else float("nan")
    return ResidualCheck(dispersion=float(dispersion), pvalue=float(pvalue), df=float(df))


# ---------------------------------------------------------------------------
# Topic alignment + stability (exploits determinism)
# ---------------------------------------------------------------------------

def _hungarian(cost):
    """Optimal min-cost assignment (Hungarian / Kuhn-Munkres). Returns a list of
    ``(row, col)`` pairs. Rectangular costs are padded to square."""
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    size = max(n, m)
    big = (cost.max() * size + 1.0) if cost.size else 1.0
    c = np.full((size, size), big)
    c[:n, :m] = cost
    inf = float("inf")
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = -1
            for j in range(1, size + 1):
                if not used[j]:
                    cur = c[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    out = []
    for j in range(1, size + 1):
        if p[j] != 0 and p[j] - 1 < n and j - 1 < m:
            out.append((p[j] - 1, j - 1))
    return sorted(out)


class AlignmentResult(list):
    """Result of cross-model topic alignment.

    Behaves as a list of ``(topic_a, topic_b, distance)`` tuples for backward
    compatibility, but exposes additional attributes for advanced analysis:
    - ``matches``: List of 1-to-1 matched pairs ``(topic_a, topic_b, similarity)``
    - ``splits``: Dictionary mapping ``topic_a -> list of (topic_b, similarity)``
    - ``merges``: Dictionary mapping ``topic_b -> list of (topic_a, similarity)``
    - ``unaligned_a``: List of unmatched topics in Model A
    - ``unaligned_b``: List of unmatched topics in Model B
    - ``similarity_matrix``: Raw pairwise similarity matrix of shape ``(K_A, K_B)``
    """
    def __init__(
        self,
        pairs,
        *,
        matches,
        splits,
        merges,
        unaligned_a,
        unaligned_b,
        similarity_matrix,
    ):
        super().__init__(pairs)
        self.matches = matches
        self.splits = splits
        self.merges = merges
        self.unaligned_a = unaligned_a
        self.unaligned_b = unaligned_b
        self.similarity_matrix = similarity_matrix

    def __repr__(self):
        return (
            f"AlignmentResult(matches={len(self.matches)}, "
            f"splits={len(self.splits)}, merges={len(self.merges)}, "
            f"unaligned_a={len(self.unaligned_a)}, unaligned_b={len(self.unaligned_b)})"
        )


def _rbo(s, t, p, depth):
    """Calculate Rank-Biased Overlap (RBO) between two lists of words."""
    s = s[:depth]
    t = t[:depth]
    h = min(len(s), len(t), depth)
    if h == 0:
        return 0.0
    
    s_set = set()
    t_set = set()
    rbo_sum = 0.0
    
    for d in range(1, h + 1):
        s_set.add(s[d - 1])
        t_set.add(t[d - 1])
        overlap = len(s_set.intersection(t_set))
        rbo_sum += (p ** (d - 1)) * (overlap / d)
        
    overlap_h = len(s_set.intersection(t_set))
    rbo_val = (1.0 - p) * rbo_sum + (p ** h) * (overlap_h / h)
    return rbo_val


def _emd_similarity(p_a, p_b, top_idx_a, top_idx_b, word_embeddings, depth):
    """Calculate Earth Mover's Distance similarity between two topic distributions."""
    try:
        from scipy.optimize import linprog
    except ImportError:
        raise ImportError(
            "SciPy is required for Optimal Transport ('emd') topic alignment. "
            "Install it via `pip install scipy` or `pip install topica[viz]`."
        )

    # 1. Take union of top indices
    union_idx = list(set(top_idx_a[:depth]).union(set(top_idx_b[:depth])))
    M = len(union_idx)
    if M == 0:
        return 0.0

    # 2. Project probability vectors onto union and re-normalize
    mass_a = p_a[union_idx]
    mass_b = p_b[union_idx]
    
    sum_a = mass_a.sum()
    sum_b = mass_b.sum()
    
    mass_a = mass_a / sum_a if sum_a > 1e-12 else np.ones(M) / M
    mass_b = mass_b / sum_b if sum_b > 1e-12 else np.ones(M) / M

    # 3. Calculate distance matrix between words in union
    if word_embeddings is not None:
        try:
            vectors = word_embeddings[union_idx]
        except Exception:
            vectors = None
        
        if vectors is not None:
            vectors = np.asarray(vectors, dtype=np.float64)
            if vectors.ndim == 2:
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                norms = np.clip(norms, 1e-12, None)
                normed_vectors = vectors / norms
                C = 1.0 - normed_vectors @ normed_vectors.T
                C = np.clip(C, 0.0, 2.0)
            else:
                C = np.ones((M, M)) - np.eye(M)
        else:
            C = np.ones((M, M)) - np.eye(M)
    else:
        C = np.ones((M, M)) - np.eye(M)

    # 4. Set up and solve the LP using linprog
    c_lp = C.flatten()
    A_eq = []
    b_eq = []
    
    for u in range(M):
        row = np.zeros((M, M))
        row[u, :] = 1.0
        A_eq.append(row.flatten())
        b_eq.append(mass_a[u])
        
    for v in range(M):
        col = np.zeros((M, M))
        col[:, v] = 1.0
        A_eq.append(col.flatten())
        b_eq.append(mass_b[v])
        
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)
    bounds = [(0.0, None) for _ in range(M * M)]
    
    res = linprog(c=c_lp, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.success:
        emd_dist = res.fun
    else:
        emd_dist = 0.5 * np.sum(np.abs(mass_a - mass_b))

    max_c = C.max()
    if max_c > 1e-12:
        similarity = np.clip(1.0 - emd_dist / max_c, 0.0, 1.0)
    else:
        similarity = 1.0 if emd_dist < 1e-9 else 0.0
    return float(similarity)


def align_topics(
    a,
    b,
    *,
    metric="cosine",
    threshold=0.3,
    depth=50,
    p=0.9,
    word_embeddings=None,
) -> AlignmentResult:
    """Match the topics of two fits one-to-one by minimal total distance
    (Hungarian on the cross-fit topic-word distance matrix). Use it to compare
    runs across seeds, across K, or train vs. resample.

    `a`, `b` are fitted models or K×V topic-word arrays (same vocabulary order, or
    automatically intersected if `.vocabulary` is available).
    `metric` is ``"cosine"``, ``"js"`` (Jensen-Shannon), ``"rbo"`` (Rank-biased overlap),
    or ``"emd"``/``"ot"`` (Earth Mover's Distance).
    Returns an `AlignmentResult` object which behaves as a list of ``(topic_a, topic_b, distance)``
    tuples sorted by ``topic_a``, but exposes additional attributes: ``matches``, ``splits``,
    ``merges``, ``unaligned_a``, ``unaligned_b``, and ``similarity_matrix``.
    """
    A = _as_topic_word(a)
    B = _as_topic_word(b)

    vocab_a = getattr(a, "vocabulary", None)
    vocab_b = getattr(b, "vocabulary", None)

    # 1. Align vocabularies if available
    if vocab_a is not None and vocab_b is not None:
        vocab_a = list(vocab_a)
        vocab_b = list(vocab_b)
        common_vocab = sorted(list(set(vocab_a).intersection(set(vocab_b))))
        if len(common_vocab) == 0:
            raise ValueError("The two fits share no common vocabulary terms.")
        
        idx_map_a = {w: i for i, w in enumerate(vocab_a)}
        idx_map_b = {w: i for i, w in enumerate(vocab_b)}
        
        idx_a = [idx_map_a[w] for w in common_vocab]
        idx_b = [idx_map_b[w] for w in common_vocab]
        
        A_proj = A[:, idx_a]
        B_proj = B[:, idx_b]
        
        A = A_proj / np.clip(A_proj.sum(axis=1, keepdims=True), 1e-12, None)
        B = B_proj / np.clip(B_proj.sum(axis=1, keepdims=True), 1e-12, None)
    else:
        if A.shape[1] != B.shape[1]:
            raise ValueError("the two fits must share a vocabulary (same V)")
        common_vocab = [str(i) for i in range(A.shape[1])]

    # Normalize word_embeddings to a matrix aligned to common_vocab
    embeddings_matrix = None
    if word_embeddings is not None:
        if isinstance(word_embeddings, dict):
            first_val = next(iter(word_embeddings.values()))
            dim = len(first_val) if hasattr(first_val, "__len__") else 1
            embeddings_matrix = np.zeros((len(common_vocab), dim))
            for i, w in enumerate(common_vocab):
                if w in word_embeddings:
                    embeddings_matrix[i] = word_embeddings[w]
        elif isinstance(word_embeddings, np.ndarray) or hasattr(word_embeddings, "__len__"):
            word_embeddings = np.asarray(word_embeddings)
            if word_embeddings.shape[0] == len(vocab_a) if vocab_a else False:
                idx_map_a = {w: i for i, w in enumerate(vocab_a)}
                idx_a = [idx_map_a[w] for w in common_vocab]
                embeddings_matrix = word_embeddings[idx_a]
            elif word_embeddings.shape[0] == len(vocab_b) if vocab_b else False:
                idx_map_b = {w: i for i, w in enumerate(vocab_b)}
                idx_b = [idx_map_b[w] for w in common_vocab]
                embeddings_matrix = word_embeddings[idx_b]
            elif word_embeddings.shape[0] == len(common_vocab):
                embeddings_matrix = word_embeddings
            else:
                raise ValueError(
                    f"word_embeddings shape {word_embeddings.shape} is not compatible with vocabulary size."
                )

    # 2. Get top word indices (descending order of probability)
    top_indices_a = [np.argsort(A[i])[::-1] for i in range(A.shape[0])]
    top_indices_b = [np.argsort(B[j])[::-1] for j in range(B.shape[0])]

    words_a = [[common_vocab[idx] for idx in top_indices_a[i]] for i in range(A.shape[0])]
    words_b = [[common_vocab[idx] for idx in top_indices_b[j]] for j in range(B.shape[0])]

    # 3. Calculate similarity matrix
    if metric == "cosine":
        an = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
        bn = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-12, None)
        similarity_matrix = an @ bn.T
    elif metric == "js":
        dist = np.zeros((A.shape[0], B.shape[0]))
        for i in range(A.shape[0]):
            pi = A[i]
            for j in range(B.shape[0]):
                qj = B[j]
                mm = 0.5 * (pi + qj)
                dist[i, j] = 0.5 * _kl(pi, mm) + 0.5 * _kl(qj, mm)
        similarity_matrix = np.clip(1.0 - dist, 0.0, 1.0)
    elif metric == "rbo":
        similarity_matrix = np.zeros((A.shape[0], B.shape[0]))
        for i in range(A.shape[0]):
            for j in range(B.shape[0]):
                similarity_matrix[i, j] = _rbo(words_a[i], words_b[j], p, depth)
    elif metric in ("emd", "ot"):
        similarity_matrix = np.zeros((A.shape[0], B.shape[0]))
        for i in range(A.shape[0]):
            for j in range(B.shape[0]):
                similarity_matrix[i, j] = _emd_similarity(
                    A[i], B[j], top_indices_a[i], top_indices_b[j],
                    embeddings_matrix, depth
                )
    else:
        raise ValueError("metric must be 'cosine', 'js', 'rbo', or 'emd'/'ot'")

    # 4. Generate legacy pairs (Hungarian matching)
    legacy_pairs = []
    for i, j in _hungarian(1.0 - similarity_matrix):
        if metric == "js":
            legacy_dist = float(dist[i, j])
        else:
            legacy_dist = float(1.0 - similarity_matrix[i, j])
        legacy_pairs.append((i, j, legacy_dist))

    # 5. Threshold-based classification (splits, merges, matches, unaligned)
    matches = []
    splits = {}
    merges = {}
    unaligned_a = []
    unaligned_b = []
    
    adj_a = {i: [] for i in range(A.shape[0])}
    adj_b = {j: [] for j in range(B.shape[0])}
    
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            sim = float(similarity_matrix[i, j])
            if sim >= threshold:
                adj_a[i].append((j, sim))
                adj_b[j].append((i, sim))
                
    for i in adj_a:
        adj_a[i].sort(key=lambda x: x[1], reverse=True)
    for j in adj_b:
        adj_b[j].sort(key=lambda x: x[1], reverse=True)
        
    for i in range(A.shape[0]):
        targets = adj_a[i]
        if len(targets) == 0:
            unaligned_a.append(i)
        elif len(targets) == 1:
            j, sim = targets[0]
            if len(adj_b[j]) == 1:
                matches.append((i, j, sim))
        else:
            splits[i] = targets
            
    for j in range(B.shape[0]):
        sources = adj_b[j]
        if len(sources) == 0:
            unaligned_b.append(j)
        elif len(sources) > 1:
            merges[j] = sources

    return AlignmentResult(
        legacy_pairs,
        matches=matches,
        splits=splits,
        merges=merges,
        unaligned_a=unaligned_a,
        unaligned_b=unaligned_b,
        similarity_matrix=similarity_matrix,
    )


def _kl(p, q):
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    return float(np.sum(p * np.log(p / q)))


def topic_stability(runs, *, topn=10, metric="cosine"):
    """Term-centric stability of topics across multiple fits (Greene, O'Callaghan
    & Cunningham 2014): a "how robust is this K?" score.

    `runs` is a list of fitted models or topic-word arrays over the *same*
    vocabulary (e.g. fits at different seeds, or on bootstrap resamples). Each
    later run's topics are matched to the first run's, and stability is the mean
    Jaccard overlap of their top-`topn` words. Returns a float in ``[0, 1]``;
    higher means more reproducible topics.
    """
    mats = [_as_topic_word(r) for r in runs]
    if len(mats) < 2:
        raise ValueError("need at least two runs to measure stability")
    ref = mats[0]
    k = ref.shape[0]
    ref_top = [set(np.argsort(ref[t])[::-1][:topn]) for t in range(k)]
    scores = []
    for mat in mats[1:]:
        for i, j, _ in align_topics(ref, mat, metric=metric):
            other = set(np.argsort(mat[j])[::-1][:topn])
            union = ref_top[i] | other
            scores.append(len(ref_top[i] & other) / len(union) if union else 0.0)
    return float(np.mean(scores)) if scores else float("nan")


# ---------------------------------------------------------------------------
# Qualitative validation: highlighted close-reading export
# ---------------------------------------------------------------------------

def find_thoughts_html(
    model,
    texts,
    *,
    topics=None,
    n_docs=3,
    n_words=8,
    max_chars=400,
    markdown=False,
):
    """Render each topic's most representative documents for close reading, with
    the topic's top words **highlighted** in the document text.

    Distant reading (top words) is only half of topic validation; the other half
    is reading the actual documents a topic loads on. This builds a self-contained
    HTML snippet (or Markdown) you can ``display`` in a notebook: per topic, its
    top words followed by its `n_docs` highest-θ documents, each truncated to
    ``max_chars`` with the topic's words marked.

    `model` is any fitted model exposing ``topic_word``, ``doc_topic`` and
    ``vocabulary``; `texts` are the original document strings, aligned to the
    rows of ``doc_topic``. Returns a string (HTML unless ``markdown=True``).
    """
    phi = _as_topic_word(model)
    theta = _as_doc_topic(model)
    vocab = list(model.vocabulary)
    if len(texts) != theta.shape[0]:
        raise ValueError("texts must be aligned with the model's documents")
    K = phi.shape[0]
    topics = range(K) if topics is None else topics

    blocks = []
    for t in topics:
        top_ids = np.argsort(phi[t])[::-1][:n_words]
        words = [vocab[i] for i in top_ids]
        docs = np.argsort(theta[:, t])[::-1][:n_docs]
        if markdown:
            blocks.append(_thoughts_md(t, words, docs, theta, texts, max_chars))
        else:
            blocks.append(_thoughts_html(t, words, docs, theta, texts, max_chars))
    if markdown:
        return "\n\n".join(blocks)
    return "<div class=\"tt-thoughts\">\n" + "\n".join(blocks) + "\n</div>"


def _keyword_pattern(words):
    # Match the readable surface form of each top word (phrase tokens use "_").
    surfaces = sorted({w.replace("_", " ") for w in words}, key=len, reverse=True)
    surfaces = [re.escape(s) for s in surfaces if s]
    if not surfaces:
        return None
    return re.compile(r"\b(" + "|".join(surfaces) + r")\b", re.IGNORECASE)


def _truncate(text, max_chars):
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + " …"


def _thoughts_html(t, words, docs, theta, texts, max_chars):
    pat = _keyword_pattern(words)
    head = (f"<h4>Topic {t}</h4>\n<p><em>"
            + ", ".join(_html.escape(w.replace('_', ' ')) for w in words)
            + "</em></p>\n<ul>")
    items = []
    for d in docs:
        body = _html.escape(_truncate(str(texts[d]), max_chars))
        if pat is not None:
            body = pat.sub(lambda m: f"<mark>{m.group(0)}</mark>", body)
        items.append(f"<li><small>doc {int(d)} (θ={theta[d, t]:.2f})</small><br>{body}</li>")
    return head + "\n" + "\n".join(items) + "\n</ul>"


def _thoughts_md(t, words, docs, theta, texts, max_chars):
    pat = _keyword_pattern(words)
    lines = [f"### Topic {t}",
             "*" + ", ".join(w.replace("_", " ") for w in words) + "*", ""]
    for d in docs:
        body = _truncate(str(texts[d]), max_chars)
        if pat is not None:
            body = pat.sub(lambda m: f"**{m.group(0)}**", body)
        lines.append(f"- **doc {int(d)}** (θ={theta[d, t]:.2f}): {body}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model-quality frontier + bootstrap stability
# ---------------------------------------------------------------------------

def quality_frontier(model, *, n=10, texts=None, coherence_type="u_mass", plot=False):
    """Per-topic coherence, exclusivity, and prevalence — the data behind stm's
    classic coherence-vs-exclusivity quality plot.

    Returns a dict of equal-length arrays: ``topic``, ``coherence``,
    ``exclusivity``, ``prevalence`` (mean θ). By default coherence is the fast
    per-topic UMass score; pass ``texts`` and a windowed ``coherence_type`` (e.g.
    ``"c_v"``) for the human-aligned measure. Feed the dict straight to pandas /
    matplotlib; with ``plot=True`` (and matplotlib installed) a labeled scatter
    ``Figure`` is returned alongside the dict as ``(data, fig)``.
    """
    from .coherence import coherence as _coherence, exclusivity as _exclusivity

    phi = _as_topic_word(model)
    theta = _as_doc_topic(model)
    K = phi.shape[0]
    if texts is not None and coherence_type != "u_mass":
        coh = np.asarray(_coherence(model, texts, coherence_type=coherence_type, topn=n))
    else:
        # The windowed coherence types need a reference corpus; without `texts`
        # the only score available is UMass. Warn rather than silently returning
        # UMass under the requested name — the scales differ (UMass ~ (-inf, 0],
        # c_v ~ [0, 1]), so a mislabeled axis invites wrong comparisons.
        if texts is None and coherence_type != "u_mass":
            warnings.warn(
                f"quality_frontier: coherence_type={coherence_type!r} needs texts "
                "(a reference corpus); without them coherence is UMass, which is on "
                "a different scale. Pass texts= or set coherence_type='u_mass'.",
                stacklevel=2,
            )
        coh = np.asarray(model.coherence(n))
    data = {
        "topic": np.arange(K),
        "coherence": coh,
        "exclusivity": _exclusivity(phi, n=n),
        "prevalence": theta.mean(axis=0),
    }
    if not plot:
        return data
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("plot=True requires matplotlib") from exc
    fig, ax = plt.subplots()
    ax.scatter(data["coherence"], data["exclusivity"],
               s=300 * data["prevalence"] + 20)
    for t in range(K):
        ax.annotate(str(t), (data["coherence"][t], data["exclusivity"][t]))
    ax.set_xlabel("Semantic coherence")
    ax.set_ylabel("Exclusivity")
    ax.set_title("Topic quality (size ∝ prevalence)")
    return data, fig


def bootstrap_stability(
    docs,
    *,
    k=None,
    n_boot=20,
    topn=10,
    seed=0,
    model_factory=None,
    reference=None,
    **fit_kwargs,
):
    """Flag fragile topics by refitting on bootstrap resamples of the corpus.

    The standard defense against "topic modeling is a fishing expedition": fit a
    reference model on the full corpus, then refit on `n_boot` resamples of the
    documents (drawn with replacement). Each bootstrap model's topics are matched
    to the reference's by top-word overlap, and a reference topic's **stability**
    is the mean Jaccard overlap of its top-`topn` words with its matched bootstrap
    topic. Topics that dissolve under resampling score low.

    Matching is on the top words as *strings*, so it is correct even though each
    resample is fit as a fresh corpus with its own vocabulary indexing.

    Parameters
    ----------
    docs : the corpus (``list[list[str]]`` or a ``Corpus``).
    k : number of topics. Required unless ``reference`` is given (then taken from
        it).
    n_boot : number of bootstrap resamples.
    model_factory : ``callable(seed) -> unfitted model``. Defaults to
        ``LDA(num_topics=k, seed=seed)``. Use it to bootstrap any model.
    reference : an already-fitted model to measure the stability *of*. When given,
        the resample topics are matched back to it (rather than to a fresh
        full-corpus fit), so the per-topic stability lines up with that model's
        topic indices. ``model_factory`` should rebuild the same model type.
    fit_kwargs : forwarded to each model's ``fit`` (e.g. ``iters=500``).

    Returns
    -------
    dict with ``topic`` (indices), ``stability`` (per-topic mean Jaccard in
    ``[0, 1]``), ``mean`` (overall), and ``reference`` (the reference model).
    """
    from . import LDA  # local import to avoid a cycle at module load

    # Accept a Corpus, matching the docstring and the sibling functions
    # (perplexity, prepare_pyldavis): pull its token lists before resampling.
    if hasattr(docs, "documents"):
        docs = docs.documents()
    docs = [list(d) for d in docs]
    D = len(docs)
    if D < 2:
        raise ValueError("need at least two documents to resample")
    if k is None:
        if reference is None:
            raise ValueError("pass k (number of topics) or a fitted reference model")
        k = int(reference.num_topics)
    factory = model_factory or (lambda s: LDA(num_topics=k, seed=s))

    def top_word_sets(model):
        phi = _as_topic_word(model)
        vocab = list(model.vocabulary)
        return [set(vocab[i] for i in np.argsort(phi[t])[::-1][:topn])
                for t in range(phi.shape[0])]

    if reference is not None:
        ref = reference
    else:
        ref = factory(seed)
        ref.fit(docs, **fit_kwargs)
    ref_sets = top_word_sets(ref)
    K = len(ref_sets)

    rng = np.random.RandomState(seed)
    per_topic = [[] for _ in range(K)]
    for b in range(n_boot):
        pick = rng.randint(0, D, size=D)
        sample = [docs[i] for i in pick]
        m = factory(seed + b + 1)
        m.fit(sample, **fit_kwargs)
        boot_sets = top_word_sets(m)
        # Match bootstrap topics to reference topics by top-word Jaccard, then
        # record each reference topic's overlap with its match.
        cost = np.empty((K, len(boot_sets)))
        for i, rs in enumerate(ref_sets):
            for j, bs in enumerate(boot_sets):
                union = rs | bs
                cost[i, j] = 1.0 - (len(rs & bs) / len(union) if union else 0.0)
        for i, j in _hungarian(cost):
            union = ref_sets[i] | boot_sets[j]
            per_topic[i].append(len(ref_sets[i] & boot_sets[j]) / len(union) if union else 0.0)

    stability = np.array([float(np.mean(s)) if s else float("nan") for s in per_topic])
    return {
        "topic": np.arange(K),
        "stability": stability,
        "mean": float(np.nanmean(stability)),
        "reference": ref,
    }


# ---------------------------------------------------------------------------
# Post-hoc topic structure: a multi-resolution dendrogram over a fitted model's
# topics (no refit), for "are these K topics really m super-themes?".
# ---------------------------------------------------------------------------

def _topic_js_matrix(beta):
    """Pairwise Jensen-Shannon divergence (base-2, in [0, 1]) over topic-word
    rows; a symmetric ``K x K`` distance with a zero diagonal."""
    P = beta / beta.sum(1, keepdims=True)
    logP = np.where(P > 0, np.log2(np.where(P > 0, P, 1.0)), 0.0)
    h_p = -(P * logP).sum(1)
    k = P.shape[0]
    D = np.zeros((k, k))
    for i in range(k):
        M = 0.5 * (P[i] + P)
        logM = np.where(M > 0, np.log2(np.where(M > 0, M, 1.0)), 0.0)
        h_m = -(M * logM).sum(1)
        D[i] = np.clip(h_m - 0.5 * (h_p[i] + h_p), 0.0, 1.0)
    np.fill_diagonal(D, 0.0)
    return 0.5 * (D + D.T)


def _topic_hellinger_matrix(beta):
    P = beta / beta.sum(1, keepdims=True)
    bc = np.clip(np.sqrt(P) @ np.sqrt(P).T, 0.0, 1.0)  # Bhattacharyya coefficient
    D = np.sqrt(np.clip(1.0 - bc, 0.0, 1.0))
    np.fill_diagonal(D, 0.0)
    return 0.5 * (D + D.T)


def _doctopic_corr_matrix(doc_topic):
    D = 1.0 - np.corrcoef(np.asarray(doc_topic, dtype=np.float64).T)
    np.fill_diagonal(D, 0.0)
    return 0.5 * (np.clip(D, 0.0, 2.0) + np.clip(D, 0.0, 2.0).T)


@dataclass
class TopicDendrogram:
    """Result of :func:`topic_dendrogram`: a hierarchical merge tree over topics.

    ``linkage`` is a SciPy linkage matrix (``(K-1) x 4``) you can pass straight to
    ``scipy.cluster.hierarchy.dendrogram``; ``distances`` is the ``K x K`` topic
    distance it was built from; ``topics`` are the per-topic top words used as leaf
    labels. Use :meth:`cut` to flatten into ``m`` super-topics and
    :meth:`merge_candidates` to list near-duplicate pairs.
    """

    linkage: np.ndarray
    distances: np.ndarray
    topics: list
    metric: str

    def cut(self, m: int) -> np.ndarray:
        """Flatten the tree into ``m`` super-topics, returning a 0-based group
        label per topic."""
        from scipy.cluster.hierarchy import fcluster
        lab = fcluster(self.linkage, t=m, criterion="maxclust")
        remap = {v: i for i, v in enumerate(sorted(set(lab)))}
        return np.array([remap[v] for v in lab])

    def merge_candidates(self, *, rel: float = 0.6, threshold=None) -> list:
        """Topic pairs close enough to be near-duplicates, as ``(i, j, distance)``
        sorted by distance.

        By default a pair is flagged when its distance is below ``rel`` times the
        median off-diagonal distance — a *relative* cutoff, because the absolute
        distance scale is corpus-dependent (shared common-word mass inflates it).
        Pass an absolute ``threshold`` to override.
        """
        D = self.distances
        k = D.shape[0]
        off = D[~np.eye(k, dtype=bool)]
        cut = float(threshold) if threshold is not None else rel * float(np.median(off))
        out = [(i, j, float(D[i, j]))
               for i in range(k) for j in range(i + 1, k) if D[i, j] < cut]
        return sorted(out, key=lambda t: t[2])

    def groups(self, m: int, *, n: int = 10) -> dict:
        """The ``m``-way cut as ``{group: (member_topics, merged_top_words)}``,
        where the words are the top ``n`` of the members' mean topic-word row."""
        labels = self.cut(m)
        beta = self._beta
        out = {}
        for g in sorted(set(labels)):
            members = [int(t) for t in np.where(labels == g)[0]]
            agg = beta[members].mean(0)
            top = np.argsort(agg)[::-1][:n]
            out[int(g)] = (members, [self._vocab[i] for i in top])
        return out


def topic_dendrogram(model, *, metric="js", method="average", n_topwords=20):
    """Agglomeratively merge a fitted model's topics into a multi-resolution tree.

    A post-hoc, no-refit answer to "are these ``K`` topics really a handful of
    super-themes, and are any of them near-duplicates?". It builds a ``K x K``
    topic distance and runs hierarchical clustering, returning a
    :class:`TopicDendrogram` you can :meth:`~TopicDendrogram.cut` at any
    resolution or query for :meth:`~TopicDendrogram.merge_candidates`. This is the
    flat-model counterpart to :class:`~topica.HLDA` (which fits a topic *tree*
    directly) and to :func:`topica.ensemble` (which merges across *runs*).

    Works on any fitted model exposing ``topic_word`` and ``vocabulary``.

    Parameters
    ----------
    model : a fitted topica model.
    metric : {"js", "hellinger", "cosine", "doctopic"}, default "js"
        Topic distance. ``js`` (Jensen-Shannon) and ``hellinger`` compare the
        full topic-word distributions; ``cosine`` compares top-``n_topwords``
        indicator sets; ``doctopic`` uses ``1 - correlation`` of the
        ``doc_topic`` columns (how often topics co-occur in documents).
    method : str, default "average"
        SciPy linkage method ("average", "ward", "complete", ...).
    n_topwords : int, default 20
        Words per topic for the ``cosine`` metric and for leaf labels.

    Returns
    -------
    :class:`TopicDendrogram`.

    Notes
    -----
    Requires SciPy (``pip install 'topica[viz]'`` or ``scipy``).
    """
    try:
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "topic_dendrogram needs SciPy; install it with "
            "`pip install 'topica[viz]'` or `pip install scipy`."
        ) from exc

    beta = np.asarray(model.topic_word, dtype=np.float64)
    vocab = list(model.vocabulary)
    if metric == "js":
        D = _topic_js_matrix(beta)
    elif metric == "hellinger":
        D = _topic_hellinger_matrix(beta)
    elif metric == "doctopic":
        D = _doctopic_corr_matrix(model.doc_topic)
    elif metric == "cosine":
        k, v = beta.shape
        ind = np.zeros((k, v))
        for t in range(k):
            ind[t, np.argsort(beta[t])[::-1][:n_topwords]] = 1.0
        norm = np.linalg.norm(ind, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        U = ind / norm
        D = 1.0 - np.clip(U @ U.T, 0.0, 1.0)
        np.fill_diagonal(D, 0.0)
        D = 0.5 * (D + D.T)
    else:
        raise ValueError(
            f"metric must be 'js', 'hellinger', 'cosine', or 'doctopic'; got {metric!r}"
        )

    Z = linkage(squareform(D, checks=False), method=method)
    topics = [[vocab[i] for i in np.argsort(beta[t])[::-1][:n_topwords]]
              for t in range(beta.shape[0])]
    result = TopicDendrogram(linkage=Z, distances=D, topics=topics, metric=metric)
    result._beta = beta      # for groups(); not part of the public dataclass fields
    result._vocab = vocab
    return result


# ---------------------------------------------------------------------------
# Per-topic quality table + automatic junk/boilerplate flag.
# ---------------------------------------------------------------------------

def flag_topics(model, texts, *, n=10, coherence_type="c_v"):
    """Score every topic on cheap quality features and flag likely junk.

    A quick "are these topics real, or did I forget to clean my corpus?" check.
    For each topic it gathers :func:`topica.coherence`,
    :func:`topica.exclusivity`, the normalized topic-word entropy (1.0 = a
    perfectly flat, uninformative topic), corpus prevalence, and the fraction of
    its top words that are stopwords, then flags a topic as junk when any of:

    - **stopword-soup** — at least 40% of the top words are stopwords;
    - **dead/tiny** — prevalence below half its uniform share (``0.5 / K``);
    - **incoherent+flat** — coherence in the run's bottom quartile *and* topic-word
      entropy in its top quartile.

    The thresholds are relative to the run, so the flag reads as "junk *for this
    model*". ``texts`` are the tokenized documents (used only for coherence; they
    need not align to ``doc_topic``).

    Returns a list of per-topic dicts (in topic order) with ``topic``,
    ``coherence``, ``exclusivity``, ``beta_entropy``, ``prevalence``,
    ``stopword_frac``, ``junk`` (bool), ``reasons`` (list of str), and
    ``top_words``.
    """
    from .stopwords import ENGLISH_STOPWORDS

    tw = np.asarray(model.topic_word, dtype=np.float64)
    dt = np.asarray(model.doc_topic, dtype=np.float64)
    vocab = list(model.vocabulary)
    k, v = tw.shape
    top_idx = np.argsort(tw, axis=1)[:, ::-1][:, :n]
    topics = [[vocab[i] for i in top_idx[t]] for t in range(k)]

    coh = np.asarray(_coherence(topics, texts, coherence_type=coherence_type, topn=n),
                     dtype=np.float64)
    excl = np.asarray(_exclusivity(model, n=n), dtype=np.float64)
    beta_ent = np.empty(k)
    for t in range(k):
        p = tw[t][tw[t] > 0]
        beta_ent[t] = (-(p * np.log(p)).sum()) / np.log(v) if v > 1 else 0.0
    prev = dt.mean(axis=0)
    prev = prev / prev.sum()
    stop_frac = np.array([np.mean([w in ENGLISH_STOPWORDS for w in topics[t]])
                          for t in range(k)])

    coh_lo = np.quantile(coh, 0.25)
    ent_hi = np.quantile(beta_ent, 0.75)
    dead = 0.5 / k

    rows = []
    for t in range(k):
        reasons = []
        if stop_frac[t] >= 0.4:
            reasons.append("stopword-soup")
        if prev[t] < dead:
            reasons.append("dead/tiny")
        if coh[t] <= coh_lo and beta_ent[t] >= ent_hi:
            reasons.append("incoherent+flat")
        rows.append({
            "topic": t,
            "coherence": float(coh[t]),
            "exclusivity": float(excl[t]),
            "beta_entropy": float(beta_ent[t]),
            "prevalence": float(prev[t]),
            "stopword_frac": float(stop_frac[t]),
            "junk": bool(reasons),
            "reasons": reasons,
            "top_words": topics[t],
        })
    return rows


# ---------------------------------------------------------------------------
# Per-document novelty / coverage residual.
# ---------------------------------------------------------------------------

def document_residuals(model, docs, *, floor=1e-12):
    """How poorly the fitted model explains each document, for outlier hunting.

    Reconstructs each document's expected word distribution as
    ``theta_d @ beta`` and compares it to the document's actual word counts. A
    high residual marks a document the current topics cannot account for: an
    off-topic intruder, an anomaly, or a sign the model is missing a theme. This
    is the per-document complement to :func:`check_residuals`, which collapses the
    whole corpus into one "is K too small?" dispersion statistic.

    ``docs`` are the tokenized documents aligned row-for-row to
    ``model.doc_topic`` (the corpus the model was fit on). To score *new*
    documents, get their ``theta`` with ``model.transform`` first.

    Returns a list of per-document dicts sorted by descending ``novelty`` (most
    anomalous first). Each has ``doc`` (row index), ``novelty`` (the headline
    score: OOV-aware per-word cross-entropy), ``cross_entropy`` (the length-robust
    in-vocabulary-only per-word log-loss; ``nan`` if the document has no in-vocab
    tokens), ``kl`` (``KL(actual || recon)``; length-confounded, use with care),
    ``cosine_dist`` (``1 - cosine``), ``oov`` (out-of-vocabulary token fraction),
    ``n_tokens`` and ``n_invocab``.

    A pure cross-entropy residual can only see in-vocabulary tokens, so a document
    written entirely in unknown words would otherwise look perfectly explained;
    ``novelty`` folds the OOV mass back in, which is what makes off-topic-vocabulary
    intruders rank at the top.
    """
    theta = np.asarray(model.doc_topic, dtype=np.float64)
    phi = np.asarray(model.topic_word, dtype=np.float64)
    vocab = list(model.vocabulary)
    n = theta.shape[0]
    if len(docs) != n:
        raise ValueError(
            f"docs has {len(docs)} entries but doc_topic has {n} rows; pass the "
            "same documents used to fit the model (use model.transform for new docs)."
        )
    vindex = {w: i for i, w in enumerate(vocab)}
    floor_ce = -np.log(floor)

    recon = np.clip(theta @ phi, floor, None)
    recon /= recon.sum(1, keepdims=True)

    rows = []
    for d in range(n):
        x = np.zeros(phi.shape[1])
        total = 0
        for w in docs[d]:
            total += 1
            i = vindex.get(w)
            if i is not None:
                x[i] += 1.0
        m = int(x.sum())
        oov = (total - m) / total if total else 0.0
        if m == 0:
            ce = float("nan")
            kl = float("nan")
            cos = float("nan")
            novelty = floor_ce
        else:
            actual = x / m
            r = recon[d]
            ce = float(-(actual * np.log(r)).sum())
            nz = actual > 0
            kl = float((actual[nz] * np.log(actual[nz] / r[nz])).sum())
            cos = float(1.0 - (actual @ r) / (np.linalg.norm(actual) * np.linalg.norm(r)))
            novelty = (1.0 - oov) * ce + oov * floor_ce
        rows.append({
            "doc": d,
            "novelty": float(novelty),
            "cross_entropy": ce,
            "kl": kl,
            "cosine_dist": cos,
            "oov": float(oov),
            "n_tokens": total,
            "n_invocab": m,
        })
    rows.sort(key=lambda r: r["novelty"], reverse=True)
    return rows


__all__ = [
    "diagnostics",
    "perplexity",
    "make_heldout",
    "eval_heldout",
    "Heldout",
    "HeldoutResult",
    "frex",
    "mmr",
    "label_topics",
    "topics_for_term",
    "topic_correlation",
    "TopicCorrelation",
    "find_thoughts",
    "find_thoughts_html",
    "topic_table",
    "quality_frontier",
    "bootstrap_stability",
    "search_k",
    "select_model",
    "SelectModelResult",
    "plot_models",
    "relevance",
    "prepare_pyldavis",
    "PyLDAvisInputs",
    "check_residuals",
    "ResidualCheck",
    "document_residuals",
    "flag_topics",
    "topic_dendrogram",
    "TopicDendrogram",
    "align_topics",
    "topic_stability",
]
