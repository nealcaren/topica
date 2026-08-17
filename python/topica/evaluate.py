"""Credibility diagnostics: coherence tables, held-out checks, residuals, stability, alignment.

Split out of the former monolithic ``topica.validation`` (issue #757). The names
here are also re-exported from :mod:`topica.validation` (a compatibility shim) and
from the workflow namespace :mod:`topica.evaluate`.
"""

from __future__ import annotations

import html as _html
import inspect
import re
import warnings
from dataclasses import dataclass, field

import numpy as np

from .coherence import (
    _as_topic_word, _as_doc_topic, _vocabulary_of, _rbo, _ref_corpus,
    coherence as _coherence, exclusivity as _exclusivity,
)

from .inspect import frex

# Coherence / diversity / exclusivity / intrusion diagnostics belong to the
# evaluate stage; re-export the public surface from topica.coherence (a leaf
# module, so no import cycle) so `topica.evaluate.coherence` etc. resolve (#757).
from .coherence import (  # noqa: F401
    coherence, coherence_ci, CoherenceCI, semantic_coherence, embedding_coherence,
    topic_diversity, topic_semantic_diversity, inverted_rbo, exclusivity,
    word_intrusion, document_intrusion,
)

__all__ = [
    'AlignmentResult',
    'CoherenceCI',
    'Heldout',
    'HeldoutResult',
    'ResidualCheck',
    'TopicDendrogram',
    'align_topics',
    'bootstrap_stability',
    'check_residuals',
    'coherence',
    'coherence_ci',
    'diagnostics',
    'document_intrusion',
    'document_residuals',
    'embedding_coherence',
    'eval_heldout',
    'exclusivity',
    'flag_topics',
    'inverted_rbo',
    'make_heldout',
    'perplexity',
    'semantic_coherence',
    'topic_diversity',
    'topic_semantic_diversity',
    'topic_dendrogram',
    'word_intrusion',
    'topic_stability',
]

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
                # top_words returns bare word strings by default (#752); take them
                # as-is so the model's own ordering/weighting is preserved.
                return list(top_method(n, topic=t))
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

    @property
    def corpus(self):
        """Directive hint: a common mis-reach is ``heldout.corpus``. The reduced
        training corpus lives on ``.documents``, and the pair is scored with
        :func:`eval_heldout` (not :func:`perplexity`)."""
        raise AttributeError(
            "Heldout has no `.corpus`; the reduced training corpus is "
            "`heldout.documents`. Fit on it and score the withheld words with "
            "eval_heldout: `model.fit(heldout.documents); "
            "eval_heldout(model, heldout)`."
        )



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
    # A Heldout is not a corpus (#767). make_heldout withholds words in place, and
    # that word-holdout is scored by eval_heldout, not by perplexity's own
    # document-completion split. Passing the object would silently score the
    # reduced training corpus under a different scheme than the user intends, so we
    # redirect — mirroring the Heldout.corpus directive guard shipped in #765.
    if isinstance(held_out, Heldout):
        raise TypeError(
            "perplexity() does not take a Heldout. make_heldout() withholds words "
            "in place, and that word-holdout is scored with eval_heldout:\n"
            "    model.fit(ho.documents); topica.eval_heldout(model, ho)\n"
            "If you instead want document-completion perplexity on the reduced "
            "corpus, pass the documents explicitly: perplexity(model, ho.documents)."
        )
    # Corpus.documents is a method; a raw token list has none. Accept both a Corpus
    # and raw token lists rather than assuming a callable (#761).
    docs_attr = getattr(held_out, "documents", None)
    if callable(docs_attr):
        held_out = docs_attr()
    elif docs_attr is not None:
        held_out = docs_attr
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
    - ``matches``: List of 1-to-1 matched pairs ``(topic_a, topic_b, similarity)`` — the
      Hungarian assignment restricted to pairs that clear ``threshold``.
    - ``splits``: Dictionary mapping ``topic_a -> list of (topic_b, similarity)``
    - ``merges``: Dictionary mapping ``topic_b -> list of (topic_a, similarity)``
    - ``unaligned_a``: List of unmatched topics in Model A
    - ``unaligned_b``: List of unmatched topics in Model B
    - ``similarity_matrix``: Raw pairwise similarity matrix of shape ``(K_A, K_B)``

    ``splits``/``merges`` are a background-relative overlay on the match/unaligned
    partition (issue #642), so a topic that has an extra close partner appears in both
    ``matches`` (or ``unaligned``) and ``splits``/``merges``. Comparing a fit with
    itself yields K matches and no splits/merges for any model whose topics are
    distinct — including correlated-topic families (STM/CTM) whose off-diagonal
    similarities are high. (Two *identical* topics are genuinely interchangeable, so a
    self-alignment of a model that contains exact-duplicate topics reports them as a
    split/merge — an honest signal that the topics cannot be told apart.)
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



# Split/merge overlay calibration (issue #642). ``_SPLIT_GAP`` is the fraction of the
# gap between a topic's own best match and the model's cross-topic background that a
# second partner must close to count as a genuine split/merge; ``_SPLIT_BG_QUANTILE``
# is the quantile of off-target similarities used as that background. Both ride the
# model's own similarity scale, so a fixed pair works across metrics (cosine/js/rbo/
# emd/jaccard) — see the module tests' per-metric self-alignment invariant.
_SPLIT_GAP = 0.25

_SPLIT_BG_QUANTILE = 0.9



def _split_background(M, argmax, split_gap, bg_quantile):
    """Robust background level of *off-target* similarities for one direction.

    ``M`` is ``(lines × candidates)`` and ``argmax`` each line's best candidate. Pools
    every non-argmax similarity (the model's own cross-topic floor, matched signal
    removed) and takes its ``bg_quantile`` quantile, then recomputes once after dropping
    provisional outliers so a genuine co-partner cannot inflate the very floor that would
    then reject it (this matters at small K, where the pool is tiny)."""
    n_lines, n_cand = M.shape
    if n_cand < 2:
        return 0.0
    rows = np.arange(n_lines)
    off = np.ones(M.shape, dtype=bool)
    off[rows, argmax] = False
    pool = M[off]
    if pool.size == 0:
        return 0.0
    q = float(np.quantile(pool, bg_quantile))
    best = M[rows, argmax]
    cut = best - split_gap * (best - q)
    kept = off & (M < cut[:, None])
    pool2 = M[kept]
    if pool2.size >= max(4, n_cand):
        q = float(np.quantile(pool2, bg_quantile))
    return q



def _co_partners(M, threshold, split_gap, bg_quantile):
    """One-to-many partners per line, calibrated to the model's own similarity scale.

    Returns ``{line: [(candidate, sim), ...]}`` (best first) for every line that keeps at
    least one *extra* candidate beyond its own best match. A candidate ``k`` qualifies
    when ``M[i, k] >= best_i - split_gap * (best_i - q)``, where ``q`` is the off-target
    background quantile: the bar is a gap measured against *this model's* cross-topic
    floor, never a fixed absolute threshold, so correlated-topic families do not
    manufacture partners. A line whose own best match is a diffuse near-background value
    (``best_i <= q``) gets a bar at or above ``best_i`` and so never splits; a line whose
    best match is itself below ``threshold`` (no real primary) is skipped entirely.

    A line need not have taken a one-to-one match to be a split/merge key: a genuine
    merge target (one topic that absorbs two, whose two sources each matched a cleaner
    counterpart elsewhere) is itself *unaligned*, so this overlay is intentionally not
    gated to matched lines — it can name a topic that also appears in
    ``unaligned_a``/``unaligned_b`` (a genuine many-to-one that is also "appeared").

    ``split_gap`` deliberately favors precision: a second partner must close a solid
    fraction of the gap from the background floor up to the best match, so a merely
    correlated neighbour is usually not called a split. This is a heuristic — a well-
    separated but weaker second partner can be missed, and on a heavily correlated draw
    an unrelated topic can occasionally be named; use the raw ``similarity_matrix`` when
    you need the exact one-to-many picture."""
    n_lines, n_cand = M.shape
    result: dict[int, list] = {}
    if n_cand < 2:
        return result
    rows = np.arange(n_lines)
    argmax = np.argmax(M, axis=1)
    best = M[rows, argmax]
    q = _split_background(M, argmax, split_gap, bg_quantile)
    cut = best - split_gap * (best - q)
    for i in range(n_lines):
        if best[i] < threshold:
            continue
        ci = cut[i]
        extra = [
            (int(k), float(M[i, k]))
            for k in range(n_cand)
            if k != argmax[i] and M[i, k] >= ci
        ]
        if extra:
            targets = [(int(argmax[i]), float(best[i]))] + extra
            targets.sort(key=lambda t: t[1], reverse=True)
            result[int(i)] = targets
    return result



def _classify_alignment(
    similarity_matrix,
    hungarian_pairs,
    threshold,
    *,
    split_gap=_SPLIT_GAP,
    bg_quantile=_SPLIT_BG_QUANTILE,
):
    """Hungarian-anchored match / split / merge classification (issue #642).

    ``matches`` are the Hungarian 1-to-1 pairs that clear ``threshold`` — the correct,
    correlation-insensitive headline: a self-alignment recovers the diagonal at
    similarity 1.0, so ``align_topics(tw, tw)`` returns K matches for *any* model,
    including correlated-topic families (STM/CTM) whose off-diagonal cosines are high.
    ``splits``/``merges`` are a background-relative *overlay* (see :func:`_co_partners`)
    that stays empty on such a self-alignment instead of the old fixed-``threshold``
    adjacency, which reported near-total instability for identical inputs.

    Returns ``(matches, splits, merges, unaligned_a, unaligned_b)`` in the historical
    shapes — ``matches`` a list of ``(i, j, sim)``, ``splits`` ``{i: [(j, sim), ...]}``,
    ``merges`` ``{j: [(i, sim), ...]}``. ``matches`` and ``splits``/``merges`` may
    overlap: a cleanly matched topic that *also* has an extra partner appears in both,
    the match giving its 1-to-1 counterpart and the split/merge naming the extra."""
    S = np.asarray(similarity_matrix, dtype=np.float64)
    na, nb = S.shape

    matches = []
    matched_a: set[int] = set()
    matched_b: set[int] = set()
    for i, j in hungarian_pairs:
        s = float(S[i, j])
        if s >= threshold:
            matches.append((int(i), int(j), s))
            matched_a.add(int(i))
            matched_b.add(int(j))
    unaligned_a = [i for i in range(na) if i not in matched_a]
    unaligned_b = [j for j in range(nb) if j not in matched_b]

    splits = _co_partners(S, threshold, split_gap, bg_quantile)
    merges = _co_partners(S.T, threshold, split_gap, bg_quantile)

    return matches, splits, merges, unaligned_a, unaligned_b



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

    ``matches`` is the Hungarian assignment restricted to pairs above ``threshold``;
    ``splits``/``merges`` are a background-relative overlay calibrated to the fit's own
    cross-topic similarity, so ``align_topics(tw, tw)`` returns K matches and no
    splits/merges for any model with distinct topics — including correlated-topic
    families (STM/CTM) whose off-diagonal cosines are high (issue #642); exact-duplicate
    topics are the honest exception (they report as a split/merge, being interchangeable).
    ``threshold`` sets the one-to-one match cut; the split/merge overlay self-calibrates
    and does not depend on it.
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

    # 5. Hungarian-anchored classification (issue #642). Matches are the Hungarian
    #    1-to-1 pairs above threshold; splits/merges are a background-relative overlay
    #    that stays empty on a correlated-topic self-alignment. The old fixed-threshold
    #    adjacency mislabelled correlated models (STM/CTM) as near-total splits/merges.
    hungarian_pairs = [(i, j) for (i, j, _d) in legacy_pairs]
    matches, splits, merges, unaligned_a, unaligned_b = _classify_alignment(
        similarity_matrix, hungarian_pairs, threshold
    )

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



def topic_stability(runs, *, num_topics=None, k=None, seeds=None, model_factory=None,
                    topn=10, metric="cosine", fit_kwargs=None, **extra_fit_kwargs):
    """Term-centric stability of topics across multiple fits (Greene, O'Callaghan
    & Cunningham 2014): a "how robust is this K?" score.

    Two call shapes:

    - **From fitted runs** (the base form): ``runs`` is a list of fitted models
      or topic-word arrays over the *same* vocabulary (e.g. fits at different
      seeds, or on bootstrap resamples). You control exactly what is compared.
    - **From a corpus** (the convenience overload, matching
      :func:`bootstrap_stability`'s ``docs`` + ``num_topics`` convention): pass
      the corpus as ``runs`` together with ``num_topics=`` (``k=`` is accepted as
      an alias, matching :func:`bootstrap_stability`) and/or ``seeds=``, and it
      fits one model per seed for you (``LDA(num_topics=k, seed=s)`` by default, or
      your ``model_factory``) before scoring. ``seeds`` defaults to ``range(5)``;
      extra keywords / ``fit_kwargs=`` forward to each ``fit``.

    Either way, each later run's topics are matched to the first run's, and
    stability is the mean Jaccard overlap of their top-`topn` words. Returns a
    float in ``[0, 1]``; higher means more reproducible topics.

    If every run is bit-identical to the first, a stability of 1.0 is
    meaningless — the runs never varied. This most often bites when the runs are
    the *same* deterministic fit repeated: models with a deterministic
    initialization (e.g. ``NMF``/``LSA`` with the default ``init="nndsvd"``)
    ignore ``seed``, so ``[NMF(seed=s).fit(docs) for s in range(5)]`` is five
    copies of one fit. A ``UserWarning`` is emitted in that case; refit with
    ``init="random"`` (which does respond to ``seed``) or measure stability on
    bootstrap resamples of the documents instead.
    """
    # k= is bootstrap_stability's name for the topic count; accept it here too so
    # the two robustness siblings share one convention (issue #755 follow-up).
    if k is not None:
        if num_topics is not None and int(k) != int(num_topics):
            raise ValueError(
                f"pass either k= or num_topics=, not both with different values "
                f"(k={k}, num_topics={num_topics}); they are aliases for the topic count")
        num_topics = k
    corpus_mode = (num_topics is not None or seeds is not None
                   or model_factory is not None)
    fit_kwargs = {**(fit_kwargs or {}), **extra_fit_kwargs}
    if corpus_mode:
        from . import LDA  # local import to avoid a cycle at module load

        # Guard the footgun of passing fitted runs *and* num_topics=/seeds= (which
        # trips corpus mode and then tries to .fit() the model list). A fitted
        # model exposes topic_word/num_topics; a corpus is a Corpus or token lists.
        if not hasattr(runs, "documents"):
            first = next(iter(runs), None)
            if first is not None and (hasattr(first, "topic_word")
                                      or hasattr(first, "num_topics")):
                raise TypeError(
                    "topic_stability got a list of fitted models together with "
                    "num_topics=/seeds=/model_factory=, which only apply to the "
                    "corpus overload. For already-fitted runs, drop those keywords: "
                    "topic_stability([m1, m2, ...]). To fit from a corpus, pass the "
                    "corpus as the first argument instead of models.")
        corpus = runs.documents() if hasattr(runs, "documents") else runs
        if model_factory is None and num_topics is None:
            raise ValueError(
                "topic_stability(corpus, ...) needs num_topics= (the topic count) "
                "or a model_factory=")
        factory = model_factory or (lambda s: LDA(num_topics=num_topics, seed=s))
        seed_list = list(range(5)) if seeds is None else list(seeds)
        if len(seed_list) < 2:
            raise ValueError("need at least two seeds to measure stability")
        runs = [factory(s).fit(corpus, **fit_kwargs) for s in seed_list]
    elif fit_kwargs:
        raise TypeError(
            "topic_stability got fit keyword(s) "
            f"{sorted(fit_kwargs)} but no num_topics=/seeds=/model_factory=, so it "
            "is in from-fitted-runs mode where the first argument must be a list of "
            "already-fitted models, not a corpus. Either pass fitted runs "
            "(topic_stability([m1, m2, ...])) or use the corpus overload "
            "(topic_stability(docs, num_topics=K, seeds=[...])).")
    else:
        corpus_hint = (
            "To measure stability from a corpus, use the overload "
            "topic_stability(docs, num_topics=K, seeds=[...]), or fit the models "
            "yourself and pass them: "
            "topic_stability([LDA(K, seed=s).fit(docs) for s in range(5)]).")
        # A Corpus passed without num_topics=/seeds= isn't corpus-mode; catch it
        # before the peek (a Corpus is not iterable) and steer to the overload.
        if hasattr(runs, "documents"):
            raise TypeError(
                "topic_stability got a Corpus but no num_topics=/seeds=, so it is "
                "in from-fitted-runs mode (the first argument must be a list of "
                f"already-fitted models). {corpus_hint}")
        # Materialize before the peek so a generator of runs isn't partly consumed
        # (next(iter(...)) would drop run 0 and silently score the rest).
        runs = list(runs)
        # Directive error for the intuitive-but-wrong topic_stability(docs) call:
        # in from-fitted-runs mode each element must be a model / topic-word array,
        # not a document (a list/tuple of tokens or ids, or a raw string).
        first = runs[0] if runs else None
        if isinstance(first, str) or (
            isinstance(first, (list, tuple))
            and (len(first) == 0
                 or isinstance(first[0], (str, int, np.integer)))
        ):
            raise TypeError(
                "topic_stability's first argument is a list of already-fitted "
                "models (or topic-word arrays), but you passed what looks like a "
                f"corpus (documents of tokens). {corpus_hint}")
    mats = [_as_topic_word(r) for r in runs]
    if len(mats) < 2:
        raise ValueError("need at least two runs to measure stability")
    ref = mats[0]
    if all(m.shape == ref.shape and np.array_equal(m, ref) for m in mats[1:]):
        warnings.warn(
            f"topic_stability: all {len(mats)} runs are identical, so the score "
            "is a trivial 1.0 that says nothing about robustness. Deterministic "
            "initializations ignore the seed (e.g. NMF/LSA with init='nndsvd'), "
            "so repeating the same fit at different seeds gives identical runs. "
            "Refit with init='random' (which responds to seed) or measure "
            "stability on bootstrap resamples of the documents.",
            UserWarning,
            stacklevel=2,
        )
    k = ref.shape[0]
    ref_top = [set(np.argsort(ref[t])[::-1][:topn]) for t in range(k)]
    scores = []
    for mat in mats[1:]:
        for i, j, _ in align_topics(ref, mat, metric=metric):
            other = set(np.argsort(mat[j])[::-1][:topn])
            union = ref_top[i] | other
            scores.append(len(ref_top[i] & other) / len(union) if union else 0.0)
    return float(np.mean(scores)) if scores else float("nan")



def _is_per_doc(value, D):
    """True when a fit-kwarg is a per-document array (first-axis length ``D``) and
    so must be resampled alongside the documents. Strings/scalars and per-feature
    arrays (a length-``p`` ``prevalence_names``) are not."""
    if isinstance(value, (str, bytes)) or value is None:
        return False
    try:
        n = value.shape[0] if hasattr(value, "shape") else len(value)
    except TypeError:
        return False
    return n == D



def _take_rows(value, pick):
    """Select rows ``pick`` (an integer index array) from a per-document
    container, preserving its type where it matters for ``fit``."""
    if hasattr(value, "iloc"):  # pandas DataFrame / Series
        return value.iloc[list(pick)].reset_index(drop=True)
    if isinstance(value, np.ndarray):
        return value[pick]
    try:  # polars and other array-likes that accept integer-array indexing
        return value[np.asarray(pick)]
    except Exception:
        return [value[i] for i in pick]



def bootstrap_stability(
    docs,
    *,
    k=None,
    num_topics=None,
    n_boot=20,
    topn=10,
    seed=0,
    model_factory=None,
    reference=None,
    fit_kwargs=None,
    **extra_fit_kwargs,
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
        it). ``num_topics=`` is accepted as an alias (the library-wide name).
    n_boot : number of bootstrap resamples.
    model_factory : ``callable(seed) -> unfitted model``. Defaults to
        ``LDA(num_topics=k, seed=seed)``. Use it to bootstrap any model.
    reference : an already-fitted model to measure the stability *of*. When given,
        the resample topics are matched back to it (rather than to a fresh
        full-corpus fit), so the per-topic stability lines up with that model's
        topic indices. ``model_factory`` should rebuild the same model type. The
        factory's single argument is the **seed** (not ``k``), e.g.
        ``model_factory=lambda seed: topica.LDA(8, seed=seed)``; the factory sets
        the topic count, so ``k=`` is ignored when a factory is given.
    fit_kwargs : dict of keyword arguments forwarded to each model's ``fit`` (e.g.
        ``fit_kwargs={"iters": 500}``), matching :func:`cross_validate`. For
        convenience the same arguments may also be passed inline as keywords
        (``bootstrap_stability(docs, k=5, iters=500)``); the two are merged, with
        inline keywords taking precedence on a clash.

        **Covariate models.** A per-document design passed here (STM
        ``prevalence`` / ``covariates``, a ``content`` label vector, dynamic
        ``timestamps``) is resampled with the *same* bootstrap draw as the
        documents, so the covariate rows stay aligned to the resampled corpus and
        a covariate model can be bootstrapped as itself — no silent downgrade to a
        covariate-free model (issue #751). Any fit-kwarg whose first-axis length
        equals the number of documents is treated as per-document and resampled;
        per-feature arrays (e.g. ``prevalence_names``, length ``p``) and scalars
        are passed through unchanged. So a full STM stability check reads::

            X, names = topica.design.one_hot(df["party"])
            stm = topica.STM(num_topics=10, seed=13).fit(
                docs, prevalence=X, prevalence_names=names)
            bs = topica.evaluate.bootstrap_stability(
                docs, reference=stm,
                model_factory=lambda s: topica.STM(num_topics=10, seed=s),
                prevalence=X, prevalence_names=names)

    Returns
    -------
    :class:`~topica._results.BootstrapStability` (a ``dict``) with ``topic``
    (indices), ``stability`` (per-topic mean Jaccard in ``[0, 1]``), ``mean``
    (overall), and ``reference`` (the reference model). Call ``.to_frame()`` for a
    per-topic ``(topic, stability)`` DataFrame.
    """
    from . import LDA  # local import to avoid a cycle at module load

    # num_topics= is the library-wide name for the topic count (every constructor
    # is Model(num_topics=...)); accept it as an alias for this function's k= so a
    # user need not remember the odd one out (issue #732).
    if num_topics is not None:
        if k is not None and int(k) != int(num_topics):
            raise ValueError(
                f"pass either k= or num_topics=, not both with different values "
                f"(k={k}, num_topics={num_topics}); they are aliases for the topic count"
            )
        k = num_topics

    # fit_kwargs= (a dict, like cross_validate) and inline **extra_fit_kwargs both
    # forward to fit; merge them so the documented dict form works and no longer
    # collides with the old **kwargs (issue #740). Inline keywords win on a clash.
    fit_kwargs = {**(fit_kwargs or {}), **extra_fit_kwargs}

    # Accept a Corpus, matching the docstring and the sibling functions
    # (perplexity, prepare_pyldavis): pull its token lists before resampling.
    if hasattr(docs, "documents"):
        docs = docs.documents()
    docs = [list(d) for d in docs]
    D = len(docs)
    if D < 2:
        raise ValueError("need at least two documents to resample")
    if model_factory is not None and k is not None:
        # The factory owns the topic count and its argument is the seed, so a
        # stray k= here is silently ignored — warn rather than build the wrong K
        # (issue #740: `lambda k: LDA(k)` reads like k but receives the seed).
        warnings.warn(
            "bootstrap_stability: k= is ignored when model_factory= is given (the "
            "factory sets the topic count). Note the factory's argument is the SEED, "
            "not k — write model_factory=lambda seed: LDA(k, seed=seed).",
            UserWarning,
            stacklevel=2,
        )
    if k is None and reference is not None:
        k = int(reference.num_topics)
    # k is only needed to build the *default* factory; a supplied model_factory
    # owns the topic count itself (the reference K is read back off the fit), so
    # factory-only usage without k= is allowed — matching the docstring (#742).
    if model_factory is None and k is None:
        raise ValueError(
            "pass k= / num_topics= (number of topics), a model_factory, or a "
            "fitted reference model")
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

    # Any per-document fit-kwarg (a covariate design: STM `prevalence`, DMR
    # `covariates`, `content` labels, `timestamps`) must be resampled with the
    # *same* draw as the documents, or the covariate rows misalign with the
    # resampled corpus. Per-feature arrays (`prevalence_names`, length p) and
    # scalars (`iters`) are passed through untouched (issue #751).
    per_doc_keys = [key for key, v in fit_kwargs.items() if _is_per_doc(v, D)]

    rng = np.random.RandomState(seed)
    per_topic = [[] for _ in range(K)]
    for b in range(n_boot):
        pick = rng.randint(0, D, size=D)
        sample = [docs[i] for i in pick]
        boot_kwargs = {
            key: (_take_rows(v, pick) if key in per_doc_keys else v)
            for key, v in fit_kwargs.items()
        }
        m = factory(seed + b + 1)
        m.fit(sample, **boot_kwargs)
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
    from ._results import BootstrapStability

    return BootstrapStability({
        "topic": np.arange(K),
        "stability": stability,
        "mean": float(np.nanmean(stability)),
        "reference": ref,
    })



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
    For each topic it gathers :func:`topica.evaluate.coherence`,
    :func:`topica.evaluate.exclusivity`, the normalized topic-word entropy (1.0 = a
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


def __dir__():
    """Show only the public workflow surface in tab-completion (#757), hiding the
    module's own imports (np, re, dataclass, ...)."""
    return sorted(__all__)
