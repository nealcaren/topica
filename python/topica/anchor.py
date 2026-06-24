"""Anchor-words spectral topic recovery (Arora et al. 2013).

A deterministic, Gibbs-free topic estimator built on separable NMF of the
word-word co-occurrence matrix. Unlike the collapsed-Gibbs and variational
models, ``AnchorLDA`` does no sampling and no EM: it finds one near-extreme
"anchor" word per topic on the co-occurrence Gram matrix and recovers each
topic's word distribution by a convex projection onto the anchor basis. The
result is reproducible bit-for-bit and fast, and each topic carries a single
human-readable anchor word.

Reference: Arora, Ge, Halpern, Mimno, Moitra, Sontag, Wu & Zhu (2013), "A
Practical Algorithm for Topic Modeling with Provable Guarantees", ICML.

Experimental: this ships before a published paper and a reference-implementation
parity check (topica's bar for a validated model), so it is gated behind
:func:`topica.enable_experimental`.
"""

from __future__ import annotations

import json

import numpy as np

from . import experimental_enabled


_GATE_MESSAGE = (
    "AnchorLDA is experimental and unvalidated: it has no published paper or "
    "reference-implementation parity yet, topica's bar for a validated model. "
    "Enable experimental models with `topica.enable_experimental()` or set the "
    "environment variable TOPICA_EXPERIMENTAL=1. Experimental models may change "
    "or be removed without a deprecation cycle."
)


def _corpus_to_token_lists(data):
    """Accept a Corpus or a list of token lists; return (docs, doc_names)."""
    if hasattr(data, "documents") and callable(getattr(data, "documents")):
        docs = [list(d) for d in data.documents()]
        names = list(getattr(data, "doc_names", None) or
                     [f"doc_{i}" for i in range(len(docs))])
    else:
        docs = [list(d) for d in data]
        names = [f"doc_{i}" for i in range(len(docs))]
    return docs, names


def _build_vocab(docs, min_count):
    from collections import Counter

    counts = Counter()
    for d in docs:
        counts.update(d)
    vocab = sorted(w for w, n in counts.items() if n >= min_count)
    return vocab, {w: i for i, w in enumerate(vocab)}


def _doc_term(docs, w2i):
    v = len(w2i)
    out = np.zeros((len(docs), v))
    for r, d in enumerate(docs):
        for w in d:
            j = w2i.get(w)
            if j is not None:
                out[r, j] += 1.0
    return out


def _build_q(counts):
    """Unbiased word-word co-occurrence ``Q`` (V x V), row-normalized so each row
    is ``p(w2 | w1)``. Per-document estimator ``(h hᵀ - diag(h)) / (n(n-1))``
    averaged over documents with at least two tokens (Arora et al. 2013)."""
    n = counts.sum(axis=1)
    keep = n >= 2
    h = counts[keep]
    nk = n[keep]
    w = 1.0 / (nk * (nk - 1))
    hw = h * w[:, None]
    qbar = hw.T @ h
    v = counts.shape[1]
    qbar[np.diag_indices(v)] -= hw.sum(axis=0)
    qbar /= int(keep.sum())
    rowsum = qbar.sum(axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1.0
    return qbar / rowsum, rowsum.ravel()


def _find_anchors(q, k, seed, candidates=None):
    """Greedy farthest-point (Gram-Schmidt) anchor selection on the rows of
    ``Q`` — Arora et al.'s FastAnchorWords. Picks the row of largest norm, then
    repeatedly the row farthest from the span of those already chosen.

    ``candidates`` (a boolean mask over words) restricts which words may be
    chosen as anchors. Restricting to well-attested (high document-frequency)
    words keeps the farthest-point search from latching onto rare, noisy rows,
    which is the standard practical refinement for real corpora."""
    rng = np.random.default_rng(seed)
    v = q.shape[0]
    if candidates is None:
        candidates = np.ones(v, dtype=bool)
    if candidates.sum() < k:               # too few candidates: fall back to all words
        candidates = np.ones(v, dtype=bool)
    norms = (q * q).sum(axis=1)
    norms = np.where(candidates, norms, -1.0)
    anchors = [int(np.argmax(norms))]
    basis = [q[anchors[0]] / np.linalg.norm(q[anchors[0]])]
    for _ in range(k - 1):
        bmat = np.array(basis)
        resid = q - (q @ bmat.T) @ bmat
        dist = (resid * resid).sum(axis=1)
        dist[~candidates] = -1.0
        dist[anchors] = -1.0
        nxt = int(np.argmax(dist))
        b = resid[nxt]
        nb = np.linalg.norm(b)
        if nb < 1e-12:
            pool = [i for i in range(v) if candidates[i] and i not in anchors]
            nxt = int(rng.choice(pool)) if pool else nxt
            b = resid[nxt]
            nb = np.linalg.norm(b)
        anchors.append(nxt)
        basis.append(b / nb if nb > 1e-12 else b)
    return anchors


def _recover_l2(q, anchors):
    """RecoverL2: for each word solve the simplex-constrained non-negative least
    squares ``Q_i ≈ Σ_k C_ik Q_{anchor_k}`` for ``C_ik = p(topic k | word i)``.
    One serial NNLS solve per word — exact but ``O(V)`` Python iterations."""
    from scipy.optimize import nnls

    qa = q[anchors]
    k = len(anchors)
    aug = np.vstack([qa.T, 1e3 * np.ones((1, k))])
    out = np.zeros((q.shape[0], k))
    for i in range(q.shape[0]):
        c, _ = nnls(aug, np.concatenate([q[i], [1e3]]))
        s = c.sum()
        out[i] = c / s if s > 0 else c
    return out


def _recover_kl(q, anchors, *, iters, eta, tol, check_every=5):
    """RecoverKL: recover ``C_ik = p(topic k | word i)`` by minimizing
    ``KL(Q_i ‖ Σ_k C_ik Q_{anchor_k})`` over the simplex with exponentiated
    gradient (Arora et al. 2013). Vectorized over all words at once — a few BLAS
    matmuls per step instead of ``O(V)`` serial solves — and stops early when the
    objective flattens. The objective is checked every ``check_every`` steps
    (it reuses the step's own reconstruction, so the check adds only a masked
    log-sum, not another matmul). Returns ``(C, history, converged)``."""
    qa = q[anchors]
    v, k = q.shape[0], len(anchors)
    c = np.full((v, k), 1.0 / k)
    mask = q > 0
    q_m = q[mask]
    q_logq = float((q_m * np.log(q_m)).sum())     # constant part of the mean KL
    history = []
    converged = False
    prev = np.inf
    for it in range(int(iters)):
        recon = np.clip(c @ qa, 1e-12, None)
        last = it == int(iters) - 1
        if it % check_every == 0 or last:
            obj = (q_logq - float((q_m * np.log(recon[mask])).sum())) / v
            history.append((it, obj))
            if prev - obj < tol * max(abs(prev), 1e-12):
                converged = True
                break
            prev = obj
        upd = eta * ((q / recon) @ qa.T)          # = -eta * dKL/dC
        upd -= upd.max(axis=1, keepdims=True)      # stabilize exp (cancels on renorm)
        # On sparse co-occurrence (large vocab), words that never co-occur with an
        # anchor drive recon -> floor and the gradient explodes (|upd| ~ 1e8), so
        # every entry but the row max underflows to 0 and the renormalization
        # divides 0/0 -> NaN. Flooring the exponent keeps a tiny mass on the
        # non-max topics (exp(-50) ~ 2e-22, negligible) so the row never collapses.
        np.maximum(upd, -50.0, out=upd)
        c *= np.exp(upd)
        c /= c.sum(axis=1, keepdims=True)
    return c, history, converged


class AnchorLDA:
    """Anchor-words spectral topic model (Arora et al. 2013), an experimental,
    deterministic, Gibbs-free estimator.

    Recovers a topic-word matrix from the word co-occurrence statistics by anchor
    selection plus convex recovery — no sampling, no EM, reproducible bit-for-bit.
    Each topic exposes a single anchor word (the near-extreme-point word that
    identifies it). The topic-word matrix is ``A`` Bayes-inverted to ``p(word |
    topic)``; the document-topic matrix is ``p(topic | doc)`` from the per-word
    topic responsibilities.

    Experimental and gated: call :func:`topica.enable_experimental` (or set
    ``TOPICA_EXPERIMENTAL=1``) before constructing one.

    Parameters
    ----------
    num_topics : int
        The number of topics (and anchors) to recover.
    recover : {"kl", "l2"}, default "kl"
        The recovery step. ``"kl"`` minimizes ``KL(Q_i ‖ C_i Q_anchors)`` with a
        vectorized exponentiated-gradient solver — faster (a few BLAS matmuls
        instead of one NNLS per word) and a closer co-occurrence fit. ``"l2"`` is
        the simplex-constrained non-negative least squares of Arora et al.'s
        RecoverL2: exact but a serial solve per word.
    min_count : int, default 5
        Drop vocabulary words occurring fewer than this many times overall.
    seed : int, default 42
        Affects only the degenerate-anchor fallback; recovery is otherwise
        deterministic.
    eta : float, default 1.0
        Exponentiated-gradient step size for ``recover="kl"`` (unused for
        ``"l2"``). Values above ~1.5 can overshoot and collapse topics.
    convergence_tol : float, default 1e-5
        Relative-objective tolerance for the ``recover="kl"`` early stop.
    frex_w : float, default 0.5
        Frequency/exclusivity balance for the default FREX top-word ranking (see
        :meth:`top_words`). ``0`` is pure exclusivity, ``1`` pure frequency.
    frequency_temper : float, default 0.5
        Exponent ``γ`` on the word frequency in the Bayes inversion ``beta ∝
        p(topic|word)·p(word)**γ``. The exact inversion (``γ=1``) weights the
        topic-word matrix by raw word frequency, so pervasive words dominate many
        topics; tempering with ``γ<1`` divides that back out and gives more
        distinctive, higher-coherence topics. The default ``0.5`` roughly tripled
        top-word diversity and raised c_v on poliblog; ``γ=1`` restores the exact
        Arora et al. estimate, ``γ=0`` is pure lift.
    anchor_min_doc_freq : float, default 0.01
        Minimum document frequency for a word to be eligible as an anchor — a
        fraction of documents when ``<1``, else an absolute count. Restricting
        anchors to well-attested words keeps the farthest-point search from
        latching onto rare, noisy rows, which sharpens topics on large
        vocabularies (it lifted c_v on 20 Newsgroups with no change on poliblog).
        ``0`` disables the restriction; if fewer than ``num_topics`` words
        qualify, it is dropped for that fit.
    """

    def __init__(self, num_topics: int, *, recover: str = "kl", min_count: int = 5,
                 seed: int = 42, eta: float = 1.0, convergence_tol: float = 1e-5,
                 frex_w: float = 0.5, frequency_temper: float = 0.5,
                 anchor_min_doc_freq: float = 0.01):
        if not experimental_enabled():
            raise RuntimeError(_GATE_MESSAGE)
        if num_topics < 2:
            raise ValueError("num_topics must be at least 2")
        if recover not in ("kl", "l2"):
            raise ValueError(f"recover must be 'kl' or 'l2'; got {recover!r}")
        self._k = int(num_topics)
        self.recover = recover
        self.min_count = int(min_count)
        self.seed = int(seed)
        self.eta = float(eta)
        self.convergence_tol = float(convergence_tol)
        self.frex_w = float(frex_w)
        self.frequency_temper = float(frequency_temper)
        self.anchor_min_doc_freq = float(anchor_min_doc_freq)
        self._fitted = False
        self._topic_word = None
        self._doc_topic = None
        self._vocab = None
        self._doc_names = None
        self._anchors = None
        self._texts = None
        self._topic_names = None
        self._fit_history = []
        self._converged = None
        self._word_counts = None

    # -- fitting ------------------------------------------------------------
    def fit(self, data, *, iters=None, min_count=None):
        """Recover topics from ``data`` (a :class:`~topica.Corpus` or a list of
        token lists). For ``recover="kl"`` ``iters`` is the maximum number of
        exponentiated-gradient steps (default 200, with an early stop on
        ``convergence_tol``);
        for ``recover="l2"`` the recovery is non-iterative and ``iters`` is
        ignored. ``min_count`` overrides the constructor value for this fit."""
        mc = self.min_count if min_count is None else int(min_count)
        docs, names = _corpus_to_token_lists(data)
        vocab, w2i = _build_vocab(docs, mc)
        if len(vocab) < self._k:
            raise ValueError(
                f"vocabulary has {len(vocab)} words after min_count={mc} but "
                f"num_topics={self._k}; lower min_count or num_topics."
            )
        counts = _doc_term(docs, w2i)
        q, pword = _build_q(counts)
        # Restrict anchor candidates to well-attested words (document frequency
        # >= anchor_min_doc_freq, read as a fraction of documents when < 1, else
        # an absolute count) so the farthest-point search skips rare, noisy rows.
        doc_freq = (counts > 0).sum(axis=0)
        n_docs = counts.shape[0]
        thresh = (self.anchor_min_doc_freq * n_docs
                  if self.anchor_min_doc_freq < 1.0 else self.anchor_min_doc_freq)
        candidates = doc_freq >= max(1.0, thresh) if thresh > 0 else None
        anchors = _find_anchors(q, self._k, self.seed, candidates)
        if self.recover == "kl":
            n_iters = 200 if iters is None else int(iters)
            c, self._fit_history, self._converged = _recover_kl(
                q, anchors, iters=n_iters, eta=self.eta, tol=self.convergence_tol)
        else:
            c = _recover_l2(q, anchors)          # V x K, p(topic | word)
            self._fit_history, self._converged = [], None
        a = c * (pword[:, None] ** self.frequency_temper)   # (tempered) Bayes inversion
        z = a.sum(axis=0, keepdims=True)
        z[z == 0] = 1.0
        beta = (a / z).T                          # K x V, p(word | topic)

        theta = counts @ c                        # D x K, unnormalized p(topic|doc)
        rs = theta.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        theta = theta / rs

        self._topic_word = beta
        self._doc_topic = theta
        self._vocab = vocab
        self._doc_names = names
        self._anchors = [vocab[i] for i in anchors]
        self._texts = docs
        self._word_counts = counts.sum(axis=0)    # per-word totals, for FREX/lift
        self._fitted = True
        return self

    def _require_fit(self):
        if not self._fitted:
            raise RuntimeError("AnchorLDA is not fitted; call fit() first.")

    # -- Tier-0 surface -----------------------------------------------------
    @property
    def num_topics(self) -> int:
        return self._k

    @property
    def topic_word(self):
        self._require_fit()
        return self._topic_word

    @property
    def doc_topic(self):
        self._require_fit()
        return self._doc_topic

    @property
    def vocabulary(self) -> list:
        self._require_fit()
        return list(self._vocab)

    @property
    def doc_names(self) -> list:
        self._require_fit()
        return list(self._doc_names)

    @property
    def anchors(self) -> list:
        """The anchor word identifying each topic, in topic order."""
        self._require_fit()
        return list(self._anchors)

    @property
    def topic_names(self) -> list:
        self._require_fit()
        if self._topic_names is not None:
            return list(self._topic_names)
        return [f"topic_{t}" for t in range(self._k)]

    @topic_names.setter
    def topic_names(self, value):
        if len(value) != self._k:
            raise ValueError(f"expected {self._k} names, got {len(value)}")
        self._topic_names = [str(v) for v in value]

    @property
    def fit_history(self) -> list:
        """For ``recover="kl"``, the ``(iteration, KL objective)`` trace of the
        exponentiated-gradient recovery; empty for the non-iterative ``"l2"``."""
        return list(self._fit_history)

    @property
    def converged(self):
        """For ``recover="kl"``, whether the recovery stopped early on
        ``convergence_tol``;
        ``None`` for the non-iterative ``"l2"`` recovery."""
        return self._converged

    def _word_count_vector(self):
        if self._word_counts is None:
            idx = {w: i for i, w in enumerate(self._vocab)}
            c = np.zeros(len(self._vocab))
            for doc in self._texts:
                for w in doc:
                    j = idx.get(w)
                    if j is not None:
                        c[j] += 1.0
            self._word_counts = c
        return self._word_counts

    def top_words(self, n: int = 10, *, topic=None, method="frex"):
        """Top ``n`` words per topic as ``(word, score)`` pairs.

        ``method`` controls the ranking. The default ``"frex"`` ranks by the
        FREX score (the frequency/exclusivity harmonic mean, balance ``frex_w``),
        which matters more for anchor-words than for the Gibbs models: the
        Bayes-inversion ``beta ∝ p(topic|word)·p(word)`` weights ``beta`` by raw
        word frequency, so ranking by ``"prob"`` surfaces pervasive words across
        many topics. ``"lift"`` ranks by ``beta / p(word)`` (pure exclusivity,
        which can over-reward rare words). Pass ``topic`` for a single topic's
        list, else a list per topic is returned.
        """
        self._require_fit()
        beta = self._topic_word
        vocab = self._vocab
        if method == "frex":
            from .validation import frex as _frex
            pairs = _frex(beta, vocab, w=self.frex_w, n=n,
                          word_counts=self._word_count_vector())
            return pairs[int(topic)] if topic is not None else pairs
        if method == "lift":
            wc = self._word_count_vector()
            score = beta / np.where(wc > 0, wc, 1.0)
        elif method == "prob":
            score = beta
        else:
            raise ValueError(f"method must be 'frex', 'prob', or 'lift'; got {method!r}")

        def one(t):
            idx = np.argsort(score[t])[::-1][:n]
            return [(vocab[i], float(score[t, i])) for i in idx]

        if topic is not None:
            return one(int(topic))
        return [one(t) for t in range(self._k)]

    def coherence(self, n: int = 10):
        """Per-topic c_v coherence over the training corpus (a ``(K,)`` array)."""
        self._require_fit()
        from .coherence import coherence as _coherence

        topics = [[w for w, _ in self.top_words(n, topic=t)] for t in range(self._k)]
        return np.asarray(_coherence(topics, self._texts, topn=n), dtype=np.float64)

    # -- persistence --------------------------------------------------------
    def save(self, path: str) -> None:
        self._require_fit()
        meta = {
            "num_topics": self._k,
            "recover": self.recover,
            "min_count": self.min_count,
            "seed": self.seed,
            "eta": self.eta,
            "convergence_tol": self.convergence_tol,
            "frex_w": self.frex_w,
            "frequency_temper": self.frequency_temper,
            "anchor_min_doc_freq": self.anchor_min_doc_freq,
            "vocabulary": list(self._vocab),
            "doc_names": list(self._doc_names),
            "anchors": list(self._anchors),
            "topic_names": self._topic_names,
            "fit_history": self._fit_history,
            "converged": self._converged,
            "texts": self._texts,
        }
        np.savez(
            path,
            topic_word=self._topic_word,
            doc_topic=self._doc_topic,
            meta=np.array(json.dumps(meta)),
        )

    @staticmethod
    def load(path: str) -> "AnchorLDA":
        if not path.endswith(".npz"):
            path = path + ".npz"
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["meta"]))
            tw = f["topic_word"]
            dt = f["doc_topic"]
        was_on = experimental_enabled()
        if not was_on:
            from . import enable_experimental
            enable_experimental(True)
        try:
            m = AnchorLDA(meta["num_topics"], recover=meta.get("recover", "l2"),
                          min_count=meta["min_count"], seed=meta["seed"],
                          eta=meta.get("eta", 1.0),
                          convergence_tol=meta.get("convergence_tol", 1e-5),
                          frex_w=meta.get("frex_w", 0.5),
                          frequency_temper=meta.get("frequency_temper", 1.0),
                          anchor_min_doc_freq=meta.get("anchor_min_doc_freq", 0.0))
        finally:
            if not was_on:
                from . import enable_experimental
                enable_experimental(False)
        m._topic_word = tw
        m._doc_topic = dt
        m._vocab = list(meta["vocabulary"])
        m._doc_names = list(meta["doc_names"])
        m._anchors = list(meta["anchors"])
        m._topic_names = meta["topic_names"]
        m._fit_history = [tuple(h) for h in meta.get("fit_history", [])]
        m._converged = meta.get("converged")
        m._texts = [list(d) for d in meta["texts"]]
        m._fitted = True
        return m

    def __repr__(self) -> str:
        if self._fitted:
            return f"AnchorLDA(num_topics={self._k}, vocab={len(self._vocab)}, fitted)"
        return f"AnchorLDA(num_topics={self._k}, unfitted)"
