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


def _find_anchors(q, k, seed):
    """Greedy farthest-point (Gram-Schmidt) anchor selection on the rows of
    ``Q`` — Arora et al.'s FastAnchorWords. Picks the row of largest norm, then
    repeatedly the row farthest from the span of those already chosen."""
    rng = np.random.default_rng(seed)
    v = q.shape[0]
    anchors = [int(np.argmax((q * q).sum(axis=1)))]
    basis = [q[anchors[0]] / np.linalg.norm(q[anchors[0]])]
    for _ in range(k - 1):
        bmat = np.array(basis)
        resid = q - (q @ bmat.T) @ bmat
        dist = (resid * resid).sum(axis=1)
        dist[anchors] = -1.0
        nxt = int(np.argmax(dist))
        b = resid[nxt]
        nb = np.linalg.norm(b)
        if nb < 1e-12:
            cand = [i for i in range(v) if i not in anchors]
            nxt = int(rng.choice(cand))
            b = resid[nxt]
            nb = np.linalg.norm(b)
        anchors.append(nxt)
        basis.append(b / nb)
    return anchors


def _recover_l2(q, anchors):
    """RecoverL2: for each word solve the simplex-constrained non-negative least
    squares ``Q_i ≈ Σ_k C_ik Q_{anchor_k}`` for ``C_ik = p(topic k | word i)``."""
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
    min_count : int, default 5
        Drop vocabulary words occurring fewer than this many times overall.
    seed : int, default 42
        Affects only the degenerate-anchor fallback; recovery is otherwise
        deterministic.
    """

    def __init__(self, num_topics: int, *, min_count: int = 5, seed: int = 42):
        if not experimental_enabled():
            raise RuntimeError(_GATE_MESSAGE)
        if num_topics < 2:
            raise ValueError("num_topics must be at least 2")
        self._k = int(num_topics)
        self.min_count = int(min_count)
        self.seed = int(seed)
        self._fitted = False
        self._topic_word = None
        self._doc_topic = None
        self._vocab = None
        self._doc_names = None
        self._anchors = None
        self._texts = None
        self._topic_names = None

    # -- fitting ------------------------------------------------------------
    def fit(self, data, *, iters=None, min_count=None):
        """Recover topics from ``data`` (a :class:`~topica.Corpus` or a list of
        token lists). ``iters`` is accepted for interface uniformity but unused:
        the recovery is non-iterative. ``min_count`` overrides the constructor
        value for this fit."""
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
        anchors = _find_anchors(q, self._k, self.seed)
        c = _recover_l2(q, anchors)              # V x K, p(topic | word)
        a = c * pword[:, None]                    # Bayes inversion
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
        """Empty: anchor-words recovery is non-iterative, so there is no
        per-iteration objective trace."""
        return []

    @property
    def converged(self):
        """``None``: the recovery is non-iterative, so convergence does not
        apply (matching the contract for non-iterative models)."""
        return None

    def top_words(self, n: int = 10, *, topic=None):
        self._require_fit()
        beta = self._topic_word
        vocab = self._vocab

        def one(t):
            idx = np.argsort(beta[t])[::-1][:n]
            return [(vocab[i], float(beta[t, i])) for i in idx]

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
            "min_count": self.min_count,
            "seed": self.seed,
            "vocabulary": list(self._vocab),
            "doc_names": list(self._doc_names),
            "anchors": list(self._anchors),
            "topic_names": self._topic_names,
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
            m = AnchorLDA(meta["num_topics"], min_count=meta["min_count"],
                          seed=meta["seed"])
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
        m._texts = [list(d) for d in meta["texts"]]
        m._fitted = True
        return m

    def __repr__(self) -> str:
        if self._fitted:
            return f"AnchorLDA(num_topics={self._k}, vocab={len(self._vocab)}, fitted)"
        return f"AnchorLDA(num_topics={self._k}, unfitted)"
