"""Embedding-guided LDA: anchor topics with pre-trained word embeddings.

The idea is a warm start, not a constraint. We cluster the vocabulary's
embeddings into ``num_topics`` semantic groups, seed each topic with the words
nearest its cluster centroid, and give those seed words a prior boost in that
topic. The Gibbs sampler then runs as ordinary LDA and can override any seed the
text data contradicts, so the embeddings shape where topics form without
dictating what they end up being.

This reuses the validated :class:`~topica.SeededLDA` sampler: an embedding-guided
fit is a seeded fit whose seeds are discovered from an embedding space instead of
typed by hand. The asymmetric topic-word prior, the seeded initialization, and
the (correctly disabled) beta optimization all come from there.

Users bring their own ``embeddings`` (a dense ``V x E`` matrix, e.g. from
``sentence-transformers`` run over the vocabulary). topica only needs the matrix
and the matching vocabulary list; it does the clustering and seeding.

    from sentence_transformers import SentenceTransformer
    import topica

    vocab = sorted({w for d in docs for w in d})
    emb = SentenceTransformer("all-MiniLM-L6-v2").encode(vocab)
    model = topica.EmbeddingLDA(num_topics=10, embeddings=emb, vocabulary=vocab)
    model.fit(docs, iters=1000)
    print(model.top_words(8))
"""

from __future__ import annotations

import hashlib
import os
from typing import Sequence

import numpy as np


def _kmeans(x: np.ndarray, k: int, *, seed: int, iters: int = 50):
    """k-means++ initialization then Lloyd iterations. Pure numpy (no sklearn),
    deterministic for a fixed ``seed``. Returns ``(labels, centroids)``."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    sq = (x * x).sum(axis=1)  # squared norms, reused in the distance identity

    # k-means++ seeding: pick centers far from those already chosen.
    centers = np.empty((k, x.shape[1]), dtype=x.dtype)
    first = int(rng.integers(n))
    centers[0] = x[first]
    d2 = sq + (centers[0] * centers[0]).sum() - 2.0 * x @ centers[0]
    np.maximum(d2, 0, out=d2)
    for c in range(1, k):
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(n, 1.0 / n)
        nxt = int(rng.choice(n, p=probs))
        centers[c] = x[nxt]
        dc = sq + (centers[c] * centers[c]).sum() - 2.0 * x @ centers[c]
        np.minimum(d2, np.maximum(dc, 0), out=d2)

    labels = np.full(n, -1, dtype=np.int64)
    for _ in range(iters):
        # (n, k) squared distances via |x|^2 - 2 x.c + |c|^2 (no n*k*d tensor).
        dists = sq[:, None] - 2.0 * x @ centers.T + (centers * centers).sum(axis=1)[None, :]
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            members = labels == c
            if members.any():
                centers[c] = x[members].mean(axis=0)
            # An emptied centroid keeps its last position (rare; harmless here).
    return labels, centers


def _cluster_words(embeddings, num_topics: int, *, seed: int):
    """Row-normalize the embeddings, k-means into ``num_topics`` groups, and
    return ``(unit_word_vectors, labels, unit_centroids)``. The unit centroids
    are comparable by cosine with any document embedding from the same space."""
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("embeddings must be a 2-D (V, E) array")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    xn = x / norms
    labels, centers = _kmeans(xn, num_topics, seed=seed)
    cnorm = np.linalg.norm(centers, axis=1, keepdims=True)
    cnorm[cnorm == 0.0] = 1.0
    return xn, labels, centers / cnorm


def _seeds_from_clusters(xn, labels, centroids, vocabulary, num_topics, top_m):
    sims = xn @ centroids.T  # (V, K) word-to-centroid cosine
    seeds: dict[str, list[str]] = {}
    for k in range(num_topics):
        members = np.where(labels == k)[0]
        if members.size == 0:
            seeds[f"topic_{k}"] = []
            continue
        order = members[np.argsort(-sims[members, k])][:top_m]
        seeds[f"topic_{k}"] = [str(vocabulary[i]) for i in order]
    return seeds


def embedding_seeds(
    embeddings,
    vocabulary: Sequence[str],
    num_topics: int,
    *,
    top_m: int = 20,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Turn word embeddings into per-topic seed-word sets.

    Clusters the (row-normalized) embeddings into ``num_topics`` groups and, for
    each cluster, returns the ``top_m`` member words closest to the centroid by
    cosine similarity. Each word seeds at most one topic (its own cluster), so
    the seed sets are disjoint and the anchors stay distinct. Returns a dict
    ``{"topic_k": [words]}`` ready for :class:`~topica.SeededLDA`; a degenerate
    empty cluster yields an empty (unseeded) topic.
    """
    if len(vocabulary) != np.asarray(embeddings).shape[0]:
        raise ValueError(
            f"vocabulary has {len(vocabulary)} words but embeddings has "
            f"{np.asarray(embeddings).shape[0]} rows"
        )
    if num_topics < 2:
        raise ValueError("num_topics must be >= 2")
    if num_topics > len(vocabulary):
        raise ValueError("num_topics cannot exceed the vocabulary size")
    if top_m < 1:
        raise ValueError("top_m must be >= 1")
    xn, labels, centroids = _cluster_words(embeddings, num_topics, seed=seed)
    return _seeds_from_clusters(xn, labels, centroids, vocabulary, num_topics, top_m)


def _npz_path(path) -> str:
    """Normalize an embedding-cache path to its `.npz` form."""
    p = str(path)
    return p if p.endswith(".npz") else p + ".npz"


def _texts_hash(texts) -> str:
    """A stable hash of a text sequence, for cache integrity checks (no pickling)."""
    h = hashlib.sha256()
    for t in texts:
        h.update(str(t).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def save_embeddings(path, embeddings, *, texts=None, model=None) -> str:
    """Save an embedding matrix to a ``.npz`` file so a costly corpus is embedded
    once and reused.

    ``embeddings`` is any ``(n, dim)`` array. When given, ``texts`` (one per row)
    is stored as a hash and ``model`` as a string, so :func:`load_embeddings` and
    :func:`llm_embed`'s ``cache=`` can confirm a cache matches the current inputs.
    The path gets a ``.npz`` suffix if it lacks one; returns the path written.
    Works on any embeddings, not just :func:`llm_embed`'s.
    """
    fields = {"embeddings": np.asarray(embeddings, dtype=float)}
    if model is not None:
        fields["model"] = np.array(str(model))
    if texts is not None:
        fields["texts_hash"] = np.array(_texts_hash([str(t) for t in texts]))
    out = _npz_path(path)
    np.savez(out, **fields)
    return out


def load_embeddings(path, *, with_meta=False):
    """Load an embedding matrix saved by :func:`save_embeddings`.

    Returns the ``(n, dim)`` array, or ``(array, meta)`` when ``with_meta=True``;
    ``meta`` carries ``model`` and ``texts_hash`` if they were saved. The ``.npz``
    suffix is added if the path lacks one and the bare path does not exist.
    """
    p = str(path)
    if not os.path.exists(p):
        p = _npz_path(p)
    with np.load(p) as data:
        emb = data["embeddings"]
        if not with_meta:
            return emb
        meta = {}
        if "model" in data:
            meta["model"] = str(data["model"])
        if "texts_hash" in data:
            meta["texts_hash"] = str(data["texts_hash"])
        return emb, meta


def llm_embed(texts, model="text-embedding-3-small", *, key=None, batch=True, cache=None):
    """Embed ``texts`` with the `llm` library's embedding models, as a dense
    ``(n, dim)`` float array.

    The embedding models in topica (``BERTopic``, ``Top2Vec``, ``ETM``,
    ``FASTopic``) and :func:`embedding_seeds` all take embeddings you supply; this
    is one way to produce them. ``model`` names any embedding model
    `llm <https://llm.datasette.io/>`_ can reach — OpenAI's
    ``"text-embedding-3-small"`` / ``"3-large"`` (needs an API key), or a local
    model such as ``"sentence-transformers/all-MiniLM-L6-v2"`` via the
    ``llm-sentence-transformers`` plugin (no API, runs offline). Pass document
    texts for document embeddings, or the vocabulary for word embeddings.

    By default the API key (for hosted embedders) is resolved by ``llm`` itself: a
    stored ``llm keys`` value, else the provider's environment variable
    (``OPENAI_API_KEY`` for OpenAI). Pass ``key`` to override it explicitly.

    Embeddings are costly, so pass ``cache=path`` to embed once and reuse: if the
    file exists and was saved for the same ``texts``, it is loaded and no model is
    called; otherwise the embeddings are computed and written there (see
    :func:`save_embeddings`).

    Requires the optional ``llm`` package (``pip install "topica[llm]"``). The
    embeddings are the only thing topica needs from a model; everything downstream
    runs in the wheel.
    """
    items = [str(t) for t in texts]
    if cache is not None:
        cp = _npz_path(cache)
        if os.path.exists(cp):
            arr, meta = load_embeddings(cp, with_meta=True)
            if arr.shape[0] == len(items) and meta.get("texts_hash") == _texts_hash(items):
                return arr

    try:
        import llm as _llm
    except ImportError as e:  # pragma: no cover - exercised via message
        raise ImportError(
            "llm_embed needs the optional `llm` package "
            '(pip install llm, or pip install "topica[llm]").'
        ) from e
    try:
        em = _llm.get_embedding_model(model)
    except Exception as e:
        from .labeling import _unknown_model_message

        unknown = getattr(_llm, "UnknownModelError", None)
        if unknown is not None and not isinstance(e, unknown):
            raise
        raise ValueError(_unknown_model_message(_llm, model, kind="embedding")) from e
    if key is not None:
        em.key = key
    vecs = list(em.embed_multi(items)) if batch else [em.embed(t) for t in items]
    arr = np.asarray(vecs, dtype=float)
    if cache is not None:
        save_embeddings(cache, arr, texts=items, model=model)
    return arr


class EmbeddingLDA:
    """LDA whose topics are anchored by pre-trained embeddings, on both sides.

    The vocabulary embeddings define the topics: k-means clusters them into
    ``num_topics`` semantic groups, and each topic is seeded with the ``top_m``
    words nearest its centroid (a prior on the **topic-word** side, via
    :class:`~topica.SeededLDA`). Optionally, at fit time, **document** embeddings
    in the same space bias each document's topic mixture toward the topics its
    own embedding is closest to (a per-document prior on the **document-topic**
    side, ``α_{d,k}``). Both are priors: the Gibbs sampler reconciles them with
    word co-occurrence and can override either.

    Word seeds alone (no ``doc_embeddings``) is the lighter mode; adding document
    embeddings is closer in spirit to embedding-clustering methods, but keeps the
    generative, mixed-membership, override-able model. The fitted-model surface
    (``topic_word``, ``doc_topic``, ``top_words``, ``coherence``, ...) is
    delegated to the underlying SeededLDA.

    No embedder of your own? :func:`~topica.llm_embed` builds the ``embeddings``
    matrix (OpenAI, or offline ``sentence-transformers``).

    Parameters
    ----------
    num_topics : int
        Number of topics (and embedding clusters) to form.
    embeddings : array (V, E)
        Dense word-embedding matrix, one row per vocabulary word.
    vocabulary : sequence of str
        The words, aligned row-for-row with ``embeddings``.
    top_m : int
        How many of each cluster's nearest words to use as seeds.
    weight : float
        Seed strength: a seed word gets ``weight * 100`` extra prior pseudocounts
        in its topic. Higher anchors the topic-word side harder.
    doc_anchor : float
        Strength of the document-embedding prior used when ``doc_embeddings`` is
        passed to :meth:`fit`. ``α_{d,k} = alpha + doc_anchor * max(cos, 0)``.
    alpha, beta : float
        Base document-topic and topic-word Dirichlet priors.
    seed : int
        Random seed for the k-means clustering and the sampler.
    """

    def __init__(
        self,
        num_topics: int,
        *,
        embeddings,
        vocabulary: Sequence[str],
        top_m: int = 20,
        weight: float = 1.0,
        doc_anchor: float = 1.0,
        alpha: float = 0.1,
        beta: float = 0.01,
        seed: int = 42,
    ) -> None:
        from . import SeededLDA

        if len(vocabulary) != np.asarray(embeddings).shape[0]:
            raise ValueError("vocabulary length must match the number of embedding rows")
        if num_topics < 2:
            raise ValueError("num_topics must be >= 2")
        if num_topics > len(vocabulary):
            raise ValueError("num_topics cannot exceed the vocabulary size")
        if top_m < 1:
            raise ValueError("top_m must be >= 1")
        if doc_anchor < 0:
            raise ValueError("doc_anchor must be >= 0")

        self.num_topics = num_topics
        self.top_m = top_m
        self.alpha = alpha
        self.doc_anchor = doc_anchor
        # One clustering pass: keep the unit centroids for the document prior.
        xn, labels, self._centroids = _cluster_words(embeddings, num_topics, seed=seed)
        self.seeds = _seeds_from_clusters(xn, labels, self._centroids, vocabulary, num_topics, top_m)
        self._model = SeededLDA(
            self.seeds, alpha=alpha, beta=beta, weight=weight, seed=seed
        )

    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict, keyword-named
        to match ``__init__`` (issue #400). The ``embeddings``/``vocabulary`` inputs
        are data, not hyperparameters, so they are not reported here."""
        sub = self._model.settings
        return {
            "num_topics": self.num_topics,
            "top_m": self.top_m,
            "weight": sub["weight"],
            "doc_anchor": self.doc_anchor,
            "alpha": self.alpha,
            "beta": sub["beta"],
            "seed": sub["seed"],
        }

    def document_topic_prior(self, doc_embeddings) -> np.ndarray:
        """The per-document Dirichlet prior ``α_{d,k}`` implied by document
        embeddings: ``alpha + doc_anchor * max(cos(doc_d, centroid_k), 0)``,
        shape ``(num_docs, num_topics)``. Useful for inspection."""
        de = np.asarray(doc_embeddings, dtype=np.float64)
        if de.ndim != 2 or de.shape[1] != self._centroids.shape[1]:
            raise ValueError(
                "doc_embeddings must be (num_docs, E) with E matching the word embeddings"
            )
        norms = np.linalg.norm(de, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        sim = (de / norms) @ self._centroids.T
        return self.alpha + self.doc_anchor * np.maximum(sim, 0.0)

    def fit(self, data, *, doc_embeddings=None, iters: int = 1000) -> "EmbeddingLDA":
        """Fit on ``data`` (a Corpus or list of token lists). If ``doc_embeddings``
        is given (one row per document, same embedding space as the vocabulary),
        each document's topic mixture is biased toward the topics its embedding is
        nearest, as a prior the sampler can still override. ``iters`` is the number
        of Gibbs sweeps for the underlying SeededLDA fit."""
        prior = self.document_topic_prior(doc_embeddings) if doc_embeddings is not None else None
        self._model.fit(data, iters=iters, doc_topic_prior=prior)
        return self

    @property
    def model(self):
        """The underlying fitted :class:`~topica.SeededLDA`."""
        return self._model

    def __getattr__(self, name):
        # Delegate the fitted-model API (topic_word, top_words, ...) to SeededLDA.
        model = self.__dict__.get("_model")
        if model is None:
            raise AttributeError(name)
        return getattr(model, name)

    def __repr__(self) -> str:
        seeded = sum(1 for s in self.seeds.values() if s)
        return (
            f"EmbeddingLDA(num_topics={self.num_topics}, top_m={self.top_m}, "
            f"{seeded} topics seeded)"
        )


def _topic_anchors(seeds, vocabulary, embeddings, num_topics, topic_embeddings):
    """Per-topic unit anchor vectors (or ``None``). By default a keyword topic's
    anchor is the mean embedding of its in-vocabulary keywords; residual topics
    have none. An explicit ``topic_embeddings`` row (finite) overrides a topic's
    anchor, which also lets residual topics be anchored.

    Returns a list of length ``num_topics`` whose entries are unit-norm
    ``(E,)`` arrays or ``None``."""
    emb = np.asarray(embeddings, dtype=np.float64)
    index = {str(w): i for i, w in enumerate(vocabulary)}
    anchors: list = [None] * num_topics
    # Keyword topics come first, in dict order (matching topica's topic order).
    for t, words in enumerate(seeds.values()):
        rows = [index[str(w)] for w in words if str(w) in index]
        if rows:
            anchors[t] = emb[rows].mean(axis=0)

    if topic_embeddings is not None:
        te = np.asarray(topic_embeddings, dtype=np.float64)
        if te.shape != (num_topics, emb.shape[1]):
            raise ValueError(
                f"topic_embeddings must be ({num_topics}, {emb.shape[1]}); got {te.shape}"
            )
        for t in range(num_topics):
            if np.all(np.isfinite(te[t])) and np.linalg.norm(te[t]) > 0:
                anchors[t] = te[t]

    # Normalize to unit length so the offset is a cosine.
    out = []
    for a in anchors:
        if a is None:
            out.append(None)
            continue
        n = np.linalg.norm(a)
        out.append(a / n if n > 0 else None)
    return out


class EmbeddingKeyATM:
    """keyATM whose document-topic prior is anchored by document embeddings.

    This is the covariate keyATM (``α_{d,k} = exp(x_d · λ_k)``) with one extra
    term: a fixed embedding offset added inside the exponent,

        α_{d,k} = exp(x_d · λ_k + doc_anchor · max(cos(emb_d, anchor_k), 0)).

    Each keyword topic's ``anchor_k`` is, by default, the mean embedding of its
    keywords, so a document leans toward the keyword topics it is semantically
    near even when it is too short for word co-occurrence to place it. The
    covariate coefficients ``λ`` are still estimated, so you keep keyATM's
    effect estimation (``feature_effects``, ``by_strata``) on top of the anchor.
    Pass ``covariates`` to estimate effects; omit them to use the anchor alone
    (an intercept-only design is synthesized).

    Unlike a hard embedding clustering, the embedding here is a prior the word
    likelihood can override, and ``doc_anchor`` dials how much it is trusted.

    Parameters
    ----------
    seeds : dict[str, list[str]]
        Keyword topics, as for :class:`~topica.KeyATM`.
    num_topics : int
        Total topics (keyword topics first, then residual topics).
    embeddings : array (V, E)
        Word-embedding matrix aligned with ``vocabulary`` (for the keyword-mean
        anchors).
    vocabulary : sequence of str
        Words aligned row-for-row with ``embeddings``.
    topic_embeddings : array (num_topics, E), optional
        Explicit per-topic anchor vectors. A finite row overrides that topic's
        keyword-mean anchor (and can anchor a residual topic); leave a row
        non-finite to keep the default.
    doc_anchor : float
        Strength of the embedding offset.
    seed : int
        Random seed for the underlying sampler.
    """

    def __init__(
        self,
        seeds,
        num_topics: int,
        *,
        embeddings,
        vocabulary: Sequence[str],
        topic_embeddings=None,
        doc_anchor: float = 5.0,
        seed: int = 42,
        **keyatm_kwargs,
    ) -> None:
        from . import KeyATM

        if len(vocabulary) != np.asarray(embeddings).shape[0]:
            raise ValueError("vocabulary length must match the number of embedding rows")
        if num_topics < len(seeds):
            raise ValueError("num_topics cannot be smaller than the number of keyword topics")
        if doc_anchor < 0:
            raise ValueError("doc_anchor must be >= 0")

        self.seeds = dict(seeds)
        self.num_topics = num_topics
        self.doc_anchor = doc_anchor
        self._E = np.asarray(embeddings, dtype=np.float64).shape[1]
        self._anchors = _topic_anchors(
            self.seeds, vocabulary, embeddings, num_topics, topic_embeddings
        )
        self._model = KeyATM(self.seeds, num_topics=num_topics, seed=seed, **keyatm_kwargs)

    def document_topic_offset(self, doc_embeddings) -> np.ndarray:
        """The fixed offset matrix ``s_{d,k} = doc_anchor · max(cos(emb_d,
        anchor_k), 0)``, shape ``(num_docs, num_topics)``. Topics without an
        anchor (residual topics, by default) get a zero column."""
        de = np.asarray(doc_embeddings, dtype=np.float64)
        if de.ndim != 2 or de.shape[1] != self._E:
            raise ValueError(
                "doc_embeddings must be (num_docs, E) with E matching the word embeddings"
            )
        norms = np.linalg.norm(de, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        den = de / norms
        offset = np.zeros((de.shape[0], self.num_topics))
        for t, a in enumerate(self._anchors):
            if a is not None:
                offset[:, t] = self.doc_anchor * np.maximum(den @ a, 0.0)
        return offset

    def fit(self, data, *, doc_embeddings=None, covariates=None, iters: int = 1500, **fit_kwargs):
        """Fit on ``data``. If ``doc_embeddings`` is given (same space as the
        word embeddings), each document's topic prior is anchored toward the
        keyword topics it is closest to. ``covariates`` (and other keyATM
        ``fit`` kwargs) are passed through, so effects are estimated jointly."""
        offset = self.document_topic_offset(doc_embeddings) if doc_embeddings is not None else None
        self._model.fit(
            data, iters=iters, covariates=covariates, prior_offset=offset, **fit_kwargs
        )
        return self

    @property
    def model(self):
        """The underlying fitted :class:`~topica.KeyATM`."""
        return self._model

    def __getattr__(self, name):
        model = self.__dict__.get("_model")
        if model is None:
            raise AttributeError(name)
        return getattr(model, name)

    def __repr__(self) -> str:
        anchored = sum(1 for a in self._anchors if a is not None)
        return (
            f"EmbeddingKeyATM(num_topics={self.num_topics}, "
            f"{anchored} topics anchored, doc_anchor={self.doc_anchor})"
        )
