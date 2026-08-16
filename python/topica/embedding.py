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

    topica.enable_experimental()  # EmbeddingLDA is experimental (see note below)
    vocab = sorted({w for d in docs for w in d})
    emb = SentenceTransformer("all-MiniLM-L6-v2").encode(vocab)
    model = topica.embeddings.EmbeddingLDA(num_topics=10, embeddings=emb, vocabulary=vocab)
    model.fit(docs, iters=1000)
    print(model.top_words(8))

Validation status (experimental). EmbeddingLDA is a topica original: it has no
published paper and no external reference implementation, so it is validated by a
*planted-recovery / determinism* gold only (``parity/embeddinglda_gold.py``), not
by cross-implementation parity. That gold cannot distinguish EmbeddingLDA from
plain LDA — on the planted corpus, plain LDA and even shuffled/random embeddings
score the same block purity — and on real labeled text (20 Newsgroups) its label
recovery sits *below* plain LDA, so it is sound but not demonstrably superior. It
is therefore gated behind :func:`~topica.enable_experimental` and may change or be
removed without a deprecation cycle (issue #660).
"""

from __future__ import annotations

import hashlib
import json
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
    seed: int = 13,
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
    ``FASTopic``), :func:`embedding_seeds`, and :func:`embedding_coherence` (pass
    ``llm_embed(model.vocabulary)`` as its word table) all take embeddings you
    supply; this is one way to produce them. ``model`` names any embedding model
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

    .. note::
       **Experimental** (issue #660). EmbeddingLDA is a topica original with no
       published paper and no external reference; its gold is planted-recovery
       only and cannot tell it apart from plain LDA, whose label recovery it does
       not beat on real text. It is gated behind
       :func:`~topica.enable_experimental` and may change without a deprecation
       cycle. The inference core it delegates to (:class:`~topica.SeededLDA`) is
       itself validated; what is unproven is the embedding-seeding *benefit*.

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

    .. note::
       The ``vocabulary=`` you pass here aligns the **embedding** rows; it is not
       the fitted output vocabulary. After :meth:`fit`, ``topic_word`` columns are
       indexed by ``model.vocabulary``, the vocabulary the underlying SeededLDA
       rebuilds from the corpus you fit on, which is typically a *subset in a
       different order* (only words that survived tokenisation/pruning). Do not
       index ``topic_word`` (or build coherence) with the ``vocabulary=`` you
       passed; use ``model.vocabulary``, or the helpers that already pair them:
       ``top_words()`` and ``label_topics(model.topic_word, model.vocabulary)``.

    No embedder of your own? :func:`~topica.embeddings.llm_embed` builds the ``embeddings``
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
        in its topic. Higher anchors the topic-word side harder. Default ``0.1``
        (was ``1.0``). The embedding-cluster seed words are semantically grouped but
        do not necessarily co-occur, so anchoring them hard pulls topics away from
        corpus co-occurrence and lowers topic coherence, increasingly at larger K:
        at ``1.0`` (100 pseudocounts/seed) coherence fell well below plain LDA on
        the 20-newsgroup benchmark (#663). Coherence rises monotonically as weight
        falls, and the effect on document-mixture (theta) recovery is small, so the
        light ``0.1`` default recovers most of the lost coherence at little cost.
        Raise it toward ``1.0`` only when you want the embedding grouping to
        dominate the data.
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
        weight: float = 0.1,
        doc_anchor: float = 1.0,
        alpha: float = 0.1,
        beta: float = 0.01,
        seed: int = 13,
    ) -> None:
        from . import SeededLDA, experimental_enabled

        if not experimental_enabled():
            raise RuntimeError(
                "EmbeddingLDA is experimental: it is a topica original with no "
                "published paper and no reference-implementation parity (its gold "
                "is planted-recovery only, and cannot distinguish it from plain "
                "LDA). Enable experimental models with `topica.enable_experimental()` "
                "or set the environment variable TOPICA_EXPERIMENTAL=1. Experimental "
                "models may change or be removed without a deprecation cycle."
            )

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
        # EmbeddingLDA anchors on embedding-derived seeds, not on the seededlda
        # package's corpus-frequency prior, so it pins the uniform scheme: every
        # seed word gets a flat ``weight * 100`` pseudocount and is anchored at
        # init (its designed behavior, independent of SeededLDA's default).
        self._model = SeededLDA(
            self.seeds, alpha=alpha, beta=beta, weight=weight,
            seed_prior="uniform", seed=seed,
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

    def fit(
        self,
        data,
        *,
        doc_embeddings=None,
        iters: int = 1000,
        convergence_tol: float = 0.0,
        check_every: int = 10,
    ) -> "EmbeddingLDA":
        """Fit on ``data`` (a Corpus or list of token lists). If ``doc_embeddings``
        is given (one row per document, same embedding space as the vocabulary),
        each document's topic mixture is biased toward the topics its embedding is
        nearest, as a prior the sampler can still override. ``iters`` is the number
        of Gibbs sweeps for the underlying SeededLDA fit.

        Convergence signal. Every ``check_every`` sweeps the collapsed
        log-likelihood is recorded, so after fitting :attr:`fit_history` holds the
        ``(iteration, log_likelihood)`` trace to eyeball (or plot) for a plateau.
        ``convergence_tol`` (default ``0.0``, off) enables early stopping: once the
        relative change in that log-likelihood falls below it the sweep loop stops
        and :attr:`converged` is ``True``. With the default ``0.0`` the fit always
        runs the full ``iters`` and :attr:`converged` stays ``False`` (it means
        "early-stopping never triggered", not "did not mix"); set e.g.
        ``convergence_tol=1e-4`` to get a genuine verdict and a shorter fit."""
        prior = self.document_topic_prior(doc_embeddings) if doc_embeddings is not None else None
        self._model.fit(
            data,
            iters=iters,
            doc_topic_prior=prior,
            convergence_tol=convergence_tol,
            check_every=check_every,
        )
        return self

    @property
    def model(self):
        """The underlying fitted :class:`~topica.SeededLDA`."""
        return self._model

    # Persistence. EmbeddingLDA is a Python-layer composition, so ``__getattr__``
    # would delegate ``save``/``load`` to the underlying SeededLDA and silently
    # drop the embedding layer (centroids, seeds, doc-anchor). We define our own
    # pair: the SeededLDA core is saved by its own serializer, and the
    # embedding-layer state goes to a sidecar ``<path>.embedding.npz`` so a
    # fresh save -> load round-trip reproduces ``document_topic_prior`` and the
    # full delegated surface.
    _SIDE_SUFFIX = ".embedding.npz"

    def save(self, path) -> None:
        """Save the model. Writes the SeededLDA core to ``path`` and the
        embedding-layer state (centroids, seeds, and the topic-word/doc-anchor
        hyperparameters) to a companion ``<path>.embedding.npz``. Both files are
        needed to reload; :meth:`load` reads them together."""
        self._model.save(str(path))
        np.savez(
            str(path) + self._SIDE_SUFFIX,
            centroids=np.asarray(self._centroids, dtype=float),
            num_topics=np.asarray(self.num_topics),
            top_m=np.asarray(self.top_m),
            alpha=np.asarray(self.alpha, dtype=float),
            doc_anchor=np.asarray(self.doc_anchor, dtype=float),
            seeds=np.asarray(json.dumps(self.seeds)),
        )

    @staticmethod
    def load(path) -> "EmbeddingLDA":
        """Reload a model saved by :meth:`save`, restoring both the SeededLDA core
        and the embedding layer so ``document_topic_prior`` and the delegated
        surface work exactly as before the save."""
        from . import SeededLDA

        obj = EmbeddingLDA.__new__(EmbeddingLDA)
        obj._model = SeededLDA.load(str(path))
        with np.load(str(path) + EmbeddingLDA._SIDE_SUFFIX, allow_pickle=False) as data:
            obj._centroids = data["centroids"]
            obj.num_topics = int(data["num_topics"])
            obj.top_m = int(data["top_m"])
            obj.alpha = float(data["alpha"])
            obj.doc_anchor = float(data["doc_anchor"])
            obj.seeds = json.loads(str(data["seeds"]))
        return obj

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
