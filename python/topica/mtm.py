"""Mechanistic Topic Models: the featurization layer (issue #575).

A Mechanistic Topic Model (MTM) is not one model but a *recipe*: replace the
bag of words with a bag of **sparse-autoencoder features**, then run an ordinary
topic model over it. Zheng, Beltran-Velez, Karlekar, Shi, Nazaret, Mallik, Feder
and Blei, "Model Directions, Not Words: Mechanistic Topic Models Using Sparse
Autoencoders" (2025, arXiv:2507.23220), §4.1.

You run an SAE over your corpus; this module turns its activations into the two
things a topic model needs:

- :func:`featurize` — the paper's eq. 5, ``c̃[d,i] = Σ_j 1{α_i(a_dj) > q_i}``:
  count, per document, how many tokens fire feature ``i`` above its threshold.
  The result goes to :meth:`topica.Corpus.from_matrix` and then to any
  count-based model (LDA, NMF, …).
- :func:`document_embeddings` — the paper's eq. 14,
  ``ẽ_d = (1/N_tok) Σ_i c̃[d,i] w_i``: a document vector built by weighting each
  SAE **decoder direction** by how often its feature fired. That goes to
  :class:`topica.BERTopic` as the embedding matrix.

:func:`feature_thresholds` is a convenience for the ``q_i``, with the caveat
recorded in its docstring: the paper takes them from the SAE's *training* data,
not from your corpus.

Nothing here is model-specific and nothing here needs a GPU — the SAE forward
pass is yours to run, in whatever framework you already use.

Why features instead of words: an SAE feature is a direction in the language
model's residual stream, so a topic over features is a claim about what the
*model* represents, and it can be intervened on (steered) rather than only read.
That is the paper's argument for the whole approach.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "FeatureCounts",
    "feature_thresholds",
    "featurize",
    "document_embeddings",
    "MechanisticLDA",
    "MechanisticBERTopic",
]


def _as_document_list(activations):
    """Normalize the accepted activation shapes to a list of 2-D (tokens, features) arrays."""
    arr = None
    if isinstance(activations, np.ndarray):
        arr = activations
    elif hasattr(activations, "detach") and hasattr(activations, "cpu"):
        # A torch tensor, without importing torch.
        arr = np.asarray(activations.detach().cpu())

    if arr is not None:
        if arr.ndim == 3:
            return [np.asarray(row) for row in arr]
        if arr.ndim == 2:
            return [arr]
        raise ValueError(
            f"activations must be 2-D (tokens, features) for one document or 3-D "
            f"(documents, tokens, features); got {arr.ndim}-D"
        )

    docs = []
    for i, doc in enumerate(activations):
        if hasattr(doc, "detach") and hasattr(doc, "cpu"):
            doc = doc.detach().cpu()
        doc = np.asarray(doc)
        if doc.ndim != 2:
            raise ValueError(
                f"each document's activations must be 2-D (tokens, features); "
                f"document {i} is {doc.ndim}-D"
            )
        docs.append(doc)
    if not docs:
        raise ValueError("activations is empty; expected at least one document")
    return docs


@dataclass(frozen=True)
class FeatureCounts:
    """What :func:`featurize` produces: the count matrix and the true token counts.

    Attributes
    ----------
    counts : ``(num_docs, num_features)`` int64 — ``c̃`` of eq. 5. Entry ``[d, i]``
        is the number of tokens of document ``d`` on which feature ``i`` fired
        above its threshold.
    n_tokens : ``(num_docs,)`` int64 — the number of tokens each count was taken
        over, **after** ``drop_first_token``. This is *not* ``counts.sum(axis=1)``:
        a token may fire many features or none, so the row sum counts feature
        activations (the paper's ``N_sae``), while this counts tokens (``N_tok``).
        The distinction is load-bearing — eq. 8 uses ``N_sae``, eqs. 12 and 14 use
        ``N_tok`` — so both travel together rather than one being recomputed from
        the other.
    """

    counts: np.ndarray
    n_tokens: np.ndarray

    @property
    def num_docs(self) -> int:
        return int(self.counts.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.counts.shape[1])

    @property
    def n_sae(self) -> np.ndarray:
        """``N_sae``: total feature activations per document, i.e. the row sum.

        Provided so the contrast with :attr:`n_tokens` is explicit at the call
        site rather than implied.
        """
        return self.counts.sum(axis=1)

    def to_corpus(self, feature_names=None, **kwargs):
        """Hand these counts to :meth:`topica.Corpus.from_matrix`.

        ``n_tokens`` is forwarded automatically; ``feature_names`` and any other
        keyword (notably ``max_doc_fraction``, the paper's App. A.1 ubiquitous-feature
        filter) pass straight through. ``feature_names`` should usually be the SAE
        feature ids as strings, so a fitted model's top words are readable ids.
        """
        from . import Corpus

        return Corpus.from_matrix(
            self.counts,
            feature_names=feature_names,
            n_tokens=[int(n) for n in self.n_tokens],
            **kwargs,
        )


def feature_thresholds(activations, q: float = 0.8, *, drop_first_token: bool = True):
    """Per-feature activation quantiles ``q_i``, pooled over every token given.

    Returns a ``(num_features,)`` array: the ``q``-quantile of feature ``i``'s
    activation across all tokens in ``activations``.

    .. warning::
       The paper computes ``q_i`` on the **SAE's training data**, not on the
       corpus being modeled (the reference reads them from a precomputed
       feature-description file). Pooling over your own corpus, as this function
       does, makes the threshold corpus-dependent: it is then a within-corpus
       quantile, so roughly the same fraction of *your* tokens clears it whatever
       the SAE thought was unusual. That is a defensible fallback when you have no
       training-data quantiles, and it is a different estimator from the paper's.
       Prefer passing training-data quantiles to :func:`featurize` directly when
       you have them; use this to approximate them, and say which you used.

    ``q=0.8`` is the reference's configured value (``discrete_quantile=0.8``,
    matching the paper's 80th percentile), not the reference class default of 0.9.

    Memory is proportional to the total token count: every token is pooled to take
    an exact quantile. For a large corpus, compute the quantiles yourself over a
    sample and pass the array to :func:`featurize`.
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0.0, 1.0]; got {q}")
    docs = _as_document_list(activations)
    n_features = docs[0].shape[1]
    rows = []
    for i, doc in enumerate(docs):
        if doc.shape[1] != n_features:
            raise ValueError(
                f"every document must have the same number of features; document 0 has "
                f"{n_features} but document {i} has {doc.shape[1]}"
            )
        rows.append(doc[1:] if drop_first_token else doc)
    pooled = np.concatenate(rows, axis=0) if len(rows) > 1 else rows[0]
    if pooled.shape[0] == 0:
        raise ValueError(
            "no tokens left to take quantiles over "
            "(every document is empty, or has one token with drop_first_token=True)"
        )
    return np.quantile(np.asarray(pooled, dtype=np.float64), q, axis=0)


def featurize(activations, thresholds, *, drop_first_token: bool = True) -> FeatureCounts:
    """Count thresholded SAE feature activations per document (paper eq. 5).

    ``c̃[d, i] = Σ_j 1{α_i(a_dj) > q_i}`` — for each document, how many of its
    tokens fired feature ``i`` **strictly above** that feature's threshold.

    Parameters
    ----------
    activations
        The SAE's encoded activations. Either one 3-D ``(documents, tokens,
        features)`` array, or — the usual case, since documents differ in length —
        any iterable of 2-D ``(tokens, features)`` arrays, one per document. A
        single 2-D array is read as one document. Torch tensors are accepted.
    thresholds
        ``(num_features,)`` of ``q_i``, or a single float applied to every feature.
        See :func:`feature_thresholds` for where these should come from.
    drop_first_token
        Drop each document's first token before counting (default ``True``,
        matching the reference: the SAE was not trained on the BOS position, so its
        activations there are out of distribution). This also shortens ``n_tokens``,
        which is what the reference does.

    Returns
    -------
    :class:`FeatureCounts` — ``counts`` for the topic model, ``n_tokens`` for the
    normalizations that need a true token count.

    Notes
    -----
    The comparison is strictly ``>``, not ``>=``. With a quantile threshold and
    the many exactly-zero activations an SAE produces, ``>=`` would count every
    silent token as a firing.
    """
    docs = _as_document_list(activations)
    n_features = docs[0].shape[1]

    thr = np.asarray(thresholds, dtype=np.float64)
    if thr.ndim == 0:
        thr = np.full(n_features, float(thr))
    elif thr.ndim != 1:
        raise ValueError(f"thresholds must be a scalar or 1-D array; got {thr.ndim}-D")
    elif thr.shape[0] != n_features:
        raise ValueError(
            f"thresholds has {thr.shape[0]} entries but the activations have "
            f"{n_features} features"
        )

    counts = np.zeros((len(docs), n_features), dtype=np.int64)
    n_tokens = np.zeros(len(docs), dtype=np.int64)
    for d, doc in enumerate(docs):
        if doc.shape[1] != n_features:
            raise ValueError(
                f"every document must have the same number of features; document 0 has "
                f"{n_features} but document {d} has {doc.shape[1]}"
            )
        tokens = doc[1:] if drop_first_token else doc
        n_tokens[d] = tokens.shape[0]
        if tokens.shape[0]:
            counts[d] = (np.asarray(tokens, dtype=np.float64) > thr).sum(axis=0)
    return FeatureCounts(counts=counts, n_tokens=n_tokens)


def document_embeddings(
    weights,
    directions,
    *,
    n_tokens=None,
    normalize: bool = False,
):
    """Document vectors in the language model's space (paper eq. 14).

    ``ẽ_d = (1/N_tok) Σ_i c̃[d, i] · w_i`` — weight each SAE feature's **decoder
    direction** by how often that feature fired, and average over the document's
    tokens. The result lives in the model's residual-stream space (``d_model``
    dimensions), not in feature space, so it is a dense embedding you can hand to
    :class:`topica.BERTopic` or any other embedding-based model.

    Parameters
    ----------
    weights
        ``(num_docs, num_features)`` of per-feature weights. Either the thresholded
        counts ``c̃`` from :func:`featurize` (eq. 14 as written), or the *mean
        continuous* activations if you have them — see Notes.
    directions
        ``(num_features, d_model)`` — the SAE decoder rows ``w_i`` for the features
        in ``weights``, **in the same column order**. If you filtered features (via
        ``max_doc_fraction`` on the corpus, or your own selection), slice the decoder
        matrix with the same indices; :attr:`topica.Corpus.kept_features` reports
        them.
    n_tokens
        ``(num_docs,)`` true token counts, i.e. :attr:`FeatureCounts.n_tokens`. When
        given, each row is divided by it. Omit to skip the division. Do **not** pass
        the row sum of ``weights``: that is ``N_sae``, a count of feature activations,
        and eq. 14 asks for tokens. Documents with ``n_tokens == 0`` yield a zero row
        rather than a division by zero.
    normalize
        L2-normalize each row afterwards (default ``False``). Irrelevant under cosine
        distance, which is BERTopic's default metric; useful under euclidean.

    Notes
    -----
    ``1/N_tok`` is a positive per-document scalar, so under **cosine** distance —
    the default for both the reference's UMAP and topica's — it cancels and
    ``n_tokens`` changes nothing. It matters under euclidean, and it matters if you
    use these embeddings for anything else.

    The reference implementation's default is not eq. 14 as written: it weights the
    same decoder directions by the *mean continuous* activations rather than by
    thresholded counts (``continuous_use_counts_for_embeddings=False``), which is
    the same computation with a different ``weights`` matrix — pass the mean
    activations here and leave ``n_tokens`` unset, since the mean already divides by
    the token count. The reference's hyperparameter sweep tunes that binary choice,
    so neither setting is canonically "the" method.
    """
    w = np.asarray(weights, dtype=np.float64)
    d = np.asarray(directions, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError(f"weights must be 2-D (documents, features); got {w.ndim}-D")
    if d.ndim != 2:
        raise ValueError(f"directions must be 2-D (features, d_model); got {d.ndim}-D")
    if w.shape[1] != d.shape[0]:
        raise ValueError(
            f"weights has {w.shape[1]} features but directions has {d.shape[0]}; "
            "slice the decoder matrix to the features you kept "
            "(see Corpus.kept_features)"
        )

    if n_tokens is not None:
        n = np.asarray(n_tokens, dtype=np.float64)
        if n.shape != (w.shape[0],):
            raise ValueError(
                f"n_tokens has shape {n.shape} but weights has {w.shape[0]} documents"
            )
        if (n < 0).any():
            raise ValueError("n_tokens must be non-negative")
        w = w / np.where(n > 0, n, 1.0)[:, None]

    emb = w @ d
    if normalize:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.where(norms > 0, norms, 1.0)
    return emb


# ---------------------------------------------------------------------------
# The models: LDA and BERTopic, run over features instead of words
# ---------------------------------------------------------------------------

_EXPERIMENTAL_NOTE = (
    "{name} is experimental: it is the {paper} recipe run over topica's "
    "reference-validated {inner} core, but topica has no end-to-end parity check "
    "against the authors' implementation yet (that needs their SAE and Gemma "
    "activations). Enable experimental models with `topica.enable_experimental()` "
    "or set TOPICA_EXPERIMENTAL=1. Experimental models may change or be removed "
    "without a deprecation cycle."
)


def _counts_to_corpus(data, feature_names, doc_names, doc_labels, max_doc_fraction):
    """Accept whatever shape of feature counts the caller has, return a Corpus.

    ``data`` may already be a :class:`topica.Corpus` (built by the caller, e.g. to
    apply their own feature filter), a :class:`FeatureCounts`, or a bare
    ``(num_docs, num_features)`` count matrix. The last two are routed through
    :meth:`topica.Corpus.from_matrix`, which is what preserves the feature-column
    contract.
    """
    from . import Corpus

    if isinstance(data, Corpus):
        if feature_names is not None or doc_names is not None or doc_labels is not None:
            raise ValueError(
                "feature_names / doc_names / doc_labels apply only when fitting on a "
                "count matrix; the Corpus already carries them"
            )
        return data
    if isinstance(data, FeatureCounts):
        return data.to_corpus(
            feature_names=feature_names,
            doc_names=doc_names,
            doc_labels=doc_labels,
            max_doc_fraction=max_doc_fraction,
        )
    counts = np.asarray(data)
    if counts.ndim != 2:
        raise ValueError(
            "data must be a Corpus, a FeatureCounts, or a 2-D (documents, features) "
            f"count matrix; got a {counts.ndim}-D array"
        )
    return Corpus.from_matrix(
        counts.astype(np.int64, copy=False),
        feature_names=feature_names,
        doc_names=doc_names,
        doc_labels=doc_labels,
        max_doc_fraction=max_doc_fraction,
    )


class _MechanisticModel:
    """Shared plumbing: forward the fitted surface to the inner topica model."""

    _inner = None

    def _require_fit(self):
        if self._inner is None:
            raise RuntimeError("Model is not fitted")
        return self._inner

    @property
    def corpus(self):
        """The feature :class:`topica.Corpus` this was fit on.

        Its ``vocabulary`` is the feature names and its
        :attr:`~topica.Corpus.kept_features` reports which of the caller's original
        feature columns survived ``max_doc_fraction`` — the indices you need to
        slice an SAE decoder matrix to match.
        """
        if self._corpus is None:
            raise RuntimeError("Model is not fitted")
        return self._corpus

    @property
    def topic_word(self):
        return self._require_fit().topic_word

    @property
    def doc_topic(self):
        return self._require_fit().doc_topic

    @property
    def vocabulary(self):
        """The feature names, in ``topic_word`` column order."""
        return self._require_fit().vocabulary

    @property
    def topic_names(self):
        return self._require_fit().topic_names

    @topic_names.setter
    def topic_names(self, value):
        self._require_fit().topic_names = value

    @property
    def fit_history(self):
        return self._require_fit().fit_history

    @property
    def converged(self):
        return self._require_fit().converged

    def top_words(self, n: int = 10, *, topic=None):
        """The features a topic puts most weight on, as ``(name, weight)`` pairs.

        These are feature *ids*, not words. Reading them as a topic label needs the
        SAE's feature descriptions (the paper uses Neuronpedia's, or an LLM
        summary of a feature's top-activating text) — the identifiers alone are
        not interpretable, which is the one real ergonomic cost of modeling
        features instead of words.
        """
        return self._require_fit().top_words(n, topic=topic)

    def coherence(self, n: int = 10):
        return self._require_fit().coherence(n)

    def save(self, path: str) -> None:
        """Save the inner fitted model. See the class docstring for what is lost."""
        self._require_fit().save(path)


class MechanisticLDA(_MechanisticModel):
    """mLDA: LDA over SAE feature counts (Zheng et al. 2025, §4.2.1). Experimental.

    Ordinary collapsed-Gibbs LDA, with the bag of words replaced by the bag of
    thresholded SAE features from :func:`featurize`. There is no new inference
    here and none is claimed: the paper's contribution at this layer is the
    *representation*, so this runs topica's MALLET-parity :class:`topica.LDA` over
    a feature corpus and carries the reference implementation's priors as defaults.

    Defaults follow the reference (``mallet_lda_bof_model.py``), not topica's
    usual LDA defaults: ``alpha_sum=5.0`` (a *sum* over topics, MALLET's
    convention), ``beta=0.01``, ``optimize_interval=10``. ``max_doc_fraction=0.9``
    is App. A.1's ubiquitous-feature filter, which the paper calls crucial — a few
    SAE features fire on nearly every token and swamp every topic if kept.

    Why this is experimental: the recipe and the priors are checked against the
    reference source, but topica has no end-to-end numeric parity run against the
    authors' pipeline, which needs their Gemma-2 activations and SAE.

    Examples
    --------
    ::

        counts = topica.mtm.featurize(activations, thresholds)
        m = topica.mtm.MechanisticLDA(50).fit(counts, feature_names=feature_ids)
        m.top_words(10, topic=0)     # -> SAE feature ids, not words
    """

    def __init__(
        self,
        num_topics: int,
        *,
        alpha_sum: float = 5.0,
        beta: float = 0.01,
        optimize_interval: int = 10,
        use_symmetric_alpha: bool = False,
        burn_in: int = 200,
        max_doc_fraction: float = 0.9,
        seed: int = 13,
        num_threads: int = 1,
        sampler: str = "sparse",
    ) -> None:
        from . import experimental_enabled

        if not experimental_enabled():
            raise RuntimeError(
                _EXPERIMENTAL_NOTE.format(
                    name="MechanisticLDA", paper="Mechanistic Topic Model", inner="LDA"
                )
            )
        if num_topics < 1:
            raise ValueError("num_topics must be >= 1")
        self._num_topics = num_topics
        self._alpha_sum = alpha_sum
        self._beta = beta
        self._optimize_interval = optimize_interval
        self._use_symmetric_alpha = use_symmetric_alpha
        self._burn_in = burn_in
        self._max_doc_fraction = max_doc_fraction
        self._seed = seed
        self._num_threads = num_threads
        self._sampler = sampler
        self._inner = None
        self._corpus = None

    @property
    def num_topics(self) -> int:
        return self._num_topics

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict, keyword-named
        to match ``__init__`` (issue #400)."""
        return {
            "num_topics": self._num_topics,
            "alpha_sum": self._alpha_sum,
            "beta": self._beta,
            "optimize_interval": self._optimize_interval,
            "use_symmetric_alpha": self._use_symmetric_alpha,
            "burn_in": self._burn_in,
            "max_doc_fraction": self._max_doc_fraction,
            "seed": self._seed,
            "num_threads": self._num_threads,
            "sampler": self._sampler,
        }

    def fit(
        self,
        data,
        *,
        feature_names=None,
        doc_names=None,
        doc_labels=None,
        iters: int = 1000,
        **kwargs,
    ) -> "MechanisticLDA":
        """Fit on feature counts.

        ``data`` is a :class:`FeatureCounts` from :func:`featurize`, a
        ``(num_docs, num_features)`` count matrix, or a :class:`topica.Corpus` you
        built yourself with :meth:`topica.Corpus.from_matrix`. ``feature_names``
        (usually the SAE feature ids as strings) applies to the first two; a Corpus
        already carries its own. Remaining keywords go to :meth:`topica.LDA.fit`.
        """
        from . import LDA

        self._corpus = _counts_to_corpus(
            data, feature_names, doc_names, doc_labels, self._max_doc_fraction
        )
        self._inner = LDA(
            self._num_topics,
            alpha_sum=self._alpha_sum,
            beta=self._beta,
            optimize_interval=self._optimize_interval,
            use_symmetric_alpha=self._use_symmetric_alpha,
            burn_in=self._burn_in,
            seed=self._seed,
            num_threads=self._num_threads,
            sampler=self._sampler,
        )
        self._inner.fit(self._corpus, iters=iters, **kwargs)
        return self

    def transform(self, data, **kwargs):
        """Infer topic proportions for held-out feature counts.

        ``data`` takes the same shapes as :meth:`fit`. A bare matrix or
        :class:`FeatureCounts` must already be in this model's feature-column
        space: pass counts sliced to :attr:`~topica.Corpus.kept_features`, or a
        Corpus built with the same feature names.
        """
        inner = self._require_fit()
        corpus = _counts_to_corpus(data, None, None, None, 1.0)
        return inner.transform(corpus, **kwargs)

    # Dirichlet-family surface, forwarded from the inner LDA.
    @property
    def alpha(self):
        return self._require_fit().alpha

    @property
    def beta(self) -> float:
        return self._require_fit().beta

    @property
    def theta_draws(self):
        return self._require_fit().theta_draws

    @property
    def doc_lengths(self):
        """Per-document totals, in ``doc_topic`` row order.

        These are **feature activation** totals (the paper's ``N_sae``), not token
        counts: this is a corpus of features, so a "token" here is one feature
        firing. The true token counts are :attr:`FeatureCounts.n_tokens`, carried
        separately on the corpus as :attr:`topica.Corpus.n_tokens`.
        """
        return self._require_fit().doc_lengths

    @property
    def doc_names(self):
        return self._require_fit().doc_names

    @staticmethod
    def load(path: str):
        """Load a previously saved fit.

        Returns a plain :class:`topica.LDA` — the saved file is the inner model, so
        the mechanistic wrapper's settings (``max_doc_fraction``, the corpus's
        ``kept_features`` and ``n_tokens``) are not restored. Keep those alongside
        it if you need to map topic columns back to SAE feature ids.
        """
        from . import LDA

        return LDA.load(path)

    def __repr__(self) -> str:
        state = "fitted" if self._inner is not None else "unfitted"
        return f"MechanisticLDA(num_topics={self._num_topics}, {state})"


class MechanisticBERTopic(_MechanisticModel):
    """mBERTopic: BERTopic over SAE feature counts and directions (§4.2.3). Experimental.

    BERTopic with both of its inputs made mechanistic. Documents are embedded by
    weighting SAE **decoder directions** by feature activations (eq. 14,
    :func:`document_embeddings`), so the reduce-and-cluster step runs in the
    language model's own residual-stream space; and the c-TF-IDF that names each
    cluster is computed over SAE features rather than words (eq. 15), so topics
    come out as feature ids.

    Defaults follow the reference (``mbertopic.py``) where they differ from
    topica's BERTopic: ``min_cluster_size=10`` (the reference's ``min_topic_size``;
    topica's own default is 15) and ``n_neighbors=15``, ``n_components=5``,
    ``min_dist=0.0``, cosine UMAP — all as upstream.

    Known divergences from the reference, none of them silent:

    - **``nr_topics`` counts differently.** The reference passes its ``num_topics``
      straight to upstream BERTopic and then treats the last slot as the ``-1``
      outlier topic, so its K *includes* the outlier cluster; topica's ``nr_topics``
      counts real topics only. To ask for the reference's K, pass ``K - 1``.
    - **c-TF-IDF excludes outliers.** topica computes tf/df over clustered
      documents only, where upstream includes the ``-1`` cluster; and topica uses
      raw ``tf`` — literally eq. 15 — where upstream L1-normalizes it first.
    - **``doc_topic`` is a different estimand.** The reference reports HDBSCAN soft
      membership with outlier mass as a final column; topica reports its
      sliding-window c-TF-IDF approximation.

    Why this is experimental: the recipe is checked against the reference source,
    but topica has no end-to-end numeric parity run against the authors' pipeline,
    and the divergences above mean topic-by-topic agreement is not the right bar
    even in principle — cluster-label agreement is.

    Examples
    --------
    ::

        counts = topica.mtm.featurize(activations, thresholds)
        m = topica.mtm.MechanisticBERTopic().fit(counts, directions=sae.W_dec)
    """

    def __init__(
        self,
        *,
        min_cluster_size: int = 10,
        min_samples=None,
        nr_topics=None,
        n_neighbors: int = 15,
        n_components: int = 5,
        reducer: str = "umap",
        max_doc_fraction: float = 0.9,
        seed: int = 13,
    ) -> None:
        from . import experimental_enabled

        if not experimental_enabled():
            raise RuntimeError(
                _EXPERIMENTAL_NOTE.format(
                    name="MechanisticBERTopic",
                    paper="Mechanistic Topic Model",
                    inner="BERTopic",
                )
            )
        self._min_cluster_size = min_cluster_size
        self._min_samples = min_samples
        self._nr_topics = nr_topics
        self._n_neighbors = n_neighbors
        self._n_components = n_components
        self._reducer = reducer
        self._max_doc_fraction = max_doc_fraction
        self._seed = seed
        self._inner = None
        self._corpus = None
        self._doc_embeddings = None

    @property
    def num_topics(self) -> int:
        """The discovered topic count. Available only after :meth:`fit`."""
        return self._require_fit().num_topics

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict, keyword-named
        to match ``__init__`` (issue #400)."""
        return {
            "min_cluster_size": self._min_cluster_size,
            "min_samples": self._min_samples,
            "nr_topics": self._nr_topics,
            "n_neighbors": self._n_neighbors,
            "n_components": self._n_components,
            "reducer": self._reducer,
            "max_doc_fraction": self._max_doc_fraction,
            "seed": self._seed,
        }

    @property
    def doc_embeddings(self):
        """The eq. 14 document embeddings this was fit on, ``(num_docs, d_model)``."""
        if self._doc_embeddings is None:
            raise RuntimeError("Model is not fitted")
        return self._doc_embeddings

    @property
    def labels(self):
        """Hard cluster assignment per document; ``-1`` is the outlier cluster."""
        return self._require_fit().labels

    def fit(
        self,
        data,
        doc_embeddings=None,
        *,
        directions=None,
        n_tokens=None,
        feature_names=None,
        doc_names=None,
        doc_labels=None,
    ) -> "MechanisticBERTopic":
        """Fit on feature counts plus one embedding per document.

        ``data`` takes the same shapes as :meth:`MechanisticLDA.fit`. Supply the
        embeddings one of two ways:

        - ``directions`` — the SAE decoder matrix, ``(num_features, d_model)``, in
          the caller's original feature-column order. Embeddings are then built by
          eq. 14, sliced to whichever features survived ``max_doc_fraction`` and
          divided by ``n_tokens`` when those are known.
        - ``doc_embeddings`` — a ``(num_docs, d_model)`` matrix you built yourself,
          for instance with :func:`document_embeddings` from *mean continuous*
          activations, which is what the reference implementation does by default.

        Pass exactly one.
        """
        from . import BERTopic

        if (directions is None) == (doc_embeddings is None):
            raise ValueError(
                "pass exactly one of directions= (build eq. 14 embeddings from the "
                "SAE decoder) or doc_embeddings= (supply them yourself)"
            )

        self._corpus = _counts_to_corpus(
            data, feature_names, doc_names, doc_labels, self._max_doc_fraction
        )

        if directions is None:
            emb = np.asarray(doc_embeddings, dtype=np.float64)
            if emb.ndim != 2 or emb.shape[0] != self._corpus.num_docs:
                raise ValueError(
                    f"doc_embeddings must be (num_docs, d_model) with "
                    f"{self._corpus.num_docs} rows; got shape {emb.shape}"
                )
        else:
            emb = self._build_embeddings(data, directions, n_tokens)
        self._doc_embeddings = emb

        self._inner = BERTopic(
            min_cluster_size=self._min_cluster_size,
            min_samples=self._min_samples,
            nr_topics=self._nr_topics,
            n_neighbors=self._n_neighbors,
            n_components=self._n_components,
            reducer=self._reducer,
            seed=self._seed,
        )
        self._inner.fit(self._corpus, emb)
        return self

    def _build_embeddings(self, data, directions, n_tokens):
        """eq. 14 from the caller's counts, aligned to the features the corpus kept."""
        if isinstance(data, FeatureCounts):
            weights, default_n = data.counts, data.n_tokens
        else:
            from . import Corpus

            if isinstance(data, Corpus):
                raise ValueError(
                    "directions= needs the count matrix to weight them by; fit on a "
                    "FeatureCounts or a count matrix, or pass doc_embeddings= built "
                    "with topica.mtm.document_embeddings"
                )
            weights, default_n = np.asarray(data), None

        d = np.asarray(directions, dtype=np.float64)
        weights = np.asarray(weights)
        kept = self._corpus.kept_features
        if kept is None:
            kept = list(range(weights.shape[1]))
        if d.shape[0] == weights.shape[1]:
            # The whole decoder: slice it the way the corpus was sliced.
            d = d[kept]
        elif d.shape[0] != len(kept):
            raise ValueError(
                f"directions has {d.shape[0]} rows; expected {weights.shape[1]} (the "
                f"full feature width) or {len(kept)} (already sliced to the features "
                "the corpus kept -- see Corpus.kept_features)"
            )
        weights = weights[:, kept]
        if n_tokens is None:
            n_tokens = default_n
        return document_embeddings(weights, d, n_tokens=n_tokens)

    def transform(self, data, doc_embeddings=None):
        """Approximate topic distribution for held-out feature counts."""
        inner = self._require_fit()
        corpus = _counts_to_corpus(data, None, None, None, 1.0)
        return inner.transform(corpus, doc_embeddings)

    @staticmethod
    def load(path: str):
        """Load a previously saved fit.

        Returns a plain :class:`topica.BERTopic` — the saved file is the inner
        model, so the wrapper's settings and the embeddings are not restored.
        """
        from . import BERTopic

        return BERTopic.load(path)

    def __repr__(self) -> str:
        if self._inner is None:
            return "MechanisticBERTopic(unfitted)"
        return f"MechanisticBERTopic(num_topics={self._inner.num_topics}, fitted)"
