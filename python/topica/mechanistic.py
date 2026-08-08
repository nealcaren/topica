"""Mechanistic Topic Model (mLDA) — LDA over sparse-autoencoder features.

An implementation of the Mechanistic Topic Model of Zheng, Beltran-Velez,
Karlekar, Shi, Nazaret, Mallik, Feder & Blei, *"Model Directions, Not Words:
Mechanistic Topic Models Using Sparse Autoencoders"* (2025; arXiv:2507.23220),
topica issue #575.

A Mechanistic Topic Model represents a document not as a bag of *words* but as a
bag of *interpretable features* from a sparse autoencoder (SAE) trained on an
LLM's activations, and then fits a topic model over that feature space. The
paper's **mLDA** variant is, mathematically, ordinary LDA with SAE features
substituted for words: the document x feature activation matrix is modeled as a
multinomial mixture of topics, each topic a distribution over features. topica
therefore implements mLDA as a thin, feature-aware wrapper over its validated
SparseLDA collapsed-Gibbs sampler — the same inference LDA uses — consuming a
feature corpus built by :func:`topica.from_feature_matrix`.

Producing the SAE feature matrix (LLM activations -> pretrained SAE -> features)
is heavy and model-specific; it stays outside the core, exactly as
``topica.llm_embed`` sits outside the embedding-cluster models. You bring the
features; ``MechanisticLDA`` models them.

This is an **experimental** model: it ships before an end-to-end
reference-parity check against the authors' pipeline (which requires a Gemma-2-9b
SAE), so it is gated behind :func:`topica.enable_experimental`. The topic model
*over a supplied feature matrix* reduces to topica's reference-validated LDA: for
a corpus with no empty documents, the fit is bit-identical to :class:`topica.LDA`
on the equivalent bag-of-words corpus (an all-zero row is retained here but
dropped by ``Corpus.from_documents``, which shifts the sampler's RNG stream, so
parity is exact only when every document has at least one active feature). The
feature-extraction pipeline and the paper's mETM / mBERTopic variants and topic
steering are tracked as follow-ups on #575.
"""
from __future__ import annotations

import pickle
from typing import Sequence

import numpy as np

from . import experimental_enabled
from ._topica import LDA, Corpus
from .features import from_feature_matrix

__all__ = ["MechanisticLDA"]

_EXPERIMENTAL_MSG = (
    "MechanisticLDA is experimental: the topic model over a supplied SAE-feature "
    "matrix reuses topica's validated LDA sampler, but the end-to-end pipeline has "
    "no reference-parity check against the authors' Gemma-2-9b implementation yet. "
    "Enable experimental models with `topica.enable_experimental()` or set the "
    "environment variable TOPICA_EXPERIMENTAL=1. Experimental models may change or "
    "be removed without a deprecation cycle."
)


class MechanisticLDA:
    """Mechanistic Topic Model (mLDA), Zheng et al. (2025) — **experimental**.

    LDA over sparse-autoencoder features: topics are distributions over
    interpretable SAE features rather than words, so each topic is described by
    feature concepts (see :meth:`top_features`) instead of a word list. Fit it on
    a feature corpus from :func:`topica.from_feature_matrix`, or pass a document x
    feature count matrix directly to :meth:`fit` with ``feature_names=``.

    Inference is topica's SparseLDA collapsed-Gibbs sampler (the same engine as
    :class:`topica.LDA`), so the constructor and ``fit`` mirror LDA's. The fitted
    surface is LDA's, plus feature-named aliases: :attr:`topic_feature` /
    :meth:`top_features` are the primary names; :attr:`topic_word` /
    :meth:`top_words` are kept as compatibility aliases (the "vocabulary" is the
    feature descriptions).
    """

    def __init__(
        self,
        num_topics: int,
        *,
        alpha_sum: float | None = None,
        beta: float = 0.01,
        optimize_interval: int = 50,
        burn_in: int = 200,
        seed: int = 13,
        num_threads: int = 1,
        sampler: str = "sparse",
    ) -> None:
        if not experimental_enabled():
            raise RuntimeError(_EXPERIMENTAL_MSG)
        if num_topics < 1:
            raise ValueError("num_topics must be >= 1")

        self._num_topics = num_topics
        self._alpha_sum = alpha_sum
        self._beta = beta
        self._optimize_interval = optimize_interval
        self._burn_in = burn_in
        self._seed = seed
        self._num_threads = num_threads
        self._sampler = sampler
        self._lda: LDA | None = None

    # -- configuration ----------------------------------------------------

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
            "burn_in": self._burn_in,
            "seed": self._seed,
            "num_threads": self._num_threads,
            "sampler": self._sampler,
        }

    # -- fit --------------------------------------------------------------

    def fit(
        self,
        data: "Corpus | np.ndarray | Sequence[Sequence[float]]",
        *,
        feature_names: Sequence[str] | None = None,
        iters: int = 1000,
        num_samples: int = 5,
        sample_interval: int = 25,
        progress: object | None = None,
        progress_interval: int = 50,
        keep_theta_draws: bool = True,
        num_theta_draws: int = 25,
        convergence_tol: float = 0.0,
        check_every: int = 10,
        num_threads: int | None = None,
    ) -> "MechanisticLDA":
        """Fit mLDA on a feature corpus (or a raw document x feature count matrix).

        ``data`` is either a :class:`~topica.Corpus` built by
        :func:`topica.from_feature_matrix`, or a document x feature count matrix
        (dense array or SciPy sparse) — in which case ``feature_names`` names its
        columns. All other arguments are LDA's; the model is fitted by the
        SparseLDA collapsed-Gibbs sampler.
        """
        if isinstance(data, Corpus):
            if feature_names is not None:
                raise ValueError(
                    "feature_names is only used when fitting a raw count matrix; the "
                    "corpus already carries its feature vocabulary"
                )
            corpus = data
        else:
            corpus = from_feature_matrix(data, feature_names)

        lda = LDA(
            self._num_topics,
            alpha_sum=self._alpha_sum,
            beta=self._beta,
            optimize_interval=self._optimize_interval,
            burn_in=self._burn_in,
            seed=self._seed,
            num_threads=self._num_threads,
            sampler=self._sampler,
        )
        lda.fit(
            corpus,
            iters=iters,
            num_samples=num_samples,
            sample_interval=sample_interval,
            progress=progress,
            progress_interval=progress_interval,
            keep_theta_draws=keep_theta_draws,
            num_theta_draws=num_theta_draws,
            convergence_tol=convergence_tol,
            check_every=check_every,
            num_threads=num_threads,
        )
        self._lda = lda
        return self

    def _require_fitted(self) -> LDA:
        if self._lda is None:
            raise RuntimeError("Model is not fitted")
        return self._lda

    # -- feature-named surface (primary) ----------------------------------

    @property
    def topic_feature(self) -> np.ndarray:
        """Topic x feature distribution (phi). The feature-space name for
        :attr:`topic_word`."""
        return self._require_fitted().topic_word

    def top_features(self, n: int = 10, *, topic: int | None = None):
        """Top ``n`` SAE features per topic as ``(feature, weight)`` pairs — the
        feature-space name for :meth:`top_words`."""
        return self._require_fitted().top_words(n, topic=topic)

    @property
    def feature_names(self) -> list[str]:
        """The SAE feature vocabulary (one description per feature). The
        feature-space name for :attr:`vocabulary`."""
        return self._require_fitted().vocabulary

    # -- word-named compatibility aliases ---------------------------------

    @property
    def topic_word(self) -> np.ndarray:
        return self._require_fitted().topic_word

    @property
    def doc_topic(self) -> np.ndarray:
        return self._require_fitted().doc_topic

    @property
    def vocabulary(self) -> list[str]:
        return self._require_fitted().vocabulary

    def top_words(self, n: int = 10, *, topic: int | None = None):
        return self._require_fitted().top_words(n, topic=topic)

    # -- everything else delegates to the inner LDA -----------------------

    def __getattr__(self, name: str):
        # Only reached for names not defined on the wrapper. Expose the rest of
        # LDA's fitted surface (topic_names, doc_names, coherence, fit_history,
        # converged, alpha, theta_draws, doc_lengths, transform, …).
        if name.startswith("_"):
            raise AttributeError(name)
        lda = self.__dict__.get("_lda")
        if lda is None:
            raise AttributeError(
                f"{name!r} is unavailable until the model is fitted"
            )
        return getattr(lda, name)

    # -- persistence ------------------------------------------------------

    # The inner LDA is written next to the wrapper file with this suffix. Only the
    # suffix is stored (not an absolute path), so the pair can be moved together
    # and still load.
    _INNER_SUFFIX = "._inner_lda"

    def save(self, path: str) -> None:
        """Persist the fitted model to ``path``.

        The inner LDA is written alongside as ``path + "._inner_lda"``; move or copy
        the two files together to relocate a saved model.
        """
        lda = self._require_fitted()
        lda.save(path + self._INNER_SUFFIX)
        state = {**self.settings, "inner_suffix": self._INNER_SUFFIX}
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @staticmethod
    def load(path: str) -> "MechanisticLDA":
        """Load a model saved by :meth:`save`. Requires experimental mode."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        # Resolve the inner LDA relative to this file so a moved pair still loads.
        # Fall back to the legacy absolute "inner_path" for models saved earlier.
        inner_path = (
            state["inner_path"]
            if "inner_path" in state
            else path + state.get("inner_suffix", MechanisticLDA._INNER_SUFFIX)
        )
        model = MechanisticLDA(
            state["num_topics"],
            alpha_sum=state["alpha_sum"],
            beta=state["beta"],
            optimize_interval=state["optimize_interval"],
            burn_in=state["burn_in"],
            seed=state["seed"],
            num_threads=state["num_threads"],
            sampler=state["sampler"],
        )
        model._lda = LDA.load(inner_path)
        return model

    def __repr__(self) -> str:
        status = "fitted" if self._lda is not None else "unfitted"
        return f"<MechanisticLDA K={self._num_topics} ({status})>"
