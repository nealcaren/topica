"""Author-Recipient-Topic (ART) model.

ART (McCallum, Wang & Corrada-Emmanuel, "Topic and Role Discovery in Social
Networks," JAIR 30 (2007) 249-272) conditions a message's topics jointly on its
**sender and one recipient**: for each token a recipient ``x`` is drawn uniformly
from the message's recipient set ``r_d``, a topic ``z`` from the pair-specific
distribution ``theta_{a_d, x}``, and a word from ``phi_z``. It captures the
language of a *directed* social network (who talks to whom about what), unlike the
Author-Topic model (AuthorTopic) which conditions on the author alone.

Implementation. ART is mathematically isomorphic to the Author-Topic model with the
"author" entity taken to be the ordered ``(sender, recipient)`` **pair**: because
the sender ``a_d`` is fixed within a message, uniform sampling of recipients over
``r_d`` is identical to uniform sampling of pairs over ``{(a_d, j) : j in r_d}``. We
therefore realize ART as a thin, provably-faithful wrapper over the compiled
``topica.AuthorTopic`` engine — mapping each document's recipients to pair labels
and delegating the collapsed-Gibbs fit. This inherits AuthorTopic's determinism and,
crucially, its held-in log-likelihood already carries the ``1/|r_d|`` recipient
average, so the likelihood is correct without any change to the core.

The base ART model is implemented here. The Role-Author-Recipient-Topic (RART)
extension — latent *roles* for senders/recipients — is future work (see the roster
notes); passing observed group labels (e.g. a subreddit, a community) as senders and
recipients is a coarsened observed-group application of ART, not latent role
discovery.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np

from ._topica import AuthorTopic


class AuthorRecipientTopic:
    """Author-Recipient-Topic model (McCallum et al. 2007).

    Parameters
    ----------
    num_topics:
        Number of topics K.
    alpha, beta:
        Dirichlet hyperparameters for the pair-topic and topic-word priors. The
        defaults are the ART paper's: ``alpha = 50/K`` (inherited from the AuthorTopic
        engine when left ``None``) and ``beta = 0.1``. (topica's AuthorTopic sibling
        defaults ``beta = 0.01``; pass that if you want to match it instead.)
    seed:
        RNG seed (topica default 13).
    """

    def __init__(self, num_topics: int, *, alpha: float | None = None,
                 beta: float | None = 0.1, seed: int = 13):
        self._num_topics = int(num_topics)
        self._alpha = alpha
        self._beta = beta
        self._seed = int(seed)
        self._at: AuthorTopic | None = None
        self._pair_labels: list[tuple] | None = None
        self._fitted = False

    # -- fit ----------------------------------------------------------------
    def fit(self, docs: Sequence[Sequence[str]], *, authors: Sequence,
            recipients: Sequence[Sequence], iters: int = 1000, progress=None):
        """Fit ART.

        ``authors`` is a length-D sequence of sender labels (one per message).
        ``recipients`` is a length-D sequence of recipient-label lists (each with at
        least one recipient). Labels are serializable categoricals (strings or ints).
        For Reddit-style reply data, the sender is the commenter and the single
        recipient is the parent comment's author.
        """
        D = len(docs)
        if len(authors) != D:
            raise ValueError(f"authors has {len(authors)} entries but there are {D} documents")
        if len(recipients) != D:
            raise ValueError(f"recipients has {len(recipients)} entries but there are {D} documents")

        # Key pairs on the original (sender, recipient) TUPLE, not a stringified join:
        # tuples of mixed types are hashable and keep (1, "2") distinct from ("1", 2),
        # so there is no separator collision and no type loss. Each distinct pair gets
        # a synthetic string id passed to the AuthorTopic engine; the id->tuple map
        # recovers the original-typed labels. Ordering is deterministic regardless of
        # whether the caller passes a list or a set (recipient order is semantically
        # irrelevant in ART — recipients are a set drawn from uniformly).
        pair_id: dict[tuple, str] = {}
        id_pair: list[tuple] = []
        pair_lists: list[list[str]] = []
        for d in range(D):
            rset = recipients[d]
            if isinstance(rset, (str, bytes)):
                raise ValueError(
                    f"document {d}'s recipients is a bare string {rset!r}; pass a list "
                    f"of recipient labels (e.g. [{rset!r}]), not a single string")
            if rset is None or len(rset) == 0:
                raise ValueError(f"document {d} has an empty recipient set; every message needs >= 1 recipient")
            keys, seen = [], set()
            for j in sorted(rset, key=repr):     # dedupe; stable order for any iterable/type
                pair = (authors[d], j)
                if pair in seen:
                    continue
                seen.add(pair)
                pid = pair_id.get(pair)
                if pid is None:
                    pid = f"p{len(id_pair)}"
                    pair_id[pair] = pid
                    id_pair.append(pair)
                keys.append(pid)
            pair_lists.append(keys)

        kw = {"seed": self._seed}
        if self._alpha is not None:
            kw["alpha"] = self._alpha
        if self._beta is not None:
            kw["beta"] = self._beta
        at = AuthorTopic(self._num_topics, **kw)
        at.fit(docs, pair_lists, iters=iters, progress=progress)
        self._at = at
        # map the engine's ordered pair ids back to original (sender, recipient)
        # tuples, preserving label types (ints stay ints)
        _to_pair = {pid: pair for pair, pid in pair_id.items()}
        self._pair_labels = [_to_pair[pid] for pid in at.authors]
        self._fitted = True
        return self

    def _check(self):
        if not self._fitted or self._at is None:
            raise RuntimeError("model is not fitted yet; call fit() first")

    # -- fitted surface -----------------------------------------------------
    @property
    def topic_word(self) -> np.ndarray:
        self._check()
        return self._at.topic_word

    @property
    def doc_topic(self) -> np.ndarray:
        """Per-document content-based topic simplex (D x K), as AuthorTopic/LDA report."""
        self._check()
        return self._at.doc_topic

    @property
    def pair_topic(self) -> np.ndarray:
        """The model-defining output: per (sender, recipient) pair topic distribution
        (P x K), aligned to :attr:`pair_labels`."""
        self._check()
        return self._at.author_topic

    @property
    def pair_labels(self) -> list[tuple]:
        """The P ordered (sender, recipient) tuples, aligned to :attr:`pair_topic`."""
        self._check()
        return list(self._pair_labels)

    @property
    def pair_counts(self) -> np.ndarray:
        """Number of *messages* each pair appears on (length P) — the observed
        data-support diagnostic (a pair seen on one message has an unstable row)."""
        self._check()
        return np.asarray(self._at.author_doc_counts)

    @property
    def vocabulary(self):
        self._check()
        return self._at.vocabulary

    @property
    def num_topics(self) -> int:
        return self._num_topics

    @property
    def settings(self) -> dict:
        """Constructor configuration (available before and after fit)."""
        return {"num_topics": self._num_topics, "alpha": self._alpha,
                "beta": self._beta, "seed": self._seed}

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def converged(self) -> bool:
        self._check()
        return self._at.converged

    @property
    def fit_history(self):
        self._check()
        return self._at.fit_history

    def top_words(self, n: int = 10, *, topic: int | None = None, weights: bool = False):
        self._check()
        return self._at.top_words(n, topic=topic, weights=weights)

    # -- persistence --------------------------------------------------------
    # The compiled AuthorTopic engine owns the heavy state; we persist it to a
    # sidecar file and pickle only the ART-level metadata (pair labels, hypers).
    def save(self, path: str) -> None:
        self._check()
        p = Path(path)
        self._at.save(str(p) + ".at")
        with open(p, "wb") as f:
            pickle.dump(
                {"num_topics": self._num_topics, "alpha": self._alpha, "beta": self._beta,
                 "seed": self._seed, "pair_labels": self._pair_labels},
                f,
            )

    @classmethod
    def load(cls, path: str) -> "AuthorRecipientTopic":
        p = Path(path)
        with open(p, "rb") as f:
            meta = pickle.load(f)
        obj = cls(meta["num_topics"], alpha=meta["alpha"], beta=meta["beta"], seed=meta["seed"])
        obj._at = AuthorTopic.load(str(p) + ".at")
        obj._pair_labels = meta["pair_labels"]
        obj._fitted = True
        return obj


# reference-package / paper alias
ART = AuthorRecipientTopic
