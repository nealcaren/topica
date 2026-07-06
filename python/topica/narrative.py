from __future__ import annotations
import math
import re
import numpy as np
import pickle
from typing import Sequence
from .gdmr import GDMR
from . import experimental_enabled

class NarrativeTM:
    """Intra-Document Narrative Trajectory Model (Experimental).

    Segments documents into chunks (e.g., sentences or fixed token intervals)
    and fits a Generalized DMR (GDMR) model over their relative positions,
    capturing the average progression of topics from the beginning to the end of texts.
    """

    def __init__(
        self,
        num_topics: int,
        *,
        degree: int = 3,
        segment_by: str = "chunk",
        chunk_size: int = 20,
        beta: float = 0.01,
        optimize_interval: int = 50,
        burn_in: int = 200,
        seed: int = 42,
        decay: float = 0.0,
        sigma: float = 1.0,
        sigma0: float = 3.0,
        sampler: str = "sparse",
    ) -> None:
        if not experimental_enabled():
            raise RuntimeError(
                "NarrativeTM is experimental: it is an original construction with no "
                "published paper or reference-implementation parity yet. Enable "
                "experimental models with `topica.enable_experimental()` or set the "
                "environment variable TOPICA_EXPERIMENTAL=1. Experimental models may "
                "change or be removed without a deprecation cycle."
            )
        if num_topics < 1:
            raise ValueError("num_topics must be >= 1")
        if degree < 0:
            raise ValueError("degree must be >= 0")
        if segment_by not in ("chunk", "sentence"):
            raise ValueError("segment_by must be 'chunk' or 'sentence'")
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

        self._num_topics = num_topics
        self._degree = degree
        self._segment_by = segment_by
        self._chunk_size = chunk_size
        self._beta = beta
        self._optimize_interval = optimize_interval
        self._burn_in = burn_in
        self._seed = seed
        self._decay = decay
        self._sigma = sigma
        self._sigma0 = sigma0
        self._sampler = sampler

        self._gdmr: GDMR | None = None
        self._doc_topic: np.ndarray | None = None
        self._vocabulary: list[str] | None = None
        self._fitted = False

    @property
    def num_topics(self) -> int:
        return self._num_topics

    @property
    def vocabulary(self) -> list[str]:
        if not self._fitted:
            raise RuntimeError("Model is not fitted")
        return self._vocabulary

    @property
    def topic_word(self) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model is not fitted")
        return self._gdmr.topic_word

    @property
    def doc_topic(self) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model is not fitted")
        return self._doc_topic

    def top_words(self, n: int = 10, *, topic: int | None = None):
        if not self._fitted:
            raise RuntimeError("Model is not fitted")
        return self._gdmr.top_words(n, topic=topic)

    def coherence(self, n: int = 10) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model is not fitted")
        return self._gdmr.coherence(n)

    def fit(
        self,
        data: "Corpus | Sequence[Sequence[str]]",
        *,
        iters: int = 1000,
        num_samples: int = 5,
        sample_interval: int = 25,
        keep_theta_draws: bool = True,
        convergence_tol: float = 0.0,
        check_every: int = 10,
    ) -> None:
        from ._topica import Corpus
        
        # 1. Convert to lists of tokens
        if isinstance(data, Corpus):
            orig_docs = data.documents()
        elif hasattr(data, "documents") and callable(data.documents):
            orig_docs = data.documents()
        else:
            orig_docs = [list(d) for d in data]

        # 2. Segment each document and record position
        flat_chunks = []
        flat_positions = []
        doc_indices = []  # mapping chunk_idx -> original_doc_idx
        chunk_lengths = []

        for doc_idx, doc in enumerate(orig_docs):
            if not doc:
                continue
            
            # Segment
            if self._segment_by == "sentence":
                # Find boundaries based on punctuation
                chunks = []
                current = []
                for token in doc:
                    current.append(token)
                    if token in (".", "?", "!", ";"):
                        chunks.append(current)
                        current = []
                if current:
                    chunks.append(current)
                # Fallback to fixed chunking if no sentence boundary found
                if len(chunks) <= 1 and not any(t in (".", "?", "!", ";") for t in doc):
                    chunks = [doc[i : i + self._chunk_size] for i in range(0, len(doc), self._chunk_size)]
            else:
                chunks = [doc[i : i + self._chunk_size] for i in range(0, len(doc), self._chunk_size)]
            
            M = len(chunks)
            for i, chunk in enumerate(chunks):
                if not chunk:
                    continue
                # Map position to [0, 1]
                if M > 1:
                    x = float(i) / (M - 1)
                else:
                    x = 0.5
                
                flat_chunks.append(chunk)
                flat_positions.append(x)
                doc_indices.append(doc_idx)
                chunk_lengths.append(len(chunk))

        if not flat_chunks:
            raise ValueError("No valid document segments found for training")

        # 3. Fit inner GDMR engine
        # D = 1 (relative position)
        self._gdmr = GDMR(
            num_topics=self._num_topics,
            degrees=[self._degree],
            beta=self._beta,
            optimize_interval=self._optimize_interval,
            burn_in=self._burn_in,
            seed=self._seed,
            sigma=self._sigma,
            sigma0=self._sigma0,
            decay=self._decay,
            metadata_range=[(0.0, 1.0)],
            lbfgs_iters=20,
            sampler=self._sampler,
        )

        positions_array = np.array(flat_positions, dtype=np.float64).reshape(-1, 1)
        self._gdmr.fit(
            flat_chunks,
            features=positions_array,
            iters=iters,
            num_samples=num_samples,
            sample_interval=sample_interval,
            keep_theta_draws=keep_theta_draws,
            convergence_tol=convergence_tol,
            check_every=check_every,
        )

        self._vocabulary = self._gdmr.vocabulary

        # 4. Reconstruct document-level doc_topic (theta_d)
        gdmr_theta = np.asarray(self._gdmr.doc_topic)  # shape (N_chunks, K)
        N_docs = len(orig_docs)
        self._doc_topic = np.zeros((N_docs, self._num_topics), dtype=np.float64)
        
        # Token-weighted average per original document
        doc_token_counts = np.zeros(N_docs, dtype=np.float64)
        for chunk_idx, doc_idx in enumerate(doc_indices):
            weight = chunk_lengths[chunk_idx]
            self._doc_topic[doc_idx] += gdmr_theta[chunk_idx] * weight
            doc_token_counts[doc_idx] += weight

        for doc_idx in range(N_docs):
            if doc_token_counts[doc_idx] > 0:
                self._doc_topic[doc_idx] /= doc_token_counts[doc_idx]
            else:
                # empty document fallback to uniform
                self._doc_topic[doc_idx] = 1.0 / self._num_topics

        self._fitted = True

    def global_trajectory(self, t: float | Sequence[float] | np.ndarray) -> np.ndarray:
        """Evaluate the learned global narrative trajectory at relative position t in [0, 1].

        Returns a numpy array of shape (len(t), K) or (K,) representing the topic
        proportions at position t.
        """
        if not self._fitted:
            raise RuntimeError("Model is not fitted")

        t_arr = np.atleast_1d(np.asarray(t, dtype=np.float64))
        if np.any((t_arr < 0.0) | (t_arr > 1.0)):
            raise ValueError("Position t must be in the interval [0, 1]")

        # Evaluate Legendre polynomial prior using the inner GDMR model's TDF
        # GDMR expects input of shape (N, D) where D = 1
        res = self._gdmr.tdf(t_arr.reshape(-1, 1))
        # If the input was a single scalar, return a 1-D array
        if np.isscalar(t) or (isinstance(t, (float, int))):
            return res[0]
        return res

    def save(self, path: str) -> None:
        """Persist the model state using pickle."""
        if not self._fitted:
            raise RuntimeError("Model is not fitted")
        gdmr_path = path + "._inner_gdmr"
        self._gdmr.save(gdmr_path)
        state = {
            "num_topics": self._num_topics,
            "degree": self._degree,
            "segment_by": self._segment_by,
            "chunk_size": self._chunk_size,
            "beta": self._beta,
            "optimize_interval": self._optimize_interval,
            "burn_in": self._burn_in,
            "seed": self._seed,
            "decay": self._decay,
            "sigma": self._sigma,
            "sigma0": self._sigma0,
            "sampler": self._sampler,
            "gdmr_path": gdmr_path,
            "doc_topic": self._doc_topic,
            "vocabulary": self._vocabulary,
            "fitted": self._fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @staticmethod
    def load(path: str) -> NarrativeTM:
        """Load a saved model state from path."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        model = NarrativeTM(
            num_topics=state["num_topics"],
            degree=state["degree"],
            segment_by=state["segment_by"],
            chunk_size=state["chunk_size"],
            beta=state["beta"],
            optimize_interval=state["optimize_interval"],
            burn_in=state["burn_in"],
            seed=state["seed"],
            decay=state["decay"],
            sigma=state["sigma"],
            sigma0=state["sigma0"],
            sampler=state["sampler"],
        )
        model._gdmr = GDMR.load(state["gdmr_path"])
        model._doc_topic = state["doc_topic"]
        model._vocabulary = state["vocabulary"]
        model._fitted = state["fitted"]
        return model

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return f"<NarrativeTM K={self._num_topics} degree={self._degree} ({status})>"
