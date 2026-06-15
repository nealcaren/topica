"""TopicGPT: LLM-driven topic discovery (Pham et al. 2024, NAACL).

TopicGPT discovers topics by *prompting a language model* rather than by fitting a
generative model. The model reads documents and proposes a topic taxonomy with
natural-language descriptions, refines it by merging near-duplicates and pruning
rare topics, and then assigns each document to one (or more) of the discovered
topics with a supporting quote. The headline output is the set of topic
*descriptions*, which the count-based clustering models (``BERTopic``,
``Top2Vec``) do not produce.

topica follows the same division of labor here as in :mod:`topica.labeling`:
topica assembles the prompts; you bring the model. The backend is any callable
``str -> str`` (your own client, an ``ollama`` endpoint, a fake for testing, or
the :func:`topica.llm_backend` adapter for Simon Willison's ``llm`` library), so
the core takes no API dependency. The prompt templates live in :data:`PROMPTS`
and are overridable, because the prompts are part of the method and a researcher
must be able to audit and adapt them.

Where TopicGPT sits in the taxonomy
-----------------------------------
TopicGPT is a *cluster-style* model: the LLM does the clustering and labeling that
HDBSCAN + class-based TF-IDF do in ``BERTopic``. Like those models it has no
generative posterior over document-topic proportions, so:

- ``topic_word`` (K x V) is synthesized post hoc by class-based TF-IDF over each
  topic's assigned documents. It is a *descriptor* for ranking salient words, not
  a generative ``P(w | topic)`` distribution, exactly as in ``BERTopic``. It lets
  ``coherence``, ``topic_diversity``, ``exclusivity``, ``label_topics``, and
  ``find_thoughts`` run.
- ``estimate_effect``, ``posterior_theta_samples``, and ``ensemble`` are declined
  with an informative error: there is no theta posterior, so there are no honest
  confidence intervals to report.

Determinism is ``llm-bounded``: with ``temperature=0`` and a backend ``seed`` the
output is *stable*, not bit-reproducible, because it depends on an external model
topica does not control. Responses are cached within a fit keyed by
``(prompt, model)``, which gives within-session reproducibility and cuts cost.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Editable prompt assets
# ---------------------------------------------------------------------------
# The prompts ARE the method (Pham et al. 2024): a researcher must be able to read
# and adapt them. They live here as a module-level, overridable dict rather than
# inlined in the orchestration, so a fit can be reproduced and audited from the
# template text alone. Override per-instance via TopicGPT(..., prompts=...).

GENERATION_PROMPT = """You are building a topic taxonomy for a document collection.

Below is the taxonomy discovered so far (it may be empty), followed by a new
document. Decide whether the document fits an existing topic or introduces a new
one. If it introduces a new topic, name it concisely and write a one-sentence
description. Do not duplicate an existing topic.

Existing topics:
{taxonomy}

Document:
{document}

Respond with a single JSON object and nothing else:
{{"topic": "<short topic name>", "description": "<one sentence>", "new": <true|false>}}
"""

REFINEMENT_PROMPT = """You are refining a topic taxonomy by merging near-duplicates.

Here is a list of candidate topics, each a name and a description:
{taxonomy}

Group topics that describe the same theme. Keep distinct themes separate. Prefer
the clearest name for each group and write one merged description per group.

Respond with a single JSON object and nothing else:
{{"topics": [{{"topic": "<name>", "description": "<one sentence>",
              "merged_from": ["<original name>", ...]}}, ...]}}
"""

ASSIGNMENT_PROMPT = """You are assigning a document to topics from a fixed taxonomy.

Taxonomy:
{taxonomy}

Document:
{document}

Assign the document to {n_label} of the topics above. For each assigned topic,
include a short verbatim quote from the document that supports the assignment.

Respond with a single JSON object and nothing else:
{{"assignments": [{{"topic": "<exact topic name from the taxonomy>",
                    "quote": "<verbatim supporting quote>"}}, ...]}}
"""

PROMPTS: dict[str, str] = {
    "generation": GENERATION_PROMPT,
    "refinement": REFINEMENT_PROMPT,
    "assignment": ASSIGNMENT_PROMPT,
}


# ---------------------------------------------------------------------------
# Small data records
# ---------------------------------------------------------------------------

@dataclass
class Topic:
    """One discovered topic: a name and a natural-language description."""

    name: str
    description: str


@dataclass
class Assignment:
    """A document's assignment: the chosen ``topic_id`` and a supporting quote."""

    topic_id: int
    quote: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_token_docs(data) -> list[list[str]]:
    """Normalize ``data`` to a list of token lists.

    Accepts a topica ``Corpus`` (``.documents()``), raw strings (whitespace
    tokenized, lowercased), or already-tokenized sequences.
    """
    if hasattr(data, "documents"):
        return [list(d) for d in data.documents()]
    out: list[list[str]] = []
    for d in data:
        if isinstance(d, str):
            out.append(re.findall(r"[a-z0-9]+", d.lower()))
        else:
            out.append([str(t) for t in d])
    return out


def _as_text_docs(data) -> list[str]:
    """Normalize ``data`` to a list of display strings for the prompts."""
    if hasattr(data, "documents"):
        return [" ".join(d) for d in data.documents()]
    return [d if isinstance(d, str) else " ".join(str(t) for t in d) for d in data]


def _extract_json(text: str):
    """Parse the first JSON object out of a model reply, tolerating prose around
    it and Markdown code fences. Returns the parsed object or ``None``."""
    if text is None:
        return None
    s = str(text).strip()
    # Strip a leading ```json / ``` fence if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    # Fall back to the first balanced {...} span.
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start : i + 1])
                except Exception:
                    return None
    return None


def _norm(name: str) -> str:
    """A loose key for deduplicating topic names (case/space/punct insensitive)."""
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class TopicGPT:
    """LLM-driven topic discovery (Pham et al. 2024).

    TopicGPT prompts a language model to (1) generate a topic taxonomy with
    natural-language descriptions, (2) refine it by merging near-duplicates, and
    (3) assign every document to one or more topics with a supporting quote. It
    presents the standard fitted-model surface (``doc_topic``, ``topic_word``,
    ``top_words``, ``coherence``, ``transform``, ``save``/``load``), plus
    ``topic_descriptions`` and ``assignments``, which the count-based models do
    not produce.

    The backend is any callable ``str -> str``. Bring your own client, an
    ``ollama`` endpoint, or pass :func:`topica.llm_backend` ``model`` for the
    ``topica[llm]`` adapter; a fake callable makes the whole pipeline testable
    without a network. Pass either ``backend=`` or ``model=`` (the latter routes
    through :func:`topica.llm_backend`); supplying both raises.

    Determinism is ``llm-bounded``: ``temperature=0`` and a backend ``seed`` give
    *stable*, not bit-reproducible, results. Responses are cached within a fit
    keyed by ``(prompt, model)`` for within-session reproducibility and lower
    cost.

    ``topic_word`` is synthesized by class-based TF-IDF over each topic's assigned
    documents, so it is a *descriptor* for word ranking, not a generative
    ``P(w | topic)``. The model declines ``estimate_effect``,
    ``posterior_theta_samples``, and ``ensemble``: with no theta posterior there
    are no honest confidence intervals.

    Parameters
    ----------
    backend : callable ``str -> str``, optional
        The model. Mutually exclusive with ``model``.
    model : str, optional
        A model name routed through :func:`topica.llm_backend` (``topica[llm]``).
    hierarchical : bool, default False
        When True, ``fit`` also induces a two-level (super/sub) grouping of the
        discovered topics. ``num_topics`` always counts the leaf topics.
    assignment : {"hard", "soft"}, default "hard"
        ``"hard"`` assigns each document to one topic (one-hot ``doc_topic``);
        ``"soft"`` allows several (row-normalized weights).
    sample : int, optional
        Use only the first ``sample`` documents in the generation stage (a cost
        control); assignment still covers every document. ``None`` uses all.
    max_topics : int, optional
        Cap on the number of discovered topics carried past refinement.
    temperature : float, default 0.0
        Forwarded to :func:`topica.llm_backend` when ``model`` is given.
    seed : int, default 42
        Forwarded to the backend where supported; records intent, not a guarantee
        (the determinism is ``llm-bounded``).
    prompts : dict, optional
        Override the editable prompt templates (keys ``"generation"``,
        ``"refinement"``, ``"assignment"``). Defaults to :data:`PROMPTS`.
    """

    # Class-level sentinels so the conformance check (which inspects the class,
    # not an instance) sees every Tier-0 attribute. Instances shadow these.
    #
    # TopicGPT is a cluster-style model with no iterative objective (like
    # BERTopic / Top2Vec), so ``fit_history`` is always ``[]`` and ``converged``
    # is always ``None``. The human-readable per-stage progress log lives in the
    # separate ``stage_log`` attribute (a list of ``(stage, value)`` records),
    # which keeps ``fit_history`` to the cross-model ``(int, float)`` contract.
    converged = None
    fit_history: list = []

    def __init__(
        self,
        *,
        backend: Optional[Callable[[str], str]] = None,
        model: Optional[str] = None,
        hierarchical: bool = False,
        assignment: str = "hard",
        sample: Optional[int] = None,
        max_topics: Optional[int] = None,
        temperature: float = 0.0,
        seed: int = 42,
        prompts: Optional[dict] = None,
    ) -> None:
        if backend is not None and model is not None:
            raise ValueError("pass either backend= or model=, not both")
        if assignment not in ("hard", "soft"):
            raise ValueError('assignment must be "hard" or "soft"')
        self._backend_arg = backend
        self._model_name = model
        self.hierarchical = bool(hierarchical)
        self.assignment = assignment
        self.sample = sample
        self.max_topics = max_topics
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.prompts = dict(prompts) if prompts is not None else dict(PROMPTS)

        # Fitted state (set by fit()).
        self._fitted = False
        self.topics: list[Topic] = []
        self.assignments: list[Assignment] = []
        self._doc_topic: Optional[np.ndarray] = None
        self._topic_word: Optional[np.ndarray] = None
        self._vocabulary: list[str] = []
        self._topic_names: list[str] = []
        self._docs_tokens: list[list[str]] = []
        self._doc_names: list[str] = []
        self.stage_log: list[tuple[str, object]] = []
        self.hierarchy: Optional[dict] = None
        self._cache: dict[str, str] = {}
        self._call_count: int = 0

    # -- backend plumbing --------------------------------------------------

    def _resolve_backend(self) -> Callable[[str], str]:
        if self._backend_arg is not None:
            return self._backend_arg
        if self._model_name is not None:
            from .labeling import llm_backend

            return llm_backend(self._model_name, temperature=self.temperature)
        raise ImportError(
            "TopicGPT needs a model. Pass a backend callable "
            "(backend=lambda prompt: my_client(prompt)) or a model name "
            "(model='gpt-4o-mini'), which routes through topica.llm_backend and "
            'needs the optional `llm` package (pip install "topica[llm]").'
        )

    def _cache_key(self, prompt: str) -> str:
        tag = self._model_name or "callable-backend"
        return hashlib.sha256(f"{tag}\x00{prompt}".encode("utf-8")).hexdigest()

    def _ask(self, backend: Callable[[str], str], prompt: str) -> str:
        """Call the backend with response caching keyed by (prompt, model)."""
        key = self._cache_key(prompt)
        if key in self._cache:
            return self._cache[key]
        reply = str(backend(prompt))
        self._call_count += 1
        self._cache[key] = reply
        return reply

    # -- cost estimate -----------------------------------------------------

    def estimated_calls(self, data) -> int:
        """The number of backend calls ``fit(data)`` will make, before caching.

        Generation makes one call per document in the (optionally sampled)
        generation set, refinement makes one, and assignment makes one per
        document. Caching can lower the realized count when prompts repeat.
        """
        docs = _as_text_docs(data)
        n_gen = len(docs) if self.sample is None else min(self.sample, len(docs))
        return n_gen + 1 + len(docs)

    # -- fit ---------------------------------------------------------------

    def fit(self, data, *, metadata=None) -> "TopicGPT":
        """Discover a topic taxonomy from ``data`` and assign every document.

        ``data`` is a topica ``Corpus``, a list of raw strings, or a list of token
        lists. ``metadata`` is accepted for API symmetry and recorded in
        ``stage_log`` but does not steer the prompting in v1. There is no
        ``iters`` argument: TopicGPT is a prompting pipeline, not an iterative
        sampler (the same reason ``BERTopic`` has none).

        Runs three stages, logging each to ``stage_log``: generation (propose
        topics over the documents or a ``sample``), refinement (merge
        near-duplicates, prune, cap at ``max_topics``), and assignment (place each
        document, ``hard`` or ``soft``, with a supporting quote). Returns ``self``.
        """
        backend = self._resolve_backend()
        text_docs = _as_text_docs(data)
        self._docs_tokens = _as_token_docs(data)
        n_docs = len(text_docs)
        self._doc_names = [str(i) for i in range(n_docs)]
        self.stage_log = []
        self._call_count = 0
        if metadata is not None:
            self.stage_log.append(("metadata", "recorded (not used to steer prompts in v1)"))

        # Stage 1: generation -------------------------------------------------
        gen_idx = range(n_docs) if self.sample is None else range(min(self.sample, n_docs))
        topics: list[Topic] = []
        seen: dict[str, int] = {}
        for i in gen_idx:
            taxonomy = self._render_taxonomy(topics) or "(none yet)"
            prompt = self.prompts["generation"].format(
                taxonomy=taxonomy, document=text_docs[i][:2000]
            )
            obj = _extract_json(self._ask(backend, prompt)) or {}
            name = str(obj.get("topic", "")).strip()
            if not name:
                continue
            key = _norm(name)
            if key in seen:
                continue
            seen[key] = len(topics)
            topics.append(Topic(name=name, description=str(obj.get("description", "")).strip()))
        self.stage_log.append(("generation", len(topics)))

        # Stage 2: refinement -------------------------------------------------
        topics = self._refine(backend, topics)
        if self.max_topics is not None and len(topics) > self.max_topics:
            topics = topics[: self.max_topics]
        if not topics:  # never leave fit with zero topics
            topics = [Topic(name="topic_0", description="(no topics generated)")]
        self.topics = topics
        self._topic_names = [t.name for t in topics]
        self.stage_log.append(("refinement", len(topics)))

        # Stage 3: assignment -------------------------------------------------
        k = len(topics)
        name_to_id = {_norm(t.name): j for j, t in enumerate(topics)}
        doc_topic = np.zeros((n_docs, k), dtype=np.float64)
        assignments: list[Assignment] = []
        n_label = "exactly one topic" if self.assignment == "hard" else "one or more topics"
        taxonomy = self._render_taxonomy(topics)
        for i in range(n_docs):
            prompt = self.prompts["assignment"].format(
                taxonomy=taxonomy, document=text_docs[i][:2000], n_label=n_label
            )
            obj = _extract_json(self._ask(backend, prompt)) or {}
            chosen = self._parse_assignments(obj, name_to_id)
            if not chosen:
                chosen = [(0, "")]  # fall back to topic 0 with no quote
            if self.assignment == "hard":
                chosen = chosen[:1]
            for tid, _quote in chosen:
                doc_topic[i, tid] += 1.0
            row = doc_topic[i]
            if row.sum() > 0:
                row /= row.sum()
            assignments.append(Assignment(topic_id=chosen[0][0], quote=chosen[0][1]))
        self.assignments = assignments
        self._doc_topic = doc_topic
        self.stage_log.append(("assignment", n_docs))

        # Optional two-level hierarchy ---------------------------------------
        if self.hierarchical:
            self.hierarchy = self._induce_hierarchy(backend, topics)
            self.stage_log.append(("hierarchy", len(self.hierarchy.get("supertopics", []))))

        # Synthesize the descriptor topic_word via class-based TF-IDF --------
        self._vocabulary, self._topic_word = self._class_tfidf(doc_topic)
        self._fitted = True
        return self

    # -- stage helpers -----------------------------------------------------

    def _render_taxonomy(self, topics: list[Topic]) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in topics)

    def _refine(self, backend, topics: list[Topic]) -> list[Topic]:
        """Merge near-duplicate topics. Asks the backend once; falls back to the
        input taxonomy if the reply is unusable."""
        if len(topics) < 2:
            return topics
        prompt = self.prompts["refinement"].format(taxonomy=self._render_taxonomy(topics))
        obj = _extract_json(self._ask(backend, prompt)) or {}
        merged = obj.get("topics")
        if not isinstance(merged, list) or not merged:
            return topics
        out: list[Topic] = []
        seen: set[str] = set()
        for m in merged:
            if not isinstance(m, dict):
                continue
            name = str(m.get("topic", "")).strip()
            if not name:
                continue
            key = _norm(name)
            if key in seen:
                continue
            seen.add(key)
            out.append(Topic(name=name, description=str(m.get("description", "")).strip()))
        return out or topics

    def _parse_assignments(self, obj: dict, name_to_id: dict) -> list[tuple[int, str]]:
        items = obj.get("assignments")
        if not isinstance(items, list):
            return []
        out: list[tuple[int, str]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            tid = name_to_id.get(_norm(it.get("topic", "")))
            if tid is None:
                continue
            out.append((tid, str(it.get("quote", "")).strip()))
        return out

    def _induce_hierarchy(self, backend, topics: list[Topic]) -> dict:
        """A two-level grouping of the discovered topics, reusing the refinement
        prompt to cluster leaf topics into supertopics."""
        prompt = self.prompts["refinement"].format(taxonomy=self._render_taxonomy(topics))
        obj = _extract_json(self._ask(backend, prompt)) or {}
        groups = obj.get("topics")
        name_to_id = {_norm(t.name): j for j, t in enumerate(topics)}
        supers = []
        if isinstance(groups, list):
            for g in groups:
                if not isinstance(g, dict):
                    continue
                children = [
                    name_to_id[_norm(c)]
                    for c in (g.get("merged_from") or [])
                    if _norm(c) in name_to_id
                ]
                supers.append({"name": str(g.get("topic", "")).strip(), "children": children})
        return {"supertopics": supers}

    def _class_tfidf(self, doc_topic: np.ndarray) -> tuple[list[str], np.ndarray]:
        """Class-based TF-IDF over assigned documents (the BERTopic descriptor).

        Each topic's documents are pooled into a class; term frequencies are
        weighted by ``log(1 + average_class_size / term_class_frequency)``. Rows
        are L1-normalized so they read like a distribution, but they remain a
        *descriptor*, not a generative ``P(w | topic)``.
        """
        vocab = sorted({w for d in self._docs_tokens for w in d})
        v = len(vocab)
        k = doc_topic.shape[1]
        if v == 0:
            return vocab, np.zeros((k, 1), dtype=np.float64)
        widx = {w: j for j, w in enumerate(vocab)}
        # Soft class term counts: a doc contributes its tokens to each topic in
        # proportion to its (possibly soft) assignment weight.
        tf = np.zeros((k, v), dtype=np.float64)
        for i, toks in enumerate(self._docs_tokens):
            w = doc_topic[i]
            if w.sum() == 0:
                continue
            for word, c in Counter(toks).items():
                tf[:, widx[word]] += w * float(c)
        # IDF across classes: words common to many topics are down-weighted.
        avg_len = tf.sum(axis=1).mean() if tf.sum() > 0 else 1.0
        term_total = tf.sum(axis=0)
        term_total[term_total == 0.0] = 1.0
        idf = np.log(1.0 + avg_len / term_total)
        ctfidf = tf * idf[None, :]
        rowsum = ctfidf.sum(axis=1, keepdims=True)
        rowsum[rowsum == 0.0] = 1.0
        return vocab, ctfidf / rowsum

    # -- fitted-model surface ---------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("TopicGPT is not fitted; call fit(data) first")

    @property
    def num_topics(self) -> int:
        """The discovered topic count (a fitted attribute, like HDP)."""
        self._check_fitted()
        return len(self.topics)

    @property
    def doc_topic(self) -> np.ndarray:
        """The (D, K) assignment matrix (one-hot for ``hard``; normalized weights
        for ``soft``). A descriptor of the LLM's assignments, not a posterior."""
        self._check_fitted()
        return self._doc_topic

    @property
    def topic_word(self) -> np.ndarray:
        """The (K, V) class-based TF-IDF descriptor. NOT a generative ``P(w|topic)``
        distribution (see the class docstring); use it for word ranking only."""
        self._check_fitted()
        return self._topic_word

    @property
    def vocabulary(self) -> list[str]:
        self._check_fitted()
        return list(self._vocabulary)

    @property
    def topic_descriptions(self) -> list[str]:
        """The LLM's natural-language description of each topic (the headline
        output, and the value-add over the count-based clustering models)."""
        self._check_fitted()
        return [t.description for t in self.topics]

    @property
    def topic_names(self) -> list[str]:
        self._check_fitted()
        return list(self._topic_names)

    @topic_names.setter
    def topic_names(self, value: Sequence[str]) -> None:
        self._check_fitted()
        value = list(value)
        if len(value) != len(self.topics):
            raise ValueError(f"expected {len(self.topics)} names, got {len(value)}")
        self._topic_names = [str(v) for v in value]

    @property
    def doc_names(self) -> list[str]:
        self._check_fitted()
        return list(self._doc_names)

    def top_words(self, n: int = 10, *, topic: Optional[int] = None):
        """Top-``n`` words per topic from the synthesized class-TF-IDF descriptor.

        Returns a list of ``(word, score)`` pairs for the given ``topic``, or one
        such list per topic when ``topic`` is None. The scores are c-TF-IDF
        weights, not probabilities.
        """
        self._check_fitted()
        phi = self._topic_word
        vocab = self._vocabulary

        def row_words(t: int):
            idx = np.argsort(phi[t])[::-1][:n]
            return [(vocab[j], float(phi[t, j])) for j in idx if phi[t, j] > 0]

        if topic is not None:
            return row_words(topic)
        return [row_words(t) for t in range(phi.shape[0])]

    def coherence(self, n: int = 10) -> np.ndarray:
        """Per-topic c_v coherence of the top-``n`` descriptor words against the
        training corpus. The same windowed measure used across topica's models."""
        self._check_fitted()
        from .coherence import coherence as _coherence

        topics = [[w for w, _ in self.top_words(n, topic=t)] for t in range(self.num_topics)]
        return _coherence(topics, self._docs_tokens, topn=n)

    # -- transform: assign held-out docs to the discovered taxonomy --------

    def transform(self, new_docs) -> np.ndarray:
        """Assign held-out documents to the discovered taxonomy via the assignment
        prompt. Returns a (n, K) assignment matrix (one-hot/soft per ``assignment``).

        This is ``llm-bounded`` and more capable than the count-based models'
        transform: the LLM places new documents directly into the named taxonomy.
        """
        self._check_fitted()
        backend = self._resolve_backend()
        texts = _as_text_docs(new_docs)
        k = self.num_topics
        out = np.zeros((len(texts), k), dtype=np.float64)
        name_to_id = {_norm(t.name): j for j, t in enumerate(self.topics)}
        n_label = "exactly one topic" if self.assignment == "hard" else "one or more topics"
        taxonomy = self._render_taxonomy(self.topics)
        for i, text in enumerate(texts):
            prompt = self.prompts["assignment"].format(
                taxonomy=taxonomy, document=text[:2000], n_label=n_label
            )
            obj = _extract_json(self._ask(backend, prompt)) or {}
            chosen = self._parse_assignments(obj, name_to_id) or [(0, "")]
            if self.assignment == "hard":
                chosen = chosen[:1]
            for tid, _q in chosen:
                out[i, tid] += 1.0
            if out[i].sum() > 0:
                out[i] /= out[i].sum()
        return out

    # -- honest declines (no theta posterior) ------------------------------

    def estimate_effect(self, *args, **kwargs):
        """Declined: TopicGPT has no theta posterior, so there are no honest
        confidence intervals on covariate effects."""
        raise NotImplementedError(_DECLINE_MSG.format(name="estimate_effect"))

    def posterior_theta_samples(self, *args, **kwargs):
        """Declined: TopicGPT has no posterior over theta to sample (its
        ``doc_topic`` is an LLM assignment, not a fitted distribution)."""
        raise NotImplementedError(_DECLINE_MSG.format(name="posterior_theta_samples"))

    def ensemble(self, *args, **kwargs):
        """Declined: ensembling assumes alignable runs with comparable topic
        distributions; TopicGPT's taxonomy is an llm-bounded, run-specific
        artifact, so a stable consensus is out of scope (v1)."""
        raise NotImplementedError(_DECLINE_MSG.format(name="ensemble"))

    # -- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        """Pickle the fitted state (topics, assignments, matrices, cache). The
        backend callable is NOT saved (it is not picklable in general); set it
        again after :meth:`load` to call ``transform``."""
        self._check_fitted()
        state = {
            "topics": self.topics,
            "assignments": self.assignments,
            "doc_topic": self._doc_topic,
            "topic_word": self._topic_word,
            "vocabulary": self._vocabulary,
            "topic_names": self._topic_names,
            "docs_tokens": self._docs_tokens,
            "doc_names": self._doc_names,
            "stage_log": self.stage_log,
            "hierarchy": self.hierarchy,
            "assignment": self.assignment,
            "hierarchical": self.hierarchical,
            "model_name": self._model_name,
            "prompts": self.prompts,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @staticmethod
    def load(path: str) -> "TopicGPT":
        """Load a fitted model saved by :meth:`save`. The backend is not restored;
        call ``model.set_backend(...)`` before ``transform``."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        m = TopicGPT(
            assignment=state["assignment"],
            hierarchical=state["hierarchical"],
            model=state.get("model_name"),
            prompts=state.get("prompts"),
        )
        m.topics = state["topics"]
        m.assignments = state["assignments"]
        m._doc_topic = state["doc_topic"]
        m._topic_word = state["topic_word"]
        m._vocabulary = state["vocabulary"]
        m._topic_names = state["topic_names"]
        m._docs_tokens = state["docs_tokens"]
        m._doc_names = state["doc_names"]
        m.stage_log = state["stage_log"]
        m.hierarchy = state["hierarchy"]
        m._fitted = True
        return m

    def set_backend(self, backend: Callable[[str], str]) -> None:
        """Attach a backend callable to a model (e.g. after :meth:`load`) so
        :meth:`transform` can run."""
        self._backend_arg = backend
        self._model_name = None

    def __repr__(self) -> str:
        if self._fitted:
            return f"TopicGPT(num_topics={len(self.topics)}, assignment={self.assignment!r})"
        return f"TopicGPT(assignment={self.assignment!r}, unfitted)"


_DECLINE_MSG = (
    "TopicGPT does not support {name}: it is an llm-bounded, cluster-style model "
    "with no posterior over theta, so there are no honest confidence intervals to "
    "report. This refusal is by design (the 'no CIs without a posterior' "
    "principle). Use a generative model (LDA, STM, ...) when you need {name}."
)
