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

The default templates are adapted from the published TopicGPT reference prompts
(github.com/chtmp223/topicGPT, MIT-licensed; Pham, Hoyle, Sun & Iyyer 2024): the
bracketed ``[level] Label: Description`` output format, the few-shot
demonstrations, and the method's rules (generalizable single topics; never invent
a topic or a quote). Override one or more stages via ``prompts=`` (a partial dict
merges over the defaults) or :meth:`TopicGPT.with_prompt`.

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
  with an informative error: there is no theta posterior, so there are no defensible
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
import warnings
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
#
# These templates are adapted from the published TopicGPT reference prompts
# (github.com/chtmp223/topicGPT, MIT-licensed; Pham, Hoyle, Sun & Iyyer 2024):
# `generation_1.txt`, `refinement.txt`, and `assignment.txt`. We keep TopicGPT's
# bracketed `[level] Label: Description` output format, its few-shot demonstration
# structure, and its rules (generalizable single topics; never invent topics or
# quotes), which is what makes "this implements TopicGPT" a defensible claim.
#
# Three documented deviations from the reference: (1) the few-shot examples are
# generalized to neutral cross-domain ones (astronomy/cuisine/music) instead of the
# reference's congressional-bills examples. The reference prompts are tuned to the
# paper's policy corpus (and ship a domain seed); used verbatim on an arbitrary
# corpus they bias the model to reject off-domain documents as "None". The format
# and rules are the paper's; only the demonstration domain is neutralized, and the
# whole template is overridable via `prompts=` for a domain-specific run.
# (2) Refinement asks for the full refined topic list rather than only the
# incremental merge edits, so the orchestration can replace the topic set directly
# (see `_refine`). (3) The assignment prompt carries topica's hard/soft `{n_label}`
# phrasing, which the single-assignment reference prompt does not have.

GENERATION_PROMPT = """You will receive a document and a set of top-level topics from a topic hierarchy. Your task is to identify generalizable topics within the document that can act as top-level topics in the hierarchy. If any relevant topics are missing from the provided set, add them. Otherwise, output the existing top-level topics identified in the document.

[Top-level topics]
{taxonomy}

[Examples]
Example 1: The relevant topic is missing, so add "[1] Astronomy"
Document:
Astronomers using a space telescope detected water vapor in the atmosphere of a distant exoplanet orbiting a faint red dwarf star.
Your response:
[1] Astronomy: Mentions the study of celestial objects and the physical universe.

Example 2: A relevant topic ("[1] Cuisine") already exists, so return it unchanged
Document:
A food writer profiled three neighborhood bakeries reviving nineteenth-century sourdough techniques.
Your response:
[1] Cuisine: Mentions cooking, food preparation, and culinary culture.

Example 3: The document is about a different theme, so add "[1] Music"
Document:
The quartet premiered a new string composition at the concert hall, blending baroque counterpoint with electronic textures.
Your response:
[1] Music: Mentions musical works, performance, and composition.

[Instructions]
Step 1: Determine the topics mentioned in the document.
- The topic labels must be as GENERALIZABLE as possible; they must not be document-specific.
- The topics must each reflect a SINGLE topic instead of a combination of topics.
- Each new topic must have a level number, a short general label, and a one-sentence description.
- The topics must be broad enough to accommodate future subtopics.
Step 2: Perform ONE of the following operations:
1. If a relevant or duplicate topic already exists in the hierarchy, output that topic and stop.
2. If the document contains no topic, return "None".
3. Otherwise, add your topic as a top-level topic and output it. Do not add any lower levels.

[Document]
{document}

Please ONLY return the relevant or modified top-level topics, one per line, in the format:
[1] Topic Label: Topic Description

Your response:
"""

REFINEMENT_PROMPT = """You will receive a list of topics that belong to the same level of a topic hierarchy. Your task is to merge topics that are paraphrases or near-duplicates of one another. Keep genuinely distinct topics separate. Return "None" if no modification is needed.

[Topic list]
{taxonomy}

[Rules]
- Each line is a topic, with a level indicator, a label, and a description.
- Merge near-duplicate topics into a single topic, choosing the clearest label and writing one merged description.
- Do nothing and return "None" if no modification is needed.

Output the complete refined list of top-level topics (both the merged topics and any left unchanged), one per line, in the format:
[1] Topic Label: Topic Description

[Your response]
"""

ASSIGNMENT_PROMPT = """You will receive a document and a topic hierarchy. Assign the document to {n_label} from the hierarchy. Then output the topic label(s), your assignment reasoning, and a supporting quote taken verbatim from the document. DO NOT make up new topics or quotes.

[Topic hierarchy]
{taxonomy}

[Example]
Document:
Astronomers using a space telescope detected water vapor in the atmosphere of a distant exoplanet.
Assignment:
[1] Astronomy: Reports the detection of an exoplanet's atmosphere ("...detected water vapor in the atmosphere of a distant exoplanet.")

[Instructions]
1. Topic labels must be present in the provided hierarchy. You MUST NOT invent new topics.
2. Each quote must be taken from the document. You MUST NOT invent quotes.

[Document]
{document}

Double-check that every assignment exists in the hierarchy.
Your response should have one assignment per line, in the format:
[Topic Level] Topic Label: Assignment reasoning (Supporting quote)

Your response:
"""

PROMPTS: dict[str, str] = {
    "generation": GENERATION_PROMPT,
    "refinement": REFINEMENT_PROMPT,
    "assignment": ASSIGNMENT_PROMPT,
}

#: Fields each stage template must contain after a user override, so a custom
#: prompt still receives the data the orchestration substitutes in.
PROMPT_FIELDS: dict[str, tuple[str, ...]] = {
    "generation": ("taxonomy", "document"),
    "refinement": ("taxonomy",),
    "assignment": ("taxonomy", "document", "n_label"),
}


def _merge_prompts(overrides: Optional[dict]) -> dict[str, str]:
    """Merge user prompt overrides over the published defaults.

    A partial dict overrides only the named stage(s); the rest fall back to
    :data:`PROMPTS`. Unknown keys raise (a typo should not be silently ignored),
    and each overridden template must keep the ``{field}`` placeholders the
    orchestration fills (see :data:`PROMPT_FIELDS`).
    """
    merged = dict(PROMPTS)
    if not overrides:
        return merged
    unknown = set(overrides) - set(PROMPTS)
    if unknown:
        raise ValueError(
            f"unknown prompt key(s) {sorted(unknown)}; valid keys are {sorted(PROMPTS)}"
        )
    for stage, template in overrides.items():
        template = str(template)
        missing = [f for f in PROMPT_FIELDS[stage] if "{" + f + "}" not in template]
        if missing:
            raise ValueError(
                f"custom {stage!r} prompt is missing required field(s) "
                f"{missing} (expected placeholders: "
                f"{['{' + f + '}' for f in PROMPT_FIELDS[stage]]})"
            )
        # Fail fast on a template that ``str.format`` cannot fill. The
        # orchestration substitutes only the stage's required fields via
        # ``str.format``, so any *other* brace (a pasted JSON snippet like
        # ``{"topic": "x"}``, a LaTeX/exponent ``O(n^{2})``, or a stray
        # ``{placeholder}``) makes the real fit crash deep in ``fit`` with an
        # opaque ``KeyError``/``IndexError``. Trial-format here with the exact
        # stage kwargs so the failure surfaces at construction with a clear
        # message and a fix (escape literal braces as ``{{`` / ``}}``).
        try:
            template.format(**{f: "" for f in PROMPT_FIELDS[stage]})
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(
                f"custom {stage!r} prompt is not brace-safe: str.format() raised "
                f"{type(exc).__name__}: {exc}. The orchestration fills only "
                f"{list(PROMPT_FIELDS[stage])} via str.format(), so escape every "
                "other literal brace by doubling it ({{ and }})."
            ) from exc
        merged[stage] = template
    return merged


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
    it and Markdown code fences. Returns the parsed object or ``None``.

    Kept as a tolerant fallback: the published TopicGPT prompts emit a bracketed
    line format (parsed by :func:`_parse_topic_lines` / :func:`_parse_assignment_lines`),
    but a backend that ignores the format and returns JSON still degrades cleanly.
    """
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


# The published TopicGPT output format: one topic per line, ``[level] Label: text``.
# The label character class follows the reference parser
# (github.com/chtmp223/topicGPT, generation_1.py: ``r"^\[(\d+)\] ([\w\s\+_#-]+):(.+)"``)
# but is widened to ``[^:]`` so arbitrary labels survive; the leading ``[level]`` and
# the first colon are what anchor a topic line.
_TOPIC_LINE = re.compile(r"^\s*\[(\d+)\]\s*([^:\]]+?)\s*:\s*(.*)$")
# A supporting quote is the trailing parenthesized span of an assignment line,
# e.g. ``[1] Trade: reasoning ("...the quote...")``.
_QUOTE_SPAN = re.compile(r"\(\s*[\"“‘]?(.*?)[\"”’]?\s*\)\s*$", re.DOTALL)

# Leading list/markdown decoration an LLM often wraps a topic line in:
# ``- ``, ``* ``, ``1. ``, ``1) ``, or a leading ``**`` bold marker. We strip
# these before matching so a decorated ``**[1] Sports: ...**`` is not silently
# dropped (github issue #509 finding #3).
_LINE_DECORATION = re.compile(r"^\s*(?:[-*+•]\s+|\d+[.)]\s+)?\*{0,2}\s*")


def _strip_decorations(line: str) -> str:
    """Remove leading bullet/number markers and surrounding ``**`` bold marks so a
    markdown-wrapped topic line still matches :data:`_TOPIC_LINE`."""
    s = _LINE_DECORATION.sub("", line)
    s = re.sub(r"\*{1,2}\s*$", "", s)  # trailing bold close
    return s.strip()


def _is_noise_line(raw: str) -> bool:
    """A blank line or a bare ``None`` reply (the reference's "no change / no
    topic" signal) is expected, not a dropped topic line."""
    s = raw.strip().rstrip(".")
    return not s or s.lower() == "none"


def _parse_topic_lines(text: str) -> tuple[list[tuple[int, str, str]], list[str]]:
    """Parse the reference ``[level] Label: Description`` format into
    ``(level, name, description)`` triples, plus the list of non-noise lines that
    did not parse (so the caller can surface silent drops, issue #509 finding #3).

    A bare ``None`` reply yields no triples and no drops. Markdown/bullet
    decoration is stripped before matching. Falls back to a tolerant JSON parse if
    no bracketed line is present (a backend that ignored the format); a reply that
    parses cleanly as JSON reports no drops.
    """
    out: list[tuple[int, str, str]] = []
    dropped: list[str] = []
    for line in str(text or "").splitlines():
        raw = line.strip()
        m = _TOPIC_LINE.match(_strip_decorations(line))
        name = m.group(2).strip() if m else ""
        if not name:
            if not _is_noise_line(raw):
                dropped.append(raw)
            continue
        out.append((int(m.group(1)), name, m.group(3).strip()))
    if out:
        return out, dropped
    # Tolerant fallback: {"topics": [{topic, description}, ...]} or {topic, description}.
    obj = _extract_json(text)
    if isinstance(obj, dict) and isinstance(obj.get("topics"), list):
        for m in obj["topics"]:
            if isinstance(m, dict) and str(m.get("topic", "")).strip():
                out.append((1, str(m["topic"]).strip(), str(m.get("description", "")).strip()))
    elif isinstance(obj, dict) and str(obj.get("topic", "")).strip():
        out.append((1, str(obj["topic"]).strip(), str(obj.get("description", "")).strip()))
    if out:  # the reply was JSON-shaped, not a dropped line format
        return out, []
    return out, dropped


def _parse_assignment_lines(text: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Parse the reference assignment format
    ``[level] Label: reasoning (Supporting quote)`` into ``(name, quote)`` pairs,
    plus the non-noise lines that did not parse. Falls back to a tolerant JSON
    parse (``{"assignments": [{topic, quote}]}``).
    """
    out: list[tuple[str, str]] = []
    dropped: list[str] = []
    for line in str(text or "").splitlines():
        raw = line.strip()
        m = _TOPIC_LINE.match(_strip_decorations(line))
        name = m.group(2).strip() if m else ""
        if not name:
            if not _is_noise_line(raw):
                dropped.append(raw)
            continue
        q = _QUOTE_SPAN.search(m.group(3))
        out.append((name, q.group(1).strip() if q else ""))
    if out:
        return out, dropped
    obj = _extract_json(text)
    if isinstance(obj, dict) and isinstance(obj.get("assignments"), list):
        for it in obj["assignments"]:
            if isinstance(it, dict) and str(it.get("topic", "")).strip():
                out.append((str(it["topic"]).strip(), str(it.get("quote", "")).strip()))
    if out:
        return out, []
    return out, dropped


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
    through :func:`topica.llm_backend`); supplying both raises. topica showcases
    open-source models (e.g. ``model="ollama/qwen3"`` or an openrouter qwen id),
    but any backend works.

    Determinism is ``llm-bounded``: ``temperature=0`` and a backend ``seed`` give
    *stable*, not bit-reproducible, results. Responses are cached within a fit
    keyed by ``(prompt, model)`` for within-session reproducibility and lower
    cost.

    ``topic_word`` is synthesized by class-based TF-IDF over each topic's assigned
    documents, so it is a *descriptor* for word ranking, not a generative
    ``P(w | topic)``. The model declines ``estimate_effect``,
    ``posterior_theta_samples``, and ``ensemble``: with no theta posterior there
    are no defensible confidence intervals.

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
        Cap on the number of discovered topics carried past refinement (an
        order-truncation, applied after refinement).
    min_topic_count : int, default 1
        Prune topics evoked by fewer than this many documents in the generation
        stage before refinement (the reference's frequency-based rare-topic
        removal). The default of 1 keeps every topic that appeared at least once
        (a no-op); raise it to discard one-off spurious topics. Pruning never
        removes the last topic.
    temperature : float, default 0.0
        Forwarded to :func:`topica.llm_backend` when ``model`` is given.
    seed : int, default 13
        Forwarded to the backend where supported; records intent, not a guarantee
        (the determinism is ``llm-bounded``).
    prompts : dict, optional
        Override the editable prompt templates. Keys are ``"generation"``,
        ``"refinement"``, and ``"assignment"``; any subset is accepted and is
        merged over the published defaults (:data:`PROMPTS`), so you can swap a
        single stage and keep the rest. A custom ``"generation"`` /
        ``"assignment"`` template must keep the ``{taxonomy}`` and ``{document}``
        fields (assignment also ``{n_label}``); ``"refinement"`` keeps
        ``{taxonomy}``. Unknown keys raise. See :meth:`with_prompt` for a
        convenience that overrides one stage.
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
        min_topic_count: int = 1,
        temperature: float = 0.0,
        seed: int = 13,
        prompts: Optional[dict] = None,
    ) -> None:
        if backend is not None and model is not None:
            raise ValueError("pass either backend= or model=, not both")
        if assignment not in ("hard", "soft"):
            raise ValueError('assignment must be "hard" or "soft"')
        if int(min_topic_count) < 1:
            raise ValueError("min_topic_count must be >= 1")
        self._backend_arg = backend
        self._model_name = model
        self.hierarchical = bool(hierarchical)
        self.assignment = assignment
        self.sample = sample
        self.max_topics = max_topics
        self.min_topic_count = int(min_topic_count)
        self.temperature = float(temperature)
        self.seed = int(seed)
        # Merge any overrides over the published defaults so a partial dict (one
        # stage) works; reject unknown keys so a typo surfaces instead of being
        # silently ignored at format time.
        self.prompts = _merge_prompts(prompts)

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
        self.parse_drops: list[str] = []

    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict, keyword-named
        to match ``__init__`` (issue #400). The ``backend`` callable is data, not a
        hyperparameter, so it is not reported; ``model`` records the named backend."""
        return {
            "model": self._model_name,
            "hierarchical": self.hierarchical,
            "assignment": self.assignment,
            "sample": self.sample,
            "max_topics": self.max_topics,
            "min_topic_count": self.min_topic_count,
            "temperature": self.temperature,
            "seed": self.seed,
            "prompts": dict(self.prompts),
        }

    # -- custom prompts ----------------------------------------------------

    def with_prompt(self, stage: str, template: str) -> "TopicGPT":
        """Override one stage's prompt template in place and return ``self``.

        A convenience over ``TopicGPT(prompts={stage: template})`` for swapping a
        single stage, e.g. to adapt the few-shot examples to your domain::

            model = TopicGPT(model="ollama/qwen3").with_prompt(
                "generation", my_generation_template)

        ``stage`` is one of ``"generation"``, ``"refinement"``, ``"assignment"``;
        the template must keep that stage's ``{field}`` placeholders (see
        :data:`PROMPT_FIELDS`). Raises if called on a fitted model (the prompts
        drove the existing fit and must stay auditable alongside it)."""
        if self._fitted:
            raise RuntimeError("set custom prompts before fit(); the fitted prompts are frozen")
        if stage not in PROMPTS:
            raise ValueError(f"unknown stage {stage!r}; valid stages are {sorted(PROMPTS)}")
        self.prompts = _merge_prompts({**{k: self.prompts[k] for k in PROMPTS}, stage: template})
        return self

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
            "(model='ollama/qwen3'), which routes through topica.llm_backend and "
            'needs the optional `llm` package (pip install "topica[llm]"). '
            "topica showcases open-source models (e.g. an ollama or openrouter "
            "qwen); any callable str -> str backend works."
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
        dropped: list[str] = []
        gen_idx = range(n_docs) if self.sample is None else range(min(self.sample, n_docs))
        topics: list[Topic] = []
        seen: dict[str, int] = {}
        counts: list[int] = []  # per-topic document count (frequency signal)
        for i in gen_idx:
            taxonomy = self._render_taxonomy(topics) or "(none yet)"
            prompt = self.prompts["generation"].format(
                taxonomy=taxonomy, document=text_docs[i][:2000]
            )
            # The reference generation prompt may return several top-level topics
            # (newly added and/or existing duplicates); add each unseen one and
            # count how many documents evoke each topic (the pruning signal).
            triples, drops = _parse_topic_lines(self._ask(backend, prompt))
            dropped.extend(drops)
            doc_keys: set[str] = set()  # count each topic at most once per document
            for lvl, name, desc in triples:
                if lvl != 1:  # generation_1 induces top-level topics only
                    continue
                key = _norm(name)
                if key in doc_keys:  # same topic listed twice in one reply
                    continue
                doc_keys.add(key)
                if key in seen:
                    counts[seen[key]] += 1
                    continue
                seen[key] = len(topics)
                topics.append(Topic(name=name, description=desc))
                counts.append(1)
        self.stage_log.append(("generation", len(topics)))

        # Prune rare topics: drop any evoked by fewer than ``min_topic_count``
        # documents (the reference's frequency-based refinement). The default of 1
        # keeps every topic that appeared at least once (a no-op); raise it to
        # discard one-off spurious topics before the LLM merge.
        if self.min_topic_count > 1 and topics:
            kept = [t for t, c in zip(topics, counts) if c >= self.min_topic_count]
            # Prune only when at least one topic survives; never leave zero topics.
            if kept and len(kept) < len(topics):
                self.stage_log.append(("pruned", len(topics) - len(kept)))
                topics = kept

        # Stage 2: refinement -------------------------------------------------
        topics, refine_drops = self._refine(backend, topics)
        dropped.extend(refine_drops)
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
            chosen, adrops = self._parse_assignments(self._ask(backend, prompt), name_to_id)
            dropped.extend(adrops)
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

        # Surface silently-dropped LLM output lines instead of discarding them.
        # A non-noise line the parser could not read is a lost topic/assignment
        # (a bolded line, a stray format, a prose refusal); record the count in
        # ``stage_log`` and warn once so a malformed backend is not invisible.
        self.parse_drops = list(dropped)
        if dropped:
            self.stage_log.append(("parse_drops", len(dropped)))
            warnings.warn(
                f"TopicGPT dropped {len(dropped)} unparseable LLM output line(s) "
                "during fit (they matched no topic/assignment format and were not "
                "a 'None' reply); see the `parse_drops` attribute. This usually "
                "means the backend ignored the requested bracketed-line format.",
                stacklevel=2,
            )

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
        # Emit the reference's bracketed format so the taxonomy the model reads in a
        # prompt matches the format it is asked to produce.
        return "\n".join(f"[1] {t.name}: {t.description}" for t in topics)

    def _refine(self, backend, topics: list[Topic]) -> tuple[list[Topic], list[str]]:
        """Merge near-duplicate topics. Asks the backend once; falls back to the
        input taxonomy if the reply is "None" or unusable. The reference returns
        only the incremental merge edits; topica's refinement prompt instead asks
        for the complete refined list, which we adopt as the new topic set. Returns
        ``(topics, dropped_lines)``."""
        if len(topics) < 2:
            return topics, []
        prompt = self.prompts["refinement"].format(taxonomy=self._render_taxonomy(topics))
        reply = self._ask(backend, prompt)
        # Anchor the "no change" signal so a "Nonetheless, ..." merge reply is not
        # mistaken for a bare "None" (issue #509 finding #7).
        if _is_noise_line(str(reply).splitlines()[0] if str(reply).strip() else ""):
            return topics, []
        out: list[Topic] = []
        seen: set[str] = set()
        triples, dropped = _parse_topic_lines(reply)
        for lvl, name, desc in triples:
            if lvl != 1:  # refinement operates on top-level topics only
                continue
            key = _norm(name)
            if key in seen:
                continue
            seen.add(key)
            out.append(Topic(name=name, description=desc))
        return (out or topics), dropped

    def _parse_assignments(self, reply: str, name_to_id: dict) -> tuple[list[tuple[int, str]], list[str]]:
        out: list[tuple[int, str]] = []
        pairs, dropped = _parse_assignment_lines(reply)
        for name, quote in pairs:
            tid = name_to_id.get(_norm(name))
            if tid is None:
                continue
            out.append((tid, quote))
        return out, dropped

    def _induce_hierarchy(self, backend, topics: list[Topic]) -> dict:
        """A two-level grouping of the discovered topics, reusing the refinement
        prompt to cluster leaf topics into supertopics. A topica convenience, not
        the reference's `generation_2` subtopic induction; each parsed group line
        becomes a supertopic, and a leaf topic whose name matches a group name is
        attached as a child."""
        prompt = self.prompts["refinement"].format(taxonomy=self._render_taxonomy(topics))
        reply = self._ask(backend, prompt)
        name_to_id = {_norm(t.name): j for j, t in enumerate(topics)}
        supers = []
        triples, _drops = _parse_topic_lines(reply)
        for lvl, name, desc in triples:
            nkey = _norm(name)
            children = [name_to_id[nkey]] if nkey in name_to_id else []
            supers.append({"name": name, "description": desc, "children": children})
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
        empty = rowsum[:, 0] == 0.0
        rowsum[rowsum == 0.0] = 1.0
        tw = ctfidf / rowsum
        # A topic that received no documents has an all-zero c-TF-IDF row. Left as
        # zeros it breaks the sum-to-1 descriptor invariant and makes `coherence`
        # return NaN (empty `top_words`). Give it a uniform row so the invariant
        # holds and every per-topic diagnostic stays finite (issue #509 finding #1).
        if empty.any():
            tw[empty, :] = 1.0 / v
        return vocab, tw

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

    def top_words(self, n: int = 10, *, topic: Optional[int] = None, weights: bool = False):
        """Top-``n`` words per topic from the synthesized class-TF-IDF descriptor.

        Returns bare word strings for the given ``topic`` (or one list per topic
        when ``topic`` is None). Pass ``weights=True`` for ``(word, score)`` pairs,
        where the scores are c-TF-IDF weights, not probabilities.
        """
        self._check_fitted()
        from .coherence import _strip_pairs
        phi = self._topic_word
        vocab = self._vocabulary

        def row_words(t: int):
            idx = np.argsort(phi[t])[::-1][:n]
            return [(vocab[j], float(phi[t, j])) for j in idx if phi[t, j] > 0]

        result = row_words(topic) if topic is not None else [row_words(t) for t in range(phi.shape[0])]
        return _strip_pairs(result, weights)

    def coherence(self, n: int = 10) -> np.ndarray:
        """Per-topic c_v coherence of the top-``n`` descriptor words against the
        training corpus. The same windowed measure used across topica's models."""
        self._check_fitted()
        from .coherence import coherence as _coherence

        topics = [list(self.top_words(n, topic=t)) for t in range(self.num_topics)]
        c = np.asarray(_coherence(topics, self._docs_tokens, topn=n), dtype=np.float64)
        # A degenerate topic (e.g. one with no assigned documents, or too few
        # distinct top words) can yield NaN from the windowed measure; report a
        # finite 0.0 sentinel so the health invariant holds (issue #509 finding #1).
        return np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)

    # -- transform: assign held-out docs to the discovered taxonomy --------

    def transform(self, new_docs) -> np.ndarray:
        """Assign held-out documents to the discovered taxonomy via the assignment
        prompt. ``new_docs`` is a list of raw document texts (or a Corpus). Returns
        a (n, K) assignment matrix (one-hot/soft per ``assignment``).

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
            chosen, _drops = self._parse_assignments(self._ask(backend, prompt), name_to_id)
            if not chosen:
                chosen = [(0, "")]
            if self.assignment == "hard":
                chosen = chosen[:1]
            for tid, _q in chosen:
                out[i, tid] += 1.0
            if out[i].sum() > 0:
                out[i] /= out[i].sum()
        return out

    # -- principled declines (no theta posterior) ------------------------------

    def estimate_effect(self, *args, **kwargs):
        """Declined: TopicGPT has no theta posterior, so there are no defensible
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
    "with no posterior over theta, so there are no defensible confidence intervals to "
    "report. This refusal is by design (the 'no CIs without a posterior' "
    "principle). Use a generative model (LDA, STM, ...) when you need {name}."
)
