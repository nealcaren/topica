"""LLM topic labeling, as plumbing.

topica assembles the labeling prompt from each topic's top words and
representative documents; you bring the model. The core is dependency-free and
takes any callable ``str -> str`` (:func:`llm_topic_labels` with ``backend=``), so
your own client, an ``ollama`` endpoint, or an API wrapper all work without
topica taking a dependency. :func:`llm_backend` is an optional adapter for Simon
Willison's `llm <https://llm.datasette.io/>`_ library, so you can name a model
instead of writing the call yourself — one optional dependency that reaches every
provider and local models via plugins.

LLM labels are a readable convenience, not a reproducible measurement. Pin the
model and set temperature to 0 for stability, and keep FREX / probability / lift
(:func:`topica.inspect.label_topics`) as the defensible, deterministic descriptors.
"""

from __future__ import annotations

import numpy as np

from .analysis import _top_words, representative_docs, set_topic_labels

def _unknown_model_message(_llm, model, kind="chat") -> str:
    """An actionable error for a model `llm` does not know — the failure a user
    hits when the needed plugin, server, or pulled model is missing."""
    try:
        getter = _llm.get_models if kind == "chat" else _llm.get_embedding_models
        known = ", ".join(m.model_id for m in list(getter())[:6])
    except Exception:  # pragma: no cover - defensive
        known = "(could not list installed models)"
    plugin = "llm-ollama" if kind == "chat" else "llm-sentence-transformers"
    return (
        f"llm has no {kind} model named {model!r}. OpenAI models work once "
        f"OPENAI_API_KEY is set; for a local model install a plugin and make the "
        f"model available — e.g. `pip install {plugin}`, and for ollama run the "
        f"server and `ollama pull {model}`. Known {kind} models include: {known}."
    )


_DEFAULT_INSTRUCTIONS = (
    "You are labeling topics from a topic model fit on a document collection. "
    "Given a topic's most characteristic words and a few representative "
    "documents, write a short, specific label of 2 to 5 words that captures the "
    "theme. Reply with only the label: no punctuation, quotes, or explanation."
)


def topic_label_prompts(model, texts=None, *, n_words=12, n_docs=3, max_chars=300,
                        instructions=None):
    """One labeling prompt per topic — exactly the text a model is asked to label.

    Each prompt lists the topic's top ``n_words`` words and, when ``texts`` is
    given, up to ``n_docs`` representative documents (each whitespace-collapsed
    and truncated to ``max_chars``). ``instructions`` overrides the default task
    framing. Returns a list of prompt strings, one per topic.

    This is the plumbing behind :func:`llm_topic_labels`; build it yourself to see
    or adjust what the model sees, or to drive a model topica does not know about.
    """
    instr = instructions or _DEFAULT_INSTRUCTIONS
    k = np.asarray(model.topic_word).shape[0]
    prompts = []
    for t in range(k):
        words = _top_words(model, t, n_words)
        lines = [instr, "", "Top words: " + ", ".join(words)]
        if texts is not None:
            docs = representative_docs(model, texts, topic=t, n=n_docs)
            if docs:
                lines += ["", "Representative documents:"]
                for d in docs:
                    d = " ".join(str(d).split())
                    if len(d) > max_chars:
                        d = d[: max_chars - 1] + "…"
                    lines.append(f"- {d}")
        lines += ["", "Label:"]
        prompts.append("\n".join(lines))
    return prompts


def llm_backend(model="gpt-4o-mini", *, key=None, system=None, **options):
    """A ``str -> str`` callable backed by the `llm` library, for the ``backend=``
    argument of :func:`llm_topic_labels`.

    ``model`` names any model ``llm`` can reach — OpenAI, Anthropic, or local
    models through plugins such as ``llm-ollama``. By default the API key is
    resolved by ``llm`` itself: a stored ``llm keys`` value, else the provider's
    environment variable (``OPENAI_API_KEY`` for OpenAI). Pass ``key`` to override
    that with an explicit key. ``options`` pass through to ``llm`` (e.g.
    ``temperature=0`` for reproducible labels where the provider supports it).
    Requires the optional ``llm`` package (``pip install llm`` or
    ``pip install "topica[llm]"``).
    """
    try:
        import llm as _llm
    except ImportError as e:  # pragma: no cover - exercised via message
        raise ImportError(
            "llm_backend needs the optional `llm` package "
            '(pip install llm, or pip install "topica[llm]").'
        ) from e
    try:
        obj = _llm.get_model(model)
    except Exception as e:
        unknown = getattr(_llm, "UnknownModelError", None)
        if unknown is not None and not isinstance(e, unknown):
            raise
        raise ValueError(_unknown_model_message(_llm, model, kind="chat")) from e
    if key is not None:
        obj.key = key

    def call(prompt: str) -> str:
        return obj.prompt(prompt, system=system, **options).text().strip()

    return call


def llm_topic_labels(model, texts=None, *, backend=None, llm_model="gpt-4o-mini",
                     n_words=12, n_docs=3, max_chars=300, instructions=None,
                     set_labels=False):
    """A short, human-readable label for each topic, generated by an LLM.

    For each topic, assembles a prompt from its top words and representative
    documents (see :func:`topic_label_prompts`) and asks a model for a concise
    label. Returns a list of labels, one per topic.

    Supply the model one of two ways:

    - ``backend``: any callable ``str(prompt) -> str(label)`` — your own client,
      ``ollama``, whatever, or :func:`topica.llm_backend` / :func:`topica.llm.backend`.
      Zero extra dependencies; you own determinism.
    - otherwise ``llm_model`` names a model used through :func:`llm_backend` (the
      ``topica[llm]`` extra). ``backend`` takes precedence when both are given.

    With ``set_labels=True`` the labels are stored via
    :func:`topica.set_topic_labels`, so they flow into :func:`topica.topic_info`,
    :func:`topica.topic_labels`, and :func:`topica.plot_report`.

    LLM labels are a convenience, not a reproducible measurement: pin the model
    and set temperature to 0, and keep :func:`topica.inspect.label_topics` (FREX /
    probability / lift) for the defensible descriptors.
    """
    fn = backend if backend is not None else llm_backend(llm_model)
    prompts = topic_label_prompts(
        model, texts, n_words=n_words, n_docs=n_docs, max_chars=max_chars,
        instructions=instructions,
    )
    labels = [str(fn(p)).strip() for p in prompts]
    if set_labels:
        set_topic_labels(model, {t: lab for t, lab in enumerate(labels)})
    return labels
