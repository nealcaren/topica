"""LLM topic labeling, as plumbing.

topica assembles the labeling prompt from each topic's top words and
representative documents; you bring the model. The core is dependency-free and
takes any callable ``str -> str`` (:func:`llm_topic_labels` with ``backend=``), so
your own client, an ``ollama`` endpoint, or an API wrapper all work without
topica taking a dependency. :func:`llm_backend` is an optional adapter that lets
you name a model instead of writing the call yourself: it dispatches by the model
name to a lightweight provider SDK (``gpt-*`` -> OpenAI, ``claude-*`` ->
Anthropic, ``gemini-*`` -> Gemini), and reaches any OpenAI-compatible endpoint
(ollama, openrouter, vLLM, ...) via ``base_url=``.

LLM labels are a readable convenience, not a reproducible measurement. Pin the
model and set temperature to 0 for stability, and keep FREX / probability / lift
(:func:`topica.label_topics`) as the defensible, deterministic descriptors.
"""

from __future__ import annotations

import numpy as np

from .analysis import _top_words, representative_docs, set_topic_labels

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


# --- Provider backends: one lightweight SDK per provider, dispatched by model --
# The distribution to `pip install` and the module to import, per provider.
_PROVIDER_DIST = {
    "openai": ("openai", "openai"),
    "anthropic": ("anthropic", "anthropic"),
    "gemini": ("google-genai", "google.genai"),
}


def _detect_provider(model: str, base_url) -> str:
    """Pick a provider from the model name. An explicit ``base_url`` always means
    an OpenAI-compatible endpoint (ollama, openrouter, vLLM, ...), so OpenAI."""
    if base_url is not None:
        return "openai"
    m = model.lower().split("/")[-1]  # tolerate "models/gemini-..." and "anthropic/claude-..."
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gemini"):
        return "gemini"
    # OpenAI is the default: gpt-*, o1/o3/o4-*, chatgpt-*, ft:*, and anything a
    # compatible server exposes.
    return "openai"


def _import_provider(provider: str, func: str):
    import importlib

    dist, mod = _PROVIDER_DIST[provider]
    try:
        return importlib.import_module(mod)
    except ImportError as e:  # pragma: no cover - exercised via message
        raise ImportError(
            f"{func} with a {provider} model needs the optional `{dist}` package "
            f'(pip install {dist}, or pip install "topica[{provider}]" '
            f'/ "topica[llm]").'
        ) from e


def _openai_chat_call(model, *, base_url, key, system, options):
    import os

    openai = _import_provider("openai", "llm_backend")
    api_key = key or os.environ.get("OPENAI_API_KEY")
    if api_key is None and base_url is not None:
        # Local OpenAI-compatible servers ignore auth, but the client refuses to
        # construct without a key.
        api_key = "not-needed"
    client = openai.OpenAI(base_url=base_url, api_key=api_key)

    def call(prompt: str) -> str:
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(model=model, messages=messages, **options)
        return (resp.choices[0].message.content or "").strip()

    return call


def _anthropic_chat_call(model, *, key, system, options):
    anthropic = _import_provider("anthropic", "llm_backend")
    client = anthropic.Anthropic(api_key=key) if key is not None else anthropic.Anthropic()
    opts = dict(options)
    max_tokens = opts.pop("max_tokens", 1024)  # required by the Messages API

    def call(prompt: str) -> str:
        kw = dict(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **opts,
        )
        if system is not None:
            kw["system"] = system
        resp = client.messages.create(**kw)
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "".join(parts).strip()

    return call


def _gemini_chat_call(model, *, key, system, options):
    genai = _import_provider("gemini", "llm_backend")
    from google.genai import types

    client = genai.Client(api_key=key) if key is not None else genai.Client()
    cfg = None
    if system is not None or options:
        cfg = types.GenerateContentConfig(system_instruction=system, **options)

    def call(prompt: str) -> str:
        resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
        return (resp.text or "").strip()

    return call


def llm_backend(model="gpt-4o-mini", *, provider=None, base_url=None, key=None,
                system=None, **options):
    """A ``str -> str`` callable for the ``backend=`` argument of
    :func:`llm_topic_labels`, dispatched to the right lightweight provider SDK by
    the model name:

    - ``gpt-*`` / ``o1-*`` / anything else -> OpenAI (the ``openai`` client)
    - ``claude-*``                         -> Anthropic (the ``anthropic`` client)
    - ``gemini-*``                         -> Gemini (the ``google-genai`` client)

    Each provider is one lightweight, wheel-only dependency — unlike the former
    ``llm`` CLI, which pulled a large tree including a pre-release of
    ``sqlite-migrate`` that breaks ``uv pip install``.

    Pass ``base_url`` to force an OpenAI-compatible endpoint regardless of the
    model name — local models via ``ollama``
    (``base_url="http://localhost:11434/v1"``), plus openrouter, vLLM, LM Studio,
    and groq. ``provider`` overrides auto-detection when a model name is
    ambiguous. ``key`` sets the API key, else the provider's environment variable
    is used (``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` / ``GEMINI_API_KEY``); a
    local server that ignores auth gets a placeholder automatically. ``system`` is
    an optional system prompt, and ``options`` pass through to the provider's call
    (``temperature=0`` is honored by all three for stable labels).

    Needs the matching optional SDK: ``pip install "topica[openai]"``,
    ``"topica[anthropic]"``, ``"topica[gemini]"``, or ``"topica[llm]"`` for all
    three.
    """
    prov = provider or _detect_provider(model, base_url)
    if prov not in ("openai", "anthropic", "gemini"):
        raise ValueError(
            f"unknown provider {prov!r}; expected 'openai', 'anthropic', or "
            "'gemini' (or pass base_url= for an OpenAI-compatible endpoint)."
        )
    if prov == "anthropic":
        return _anthropic_chat_call(model, key=key, system=system, options=options)
    if prov == "gemini":
        return _gemini_chat_call(model, key=key, system=system, options=options)
    return _openai_chat_call(
        model, base_url=base_url, key=key, system=system, options=options
    )


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
    and set temperature to 0, and keep :func:`topica.label_topics` (FREX /
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
