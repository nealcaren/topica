"""The agent/first-timer cheat sheet, rendered live so it cannot drift.

``topica.guide()`` compresses the three artifacts a newcomer (human or LLM) has
to find on their own — the canonical workflow, the goal-to-model chooser, and the
uniform read surface every fitted model exposes — into one screen. It is built at
call time from :mod:`topica.registry` (the roster + the front-door chooser) and
from live ``inspect.signature`` on each model class, so it reflects the compiled
extension exactly and needs no regeneration when a signature changes.

Three views:

* ``build_guide()`` — the one-screen essentials.
* ``build_guide("STM")`` — one model's card (summary, signatures, first calls).
* ``build_guide(full=True)`` — every validated model, grouped, with signatures.

``scripts/gen_guide.py`` renders the same builder into
``docs/guides/agent-cheatsheet.md`` for readers who are not at a REPL; a preflight
``--check`` keeps that page in sync. Because the runtime view is generated on the
fly, only the committed Markdown can go stale, and the check guards it.
"""
from __future__ import annotations

import inspect

from .registry import CHOOSER, GROUPS, REGISTRY

DOCS_BASE = "https://nealcaren.github.io/topica/"

# The uniform read surface (docs/contributing/estimator-contract.md): the same
# attribute names carry the same meaning on every fitted model, so this is one
# block, not a per-model column.
READ_SURFACE = [
    ("model.topic_word", "(K, V) topic-word matrix; rows sum to 1 (generative models)"),
    ("model.doc_topic", "(D, K) document-topic matrix; rows sum to 1"),
    ("model.vocabulary", "V words, aligned to topic_word columns"),
    ("model.top_words(n)", "list[list[str]] top n per topic; weights=True for (word, prob)"),
    ("model.num_topics / .doc_names / .settings", "K, row labels, and the fit config as a dict"),
    ("model.save(path) / Model.load(path)", "round-trip a fitted model to disk"),
]

# The workflow namespaces (issue #757): the taught home of every helper.
NAMESPACES = [
    ("select", "choosing K (search_k, select_model)"),
    ("inspect", "reading topics (label_topics, topic_table, frex, find_thoughts)"),
    ("evaluate", "validation (coherence, exclusivity, topic_stability, perplexity)"),
    ("effects", "covariate effects (estimate_effect, predicted_prevalence)"),
    ("design", "design matrices (one_hot, design_matrix, spline)"),
    ("data", "corpus + bundled datasets (from_dataframe, tokenize, datasets)"),
    ("compare / provenance / embeddings", "two-fit drift, analysis manifest, embedding I/O"),
]


def _model_class(name: str):
    """Resolve a registry name to its class on the ``topica`` namespace.

    Imported lazily: this module is imported while ``topica.__init__`` is still
    executing, so ``topica`` is only fully populated by call time. Returns
    ``None`` if the name does not resolve (e.g. an experimental model behind the
    gate), so the caller can degrade to a bare signature-less line.
    """
    import topica

    return getattr(topica, name, None)


def _sig(obj) -> str | None:
    """A one-line ``inspect.signature`` string, or ``None`` if unavailable.

    PyO3 sets ``text_signature`` on the Rust-backed models, so this works for the
    whole roster; the pure-Python wrappers introspect natively. Anything without a
    signature (a class the interpreter cannot introspect) degrades to ``None``.
    """
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None


def _fit_sig(cls) -> str | None:
    """The ``fit`` signature with the leading ``self`` dropped, if introspectable."""
    fit = getattr(cls, "fit", None)
    if fit is None:
        return None
    s = _sig(fit)
    if s is None:
        return None
    # Drop the bound-method receiver so the rendered call reads as `.fit(data, ...)`.
    return s.replace("(self, /, ", "(").replace("(self, ", "(").replace("(self)", "()")


def _chooser_lines() -> list[str]:
    """The goal-to-model matrix (:data:`CHOOSER`) as aligned plain-text rows."""
    lines = []
    for section, header in (("common", "Common openings"), ("specialized", "Specialized (start here when your design calls for it)")):
        rows = [r for r in CHOOSER if r.section == section]
        if not rows:
            continue
        lines.append(f"  {header}:")
        for r in rows:
            also = f" (or {r.also})" if r.also else ""
            lines.append(f"    {r.goal}")
            lines.append(f"        -> {r.primary}{also}: {r.calls}")
    return lines


def _essentials(show_version: bool = True) -> list[str]:
    import topica

    version = f" {topica.__version__}" if show_version else ""
    out = [
        f"topica{version} - quick guide for agents and first-time users",
        "",
        "THE WORKFLOW (a full LDA analysis in five lines)",
        "    import topica",
        '    corpus = topica.from_dataframe(df, text_col="text")      # build + prune vocab',
        "    res    = topica.select.search_k(corpus, ks=[10, 20, 30])  # choose K; res.best_k()",
        "    model  = topica.LDA(num_topics=res.best_k()).fit(corpus)",
        "    topica.inspect.topic_table(model)                         # labelled topics",
        "    # with metadata:",
        "    topica.effects.estimate_effect(model, X=X, corpus=corpus) # covariate effects + CIs",
        "",
        "EVERY FITTED MODEL EXPOSES THE SAME SURFACE",
    ]
    width = max(len(name) for name, _ in READ_SURFACE)
    for name, desc in READ_SURFACE:
        out.append(f"    {name.ljust(width)}  {desc}")
    out += ["", "PICK A MODEL BY GOAL"]
    out += _chooser_lines()
    out += ["", "HELPER NAMESPACES (topica.<stage>.*)"]
    width = max(len(name) for name, _ in NAMESPACES)
    for name, desc in NAMESPACES:
        out.append(f"    {name.ljust(width)}  {desc}")
    out += [
        "",
        "GO DEEPER",
        '    topica.guide("STM")      one model: signatures + first calls',
        "    topica.guide(full=True)  every model, grouped",
        "    help(topica.STM)         full constructor / fit docstrings",
        "    topica.list_models()     the roster (list_models(group=...) to filter)",
        f"    docs: {DOCS_BASE}",
    ]
    return out


def _card(name: str) -> list[str]:
    """One model's card: taxonomy line, constructor + fit signatures, first calls."""
    info = REGISTRY[name]
    cls = _model_class(name)
    group = GROUPS.get(info.group, info.group)
    out = [f"{name} - {group}   (topica.{name})", "", f"  {info.summary}", ""]
    out.append(
        f"  brings: {', '.join(info.brings) or 'text'}"
        f"    inference: {info.inference}    determinism: {info.determinism}"
    )
    if cls is not None:
        csig = _sig(cls)
        fsig = _fit_sig(cls)
        if csig:
            out.append(f"  {name}{csig}")
        if fsig:
            out.append(f"  model.fit{fsig}")
    # The chooser rows that name this model, so the reader gets the first calls.
    for r in CHOOSER:
        if name in (r.primary, r.also):
            role = "start" if name == r.primary else "alternative"
            out += ["", f'  goal ({role}): "{r.goal}"', f"    first calls: {r.calls}", f"    note: {r.note}"]
    out += ["", f"  docs: {DOCS_BASE}{info.doc}"]
    return out


def _full() -> list[str]:
    """Every validated model, grouped by purpose, with signatures."""
    out = ["topica model reference (validated roster)", ""]
    by_group: dict[str, list[str]] = {}
    for name, info in REGISTRY.items():
        if info.experimental:
            continue
        by_group.setdefault(info.group, []).append(name)
    for key, label in GROUPS.items():
        names = by_group.get(key)
        if not names:
            continue
        out += [f"### {label}", ""]
        for name in names:
            info = REGISTRY[name]
            cls = _model_class(name)
            csig = _sig(cls) if cls is not None else None
            out.append(f"{name}{csig or '(...)'}")
            out.append(f"    {info.summary}")
            fsig = _fit_sig(cls) if cls is not None else None
            if fsig:
                out.append(f"    .fit{fsig}")
            out.append("")
    out += [
        "Experimental models are omitted; enable_experimental() then "
        'list_models(experimental=True) to see them.',
    ]
    return out


def build_guide(topic: str | None = None, *, full: bool = False, show_version: bool = True) -> str:
    """Render the cheat sheet as a string (see :func:`topica.guide`).

    Parameters
    ----------
    topic : a model name (case-insensitive) for that model's card, or ``None``
        for the one-screen essentials.
    full : render every validated model with signatures, grouped by purpose.
        Ignored when ``topic`` is given.
    show_version : include the running version in the essentials header. The docs
        generator passes ``False`` so the committed page does not churn on a
        version bump.
    """
    if topic is not None:
        match = _resolve_name(topic)
        if match is None:
            near = _did_you_mean(topic)
            hint = f" Did you mean: {', '.join(near)}?" if near else ""
            return (
                f"No model named {topic!r} in the roster.{hint}\n"
                "Run topica.list_models() for the full list."
            )
        return "\n".join(_card(match))
    if full:
        return "\n".join(_full())
    return "\n".join(_essentials(show_version=show_version))


def _resolve_name(topic: str) -> str | None:
    """Case-insensitive exact match of ``topic`` to a registry key."""
    lowered = topic.lower()
    for name in REGISTRY:
        if name.lower() == lowered:
            return name
    return None


def _did_you_mean(topic: str) -> list[str]:
    """Up to three registry names whose lowercase form contains, or is contained
    by, the query — a cheap substring suggestion, no fuzzy-match dependency."""
    lowered = topic.lower()
    hits = [n for n in REGISTRY if lowered in n.lower() or n.lower() in lowered]
    return hits[:3]
