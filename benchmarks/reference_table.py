"""Companion 'validated against reference' table, built from the FROZEN parity
gold (`parity/*_gold.json`) -- no re-fitting.

The core speed/accuracy table (`full_model_run.py`) covers the models that share a
comparable corpus (20NG / poliblog) so their agreement and speed tabulate together.
The specialised models -- relational (RTM), cross-lingual (PLTM/PolylingualLDA),
contextual-neural (CombinedTM/ZeroShotTM/InfoCTM), sentiment (STS), etc. -- each
live on their own corpus with their own metric, already computed and frozen by the
parity suite. This reads those numbers and renders one honest row per model, split
into:

  * cross-implementation -- agreement against a genuine EXTERNAL reference
    (another package / language), reported as topica-vs-reference cosine and the
    reference's own seed-to-seed self cosine (the ceiling).
  * self-consistency -- no external implementation exists (or it is
    non-reproducible); the gold freezes topica's own output and checks a stability
    bar. Reported as such, never dressed up as external agreement.

Usage:  python benchmarks/reference_table.py [--latex]
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARITY = os.path.join(os.path.dirname(HERE), "parity")

# Field name for the topica-vs-reference agreement varies by gold; try in order.
_AGREE_FIELDS = [
    "topica_vs_reference_cosine", "topica_vs_r_spectral_cosine", "topica_topic_cosine",
    "topica_word_cosine", "topica_mean_cosine", "topica_keyword_cosine",
    "topica_cosine", "cross_ari",
    # NB: purity fields are deliberately NOT agreement (topica_purity can exceed the
    # reference's -- see #610); InfoCTM is recorded via tail_results.json instead.
]
# The reference's own self/ceiling baseline, same order of preference.
_BASE_FIELDS = [
    "reference_self_cosine", "r_self_cosine", "topic_r_self_cosine",
    "mallet_self_cosine", "keyword_r_self_cosine", "reference_purity_a",
    "r_spectral_vs_random", "cross_ari_min", "cosine_bar",
]


def _first(d, fields):
    for f in fields:
        if f in d and isinstance(d[f], (int, float)):
            return f, float(d[f])
    return None, None


def _is_external(d):
    kind = str(d.get("kind", "")).lower()
    if "cross-impl" in kind or "cross implementation" in kind:
        return True
    # A genuine external reference names another package/language, and the gold
    # carries a topica-vs-reference agreement field (not just a self bar).
    ref = str(d.get("reference", "")).lower()
    # An external reference names another package/language; self-gold references
    # name topica itself (or nothing). If the gold carries a topica-vs-reference
    # agreement (which every kept row does) and names a real external tool, it is
    # a cross-implementation comparison.
    return any(t in ref for t in ("r ", "r/", "mallet", "torch", "tomotopy",
                                  "gensim", "sklearn", "cran", "java", "bertopic",
                                  "fastopic", "package", "lda "))


def collect():
    rows = []
    for path in sorted(glob.glob(os.path.join(PARITY, "*_gold.json"))):
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        af, agree = _first(d, _AGREE_FIELDS)
        if agree is None:
            continue
        bf, base = _first(d, _BASE_FIELDS)
        rows.append({
            "file": os.path.basename(path).replace("_gold.json", ""),
            "model": d.get("model", "?"),
            "reference": d.get("reference", "?"),
            "corpus": d.get("corpus", "?"),
            "num_docs": d.get("num_docs") or d.get("n_docs"),
            "k": d.get("num_topics") or d.get("K"),
            "agree": round(agree, 3),
            "agree_field": af,
            "ceiling": round(base, 3) if base is not None else None,
            "external": _is_external(d),
        })
    # Merge live-run tail results (models with no frozen parity gold: BTM,
    # Wordfish, PLTM, ETM, GSDMM) captured once into benchmarks/tail_results.json.
    tail = os.path.join(HERE, "tail_results.json")
    if os.path.exists(tail):
        for t in json.load(open(tail, encoding="utf-8")):
            rows.append({
                "file": t.get("model", "?").split(" ")[0].lower(),
                "model": t.get("model", "?"), "reference": t.get("reference", "?"),
                "corpus": t.get("corpus", "?"), "num_docs": t.get("num_docs"),
                "k": t.get("k"), "agree": t.get("agree"), "agree_field": "live",
                "ceiling": t.get("ceiling"), "external": t.get("external", True),
            })
    rows.sort(key=lambda r: (not r["external"], r["file"]))
    return rows


def render_md(rows):
    def block(title, rs):
        out = [f"\n### {title}\n",
               "| Model | Reference | Corpus | docs | K | Agreement | Ref. ceiling |",
               "|---|---|---|--:|--:|--:|--:|"]
        for r in rs:
            out.append(f"| {r['model']} | {r['reference']} | {r['corpus'][:40]} "
                       f"| {r['num_docs'] or ''} | {r['k'] or ''} | {r['agree']} "
                       f"| {r['ceiling'] if r['ceiling'] is not None else ''} |")
        return "\n".join(out)
    ext = [r for r in rows if r["external"]]
    self_ = [r for r in rows if not r["external"]]
    parts = ["# Validated against reference (frozen parity gold)\n",
             f"{len(ext)} models with an external cross-implementation reference; "
             f"{len(self_)} validated by self-consistency (no reproducible external "
             f"implementation). Agreement = topica-vs-reference topic-word cosine "
             f"(or the metric named in each gold); ceiling = the reference's own "
             f"seed-to-seed self cosine."]
    parts.append(block("Cross-implementation (external reference)", ext))
    parts.append(block("Self-consistency (no reproducible external reference)", self_))
    return "\n".join(parts)


def main():
    rows = collect()
    md = render_md(rows)
    out = os.path.join(HERE, "reference_table.md")
    open(out, "w", encoding="utf-8").write(md + "\n")
    print(md)
    print(f"\n[wrote {out}]")


if __name__ == "__main__":
    main()
