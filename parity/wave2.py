"""Shared scaffolding for the Wave 2 planted self-consistency golds (issue #271).

Wave 2 covers the eleven models that have NO external reference implementation:
HDP, ECTM, DETM, HLDA, PA, ETM, SupervisedLDA, SeededLDA, GSDMM, PT,
EmbeddingLDA. For each, the committed gold is topica's OWN frozen output on a
fixed-seed planted corpus (the SAGE template, ``parity/sage_gold.py``). There is
nothing external to compare against, so the gold locks the exact result against
regression: a refit (same seed, same corpus, same threads) must reproduce the
frozen topic-word matrix in cosine, the planted block structure must be
recovered, and the validity invariants (``tests/invariants.py``) must hold.

The fit CONTRACT for each model (which planted corpus, which covariates / labels /
seeds / times / embeddings) is already encoded, small and known-healthy, in
``tests/test_model_invariants.py``. Rather than re-derive it, this module imports
that module's corpus builders and re-runs the same construction, returning the
fitted MODEL object (the adapters there return only arrays, but the golds need
``vocabulary`` / ``top_words`` / ``doc_paths`` too). The model-construction lines
are copied verbatim from the matching ``_fit_*`` adapter so the two stay aligned.

Recovery metric. The planted corpora label every word with its block in the word
name (``b{B}w{i}``; SupervisedLDA uses ``a*`` / ``g*``). For the topic-word
models we measure *block purity*: align each recovered topic to a planted block
and score the fraction of its top words drawn from that block, plus how many
distinct blocks the topics collectively cover. HLDA has no per-topic word
simplex over blocks (it is a tree), so it uses a path-recovery score instead.
"""

from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np

# Make the test corpus builders importable (they live under tests/).
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TESTS = os.path.join(ROOT, "tests")
if TESTS not in sys.path:
    sys.path.insert(0, TESTS)

import test_model_invariants as tmi  # noqa: E402

K = tmi.K  # 4 — the default planted-block count for the suite


# --------------------------------------------------------------------------- #
# Model-returning fitters. Each reuses tmi's corpus builders verbatim and copies
# only the (one-line) model construction from the matching ``_fit_*`` adapter,
# then returns the fitted model so the gold can read vocabulary / top_words.
# --------------------------------------------------------------------------- #
def fit_hdp(iters=150):
    import topica

    docs, _ = tmi._planted_blocks(k=K, seed=0)
    m = topica.HDP(seed=1, alpha=1.0, gamma=1.0)
    m.fit(docs, iters=iters)
    return m


def fit_ectm(iters=60):
    import topica

    was = topica.experimental_enabled()
    topica.enable_experimental(True)
    try:
        docs, _, _, levels = tmi._covariate_corpus()
        groups = [f"g{l % 2}" for l in levels]
        times = [2000 + (i % 3) for i in range(len(docs))]
        m = topica.ECTM(num_topics=K, seed=1, init="spectral")
        m.fit(docs, times=times, content=groups, iters=iters,
              period_smooth=5.0, interaction_shrink=2.0)
        return m
    finally:
        topica.enable_experimental(was)


def fit_detm(iters=40):
    import topica

    docs, vocab = tmi._planted_blocks(k=K, block=6, n=240, length=20, seed=0)
    _, word_emb = tmi._planted_embeddings(k=K, block=6, seed=0)
    times = np.array([i % 4 for i in range(len(docs))])
    m = topica.DETM(K, delta=0.005, hidden_size=32, lr=0.02, seed=42)
    m.fit(docs, word_emb, vocab, times=times, iters=iters)
    return m


def fit_hlda(iters=300):
    import topica

    shared = ["the", "of", "and"]
    blocks = [[f"b{b}w{i}" for i in range(4)] for b in range(K)]
    docs = []
    for d in range(300):
        docs.append(shared + [blocks[d % K][i] for i in range(4)])
    m = topica.HLDA(depth=2, seed=1)
    m.fit(docs, iters=iters)
    return m


def fit_pa(iters=300):
    import topica

    rng = np.random.default_rng(0)
    blocks = [[f"b{g}w{i}" for i in range(5)] for g in range(4)]
    docs = []
    for _ in range(160):
        pair = (blocks[0], blocks[1]) if rng.random() < 0.5 else (blocks[2], blocks[3])
        doc = []
        for blk in pair:
            doc += [blk[int(rng.integers(5))] for _ in range(6)]
        docs.append(doc)
    m = topica.PA(num_super=2, num_sub=4, seed=1)
    m.fit(docs, iters=iters)
    return m


def fit_etm(iters=80):
    import topica

    docs, vocab = tmi._planted_blocks(k=K, block=8, n=240, length=12, seed=0)
    _, word_emb = tmi._planted_embeddings(k=K, block=8, seed=0)
    m = topica.ETM(num_topics=K, seed=1)
    m.fit(docs, word_emb, vocab, iters=iters)
    return m


def fit_supervisedlda(iters=25):
    import topica

    docs, y = tmi._supervised_corpus()
    m = topica.SupervisedLDA(num_topics=2, seed=7)
    m.fit(docs, y, iters=iters, var_iters=15)
    return m, y


def fit_seededlda(iters=400):
    import topica

    docs, vocab = tmi._planted_blocks(k=K, seed=0)
    seeds = tmi._block_keywords(vocab, k=K)
    m = topica.SeededLDA(seeds, seed=1)
    m.fit(docs, iters=iters)
    return m


def fit_gsdmm(iters=60):
    import topica

    docs = tmi._short_corpus()
    m = topica.GSDMM(num_topics=15, seed=1)
    m.fit(docs, iters=iters)
    return m


def fit_pt(iters=300):
    import topica

    docs = tmi._short_corpus()
    m = topica.PT(num_topics=K, num_pseudo=10, seed=1)
    m.fit(docs, iters=iters)
    return m


# Planted word embeddings frozen into the embedding-model golds (self-contained).
def emb_detm():
    return tmi._planted_embeddings(k=K, block=6, seed=0)[1]


def emb_etm():
    return tmi._planted_embeddings(k=K, block=8, seed=0)[1]


def emb_embeddinglda():
    return tmi._planted_embeddings(k=K, block=8, seed=0)[1]


def fit_embeddinglda(iters=300):
    import topica

    docs, vocab = tmi._planted_blocks(k=K, block=8, n=300, seed=0)
    _, word_emb = tmi._planted_embeddings(k=K, block=8, seed=0)
    m = topica.EmbeddingLDA(num_topics=K, embeddings=word_emb, vocabulary=vocab,
                            top_m=5, seed=1)
    m.fit(docs, iters=iters)
    return m


# --------------------------------------------------------------------------- #
# Topic-word extraction aligned to a fixed vocab order.
# --------------------------------------------------------------------------- #
def topic_word_on(model, vocab_order):
    """Return ``model.topic_word`` reordered onto ``vocab_order`` (K, len(vocab))."""
    vocab = list(model.vocabulary)
    idx = {w: i for i, w in enumerate(vocab)}
    tw = np.asarray(model.topic_word, dtype=float)
    out = np.zeros((tw.shape[0], len(vocab_order)))
    for j, w in enumerate(vocab_order):
        if w in idx:
            out[:, j] = tw[:, idx[w]]
    return out


# --------------------------------------------------------------------------- #
# Planted-block recovery.
# --------------------------------------------------------------------------- #
def block_of_bw(word):
    """``b{B}w{i}`` -> B; returns None for non-block tokens (e.g. HLDA shared)."""
    if word.startswith("b") and "w" in word:
        try:
            return int(word.split("w")[0][1:])
        except ValueError:
            return None
    return None


def block_of_supervised(word):
    """SupervisedLDA corpus: ``a*`` -> block 0, ``g*`` -> block 1."""
    if word.startswith("a"):
        return 0
    if word.startswith("g"):
        return 1
    return None


def block_purity(topic_word, vocab, block_fn, top_n=5):
    """Mean fraction of each topic's top-``top_n`` words drawn from its dominant
    planted block, plus the count of distinct dominant blocks covered.

    A topic whose top words are scattered across blocks (a degenerate / shuffled
    fit) scores low purity; a clean planted recovery scores ~1.0 with coverage
    equal to the number of blocks.
    """
    tw = np.asarray(topic_word, dtype=float)
    purities = []
    dominant = []
    for k in range(tw.shape[0]):
        top = np.argsort(tw[k])[::-1][:top_n]
        blocks = [block_fn(vocab[j]) for j in top]
        blocks = [b for b in blocks if b is not None]
        if not blocks:
            purities.append(0.0)
            continue
        cnt = Counter(blocks)
        dom, n = cnt.most_common(1)[0]
        purities.append(n / len(blocks))
        dominant.append(dom)
    return float(np.mean(purities)), len(set(dominant))


def theta_from_doc_paths(model):
    """HLDA: per-document distribution over tree nodes (spread mass over the path).
    Mirrors ``tests/test_model_invariants.py``'s ``_theta_from_doc_paths``."""
    n_nodes = model.num_nodes
    paths = model.doc_paths
    theta = np.zeros((len(paths), n_nodes))
    for d, path in enumerate(paths):
        for node in path:
            theta[d, node] += 1.0 / len(path)
    return theta, n_nodes


# --------------------------------------------------------------------------- #
# Generic regenerate / compare driver shared by all eleven Wave 2 gold scripts.
#
# A ``Spec`` fully describes one model's gold: how to fit it, how to pull doc-topic
# out, how to score planted recovery, and the frozen pass bars. The per-model
# ``parity/<model>_gold.py`` is then a thin config + ``regenerate()`` / ``run()``
# dispatch onto :func:`regenerate` / :func:`run` below — exactly the SAGE split.
# --------------------------------------------------------------------------- #
import datetime  # noqa: E402

import harness  # noqa: E402  (parity/ is on sys.path when these scripts run)
from invariants import (  # noqa: E402
    assert_finite,
    assert_healthy_theta,
    effective_topics,
)


class Spec:
    """Per-model gold configuration."""

    def __init__(
        self,
        name,
        model_label,
        fit,
        *,
        block_fn=None,
        top_n=5,
        purity_bar=0.6,
        coverage_bar=2,
        cosine_bar=0.99,
        is_hlda=False,
        eff_paths_bar=2.0,
        corpus_desc="",
        embeddings=None,
        extra_meta=None,
    ):
        self.name = name
        self.model_label = model_label
        self.fit = fit
        self.block_fn = block_fn if block_fn is not None else block_of_bw
        self.top_n = top_n
        self.purity_bar = purity_bar
        self.coverage_bar = coverage_bar
        self.cosine_bar = cosine_bar
        self.is_hlda = is_hlda
        self.eff_paths_bar = eff_paths_bar
        self.corpus_desc = corpus_desc
        # ``embeddings``: a zero-arg callable returning the (V, E) word-embedding
        # matrix to FREEZE into the npz for the embedding models (ETM/DETM/
        # EmbeddingLDA), so the committed gold is self-contained.
        self.embeddings = embeddings
        self.extra_meta = extra_meta or {}

    def _fit_model(self):
        out = self.fit()
        return out[0] if isinstance(out, tuple) else out


def _measure(spec, model):
    """Return (vocab, topic_word_on_vocab, theta, n_topics, recovery_dict)."""
    vocab = list(model.vocabulary)
    tw = topic_word_on(model, vocab)
    if spec.is_hlda:
        theta, n_nodes = theta_from_doc_paths(model)
        eff = effective_topics(theta)
        recovery = {"effective_paths": float(eff), "num_nodes": int(n_nodes)}
        return vocab, tw, theta, int(n_nodes), recovery
    theta = np.asarray(model.doc_topic, dtype=float)
    pur, cov = block_purity(tw, vocab, spec.block_fn, spec.top_n)
    recovery = {"block_purity": float(pur), "blocks_covered": int(cov),
                "n_topics": int(tw.shape[0])}
    return vocab, tw, theta, int(tw.shape[0]), recovery


def _recovered(spec, recovery):
    if spec.is_hlda:
        return recovery["effective_paths"] >= spec.eff_paths_bar
    return (recovery["block_purity"] >= spec.purity_bar
            and recovery["blocks_covered"] >= spec.coverage_bar)


def regenerate(spec) -> None:
    model = spec._fit_model()
    vocab, tw, theta, n_topics, recovery = _measure(spec, model)

    arrays = {
        "topic_word": tw.astype(np.float64),       # (K, V) on vocab order
        "doc_topic": theta.astype(np.float64),     # (D, K|nodes)
        "vocab": np.array(vocab, dtype=object),
    }
    if spec.embeddings is not None:
        # Freeze the planted word embeddings so the gold is self-contained.
        arrays["word_embeddings"] = np.asarray(spec.embeddings(), dtype=np.float64)

    harness.save_gold(
        spec.name,
        arrays=arrays,
        meta={
            "reference": f"topica {spec.model_label} (self-consistency / planted-recovery gold)",
            "model": spec.model_label,
            "corpus": spec.corpus_desc,
            "num_topics": n_topics,
            "vocab_size": len(vocab),
            "cosine_bar": spec.cosine_bar,
            "purity_bar": spec.purity_bar,
            "coverage_bar": spec.coverage_bar,
            "eff_paths_bar": spec.eff_paths_bar if spec.is_hlda else None,
            "recovery": recovery,
            "date": datetime.date.today().isoformat(),
            "kind": (
                "PLANTED self-consistency gold (NO external reference exists for "
                f"{spec.model_label}). Locks topica's own fixed-seed output: a refit "
                "must reproduce the frozen topic-word matrix in cosine, the planted "
                "block structure must be recovered, and the validity invariants must "
                "hold; non-vacuous via the shuffle check."
            ),
            "pass_bar": (
                "refit-vs-gold aligned topic-word cosine >= cosine_bar AND planted "
                "recovery clears its bar"
            ),
            **spec.extra_meta,
        },
    )
    npz, js = harness.gold_paths(spec.name)
    print(f"wrote {npz.name} + {js.name} ({npz.stat().st_size} bytes)")
    print(f"  recovery: {recovery}")


def run(spec, verbose: bool = True) -> dict:
    arrays, meta = harness.load_gold(spec.name)
    gold_tw = arrays["topic_word"]
    gold_vocab = list(arrays["vocab"])
    cosine_bar = float(meta.get("cosine_bar", spec.cosine_bar))

    model = spec._fit_model()
    refit_tw = topic_word_on(model, gold_vocab)
    _, _, theta, n_topics, recovery = _measure(spec, model)

    cosine, _ = harness.align_cosine(gold_tw, refit_tw)
    jaccard = harness.top_word_jaccard(gold_tw, refit_tw, n=spec.top_n)
    recovered = _recovered(spec, recovery)

    # Validity invariants (Wave 0), applied to the refit's theta.
    assert_finite(theta, model=spec.name)
    if n_topics > 1:
        assert_healthy_theta(theta, n_topics, model=spec.name)

    result = {
        "cosine": cosine,
        "jaccard": jaccard,
        "cosine_bar": cosine_bar,
        "margin_over_bar": cosine - cosine_bar,
        "recovery": recovery,
        "recovered": bool(recovered),
        "passes": bool(cosine >= cosine_bar and recovered),
    }
    if verbose:
        print(f"gold: {meta.get('reference')}")
        print(f"  refit-vs-gold cosine : {cosine:.4f} (bar {cosine_bar:.2f}, "
              f"jaccard {jaccard:.3f})")
        print(f"  recovery             : {recovery} -> {'OK' if recovered else 'FAIL'}")
        print(f"  verdict: {'PASS' if result['passes'] else 'FAIL'} "
              f"(margin {result['margin_over_bar']:+.4f})")
    return result
