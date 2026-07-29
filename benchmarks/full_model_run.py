"""Full-model accuracy + timing run.

For every topica model that has a reference implementation, fit topica and the
reference on a shared corpus and record BOTH the aligned topic-word accuracy and
the wall-clock time (+ peak RSS). Some models appear more than once (option or K
variants) where those materially change accuracy or speed.

Reference results (the expensive non-topica fits) are CACHED to
``benchmarks/refcache/<key>.npz`` so the reference is fit once; the harness then
loads the cached topic-word matrix (for the machine-independent accuracy metric)
and the cached wall-clock/RSS (for the timing ratio, valid because the cache was
built on this machine). Rebuild the cache only when a reference version changes.

Modes:
  --build-refs [KEY ...]   fit the reference(s), populate the cache (slow)
  --accuracy   [KEY ...]   load cached refs, fit topica, score accuracy + topica time (fast)
  --time       [KEY ...]   fit topica AND reference fresh (same machine) for the timing ratio
  --list                   list registry keys
With no KEY, all registered entries run. Missing reference toolchains SKIP cleanly.

Output: a markdown table  model | reference | K | accuracy | topica_s | ref_s | speedup | topica_MB / ref_MB
written to benchmarks/full_model_run.md (+ the raw JSON alongside).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import resource
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper"))
sys.path.insert(0, str(ROOT / "parity"))
import gen_validation_appendix as G  # poliblog(), realign_to(), align_pairs()

CACHE = Path(__file__).resolve().parent / "refcache"
CACHE.mkdir(exist_ok=True)
OUT_MD = Path(__file__).resolve().parent / "full_model_run.md"
OUT_JSON = Path(__file__).resolve().parent / "full_model_run.json"
OUT_TEX = Path(__file__).resolve().parent / "full_model_run.tex"

MULTI = min(8, os.cpu_count() or 1)   # fixed multi-thread count


def _configs(ks, threadable):
    """Yield (k, threads) configs: each k single-threaded, plus multi for threadable models."""
    for k in ks:
        yield (k, 1)
        if threadable and MULTI > 1 and not os.environ.get("BENCH_SMOKE") == "1":
            yield (k, MULTI)


def _rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024  # bytes vs KB


def timed(fn):
    """Run fn(), return (result, wall_seconds, peak_rss_delta_mb)."""
    r0 = _rss_mb()
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    return out, dt, max(0.0, _rss_mb() - r0)


from contextlib import contextmanager


@contextmanager
def pin_threads(n):
    """Pin BLAS/OpenMP (and torch intra-op) thread counts to `n` for the enclosed
    fit. Reference implementations (sklearn NMF/SVD, torch, R via BLAS) have no
    `workers` argument and otherwise silently use every core, so a row labelled
    "single-threaded" would really be topica-1-core vs reference-all-cores. This
    makes the thread count honest on both sides. topica's Rust core does not read
    these limits (it threads via its own `num_threads=` and a single-threaded
    matmul), so pinning only constrains the reference/numpy side, which is the
    point."""
    cm = None
    try:
        from threadpoolctl import threadpool_limits
        cm = threadpool_limits(limits=n)
    except Exception:
        cm = None
    torch_prev = None
    try:
        import torch
        torch_prev = torch.get_num_threads()
        torch.set_num_threads(n)
    except Exception:
        torch_prev = None
    try:
        if cm is not None:
            with cm:
                yield
        else:
            yield
    finally:
        if torch_prev is not None:
            try:
                import torch
                torch.set_num_threads(torch_prev)
            except Exception:
                pass


def accuracy(ref_tw, ref_vocab, top_tw, top_vocab):
    """Aligned topic-word cosine on the shared vocabulary."""
    top_on_ref = G.realign_to(ref_vocab, top_vocab, np.asarray(top_tw))
    _, cos = G.align_pairs(np.asarray(ref_tw), top_on_ref)
    return float(cos)


def _topwords(tw, vocab, n=10):
    tw = np.asarray(tw)
    vocab = list(vocab)
    return [[vocab[i] for i in row.argsort()[::-1][:n]] for row in tw]


def _quality(tw, vocab, texts, topn=10):
    """Intrinsic topic quality for a fitted (topic-word, vocab) on `texts`:
    (c_v, c_npmi, topic-diversity/TU) using topica's own scorers -- the same
    measures the neural-topic-model papers report. Recorded for BOTH topica and
    the reference so a coherence/diversity leaderboard can be rendered later
    without re-fitting anything. Best-effort: any scorer hiccup yields None
    rather than sinking the row."""
    if tw is None or texts is None:
        return {"cv": None, "npmi": None, "tu": None}
    try:
        import topica as _t
        topics = _topwords(tw, vocab, topn)
        cv = np.mean(_t.coherence(topics, texts, coherence_type="c_v", topn=topn))
        npmi = np.mean(_t.coherence(topics, texts, coherence_type="c_npmi", topn=topn))
        return {
            "cv": round(float(cv), 4),
            "npmi": round(float(npmi), 4),
            "tu": round(float(_t.topic_diversity(topics, topn=topn)), 4),
        }
    except Exception:
        return {"cv": None, "npmi": None, "tu": None}


def _texts(corpus):
    """Tokenized documents for a corpus tag, for the coherence scorers. Returns
    None for the specialised per-model corpora (network/bilingual/planted) whose
    texts live in their parity module -- quality is then skipped for that row."""
    if corpus == "ng":
        return _ng_docs()
    if corpus == "poliblog":
        return _poliblog()[0]
    return None


# --------------------------------------------------------------------------- #
# Reference cache
# --------------------------------------------------------------------------- #
def cache_path(key):
    return CACHE / f"{key}.npz"


def store_ref(key, tw, vocab, sec, rss):
    np.savez(cache_path(key), tw=np.asarray(tw, float),
             vocab=np.array(list(vocab), dtype=object), sec=sec, rss=rss)


def load_ref(key):
    p = cache_path(key)
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return dict(tw=d["tw"], vocab=list(d["vocab"]), sec=float(d["sec"]), rss=float(d["rss"]))


# --------------------------------------------------------------------------- #
# Registry: each entry provides topica() and ref() -> (topic_word, vocab).
# `avail()` gates on the reference toolchain. `k` is reported.
# --------------------------------------------------------------------------- #
REGISTRY = {}


def register(key, model, k, topica, refs, note="", metric_fn=None, threads=1,
             corpus="poliblog"):
    """Register a model entry.

    `refs` maps a reference label -> (avail_fn, ref_fn), so a model can be scored
    against several originals (e.g. LDA vs both tomotopy and MALLET); each produces
    its own row and its own cached reference. `topica` and each `ref_fn` return
    whatever `metric_fn` consumes — by default `(topic_word_KxV, vocab)` scored by
    aligned topic-word cosine. `metric_fn(ref_result, topica_result) -> float`
    overrides that (e.g. adjusted Rand index for the embedding-cluster models).

    The entry key embeds k and threads so distinct (k, threads) configs of the same
    model do not collide; the ref cache key (`f"{entry_key}__{ref_label}"`) then
    stays unique per config automatically.
    """
    entry_key = f"{key}__k{k}__t{threads}"
    REGISTRY[entry_key] = dict(key=entry_key, model=model, k=k, threads=threads,
                               topica=topica, refs=refs, note=note,
                               metric_fn=metric_fn, corpus=corpus)


_POLIBLOG5K = os.path.join(os.path.dirname(os.path.abspath(__file__)), "poliblog5k_prepped.csv")
_MIN_DF = 3
_poliblog_cache = None


def _poliblog():
    """The whole benchmark runs on the 5000-doc poliblog (`poliblog5k_prepped.csv`)
    rather than the 2000-doc sample, so K=25 has real support (~200 docs/topic) and
    the speed numbers are representative. Same schema (rating/day/text, already
    stemmed) and the same light doc-frequency prune (>=3) the parity loaders use.
    Returns (docs, rating_lib, day). Cached across the run."""
    global _poliblog_cache
    if _poliblog_cache is not None:
        return _poliblog_cache
    import csv as _csv
    from collections import Counter as _Counter
    if not os.path.exists(_POLIBLOG5K):
        # Fall back to the 2000-doc sample if the 5k export isn't present.
        docs, rating_lib, day, _ = G.poliblog()
        _poliblog_cache = (docs, rating_lib, day)
        return _poliblog_cache
    rows = list(_csv.DictReader(open(_POLIBLOG5K, newline="")))
    toks = [r["text"].split() for r in rows]
    df = _Counter()
    for d in toks:
        df.update(set(d))
    vocab = {w for w, c in df.items() if c >= _MIN_DF}
    kept = [(i, [w for w in d if w in vocab]) for i, d in enumerate(toks)]
    kept = [(i, d) for i, d in kept if d]
    docs = [d for _, d in kept]
    rating_lib = np.array([1.0 if rows[i]["rating"] == "Liberal" else 0.0 for i, _ in kept])
    day = np.array([float(rows[i]["day"]) for i, _ in kept])
    if _SMOKE:
        idx = _smoke_idx(len(docs))
        docs = [docs[i] for i in idx]
        rating_lib, day = rating_lib[idx], day[idx]
    _poliblog_cache = (docs, rating_lib, day)
    return _poliblog_cache


# 20 Newsgroups: the large, diverse corpus where K=50 is substantively meaningful
# (~376 docs/topic at K=50). Covariate/supervised models stay on poliblog (they
# need rating/day); every plain topic model runs here. Cached across the run.
_NG_MIN_DF = 10          # 18k docs -> prune words in <10 docs
_NG_MAX_DF = 0.5         # and words in >50% of docs
_twentyng_cache = None
_ng_names = None         # 20 newsgroup names (for LabeledLDA labels)

# Smoke mode (BENCH_SMOKE=1): subsample every corpus to ~400 docs and register
# every model at K=5, single-thread -- a fast end-to-end pass to prove each model
# produces a row before the multi-hour real run. Corpora are strided (not head-
# sliced) so 20NG's group-ordered docs stay diverse.
_SMOKE = os.environ.get("BENCH_SMOKE") == "1"
_SMOKE_N = int(os.environ.get("BENCH_SMOKE_N", "400"))


def _smoke_idx(n):
    return np.linspace(0, n - 1, min(_SMOKE_N, n)).astype(int)


def _twentyng():
    """Returns (docs, labels): tokenized 20NG documents and their newsgroup index
    (0..19, for LabeledLDA / DMR-categorical). Light preprocessing: lowercase,
    alphabetic tokens of length >=3, sklearn English stopwords, document-frequency
    prune [_NG_MIN_DF, _NG_MAX_DF]. No stemming (topica does not require it)."""
    global _twentyng_cache, _ng_names
    if _twentyng_cache is not None:
        return _twentyng_cache
    import re as _re
    from collections import Counter as _Counter
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    d = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    _ng_names = list(d.target_names)
    tok = _re.compile(r"[a-z]{3,}")
    toks = [[w for w in tok.findall(t.lower()) if w not in ENGLISH_STOP_WORDS]
            for t in d.data]
    n = len(toks)
    df = _Counter()
    for t in toks:
        df.update(set(t))
    lo, hi = _NG_MIN_DF, int(_NG_MAX_DF * n)
    vocab = {w for w, c in df.items() if lo <= c <= hi}
    kept = [(i, [w for w in t if w in vocab]) for i, t in enumerate(toks)]
    kept = [(i, t) for i, t in kept if t]
    docs = [t for _, t in kept]
    labels = np.array([int(d.target[i]) for i, _ in kept])
    if _SMOKE:
        idx = _smoke_idx(len(docs))
        docs = [docs[i] for i in idx]
        labels = labels[idx]
    _twentyng_cache = (docs, labels)
    return _twentyng_cache


def _ng_docs():
    """Just the 20NG token lists (the plain topic models ignore covariates)."""
    return _twentyng()[0]


def _ng_docs_capped(n=2000):
    """A fixed, seed-stable subset of the 20NG token lists. HLDA's tree fit is only
    tractable on a capped corpus (#611); topica and the reference both call this, so
    they see the same documents in the same order (required by the cross-NMI metric)."""
    docs = _ng_docs()
    if _SMOKE or n >= len(docs):
        return docs
    idx = np.sort(np.random.default_rng(0).choice(len(docs), n, replace=False))
    return [docs[i] for i in idx]


def _ng_labeled():
    """(docs, per-doc [label], label_names) for LabeledLDA on the 20 newsgroups."""
    docs, labels = _twentyng()
    return docs, [[_ng_names[i]] for i in labels], list(_ng_names)


# ---- tomotopy family (shared reference fitter) ---------------------------- #
def _tomo_available():
    try:
        import tomotopy  # noqa
        return True
    except ImportError:
        return False


def _tomo_fit(build, k, workers=1):
    """build(tp, k) -> a trained tomotopy model exposing get_topic_word_dist/used_vocabs."""
    import tomotopy as tp
    m = build(tp, k)
    m.burn_in = 200
    m.train(1000, workers=workers, show_progress=False)
    vocab = list(m.used_vocabs)
    tw = np.array([m.get_topic_word_dist(t) for t in range(k)])
    return tw, vocab


def _mallet_available():
    try:
        import mallet_parity as mp
        return mp.mallet_available()
    except Exception:
        return False


def _mallet_lda(k):
    import mallet_parity as mp
    docs = _ng_docs()
    mphi, mvocab = mp._mallet_phi(docs, k, iters=1000, seed=1)
    return np.asarray(mphi), list(mvocab)


def _register_lda(k=10, threads=1):
    def topica():
        from topica import LDA
        docs = _ng_docs()
        m = LDA(num_topics=k, seed=1, optimize_interval=0, num_threads=threads)
        m.fit(docs, iters=1000, num_samples=5, sample_interval=25)
        return np.asarray(m.topic_word), list(m.vocabulary)

    def ref_tomo():
        docs = _ng_docs()
        def build(tp, k):
            m = tp.LDAModel(tw=tp.TermWeight.ONE, k=k, seed=1)
            for d in docs:
                m.add_doc(d)
            return m
        return _tomo_fit(build, k, workers=threads)

    refs = {"tomotopy": (_tomo_available, ref_tomo)}
    # MALLET: mallet_parity._mallet_phi takes no threads arg, so it can only run
    # single-threaded. Include it only in the t=1 configs -- a MALLET row under a
    # t>1 config would be topica-multi vs MALLET-single, which is misleading.
    if threads == 1:
        refs["MALLET"] = (_mallet_available, lambda: _mallet_lda(k))
    register("lda", "LDA", k, topica, refs=refs, threads=threads, corpus="ng")


def _register_dmr(k=10, threads=1):
    def topica():
        from topica import DMR
        docs, rating_lib, _ = _poliblog()
        X = np.array(rating_lib, dtype=float).reshape(-1, 1)
        # DMR exposes no num_threads knob (neither constructor nor fit), so the
        # topica side runs single-threaded even at threads>1; only the ref threads.
        m = DMR(num_topics=k, seed=1)
        m.fit(docs, X, feature_names=["ratingLiberal"], iters=1000)
        return np.asarray(m.topic_word), list(m.vocabulary)

    def ref_tomo():
        docs, rating_lib, _ = _poliblog()
        rating = ["Liberal" if r else "Conservative" for r in rating_lib]
        def build(tp, k):
            m = tp.DMRModel(tw=tp.TermWeight.ONE, k=k, seed=1)
            for d, r in zip(docs, rating):
                m.add_doc(d, metadata=r)
            return m
        return _tomo_fit(build, k, workers=threads)

    register("dmr", "DMR", k, topica, refs={"tomotopy": (_tomo_available, ref_tomo)},
             note="covariate prior; at/above ref seed ceiling (#564)", threads=threads)


def _register_gdmr(k=10, threads=1):
    deg = [3]

    def topica():
        from topica import GDMR
        docs, _, day = _poliblog()
        day = np.asarray(day, dtype=float)
        x = (day - day.min()) / (day.max() - day.min() + 1e-12)
        # GDMR (Python wrapper over DMR) exposes no num_threads knob either, so the
        # topica side runs single-threaded even at threads>1; only the ref threads.
        g = GDMR(num_topics=k, degrees=deg, seed=1)
        g.fit(docs, x.reshape(-1, 1), iters=1000)
        return np.asarray(g.topic_word), list(g.vocabulary)

    def ref():
        import tomotopy as tp
        docs, _, day = _poliblog()
        day = np.asarray(day, dtype=float)
        x = (day - day.min()) / (day.max() - day.min() + 1e-12)
        m = tp.GDMRModel(tw=tp.TermWeight.ONE, k=k, degrees=deg, seed=1)
        for d, xx in zip(docs, x):
            m.add_doc(d, numeric_metadata=[float(xx)])
        m.burn_in = 200
        m.train(800, workers=threads, show_progress=False)
        return np.array([m.get_topic_word_dist(t) for t in range(k)]), list(m.used_vocabs)

    register("gdmr", "GDMR", k, topica, refs={"tomotopy": (_tomo_available, ref)},
             note="covariate prior; at/above ref seed ceiling (#564)", threads=threads)


def _register_slda(k=10, threads=1):
    def topica():
        from topica import SupervisedLDA
        docs, rating_lib, _ = _poliblog()
        m = SupervisedLDA(num_topics=k, seed=1)  # no num_threads knob
        m.fit(docs, [float(r) for r in rating_lib], iters=40)
        return np.asarray(m.topic_word), list(m.vocabulary)

    def ref():
        import tomotopy as tp
        docs, rating_lib, _ = _poliblog()
        m = tp.SLDAModel(tw=tp.TermWeight.ONE, k=k, vars=["l"], seed=1)
        for d, r in zip(docs, rating_lib):
            m.add_doc(d, y=[float(r)])
        m.burn_in = 200
        m.train(1000, workers=threads, show_progress=False)
        return np.array([m.get_topic_word_dist(t) for t in range(k)]), list(m.used_vocabs)

    register("slda", "SupervisedLDA (var)", k, topica,
             refs={"tomotopy": (_tomo_available, ref)},
             note="variational EM (Blei-McAuliffe original)", threads=threads)


def _register_slda_gibbs(k=10, threads=1):
    def topica():
        from topica import SupervisedLDA
        docs, rating_lib, _ = _poliblog()
        m = SupervisedLDA(num_topics=k, seed=1, inference="gibbs")  # no num_threads knob
        m.fit(docs, [float(r) for r in rating_lib], iters=1000)
        return np.asarray(m.topic_word), list(m.vocabulary)

    def ref():
        import tomotopy as tp
        docs, rating_lib, _ = _poliblog()
        m = tp.SLDAModel(tw=tp.TermWeight.ONE, k=k, vars=["l"], seed=1)
        for d, r in zip(docs, rating_lib):
            m.add_doc(d, y=[float(r)])
        m.burn_in = 200
        m.train(1000, workers=threads, show_progress=False)
        return np.array([m.get_topic_word_dist(t) for t in range(k)]), list(m.used_vocabs)

    register("slda_gibbs", "SupervisedLDA (gibbs)", k, topica,
             refs={"tomotopy": (_tomo_available, ref)},
             note="collapsed Gibbs, matches tomotopy", threads=threads)


def _register_pa(k=10, threads=1):
    num_super = 3

    def topica():
        from topica import PA
        docs = _ng_docs()
        p = PA(num_super, k, seed=1)  # no num_threads knob
        p.fit(docs, iters=1000)
        return np.asarray(p.topic_word), list(p.vocabulary)

    def ref():
        docs = _ng_docs()
        def build(tp, k):
            m = tp.PAModel(tw=tp.TermWeight.ONE, k1=num_super, k2=k, seed=1)
            for d in docs:
                m.add_doc(d)
            return m
        return _tomo_fit(build, k, workers=threads)

    register("pa", "PA", k, topica, refs={"tomotopy": (_tomo_available, ref)},
             note=f"{num_super} super over {k} sub; sub compared", threads=threads,
             corpus="ng")


def _register_ctm(k=10, threads=1):
    def topica():
        from topica import CTM
        docs = _ng_docs()
        m = CTM(num_topics=k, seed=1)
        m.fit(docs, iters=500)
        return np.asarray(m.topic_word), list(m.vocabulary)

    def ref():
        docs = _ng_docs()
        def build(tp, k):
            m = tp.CTModel(tw=tp.TermWeight.ONE, k=k, seed=1)
            for d in docs:
                m.add_doc(d)
            return m
        return _tomo_fit(build, k, workers=threads)

    register("ctm", "CTM", k, topica, refs={"tomotopy": (_tomo_available, ref)},
             note="topica variational vs tomotopy Gibbs CTM (same model, diff. inference)",
             threads=threads, corpus="ng")


def _register_hdp():
    """HDP discovers its own K on both sides, so an aligned topic-word cosine over
    two different topic sets is not meaningful (#611). Score by cross-NMI of the
    per-document dominant-topic assignments — agreement on the document clustering.
    Run on a capped 2k-doc subset: topica now estimates the DP concentrations by
    default (#617, faithful to blei-lab/hdp), so it discovers a rich topic set and
    each Gibbs sweep is O(K) per token — the full corpus is impractical."""
    INIT_K = 10
    N = 2000

    def topica():
        from topica import HDP
        docs = _ng_docs_capped(N)
        m = HDP(seed=1)
        m.fit(docs, iters=150)
        return (np.asarray(m.doc_topic).argmax(axis=1).astype(int),)

    def ref():
        import tomotopy as tp
        docs = _ng_docs_capped(N)
        m = tp.HDPModel(tw=tp.TermWeight.ONE, initial_k=INIT_K, seed=1)
        for d in docs:
            m.add_doc(d)
        m.burn_in = 200
        m.train(1000, workers=1, show_progress=False)
        assign = [int(np.asarray(doc.get_topic_dist()).argmax()) for doc in m.docs]
        return (np.asarray(assign, dtype=int),)

    register("hdp", "HDP", 0, topica, refs={"tomotopy": (_tomo_available, ref)},
             note=f"nonparametric; K discovered both sides on a {N}-doc subset; "
                  "cross-NMI of doc assignments (topic-word cosine not comparable). "
                  "topica estimates the DP concentrations by default (#617); tomotopy "
                  "finds fewer topics via a simplified new-table weight, so the two "
                  "clusterings agree only partially.",
             metric_fn=_cross_nmi, corpus="ng")


def _register_hlda(threads=1):
    """HLDA grows a depth-3 topic tree and discovers its own node count, so an
    aligned topic-word cosine across two independently grown trees is not
    meaningful (#611). Score by cross-NMI of the per-document deepest-topic
    assignments instead — how much the two implementations agree on the document
    clustering. Run on a capped 2k-doc subset so the tree fit is tractable; HLDA
    now parallelises via `num_threads`."""
    DEPTH = 3
    N = 2000

    def topica():
        from topica import HLDA
        docs = _ng_docs_capped(N)
        m = HLDA(depth=DEPTH, seed=1)
        m.fit(docs, iters=150, num_threads=threads)
        levels = list(m.node_levels)
        assign = [max(path, key=lambda nd: levels[nd]) for path in m.doc_paths]
        return (np.asarray(assign, dtype=int),)

    def ref():
        import tomotopy as tp
        docs = _ng_docs_capped(N)
        m = tp.HLDAModel(tw=tp.TermWeight.ONE, depth=DEPTH, seed=1)
        for d in docs:
            m.add_doc(d)
        m.burn_in = 200
        m.train(150, workers=1, show_progress=False)
        assign = [max(set(doc.topics), key=lambda t: m.level(t)) for doc in m.docs]
        return (np.asarray(assign, dtype=int),)

    register("hlda", "HLDA", 0, topica, refs={"tomotopy": (_tomo_available, ref)},
             note=f"depth {DEPTH} tree on {N}-doc subset; cross-NMI of doc "
                  f"assignments (discovered-K, cosine not comparable). At the sharp "
                  f"default beta=0.01 topica fits a far finer, higher-posterior tree "
                  f"than tomotopy (~100x more nodes), so it is slower per fit; raise "
                  f"beta for a compact reference-scale tree (#615). num_threads speeds "
                  f"the per-fit work (see the threaded row).",
             metric_fn=_cross_nmi, threads=threads, corpus="ng")


def _register_pt(k=10, threads=1):
    num_pseudo = 50

    def topica():
        from topica import PT
        docs = _ng_docs()
        m = PT(num_topics=k, num_pseudo=num_pseudo, seed=1)
        m.fit(docs, iters=1000)
        return np.asarray(m.topic_word), list(m.vocabulary)

    def ref():
        docs = _ng_docs()
        def build(tp, k):
            m = tp.PTModel(tw=tp.TermWeight.ONE, k=k, p=num_pseudo, seed=1)
            for d in docs:
                m.add_doc(d)
            return m
        return _tomo_fit(build, k, workers=threads)

    register("pt", "PT", k, topica, refs={"tomotopy": (_tomo_available, ref)},
             note=f"pseudo-document short-text LDA (p={num_pseudo})", threads=threads,
             corpus="ng")


def _register_dtm(k=10, threads=1):
    T = 4

    def _slices():
        docs, _, day = _poliblog()
        day = np.asarray(day, dtype=float)
        edges = np.quantile(day, np.linspace(0, 1, T + 1))[1:-1]
        return docs, np.searchsorted(edges, day)

    def topica():
        from topica import DTM
        docs, slc = _slices()
        m = DTM(num_topics=k, seed=1)
        m.fit(docs, slc.tolist(), iters=30)
        return np.asarray(m.topic_word(T - 1)), list(m.vocabulary)

    def ref():
        import tomotopy as tp
        docs, slc = _slices()
        m = tp.DTModel(tw=tp.TermWeight.ONE, k=k, t=T, seed=1)
        for d, s in zip(docs, slc):
            m.add_doc(d, timepoint=int(s))
        m.train(1000, workers=threads, show_progress=False)
        return (np.array([m.get_topic_word_dist(t, T - 1) for t in range(k)]),
                list(m.used_vocabs))

    register("dtm", "DTM", k, topica, refs={"tomotopy": (_tomo_available, ref)},
             note=f"{T} slices; final slice compared", threads=threads)


def _register_labeledlda(k=2, threads=1):
    def topica():
        from topica import LabeledLDA
        docs, labs, names = _ng_labeled()
        m = LabeledLDA(seed=1)
        m.fit(docs, labs, label_names=names, iters=1000)
        return np.asarray(m.topic_word), list(m.vocabulary)

    def ref():
        import tomotopy as tp
        docs, labs, names = _ng_labeled()
        m = tp.LLDAModel(tw=tp.TermWeight.ONE, seed=1)
        for d, lab in zip(docs, labs):
            m.add_doc(d, labels=lab)
        m.burn_in = 200
        m.train(1000, workers=threads, show_progress=False)
        ldict = list(m.topic_label_dict)
        by = {ldict[t]: np.array(m.get_topic_word_dist(t)) for t in range(len(ldict))}
        # align to the shared label order; skip labels tomotopy dropped (rare)
        keep = [n for n in names if n in by]
        return np.array([by[n] for n in keep]), list(m.used_vocabs)

    k = 20  # one topic per newsgroup

    register("labeledlda", "LabeledLDA", k, topica,
             refs={"tomotopy": (_tomo_available, ref)},
             note="one topic per newsgroup (20NG labels)",
             threads=threads, corpus="ng")


# ---- R-reference family (reuse the validated parity drivers) --------------- #
def _register_stm(k=10, threads=1):
    import stm_poliblog_compare as stmp
    K = k

    def _design():
        from topica.stm import spline
        docs, rating_lib, day = _poliblog()  # 5k corpus
        sb, _ = spline(day, df=stmp.SPLINE_DF)
        X = np.column_stack([rating_lib, sb])
        names = ["ratingLiberal"] + [f"day_s{j}" for j in range(sb.shape[1])]
        return docs, X, names

    def topica():
        from topica import STM
        docs, X, names = _design()
        m = STM(num_topics=K, init="spectral")
        m.fit(docs, X, prevalence_names=names, iters=stmp.EM_ITERS,
              convergence_tol=1e-5, num_threads=threads)
        return np.asarray(m.topic_word), list(m.vocabulary)

    def ref():
        import csv as _csv, os as _os, subprocess as _sp, tempfile as _tf
        docs, X, names = _design()
        design = np.column_stack([np.ones(len(docs)), X])
        driver = (
            'suppressMessages(library(stm))\n'
            'lines<-readLines(file.path(dir,"vdocs.txt")); toks<-strsplit(lines," ")\n'
            'vocab<-sort(unique(unlist(toks))); vmap<-setNames(seq_along(vocab),vocab)\n'
            'documents<-lapply(toks,function(d){tb<-table(d);idx<-as.integer(vmap[names(tb)]);'
            'o<-order(idx);matrix(as.integer(rbind(idx[o],as.integer(tb)[o])),nrow=2)})\n'
            'X<-as.matrix(read.csv(file.path(dir,"design.csv"))); set.seed(1)\n'
            'f<-stm(documents,vocab,K=KVAL,prevalence=X,init.type="Spectral",verbose=FALSE)\n'
            'b<-exp(f$beta$logbeta[[1]]); colnames(b)<-vocab\n'
            'write.csv(b,file.path(dir,"r_spectral.csv"),row.names=FALSE)\n'
            'write(vocab,file.path(dir,"r_vocab.txt")); cat("ok\\n")\n')
        with _tf.TemporaryDirectory() as d:
            open(_os.path.join(d, "vdocs.txt"), "w").write(
                "\n".join(" ".join(x) for x in docs) + "\n")
            with open(_os.path.join(d, "design.csv"), "w", newline="") as f:
                w = _csv.writer(f); w.writerow(["intercept"] + names)
                for r in design:
                    w.writerow(list(r))
            script = f'dir<-"{d}"\nKVAL<-{K}\n' + driver
            p = _sp.run(["Rscript", "-e", script], capture_output=True, text=True, timeout=1800)
            if "ok" not in p.stdout:
                raise RuntimeError(f"R stm failed:\n{p.stdout}\n{p.stderr}")
            rv = open(_os.path.join(d, "r_vocab.txt")).read().split()
            return stmp._read_r_beta(_os.path.join(d, "r_spectral.csv"), rv), rv

    register("stm", "STM", K, topica, refs={"R stm": (stmp.r_stm_available, ref)},
             note="prevalence ~rating+s(day); Spectral", threads=threads)


def _register_stm_content(k=10, threads=1):
    # Real corpus (poliblog, content=~rating), mirroring the validation appendix's
    # content leg -- so the row is a genuine speed+accuracy measurement, not the old
    # 20-word synthetic fixture whose R "time" was just subprocess startup. Compares
    # the marginal topic-word (averaged over rating levels) via aligned cosine.
    import stm_poliblog_compare as stmp

    def topica():
        from topica import STM
        docs, rating_lib, _ = _poliblog()
        rating = ["Liberal" if r else "Conservative" for r in rating_lib]
        m = STM(num_topics=k, init="spectral")
        m.fit(docs, content=rating, iters=150, convergence_tol=1e-5, num_threads=threads)
        return np.asarray(m.topic_word), list(m.vocabulary)

    def ref():
        import tempfile
        docs, rating_lib, _ = _poliblog()
        rating = ["Liberal" if r else "Conservative" for r in rating_lib]
        with tempfile.TemporaryDirectory() as d:
            rvocab, rbeta, _t, _extra = G.run_r_stm(
                docs, k, meta_rating=rating, content=True, workdir=d)
        return np.asarray(rbeta), list(rvocab)

    register("stm_content", "STM (content/SAGE)", k, topica,
             refs={"R stm": (stmp.r_stm_available, ref)},
             note="poliblog content=~rating; marginal topic-word", threads=threads)


def _register_keyatm(k=10, threads=1):
    import keyatm_r_compare as kac

    def _kdata():
        # keyATM stays on poliblog (covariate/supervised family), keyword sets kept
        # to the surviving vocabulary.
        docs = _poliblog()[0]
        vocab = {w for d in docs for w in d}
        keywords = {name: [w for w in ws if w in vocab]
                    for name, ws in kac.KEYWORD_SETS.items()}
        keywords = {n: ws for n, ws in keywords.items() if ws}
        return docs, keywords

    _, kw0 = _kdata()
    nkw = len(kw0)              # keyword-seeded topics (one per keyword set)
    nt = nkw + kac.NUM_REGULAR
    # keyATM's signature is keyword anchoring, so we score ONLY the keyword topics
    # (the first nkw, ordered identically in both engines). The no-keyword topics
    # are unseeded and seed-unstable -- exactly like plain LDA free topics, where two
    # independent Gibbs samplers only agree ~0.5 -- so including them would penalise
    # keyATM for a non-issue. Both fits still learn all nt topics; we just compare the
    # keyword block. (Split on poliblog: keyword 0.89 vs no-keyword 0.51.)

    def topica():
        from topica import KeyATM
        docs, keywords = _kdata()
        m = KeyATM(keywords, num_topics=len(keywords) + kac.NUM_REGULAR, seed=1,
                   num_threads=threads)
        m.fit(docs, iters=kac.ITERS)
        return np.asarray(m.topic_word)[:nkw], list(m.vocabulary)

    def ref():
        import os as _os, json as _json, csv as _csv, subprocess as _sp, tempfile as _tf
        docs, keywords = _kdata()
        driver = (
            'suppressMessages(library(keyATM)); suppressMessages(library(quanteda))\n'
            'lines<-readLines(file.path(dir,"vdocs.txt"))\n'
            'toks<-quanteda::as.tokens(strsplit(lines," ",fixed=TRUE))\n'
            'docs<-keyATM_read(texts=quanteda::dfm(toks))\n'
            'kw<-jsonlite::fromJSON(file.path(dir,"keywords.json"),simplifyVector=FALSE)\n'
            'kw<-lapply(kw,function(x) unlist(x))\n'
            'out<-keyATM(docs=docs,model="base",no_keyword_topics=NREG,keywords=kw,'
            'options=list(seed=1,iterations=ITERS,verbose=FALSE))\n'
            'write.csv(out$phi,file.path(dir,"r_phi1.csv")); cat("ok\\n")\n')
        with _tf.TemporaryDirectory() as d:
            open(_os.path.join(d, "vdocs.txt"), "w").write(
                "\n".join(" ".join(x) for x in docs) + "\n")
            _json.dump(keywords, open(_os.path.join(d, "keywords.json"), "w"))
            script = f'dir<-"{d}"\nNREG<-{kac.NUM_REGULAR}\nITERS<-{kac.ITERS}\n' + driver
            p = _sp.run(["Rscript", "-e", script], capture_output=True, text=True, timeout=3600)
            if "ok" not in p.stdout:
                raise RuntimeError(f"R keyATM failed:\n{p.stdout}\n{p.stderr}")
            with open(_os.path.join(d, "r_phi1.csv"), newline="") as f:
                header = next(_csv.reader(f))
            rv = [h.strip('"') for h in header[1:]]
            return kac._read_r_phi(_os.path.join(d, "r_phi1.csv"), rv)[:nkw], rv

    register("keyatm", "KeyATM", nt, topica,
             refs={"R keyATM": (kac.r_keyatm_available, ref)},
             note=f"{nkw} keyword topics scored (of {nt}); no-keyword topics excluded "
                  "(unseeded, seed-unstable like LDA free topics)", threads=threads)


# ---- sklearn / embedding / neural ----------------------------------------- #
def _sklearn_available():
    try:
        import sklearn, scipy  # noqa
        return True
    except Exception:
        return False


def _register_nmf(k=10, threads=1):
    def topica():
        import topica
        docs = _ng_docs()
        m = topica.NMF(k, beta_loss="frobenius", init="nndsvd", weighting="count",
                       convergence_tol=0.0)
        m.fit(docs, iters=300)
        return np.asarray(m.topic_word), list(m.vocabulary)

    def ref():
        import topica
        from sklearn.decomposition import NMF as SkNMF
        docs = _ng_docs()
        tm = topica.NMF(k, beta_loss="frobenius", init="nndsvd", weighting="count",
                        convergence_tol=0.0)
        tm.fit(docs, iters=1)
        vocab = list(tm.vocabulary)
        X = G.doc_term(docs, vocab)
        sk = SkNMF(n_components=k, init="nndsvda", solver="mu", beta_loss="frobenius",
                   max_iter=300, tol=0.0, random_state=0)
        sk.fit_transform(X)
        h = sk.components_
        return h / h.sum(1, keepdims=True).clip(min=1e-300), vocab

    register("nmf", "NMF", k, topica, refs={"sklearn": (_sklearn_available, ref)},
             threads=threads, corpus="ng")


def _register_lsa(k=10, threads=1):
    def topica():
        import topica
        docs = _ng_docs()
        m = topica.LSA(k, weighting="count", seed=42)
        m.fit(docs)
        return np.asarray(m.topic_word), list(m.vocabulary)

    def ref():
        import topica
        from sklearn.decomposition import TruncatedSVD
        docs = _ng_docs()
        tm = topica.LSA(k, weighting="count", seed=42)
        tm.fit(docs)
        vocab = list(tm.vocabulary)
        X = G.doc_term(docs, vocab)
        svd = TruncatedSVD(n_components=k, algorithm="randomized", random_state=0)
        svd.fit_transform(X)
        tw = svd.components_
        mx = np.argmax(np.abs(tw), axis=1)
        signs = np.sign(tw[np.arange(tw.shape[0]), mx]); signs[signs == 0] = 1.0
        return tw * signs[:, None], vocab

    register("lsa", "LSA", k, topica, refs={"sklearn": (_sklearn_available, ref)},
             threads=threads, corpus="ng")


def _truth_ari(ref_obj, top_obj):
    from sklearn.metrics import adjusted_rand_score
    labels = np.asarray(top_obj[0]).astype(int); truth = np.asarray(top_obj[1]).astype(int)
    mask = labels >= 0
    return float(adjusted_rand_score(truth[mask], labels[mask])) if mask.sum() else 0.0


def _cross_nmi(ref_obj, top_obj):
    from sklearn.metrics import normalized_mutual_info_score
    r = np.asarray(ref_obj[0]).astype(int); t = np.asarray(top_obj[0]).astype(int)
    n = min(len(r), len(t))
    return float(normalized_mutual_info_score(r[:n], t[:n]))


def _bertopic_available():
    try:
        import top2vec_compare as tc
        return tc.bertopic_available()
    except Exception:
        return False


_ng20_cache = None


def _ng20():
    """BERTopic/Top2Vec run on the same real 20-Newsgroups corpus as ProdLDA and
    FASTopic (via `fastopic_compare.load`), NOT a planted toy: ~2.6k docs over 5
    groups. MiniLM sentence embeddings are computed once and shared by topica and
    the reference, so the comparison scores the *clustering*, not the encoder, and
    the timing reflects real UMAP+HDBSCAN work at scale. Cached across the run."""
    global _ng20_cache
    if _ng20_cache is not None:
        return _ng20_cache
    import fastopic_compare as fc
    docs, texts, _, vocab = fc.load()
    doc_emb = fc.embed(texts)
    word_emb = fc.embed(vocab)
    _ng20_cache = (docs, texts, doc_emb, vocab, word_emb)
    return _ng20_cache


def _bertopic_ref():
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP
    _, texts, doc_emb, _, _ = _ng20()
    bt = BERTopic(umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                                  metric="cosine", random_state=42),
                  hdbscan_model=HDBSCAN(min_cluster_size=15, prediction_data=True),
                  calculate_probabilities=False)
    topics, _ = bt.fit_transform(texts, embeddings=doc_emb)
    return np.asarray(topics).astype(float), np.array(["_"], dtype=object)


def _register_bertopic():
    def topica():
        import topica
        docs, _, doc_emb, _, _ = _ng20()
        # UMAP (not PCA) to match the reference's reducer, so the timing is a
        # like-for-like UMAP+HDBSCAN comparison rather than PCA-vs-UMAP.
        bt = topica.BERTopic(reducer="umap", n_components=5, n_neighbors=15,
                             min_cluster_size=15, seed=42)
        bt.fit(docs, doc_emb)
        return np.asarray(bt.labels).astype(float), np.array(["_"], dtype=object)

    register("bertopic", "BERTopic", 5, topica,
             refs={"bertopic": (_bertopic_available, _bertopic_ref)},
             note="cross-NMI of doc assignments (20-News, UMAP+HDBSCAN)",
             metric_fn=_cross_nmi)


def _register_top2vec():
    def topica():
        import topica
        docs, _, doc_emb, vocab, word_emb = _ng20()
        tv = topica.Top2Vec(n_components=5, min_cluster_size=15, seed=1)
        tv.fit(docs, doc_emb, word_embeddings=word_emb, vocabulary=vocab)
        return np.asarray(tv.labels).astype(float), np.array(["_"], dtype=object)

    register("top2vec", "Top2Vec", 5, topica,
             refs={"bertopic": (_bertopic_available, _bertopic_ref)},
             note="cross-NMI of doc assignments (20-News, vs BERTopic)",
             metric_fn=_cross_nmi)


def _register_fastopic(k=10, threads=1):
    def topica():
        import topica, fastopic_compare as fc
        td, texts, _, _ = fc.load()
        emb = fc.embed(texts)
        tm = topica.FASTopic(num_topics=fc.NUM_TOPICS, lr=0.002, seed=fc.SEED)
        th = np.asarray(tm.fit_transform(td, emb))
        return th.argmax(1).astype(float), np.array(["_"], dtype=object)

    def ref():
        from fastopic import FASTopic
        import fastopic_compare as fc
        td, texts, _, _ = fc.load()
        emb = fc.embed(texts)
        rm = FASTopic(fc.NUM_TOPICS, verbose=False)
        _, th = rm.fit_transform(texts, epochs=200, learning_rate=0.002,
                                 preset_doc_embeddings=emb)
        return np.asarray(th).argmax(1).astype(float), np.array(["_"], dtype=object)

    import fastopic_compare as fc
    register("fastopic", "FASTopic", fc.NUM_TOPICS, topica,
             refs={"fastopic": (fc.available, ref)},
             note="accuracy = cross-NMI of doc assignments", metric_fn=_cross_nmi,
             threads=threads)


def _register_prodlda(k=10, threads=1):
    def topica():
        import prodlda_compare as pc
        td, _, _ = pc.load()
        _, a = pc._topica_fit(td, pc.SEED)
        return np.asarray(a).astype(float), np.array(["_"], dtype=object)

    def ref():
        import prodlda_compare as pc
        td, _, vocab = pc.load()
        counts = pc._counts_matrix(td, vocab)
        _, a = pc._ref_fit(counts, vocab, pc.SEED)
        return np.asarray(a).astype(float), np.array(["_"], dtype=object)

    import prodlda_compare as pc
    register("prodlda", "ProdLDA", pc.NUM_TOPICS, topica,
             refs={"pytorch-avitm": (pc.available, ref)},
             note="accuracy = cross-NMI of doc assignments (seed 0)", metric_fn=_cross_nmi,
             threads=threads)


def _turftopic_available():
    try:
        import turftopic  # noqa: F401
        import fastopic_compare as fc
        return fc.available()
    except Exception:
        return False


class _S3LookupEncoder:
    """A turftopic ExternalEncoder that returns precomputed embeddings by lookup, so
    turftopic's own SemanticSignalSeparation can run on planted (text-free) data
    without a real sentence-transformer. Lets the planted fidelity row compare
    against turftopic rather than raw scikit-learn."""
    def __init__(self, table, dim):
        self.table, self.dim = table, dim

    def encode(self, sentences, **kw):
        import numpy as _np
        return _np.array([self.table.get(s, _np.zeros(self.dim)) for s in sentences], dtype=float)


def _s3_assign(sources):
    """Dominant-axis doc assignment: the axis each document loads most strongly on.
    |.| makes it sign-invariant, matching ICA's sign/permutation ambiguity so the
    cross-NMI does not depend on either side's arbitrary axis sign."""
    return np.abs(np.asarray(sources)).argmax(1).astype(float)


def _register_s3_real(k=10):
    """S³ on the real 20-News/MiniLM corpus, against turftopic's *own*
    SemanticSignalSeparation (the reference package, Kardos et al.) fed the same
    precomputed document embeddings. This is a package-vs-package comparison of the
    real reference, not our hand-rolled FastICA.

    Caveat baked into the note: FastICA does NOT converge on real contextual
    embeddings at these K (both sides hit max_iter), so the axis solution is
    unstable and the cross-NMI is low for *everyone* — turftopic does not even agree
    with itself across seeds here. Read this row for TIMING, not accuracy; the
    converging-data fidelity check is `s3_planted`. Also note turftopic re-encodes
    the vocabulary with MiniLM internally (topica takes those embeddings from the
    caller), so its wall-clock includes term-encoding that topica offloads."""
    def topica():
        import topica
        docs, _, doc_emb, vocab, word_emb = _ng20()
        m = topica.SemanticSignalSeparation(num_topics=k, seed=1).fit(
            docs, doc_emb, word_emb, vocabulary=vocab)
        return _s3_assign(m.source_scores), np.array(["_"], dtype=object)

    def ref():
        from turftopic import SemanticSignalSeparation as TT
        _, texts, doc_emb, _, _ = _ng20()
        m = TT(n_components=k, max_iter=200, random_state=1)
        theta = m.fit_transform(list(texts), embeddings=doc_emb)
        return _s3_assign(theta), np.array(["_"], dtype=object)

    register("s3", "SemanticSignalSeparation", k, topica,
             refs={"turftopic": (_turftopic_available, ref)},
             note="NOT algorithm speed: ref time/RSS is turftopic loading MiniLM + re-encoding "
                  "the vocab (topica gets those embeddings from the caller). ICA also non-convergent "
                  "here => cross-NMI is noise. Fair algorithm speed + fidelity are in s3_planted.",
             metric_fn=_cross_nmi)


def _s3_planted(k=10, seed=0):
    """Planted independent-axis embeddings where FastICA genuinely converges: K
    independent non-Gaussian sources mixed into a low-dim space, plus a vocabulary
    aligned to the mixing directions. Both implementations recover the same axes, so
    the cross-NMI is a real fidelity signal (~1.0), unlike the real-corpus row.
    Returns space-joined doc strings + a word->embedding table so turftopic (via a
    lookup encoder) can serve as the reference on this text-free data."""
    rng = np.random.default_rng(seed)
    D, M, V = 2000, 64, 600
    src = rng.uniform(-1.0, 1.0, size=(D, k))
    mix = rng.standard_normal((k, M))
    doc_emb = src @ mix + 0.02 * rng.standard_normal((D, M))
    wpa = V // k
    table, vocab = {}, []
    for a in range(k):
        for j in range(wpa):
            w = f"axis{a}word{j}"          # single \w+ token, survives CountVectorizer
            table[w] = mix[a] + 0.15 * rng.standard_normal(M)
            vocab.append(w)
    trng = np.random.default_rng(seed + 7)
    doc_words = [[vocab[i] for i in trng.integers(0, len(vocab), 12)] for _ in range(D)]
    return doc_words, doc_emb, table, vocab


def _register_s3_planted(k=10):
    """Fidelity check for S³ on converging planted data, against turftopic's own
    SemanticSignalSeparation (fed the same planted embeddings via a lookup encoder)."""
    def topica():
        import topica
        doc_words, doc_emb, table, vocab = _s3_planted(k)
        vocab_emb = np.array([table[w] for w in vocab])
        m = topica.SemanticSignalSeparation(num_topics=k, seed=1).fit(
            doc_words, doc_emb, vocab_emb, vocabulary=vocab)
        return _s3_assign(m.source_scores), np.array(["_"], dtype=object)

    def ref():
        from turftopic import SemanticSignalSeparation as TT
        from sklearn.feature_extraction.text import CountVectorizer
        doc_words, doc_emb, table, vocab = _s3_planted(k)
        docs = [" ".join(w) for w in doc_words]
        enc = _S3LookupEncoder(table, doc_emb.shape[1])
        cv = CountVectorizer(vocabulary=vocab, token_pattern=r"(?u)\b\w+\b", lowercase=False)
        m = TT(n_components=k, max_iter=200, random_state=1, encoder=enc, vectorizer=cv)
        theta = m.fit_transform(docs, embeddings=doc_emb)
        return _s3_assign(theta), np.array(["_"], dtype=object)

    register("s3_planted", "SemanticSignalSeparation (planted)", k, topica,
             refs={"turftopic": (_turftopic_available, ref)},
             note="fidelity = cross-NMI of dominant-axis doc assignments on converging planted axes",
             metric_fn=_cross_nmi)


def _gensim_available():
    try:
        import gensim  # noqa
        return True
    except Exception:
        return False


def _register_online_lda(k=10, threads=1):
    # OnlineLDA is the streaming online-VB analogue of gensim's LdaModel, so gensim
    # is the reference. Hyperparameters are matched on both sides (chunksize/offset/
    # decay/eta/iterations/passes and, explicitly, the per-topic alpha, which
    # otherwise differs: topica alpha_sum=K -> 1.0 vs gensim 'symmetric' -> 1/K).
    ALPHA = 1.0

    def topica():
        from topica import OnlineLDA
        docs = _ng_docs()
        m = OnlineLDA(num_topics=k, alpha_sum=ALPHA * k, beta=0.01, tau=1.0,
                      kappa=0.7, batch_size=256, inner_iters=100, seed=1)
        m.fit(docs, iters=20)
        return np.asarray(m.topic_word), list(m.vocabulary)

    def ref_gensim():
        from gensim.corpora import Dictionary
        from gensim.models import LdaModel
        docs = _ng_docs()
        dictionary = Dictionary(docs)
        bow = [dictionary.doc2bow(d) for d in docs]
        gm = LdaModel(corpus=bow, id2word=dictionary, num_topics=k, alpha=ALPHA,
                      chunksize=256, offset=1.0, decay=0.7, eta=0.01, passes=20,
                      iterations=100, random_state=1)
        g_tw = gm.get_topics()  # (K, V_gensim)
        g_vocab = [dictionary[i] for i in range(len(dictionary))]
        return g_tw, g_vocab

    register("online_lda", "OnlineLDA", k, topica,
             refs={"gensim": (_gensim_available, ref_gensim)}, threads=threads,
             corpus="ng")


# Registry population (extend as coverage grows).
# threadable=True only for models whose reference honors a thread count (tomotopy
# workers= / MALLET): lda, dmr, gdmr. Everything else runs single-thread only.
# Threads axis (t=1 and t=MULTI) only where BOTH sides can thread. LDA: topica
# num_threads (AD-LDA) + tomotopy workers + MALLET. DMR/GDMR expose no topica
# threads knob (GDMR is a pure-Python DMR wrapper), so a t>1 row would be
# topica-single vs tomotopy-multi -- misleading. They run single-threaded only.
# K sweeps: 20NG models at K=50 (large-K meaningful), poliblog covariate family at
# canonical K=10/25. Smoke mode collapses both to K=5, single config.
KS_NG = (5,) if _SMOKE else (50,)
KS_POLI = (5,) if _SMOKE else (10, 25)
for k, t in _configs(KS_NG, True): _register_lda(k, t)
for k, t in _configs(KS_POLI, False):  _register_dmr(k, t)
for k, t in _configs(KS_POLI, False):  _register_gdmr(k, t)
for k, t in _configs(KS_POLI, False): _register_slda(k, t)
for k, t in _configs(KS_POLI, False): _register_slda_gibbs(k, t)
for k, t in _configs(KS_NG, False): _register_pa(k, t)
for k, t in _configs(KS_POLI, False): _register_dtm(k, t)
_register_labeledlda(2, 1)          # K fixed by design (one topic per label)
for k, t in _configs(KS_POLI, False): _register_stm(k, t)
for k, t in _configs(KS_POLI, False): _register_stm_content(k, t)
for k, t in _configs(KS_POLI, False): _register_keyatm(k, t)
for k, t in _configs(KS_NG, False): _register_nmf(k, t)
for k, t in _configs(KS_NG, False): _register_lsa(k, t)
_register_bertopic()   # K discovered; keep single config
_register_top2vec()    # K discovered; single config
_register_s3_real()    # real 20-News/MiniLM vs turftopic (timing; ICA non-convergent)
_register_s3_planted() # converging planted axes vs turftopic (fidelity)
for k, t in _configs(KS_NG, False): _register_fastopic(k, t)
for k, t in _configs(KS_NG, False): _register_prodlda(k, t)
for k, t in _configs(KS_NG, False): _register_online_lda(k, t)  # gensim online-VB ref
# Tier-1 additions (external reference runs locally) -- tomotopy family
for k, t in _configs(KS_NG, False): _register_ctm(k, t)
_register_hdp()        # K discovered; single config
for _, t in _configs(KS_NG, True):
    _register_hlda(t)  # K discovered (tree); 1- and multi-threaded configs
for k, t in _configs(KS_NG, False): _register_pt(k, t)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def _score(e, ref, ttw_or_out, tvocab_or_none):
    if e["metric_fn"] is not None:
        return e["metric_fn"](ref, ttw_or_out)
    return accuracy(ref["tw"], ref["vocab"], ttw_or_out, tvocab_or_none)


def run_entry(e, mode):
    """One entry -> one row per available reference."""
    rows = []
    # Fit topica once (shared across this model's references); timing is per model.
    topica_out = topica_sec = topica_rss = None
    for ref_label, (avail, ref_fn) in e["refs"].items():
        ckey = f"{e['key']}__{ref_label}"
        base = dict(key=e["key"], model=e["model"], reference=ref_label, k=e["k"],
                    threads=e.get("threads", 1), note=e["note"])
        try:
            rows.append(_run_ref(e, ref_label, ref_fn, avail, ckey, base, mode,
                                 topica_out, topica_sec, topica_rss))
            # cache topica fit across refs
            if rows[-1].get("_topica") is not None:
                topica_out, topica_sec, topica_rss = rows[-1].pop("_topica")
            else:
                rows[-1].pop("_topica", None)
        except Exception as exc:  # one model's failure must not sink the run
            import traceback
            rows.append({**base, "status": f"ERROR: {type(exc).__name__}: {exc}"})
            traceback.print_exc()
    return rows


def _run_ref(e, ref_label, ref_fn, avail, ckey, base, mode,
             topica_out, topica_sec, topica_rss):
    """Run one (model, reference) pair. Returns one row dict; the row carries a
    '_topica' key with the (out, sec, rss) tuple so the caller can reuse the
    single topica fit across this model's references."""
    if not avail():
        return {**base, "status": f"SKIP (no {ref_label})", "_topica": None}
    nthreads = e.get("threads", 1)

    if mode in ("build-refs", "time"):
        with pin_threads(nthreads):
            out, sec, rss = timed(ref_fn)
        # Cache format: for cosine models out=(tw,vocab); store both. For custom
        # metric models, store the raw object under 'obj'.
        if isinstance(out, tuple) and len(out) == 2 and e["metric_fn"] is None:
            store_ref(ckey, out[0], out[1], sec, rss)
            ref = dict(tw=np.asarray(out[0]), vocab=list(out[1]), sec=sec, rss=rss)
        else:
            np.savez(cache_path(ckey), obj=np.array([out], dtype=object),
                     vocab=np.array([], dtype=object), sec=sec, rss=rss,
                     tw=np.array([]))
            ref = dict(obj=out, sec=sec, rss=rss)
        if mode == "build-refs":
            return {**base, "status": "ref cached", "ref_s": round(sec, 2),
                    "ref_MB": round(rss, 1), "_topica": None}
    else:  # accuracy: load cached ref
        d = load_ref(ckey)
        if d is None:
            return {**base, "status": "no cached ref (run --build-refs)",
                    "_topica": None}
        ref = d if e["metric_fn"] is None else dict(
            obj=np.load(cache_path(ckey), allow_pickle=True)["obj"][0],
            sec=d["sec"], rss=d["rss"])

    if topica_out is None:
        with pin_threads(nthreads):
            topica_out, topica_sec, topica_rss = timed(e["topica"])
    # Intrinsic quality (c_v / c_npmi / TU) for BOTH sides, computed once on the
    # fitted models -- captured now so later tables need no re-fit (issue: "no
    # reason to run everything twice"). Only the cosine models expose a comparable
    # (tw, vocab); the metric_fn (embedding) models store the topica object only.
    texts = _texts(e.get("corpus", "poliblog"))
    if e["metric_fn"] is None:
        acc = accuracy(ref["tw"], ref["vocab"], topica_out[0], topica_out[1])
        topica_q = _quality(topica_out[0], topica_out[1], texts)
        ref_q = _quality(ref["tw"], ref["vocab"], texts)
    else:
        acc = e["metric_fn"](ref["obj"], topica_out)
        topica_q = ref_q = {"cv": None, "npmi": None, "tu": None}
    return {**base, "status": "ok", "accuracy": round(acc, 3),
            "topica_s": round(topica_sec, 2), "ref_s": round(ref["sec"], 2),
            "speedup": round(ref["sec"] / topica_sec, 2) if topica_sec else None,
            "topica_MB": round(topica_rss, 1), "ref_MB": round(ref["rss"], 1),
            "topica_cv": topica_q["cv"], "topica_npmi": topica_q["npmi"],
            "topica_tu": topica_q["tu"], "ref_cv": ref_q["cv"],
            "ref_npmi": ref_q["npmi"], "ref_tu": ref_q["tu"],
            "_topica": (topica_out, topica_sec, topica_rss)}


def render(rows):
    hdr = ("| model | reference | K | threads | accuracy | topica_s | ref_s | speedup "
           "| topica_MB / ref_MB | note |")
    sep = "|" + "---|" * 10
    lines = [hdr, sep]
    for r in rows:
        if r.get("status", "ok") != "ok":
            lines.append(f"| {r.get('model', r['key'])} | | {r.get('k','')} "
                         f"| {r.get('threads','')} | | | | | | {r['status']} |")
            continue
        mem = f"{r.get('topica_MB','?')} / {r.get('ref_MB','?')}"
        lines.append(f"| {r['model']} | {r['reference']} | {r['k']} | {r.get('threads',1)} "
                     f"| {r.get('accuracy','')} "
                     f"| {r.get('topica_s','')} | {r.get('ref_s','')} | {r.get('speedup','')}x "
                     f"| {mem} | {r.get('note','')} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Article table (booktabs LaTeX)
# --------------------------------------------------------------------------- #
# Row order and family blocks for the paper table. Each family is a list of the
# `model` names (as registered) that belong under it, in display order.
_FAMILIES = [
    ("Classical (collapsed Gibbs)",
     ["LDA", "DMR", "GDMR", "PA", "DTM", "LabeledLDA", "OnlineLDA", "KeyATM"]),
    ("Nonparametric / hierarchical",
     ["HDP", "HLDA"]),
    ("Supervised",
     ["SupervisedLDA (var)", "SupervisedLDA (gibbs)", "SeededLDA"]),
    ("Structural / correlated",
     ["STM", "STM (content/SAGE)", "CTM"]),
    ("Matrix factorization",
     ["NMF", "LSA"]),
    ("Neural / embedding",
     ["ProdLDA", "CombinedTM", "ZeroShotTM", "FASTopic",
      "SemanticSignalSeparation", "BERTopic", "Top2Vec"]),
]
# Models whose fidelity metric is document-assignment cross-NMI (dagger) rather
# than the default aligned topic-word cosine agreement.
_NMI_MODELS = {"BERTopic", "Top2Vec", "FASTopic", "ProdLDA",
               "SemanticSignalSeparation", "SemanticSignalSeparation (planted)"}
# Models whose K is discovered rather than fixed (shown as "auto" in the K column).
_AUTO_K = {"HDP", "HLDA", "BERTopic", "Top2Vec"}


def _tex_esc(s):
    return str(s).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def render_latex(rows):
    """Emit the combined fidelity+speed table as booktabs LaTeX.

    One row per model at a representative single-thread config (the largest K
    that produced an ``ok`` row). Columns: reference, K, agreement, topica and
    reference wall-clock, and their ratio. Models are grouped into families;
    models with no ``ok`` row are skipped. The per-K scaling story lives in the
    text and the tomotopy scaling table, not here."""
    # Pick, per model, the representative ok single-thread row (largest K).
    rep = {}
    for r in rows:
        if r.get("status", "ok") != "ok" or r.get("threads", 1) != 1:
            continue
        m = r["model"]
        if m not in rep or (r.get("k") or 0) > (rep[m].get("k") or 0):
            rep[m] = r

    def fmt_sp(x):
        return "--" if x in (None, "") else f"{float(x):.1f}$\\times$"

    def fmt_s(x):
        return "--" if x in (None, "") else f"{float(x):.2f}"

    lines = [
        r"\begin{table}[t!]",
        r"\centering",
        r"\caption{\pkg{topica} versus each model's reference implementation on a "
        r"shared \code{poliblog} corpus (single core, both sides thread-pinned). "
        r"\emph{Agreement} is aligned topic-word cosine against the reference, or "
        r"document-assignment cross-NMI ($\dagger$) for the embedding and neural "
        r"models whose topic-word matrix is not directly comparable. \emph{Speedup} "
        r"is the wall-clock ratio reference$/$\pkg{topica}. ``auto'' marks models "
        r"that discover $K$. Models validated only through the \code{parity/} "
        r"scripts (no reference runnable in this harness) are omitted.}",
        r"\label{tab:fidelity-speed}",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Model & Reference & $K$ & Agreement & "
        r"\multicolumn{2}{c}{Wall-clock (s)} & Speedup \\",
        r"\cmidrule(lr){5-6}",
        r" & & & & \pkg{topica} & ref. & \\",
        r"\midrule",
    ]
    for fam, models in _FAMILIES:
        emitted = [m for m in models if m in rep]
        if not emitted:
            continue
        lines.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{_tex_esc(fam)}}}}} \\")
        for m in emitted:
            r = rep[m]
            dag = r"$^\dagger$" if m in _NMI_MODELS else ""
            agree = r.get("accuracy")
            aval = "--" if agree in (None, "") else f"{float(agree):.2f}{dag}"
            kval = "auto" if m in _AUTO_K else _tex_esc(r.get("k", ""))
            lines.append(
                rf"\quad {_tex_esc(m)} & {_tex_esc(r.get('reference',''))} & {kval} "
                rf"& {aval} & {fmt_s(r.get('topica_s'))} & {fmt_s(r.get('ref_s'))} "
                rf"& {fmt_sp(r.get('speedup'))} \\")
        lines.append(r"\addlinespace")
    if lines and lines[-1] == r"\addlinespace":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-refs", nargs="*", metavar="KEY")
    ap.add_argument("--accuracy", nargs="*", metavar="KEY")
    ap.add_argument("--time", nargs="*", metavar="KEY")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--latex", action="store_true",
                    help="re-render the booktabs article table from the existing "
                         "JSON (no refitting) and exit")
    args = ap.parse_args()
    if args.list:
        for k, e in REGISTRY.items():
            print(f"{k:24s} {e['model']} vs {'/'.join(e['refs'])} "
                  f"(K={e['k']}, threads={e.get('threads', 1)}) {e['note']}")
        return
    if args.latex:
        rows = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        tex = render_latex(rows)
        print(tex)
        OUT_TEX.write_text(tex + "\n", encoding="utf-8")
        return
    if args.build_refs is not None:
        mode, keys = "build-refs", args.build_refs
    elif args.time is not None:
        mode, keys = "time", args.time
    else:
        mode, keys = "accuracy", (args.accuracy or [])
    keys = keys or list(REGISTRY)
    rows = []
    n = len([k for k in keys if k in REGISTRY])
    i = 0
    for k in keys:
        if k in REGISTRY:
            i += 1
            print(f"[{i}/{n}] {k} ...", file=sys.stderr, flush=True)
            t0 = time.perf_counter()
            out = run_entry(REGISTRY[k], mode)
            rows.extend(out)
            st = "/".join(r.get("status", "ok")[:12] for r in out)
            print(f"[{i}/{n}] {k} done in {time.perf_counter()-t0:.1f}s ({st})",
                  file=sys.stderr, flush=True)
            # incremental save so a later hang never loses completed rows
            OUT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    md = render(rows)
    print(f"\n[mode={mode}]\n{md}")
    OUT_MD.write_text(md + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    OUT_TEX.write_text(render_latex(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
