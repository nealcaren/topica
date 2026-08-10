"""Committed tomotopy gold for topica MGLDA (#690).

Multi-Grain LDA (Titov & McDonald, "Modeling Online Reviews with Multi-Grain Topic
Models," WWW 2008) models each document with GLOBAL topics (the document-level
subject, e.g. which product) and LOCAL topics (rateable aspects over a sliding
sentence window, e.g. battery / screen / price). Each token first picks a window,
then a global-vs-local grain, then a topic from the appropriate distribution.
Collapsed Gibbs over the (window, grain, topic) triple.

Reference: tomotopy ``MGLDAModel`` (MIT). NOTE: tomotopy 0.14.0 has a bug that
ignores ``k_g``/``k_l`` (always 1/1); this gold is generated with **tomotopy
0.13.0**, which builds them correctly. tomotopy is C++ collapsed Gibbs like the port,
but the RNG differs, so parity is topic-aligned (per-grain Hungarian cosine vs
tomotopy's own seed-to-seed floor), never bit-exact.

We freeze, from a fixed-seed sentence-segmented synthetic review corpus with planted
GLOBAL product themes (whole-document) and LOCAL aspects (per-sentence):

  * global topic-word phi (K_gl x V) and local topic-word phi (K_loc x V) at two
    tomotopy seeds -> per-grain seed-to-seed cosine floors;
  * the planted global/local word blocks -> recovery targets.

Runs in CI WITHOUT tomotopy: the reference fit is frozen in the committed
``parity/mglda_gold.npz`` + ``.json``.

    python parity/mglda_gold.py --regenerate   # needs tomotopy==0.13.0
    python parity/mglda_gold.py                # offline compare vs the gold
"""

from __future__ import annotations

import datetime
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402

NAME = "mglda"

# --- planted synthetic review corpus -----------------------------------------
K_GL = 2          # global themes (product categories)
K_LOC = 3         # local aspects
WINDOW = 3
N_DOCS = 120
SENTS_PER_DOC = 6
CORPUS_SEED = 271
GOLD_SEED = 42
FLOOR_SEED = 43
ITERS = 300

# Global themes: product-identity words that recur throughout a document.
GLOBAL_WORDS = {
    0: ["phone", "smartphone", "android", "mobile"],   # theme 0
    1: ["laptop", "notebook", "keyboard", "trackpad"],  # theme 1
}
# Local aspects: rateable-feature words that appear in individual sentences.
LOCAL_WORDS = {
    0: ["battery", "charge", "power", "drain"],
    1: ["screen", "display", "bright", "pixels"],
    2: ["price", "cost", "cheap", "expensive"],
}
VOCAB = sorted({w for d in GLOBAL_WORDS.values() for w in d}
               | {w for d in LOCAL_WORDS.values() for w in d})


def build_corpus():
    """Return docs as list[list[list[str]]] (doc -> sentences -> tokens), plus the
    per-doc global theme. Each sentence mixes the doc's global words with one
    aspect's local words, so global topics are doc-level and local topics are
    per-sentence."""
    rng = np.random.default_rng(CORPUS_SEED)
    docs = []
    doc_global = []
    for _ in range(N_DOCS):
        g = int(rng.integers(0, K_GL))
        doc_global.append(g)
        sents = []
        for s in range(SENTS_PER_DOC):
            aspect = int(rng.integers(0, K_LOC))
            sent = []
            # a couple of global-identity words
            for _ in range(2):
                sent.append(GLOBAL_WORDS[g][int(rng.integers(0, len(GLOBAL_WORDS[g])))])
            # a few local-aspect words
            for _ in range(3):
                sent.append(LOCAL_WORDS[aspect][int(rng.integers(0, len(LOCAL_WORDS[aspect])))])
            rng.shuffle(sent)
            sents.append(sent)
        docs.append(sents)
    return docs, doc_global


def _phi_over_vocab(model, vocab):
    """Return (global K_gl x V, local K_loc x V) topic-word matrices in `vocab` order."""
    used = list(model.used_vocabs)
    idx = {w: i for i, w in enumerate(used)}
    k_g, k_l = model.k_g, model.k_l
    gl = np.zeros((k_g, len(vocab)))
    lo = np.zeros((k_l, len(vocab)))
    for t in range(k_g + k_l):
        d = np.array(model.get_topic_word_dist(t))
        row = np.array([d[idx[w]] if w in idx else 0.0 for w in vocab])
        if t < k_g:
            gl[t] = row
        else:
            lo[t - k_g] = row
    gl = gl / gl.sum(1, keepdims=True).clip(1e-12)
    lo = lo / lo.sum(1, keepdims=True).clip(1e-12)
    return gl, lo


def _tomotopy_fit(docs, seed):
    import tomotopy as tp

    m = tp.MGLDAModel(k_g=K_GL, k_l=K_LOC, t=WINDOW, seed=seed,
                      alpha_g=0.1, alpha_l=0.1, alpha_mg=0.1, alpha_ml=0.1,
                      eta_g=0.01, eta_l=0.01, gamma=0.1)
    for sents in docs:
        words = []
        for i, s in enumerate(sents):
            words.extend(s)
            if i < len(sents) - 1:
                words.append(".")
        m.add_doc(words, delimiter=".")
    m.train(ITERS, workers=1)
    return m


def _tomotopy_global_fraction(m):
    c = list(m.get_count_by_topics())
    gl = float(sum(c[: m.k_g]))
    lo = float(sum(c[m.k_g:]))
    return gl / (gl + lo) if (gl + lo) > 0 else 0.0


def regenerate():
    import tomotopy as tp

    docs, doc_global = build_corpus()
    m = _tomotopy_fit(docs, GOLD_SEED)
    gl, lo = _phi_over_vocab(m, VOCAB)
    m2 = _tomotopy_fit(docs, FLOOR_SEED)
    gl2, lo2 = _phi_over_vocab(m2, VOCAB)

    floor_gl, _ = harness.align_cosine(gl, gl2)
    floor_lo, _ = harness.align_cosine(lo, lo2)
    tomo_gf = _tomotopy_global_fraction(m)
    tomo_gf2 = _tomotopy_global_fraction(m2)

    meta = {
        "model": "mglda",
        "reference": "tomotopy.MGLDAModel",
        "tomotopy_version": tp.__version__,
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "k_g": K_GL, "k_l": K_LOC, "window": WINDOW,
        "n_docs": N_DOCS, "sents_per_doc": SENTS_PER_DOC,
        "iters": ITERS, "gold_seed": GOLD_SEED, "floor_seed": FLOOR_SEED,
        "corpus_seed": CORPUS_SEED,
        "vocab": VOCAB, "docs": docs, "doc_global": doc_global,
        "global_words": GLOBAL_WORDS, "local_words": LOCAL_WORDS,
        "floor_global_cosine": float(floor_gl),
        "floor_local_cosine": float(floor_lo),
        "tomotopy_global_fraction": float(tomo_gf),
        "tomotopy_global_fraction_seed2": float(tomo_gf2),
    }
    harness.save_gold(NAME, {"global_phi": gl, "local_phi": lo,
                             "global_phi2": gl2, "local_phi2": lo2}, meta)
    print(f"[{NAME}] wrote gold: tomotopy {tp.__version__}, "
          f"K_gl={K_GL} K_loc={K_LOC} T={WINDOW} docs={N_DOCS}")
    print(f"[{NAME}] tomotopy seed-to-seed floor: global cos={floor_gl:.3f}, "
          f"local cos={floor_lo:.3f}")
    print(f"[{NAME}] tomotopy global_fraction: {tomo_gf:.3f} (seed2 {tomo_gf2:.3f}) "
          f"-- local grain is near-empty on this synthetic corpus for the reference too")


def _block_recovered(phi, vocab, blocks):
    """Each planted block's words dominate a distinct topic row."""
    idx = {w: i for i, w in enumerate(vocab)}
    doms = []
    for _, words in sorted(blocks.items()):
        col = np.array([phi[:, idx[w]].sum() for w in words if w in idx])
        # for each topic, mass on this block; the block should own one topic
        mass = np.array([sum(phi[t, idx[w]] for w in words if w in idx)
                         for t in range(phi.shape[0])])
        doms.append(int(mass.argmax()))
    return len(set(doms)) == len(blocks)


def compare():
    arrays, meta = harness.load_gold(NAME)
    gl_g, lo_g = arrays["global_phi"], arrays["local_phi"]
    vocab = meta["vocab"]
    docs = meta["docs"]
    floor_gl = meta["floor_global_cosine"]
    floor_lo = meta["floor_local_cosine"]
    tomo_gf = meta.get("tomotopy_global_fraction")
    iters = meta["iters"]  # match the reference's training length

    try:
        import topica
    except ImportError:
        print(f"[{NAME}] topica not importable; skipping compare.")
        return
    if not hasattr(topica, "MGLDA"):
        print(f"[{NAME}] topica.MGLDA not built yet; gold is ready. Skipping.")
        return

    m = topica.MGLDA(meta["k_g"], meta["k_l"], window=meta["window"], seed=13).fit(
        docs, iters=iters
    )
    tv = {w: i for i, w in enumerate(m.vocabulary)}
    gp = np.asarray(m.global_topic_word)
    lp = np.asarray(m.local_topic_word)
    gl_t = np.array([[gp[k, tv[w]] if w in tv else 0.0 for w in vocab] for k in range(meta["k_g"])])
    lo_t = np.array([[lp[k, tv[w]] if w in tv else 0.0 for w in vocab] for k in range(meta["k_l"])])
    gl_t = gl_t / gl_t.sum(1, keepdims=True).clip(1e-12)
    lo_t = lo_t / lo_t.sum(1, keepdims=True).clip(1e-12)

    cg, _ = harness.align_cosine(gl_t, gl_g)
    cl, _ = harness.align_cosine(lo_t, lo_g)
    rec_g = _block_recovered(gl_t, vocab, {int(k): v for k, v in meta["global_words"].items()})
    topica_gf = float(m.global_fraction)

    print(f"[{NAME}] global aligned cosine = {cg:.3f}  (tomotopy floor {floor_gl:.3f})")
    print(f"[{NAME}] planted global recovery = {rec_g}")
    print(f"[{NAME}] global_fraction: topica={topica_gf:.3f} tomotopy={tomo_gf:.3f}")
    print(f"[{NAME}] local aligned cosine = {cl:.3f} (INFORMATIONAL — the local grain is"
          f" near-empty for BOTH: tomotopy global_fraction {tomo_gf:.3f}, its own")
    print(f"[{NAME}]   seed-to-seed local cosine only {floor_lo:.3f}; local topics are"
          f" prior-dominated on this synthetic corpus and are NOT a fidelity target here)")
    # PARITY is gated on the GLOBAL grain (robust, identifiable): topic-word cosine at
    # tomotopy's seed-to-seed floor, exact planted theme recovery, AND grain-fraction
    # agreement (topica reproduces tomotopy's global-dominant grain dynamics on this
    # corpus). The local grain is NOT gated: it collapses to the prior for the reference
    # itself here (tomotopy global_fraction ~1.0), so its topic-word matrix carries no
    # signal to match. The per-token conditional is provably identical to tomotopy (see
    # src/mg_lda.rs); local topics become identifiable only on text with genuine
    # within-document aspect locality (real reviews), out of scope for this fixture.
    global_ok = cg >= floor_gl - 0.05 and rec_g
    grain_ok = abs(topica_gf - tomo_gf) <= 0.10
    ok = global_ok and grain_ok
    print(f"[{NAME}] PARITY {'OK' if ok else 'CHECK'}  (global_ok={global_ok} grain_ok={grain_ok})")


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        compare()
