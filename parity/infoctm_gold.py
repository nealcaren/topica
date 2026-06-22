"""Committed-gold parity for topica InfoCTM vs a paper-derived PyTorch reference (#271, Wave 1).

InfoCTM (Wu et al. 2023, AAAI; arXiv:2304.03544) is two ProdLDA / AVITM models --
one per language, over independent vocabularies, sharing the topic space -- coupled
by a Topic-Alignment Mutual-Information (TAMI) term: a masked cross-lingual InfoNCE
over the topic-word columns, with positive word pairs from a bilingual dictionary.
The reference is the paper recipe in PyTorch (the authors' license is unclear), the
same backbone topica implements in Rust, exactly as in the live script
``parity/infoctm_compare.py``.

The defining behaviour is **cross-lingual topic alignment**: with a block-aligned
dictionary, topic ``k`` must concentrate on the same planted block in both
languages. The live script's bar is exactly that (``alignment >= 0.8`` for both
implementations); full-vector topic-word cosine is deliberately NOT the bar, because
topica produces sharper, lower-entropy topic distributions than this reference --
they rank identically but differ in contrast. This gold keeps the live bar exactly:
it freezes the reference's per-language topic-word matrices and its own seed-to-seed
alignment floor, and the pass condition is topica's own cross-lingual alignment
>= 0.8 (the live ``infoctm_compare`` assert). Per-language top-word block agreement
against the frozen reference is reported as a diagnostic but is NOT the bar -- on
this clean design a single topic can fail to separate in either implementation
without breaking the cross-lingual alignment that defines a correct fit.

Two phases (mirrors parity/combinedtm_gold.py):

  * ``--regenerate`` (needs torch): fits the reference twice (two seeds) for the
    alignment floor, freezes one run's per-language topic-word matrices, the two
    vocabularies, and the dictionary, and writes the committed gold.
  * default (no torch): loads the committed gold, fits topica InfoCTM on the same
    bilingual corpus + dictionary, and checks the bar.

Run directly::

    python parity/infoctm_gold.py               # offline compare against committed gold
    python parity/infoctm_gold.py --regenerate  # run torch once, write the gold
"""

from __future__ import annotations

import datetime
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402

NAME = "infoctm"

# Config (verbatim from parity/infoctm_compare.py).
K = 5
PER = 6
N_A, N_B = 300, 270
LENGTH = 15
HIDDEN = 64
ITERS = 200
DATA_SEED = 0

GOLD_SEED = 1
FLOOR_SEED = 2
# topica's own fit seed + the live script's alignment bar.
TOPICA_SEED = 1
ALIGN_BAR = 0.8


# --------------------------------------------------------------------------- #
# Synthetic matched bilingual corpus + block-aligned dictionary (verbatim)
# --------------------------------------------------------------------------- #
def _corpus(prefix: str, n: int, rng):
    docs = []
    for d in range(n):
        b = d % K
        docs.append([f"{prefix}{b}_{int(rng.integers(PER))}" for _ in range(LENGTH)])
    return docs


def _make_data():
    rng = np.random.default_rng(DATA_SEED)
    docs_a = _corpus("a", N_A, rng)
    docs_b = _corpus("b", N_B, rng)
    vocab_a = [f"a{b}_{i}" for b in range(K) for i in range(PER)]
    vocab_b = [f"b{b}_{i}" for b in range(K) for i in range(PER)]
    dictionary = [
        (f"a{b}_{i}", f"b{b}_{j}")
        for b in range(K) for i in range(PER) for j in range(PER)
    ]
    return docs_a, docs_b, vocab_a, vocab_b, dictionary


def _counts(docs, vocab):
    idx = {w: i for i, w in enumerate(vocab)}
    m = np.zeros((len(docs), len(vocab)), dtype=np.float32)
    for d, doc in enumerate(docs):
        for w in doc:
            m[d, idx[w]] += 1.0
    return m


# --------------------------------------------------------------------------- #
# Metrics (verbatim from the live script)
# --------------------------------------------------------------------------- #
def _alignment(beta_a, beta_b, per=PER):
    block_a = [int(np.argmax(beta_a[t]) // per) for t in range(beta_a.shape[0])]
    block_b = [int(np.argmax(beta_b[t]) // per) for t in range(beta_b.shape[0])]
    return sum(a == b for a, b in zip(block_a, block_b)) / len(block_a)


def _purity(beta, per=PER):
    out = []
    for row in beta:
        b = int(np.argmax(row) // per)
        out.append(float(row[b * per:(b + 1) * per].sum()))
    return float(np.mean(out))


# --------------------------------------------------------------------------- #
# PyTorch reference (paper-derived; identical to the live script)
# --------------------------------------------------------------------------- #
def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _torch_version() -> str:
    try:
        import torch
        return f"torch {torch.__version__} (paper-derived InfoCTM, parity/refs/avitm.py backbone)"
    except Exception:
        return "torch (version unknown)"


def _train_reference(counts_a, counts_b, trans_ab, seed):
    import torch
    import torch.nn.functional as F

    from refs.avitm import build_encoder, dirichlet_laplace_kl, laplace_prior, reparameterize

    torch.manual_seed(seed)
    va, vb = counts_a.shape[1], counts_b.shape[1]
    mu2, var2 = laplace_prior(K, 1.0)
    trans = torch.tensor(trans_ab)
    trans_t = trans.T
    neg_ab = (trans <= 0).float()
    neg_ba = (trans_t <= 0).float()
    pos_sum = trans.sum() + trans_t.sum()
    temp, weight = 0.2, 30.0

    enc_a = build_encoder(va, K, HIDDEN)
    enc_b = build_encoder(vb, K, HIDDEN)
    dec_bn_a = torch.nn.BatchNorm1d(va, affine=False)
    dec_bn_b = torch.nn.BatchNorm1d(vb, affine=False)
    phi_a = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(K, va)))
    phi_b = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(K, vb)))

    mods = [enc_a, enc_b, dec_bn_a, dec_bn_b]
    params = [phi_a, phi_b]
    for m in mods:
        params += list(m.parameters())
    opt = torch.optim.Adam(params, lr=0.002)

    xa = torch.tensor(counts_a)
    xb = torch.tensor(counts_b)

    def elbo(x, enc, dbn, phi):
        mu, lv = enc.encode(x)
        z = reparameterize(mu, lv)
        theta = F.softmax(z, dim=1)
        recon = F.softmax(dbn(theta @ phi), dim=1)
        rec = -(x * (recon + 1e-10).log()).sum(1)
        kld = dirichlet_laplace_kl(mu, lv, mu2, var2, K)
        return (rec + kld).mean()

    def mutual_info(fa, fb, pos, neg):
        na = F.normalize(fa, dim=1)
        nb = F.normalize(fb, dim=1)
        s = (na @ nb.T) / temp
        s = s - s.max(1, keepdim=True).values.detach()
        exps = s.exp()
        denom = (exps * neg).sum(1, keepdim=True) + exps + 1e-10
        log_prob = s - denom.log()
        return -(pos * log_prob).sum()

    for _ in range(ITERS):
        opt.zero_grad()
        la = elbo(xa, enc_a, dec_bn_a, phi_a)
        lb = elbo(xb, enc_b, dec_bn_b, phi_b)
        tami = mutual_info(phi_a.T, phi_b.T, trans, neg_ab)
        tami = tami + mutual_info(phi_b.T, phi_a.T, trans_t, neg_ba)
        loss = la + lb + weight * tami / pos_sum
        loss.backward()
        opt.step()

    with torch.no_grad():
        ba = F.softmax(phi_a, dim=1).cpu().numpy()
        bb = F.softmax(phi_b, dim=1).cpu().numpy()
    return ba.astype(np.float64), bb.astype(np.float64)


# --------------------------------------------------------------------------- #
# topica fit (shared by regenerate + offline compare)
# --------------------------------------------------------------------------- #
def _topica_topic_word(docs_a, docs_b, dictionary, vocab_a, vocab_b, seed):
    import topica

    m = topica.InfoCTM(num_topics=K, seed=seed, hidden_size=HIDDEN, lr=0.01)
    m.fit(docs_a, docs_b, dictionary=dictionary, iters=ITERS, batch_size=64)
    t_a = np.asarray(m.topic_word(lang="a"), dtype=np.float64)
    t_b = np.asarray(m.topic_word(lang="b"), dtype=np.float64)
    t_a = t_a[:, [list(m.vocabulary(lang="a")).index(w) for w in vocab_a]]
    t_b = t_b[:, [list(m.vocabulary(lang="b")).index(w) for w in vocab_b]]
    return t_a, t_b


def _block_agreement(beta_ref, beta_topica, per=PER):
    """Fraction of topics whose top-word block matches, after aligning topica's
    topics to the reference's by top-word block."""
    ref_block = np.array([int(np.argmax(beta_ref[t]) // per) for t in range(beta_ref.shape[0])])
    top_block = np.array([int(np.argmax(beta_topica[t]) // per) for t in range(beta_topica.shape[0])])
    # Both should be a permutation of 0..K-1 on this clean design; count topica
    # topics whose block appears among the reference's blocks (set equality check).
    return float(len(set(top_block.tolist()) & set(ref_block.tolist())) / beta_ref.shape[0])


# --------------------------------------------------------------------------- #
# regenerate
# --------------------------------------------------------------------------- #
def regenerate() -> None:
    if not _torch_available():
        print("torch not available; cannot regenerate.")
        sys.exit(1)

    docs_a, docs_b, vocab_a, vocab_b, dictionary = _make_data()
    ca, cb = _counts(docs_a, vocab_a), _counts(docs_b, vocab_b)
    ia = {w: i for i, w in enumerate(vocab_a)}
    ib = {w: i for i, w in enumerate(vocab_b)}
    trans = np.zeros((len(vocab_a), len(vocab_b)), dtype=np.float32)
    for wa, wb in dictionary:
        trans[ia[wa], ib[wb]] = 1.0

    r_a1, r_b1 = _train_reference(ca, cb, trans, GOLD_SEED)
    r_a2, r_b2 = _train_reference(ca, cb, trans, FLOOR_SEED)
    ref_align = _alignment(r_a1, r_b1)
    ref_align_floor = _alignment(r_a2, r_b2)

    # topica summary at regenerate time for the provenance log.
    t_a, t_b = _topica_topic_word(docs_a, docs_b, dictionary, vocab_a, vocab_b, TOPICA_SEED)
    topica_align = _alignment(t_a, t_b)

    harness.save_gold(
        NAME,
        arrays={
            "topic_word_a": r_a1,
            "topic_word_b": r_b1,
            "vocab_a": np.array(vocab_a, dtype=object),
            "vocab_b": np.array(vocab_b, dtype=object),
            "dictionary": np.array(dictionary, dtype=object),
        },
        meta={
            "reference": _torch_version(),
            "model": "InfoCTM (Wu et al. 2023, AAAI)",
            "corpus": ("synthetic matched bilingual planted-block, A=300/B=270 docs x "
                       "15 tokens, K=5 blocks of 6 words/lang, block-aligned dictionary "
                       "(from parity/infoctm_compare.py)"),
            "num_docs_a": N_A,
            "num_docs_b": N_B,
            "vocab_size_a": len(vocab_a),
            "vocab_size_b": len(vocab_b),
            "num_topics": K,
            "per": PER,
            "hidden": HIDDEN,
            "iters": ITERS,
            "seeds": {"gold": GOLD_SEED, "noise_floor": FLOOR_SEED, "topica": TOPICA_SEED},
            "align_bar": ALIGN_BAR,
            "reference_alignment": ref_align,
            "reference_alignment_floor": ref_align_floor,
            "topica_alignment": topica_align,
            "reference_purity_a": _purity(r_a1),
            "reference_purity_b": _purity(r_b1),
            "topica_purity_a": _purity(t_a),
            "topica_purity_b": _purity(t_b),
            "date": datetime.date.today().isoformat(),
            "pass_bar": ("topica cross-lingual alignment >= align_bar (0.8, the live "
                         "infoctm_compare bar); per-language top-word block agreement vs "
                         "the reference is a reported diagnostic, not the bar"),
            "kind": ("cross-implementation (paper-derived PyTorch InfoCTM); alignment / "
                     "top-word bar, NOT topic-word cosine -- topica is sharper by design"),
        },
    )
    npz, js = harness.gold_paths(NAME)
    print(f"wrote {npz.name} + {js.name} ({npz.stat().st_size} bytes)")
    print(f"  reference alignment : {ref_align:.2f} (seed2 floor {ref_align_floor:.2f})")
    print(f"  topica alignment    : {topica_align:.2f}  (bar {ALIGN_BAR:.2f})")


# --------------------------------------------------------------------------- #
# offline compare
# --------------------------------------------------------------------------- #
def run(verbose: bool = True) -> dict:
    arrays, meta = harness.load_gold(NAME)
    r_a = arrays["topic_word_a"].astype(np.float64)
    r_b = arrays["topic_word_b"].astype(np.float64)
    vocab_a = [str(w) for w in arrays["vocab_a"]]
    vocab_b = [str(w) for w in arrays["vocab_b"]]
    dictionary = [(str(wa), str(wb)) for wa, wb in arrays["dictionary"]]
    bar = float(meta["align_bar"])

    docs_a, docs_b, va, vb, _ = _make_data()
    assert va == vocab_a and vb == vocab_b, "corpus vocab drift vs frozen gold"

    t_a, t_b = _topica_topic_word(docs_a, docs_b, dictionary, vocab_a, vocab_b, TOPICA_SEED)
    topica_align = _alignment(t_a, t_b)
    block_agree_a = _block_agreement(r_a, t_a)
    block_agree_b = _block_agreement(r_b, t_b)

    result = {
        "topica_alignment": topica_align,
        "reference_alignment": float(meta["reference_alignment"]),
        "block_agreement_a": block_agree_a,
        "block_agreement_b": block_agree_b,
        "bar": bar,
        "passes": bool(topica_align >= bar),
    }
    if verbose:
        print(f"gold: {meta.get('reference')}")
        print(f"  topica cross-lingual alignment : {topica_align:.2f} "
              f"(reference {result['reference_alignment']:.2f}, bar {bar:.2f})")
        print(f"  per-language top-word block agreement vs reference : "
              f"A {block_agree_a:.2f} / B {block_agree_b:.2f}")
        print(f"  verdict: {'PASS' if result['passes'] else 'FAIL'}")
    return result


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        run(verbose=True)
