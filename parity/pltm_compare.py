"""Parity check: topica's PolylingualLDA vs MALLET's PolylingualTopicModel.

The reference is `cc.mallet.topics.PolylingualTopicModel` (Mimno, Wallach,
Naradowsky, Smith & McCallum 2009) shipped with Java MALLET -- the implementation
from the paper's own authors. MALLET is weak-copyleft, so topica's port is derived
from the paper and MALLET is used here only as a black-box oracle.

Both engines fit the same fixed synthetic multilingual corpus: K planted topics,
each a disjoint per-language word block, with tuples aligned by index across L
languages. We reconstruct MALLET's per-language topic-word matrices from its Gibbs
state file, align topica's topics to MALLET's (Hungarian assignment on language 0),
apply that one permutation to every language, and assert the mean aligned per-language
topic-word cosine clears a floor. The RNGs differ (topica ChaCha8, MALLET Java), so
this is a distributional parity check, not bit-exact; on well-separated blocks both
converge to the same aligned topics.

Skips cleanly when `mallet` is not on PATH or SciPy is unavailable.
"""

import gzip
import os
import shutil
import subprocess
import tempfile

import numpy as np

import topica

# ---- shared corpus + fit configuration -------------------------------------
K, BLOCK, NUM_LANGS = 4, 8, 3
NDOCS, DLEN = 160, 16
ITERS, OPT_INTERVAL, SEED = 600, 10, 1
COSINE_MIN = 0.90
LANG_NAMES = [f"lang{i}" for i in range(NUM_LANGS)]


def build_corpus(seed=0):
    """K topics; language l has its own vocabulary ``l{l}_b{block}_w{i}``. Tuple d
    is topic d % K in every language, so topics are aligned by index."""
    rng = np.random.default_rng(seed)
    vocabs = [
        [f"l{l}_b{b}_w{i}" for b in range(K) for i in range(BLOCK)]
        for l in range(NUM_LANGS)
    ]
    data = {name: [] for name in LANG_NAMES}
    for d in range(NDOCS):
        t = d % K
        for l, name in enumerate(LANG_NAMES):
            doc = [vocabs[l][t * BLOCK + int(rng.integers(BLOCK))] for _ in range(DLEN)]
            data[name].append(doc)
    return data


def mallet_topic_word(data, workdir):
    """Run MALLET's PolylingualTopicModel and reconstruct per-language topic-word
    matrices from its state file. Returns (list_of_phi, list_of_vocab)."""
    lang_files = []
    for l, name in enumerate(LANG_NAMES):
        txt = os.path.join(workdir, f"{name}.txt")
        with open(txt, "w", encoding="utf-8") as fh:
            for i, doc in enumerate(data[name]):
                fh.write(f"{i}\ten\t{' '.join(doc)}\n")
        mallet_in = os.path.join(workdir, f"{name}.mallet")
        subprocess.run(
            ["mallet", "import-file", "--input", txt, "--output", mallet_in,
             "--keep-sequence", "--token-regex", r"\S+"],
            check=True, capture_output=True,
        )
        lang_files.append(mallet_in)

    state = os.path.join(workdir, "state.gz")
    subprocess.run(
        ["mallet", "run", "cc.mallet.topics.PolylingualTopicModel",
         "--language-inputs", *lang_files,
         "--num-topics", str(K), "--num-iterations", str(ITERS),
         "--optimize-interval", str(OPT_INTERVAL), "--random-seed", str(SEED),
         "--output-state", state],
        check=True, capture_output=True,
    )

    # State rows: "#doc lang pos typeindex type topic"; `type` (col 5) is the word.
    vocabs = [{} for _ in range(NUM_LANGS)]
    counts = [{} for _ in range(NUM_LANGS)]  # (topic, word_id) -> count
    with gzip.open(state, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            lang, word, topic = int(parts[1]), parts[4], int(parts[5])
            wid = vocabs[lang].setdefault(word, len(vocabs[lang]))
            counts[lang][(topic, wid)] = counts[lang].get((topic, wid), 0) + 1

    phis, vocab_lists = [], []
    for l in range(NUM_LANGS):
        v = len(vocabs[l])
        mat = np.full((K, v), 0.01)  # + beta smoothing (MALLET default 0.01)
        for (t, wid), c in counts[l].items():
            mat[t, wid] += c
        mat /= mat.sum(1, keepdims=True)
        phis.append(mat)
        inv = {wid: w for w, wid in vocabs[l].items()}
        vocab_lists.append([inv[i] for i in range(v)])
    return phis, vocab_lists


def align_cosine(topica_phi, topica_vocab, ref_phi, ref_vocab):
    """Cosine matrix (K x K) between topica and reference topics on a shared vocab."""
    common = sorted(set(topica_vocab) & set(ref_vocab))
    ti = {w: i for i, w in enumerate(topica_vocab)}
    ri = {w: i for i, w in enumerate(ref_vocab)}
    a = topica_phi[:, [ti[w] for w in common]]
    b = ref_phi[:, [ri[w] for w in common]]
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


def test_pltm_matches_mallet():
    if shutil.which("mallet") is None:
        print("SKIP: `mallet` not on PATH")
        return
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        print("SKIP: SciPy not available")
        return

    data = build_corpus(seed=0)
    workdir = tempfile.mkdtemp(prefix="pltm_parity_")
    try:
        ref_phis, ref_vocabs = mallet_topic_word(data, workdir)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"SKIP: MALLET run failed ({e})")
        return
    finally:
        pass

    m = topica.PolylingualLDA(K, iters=ITERS, optimize_interval=OPT_INTERVAL, seed=SEED)
    m.fit(data)

    # Align topica -> MALLET using language 0, then apply the same permutation to
    # every language (topics are shared, so one assignment holds across languages).
    cos0 = align_cosine(
        m.topic_word(lang=LANG_NAMES[0]), m.vocabulary(lang=LANG_NAMES[0]),
        ref_phis[0], ref_vocabs[0],
    )
    row, col = linear_sum_assignment(-cos0)
    perm = {r: c for r, c in zip(row, col)}

    per_lang = []
    for l, name in enumerate(LANG_NAMES):
        cos = align_cosine(
            m.topic_word(lang=name), m.vocabulary(lang=name), ref_phis[l], ref_vocabs[l]
        )
        aligned = np.mean([cos[t, perm[t]] for t in range(K)])
        per_lang.append(aligned)
        print(f"{name}: mean aligned topic-word cosine = {aligned:.3f}")

    mean_cos = float(np.mean(per_lang))
    print(f"overall mean aligned cosine = {mean_cos:.3f} (floor {COSINE_MIN})")
    shutil.rmtree(workdir, ignore_errors=True)
    assert mean_cos >= COSINE_MIN, (
        f"aligned topic-word cosine {mean_cos:.3f} below floor {COSINE_MIN}"
    )


if __name__ == "__main__":
    test_pltm_matches_mallet()
