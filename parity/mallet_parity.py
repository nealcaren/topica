"""Cross-implementation parity: topica vs. the original Java MALLET.

Java MALLET and topica are independent implementations with different RNGs,
so they are never byte-identical — parity here means *statistical* agreement:
given the same tokenized corpus and hyperparameters, do both recover the same
topics? On a corpus with planted (disjoint-vocabulary) topics they should, and
in practice the alignment is exact.

`lda_parity` runs Java MALLET's `train-topics` and topica's LDA on one
shared, pre-tokenized corpus, aligns the resulting topics, and reports the mean
top-word Jaccard overlap and cosine similarity of the aligned topic pairs.

This module shells out to the `mallet` CLI; callers should check
`mallet_available()` first.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))


def mallet_available() -> bool:
    return shutil.which("mallet") is not None


def mallet_home() -> str | None:
    """Locate MALLET_HOME (the libexec dir with class/ and dist/ jars)."""
    env = os.environ.get("MALLET_HOME")
    if env and os.path.isdir(os.path.join(env, "class")):
        return env
    import glob
    for base in ("/opt/homebrew/Cellar/mallet/*/libexec", "/usr/local/Cellar/mallet/*/libexec"):
        hits = sorted(glob.glob(base))
        if hits:
            return hits[-1]
    return None


def _classpath() -> str | None:
    mh = mallet_home()
    if mh is None:
        return None
    return f"{mh}/class:{mh}/dist/mallet.jar:{mh}/dist/mallet-deps.jar"


def java_drivers_available() -> bool:
    return (
        mallet_available()
        and shutil.which("javac") is not None
        and shutil.which("java") is not None
        and _classpath() is not None
    )


def _ensure_compiled(java_name: str) -> bool:
    """Compile parity/<java_name>.java against the MALLET classpath if needed."""
    cp = _classpath()
    if cp is None or shutil.which("javac") is None:
        return False
    src = os.path.join(HERE, f"{java_name}.java")
    cls = os.path.join(HERE, f"{java_name}.class")
    if not os.path.exists(src):
        return False
    if not os.path.exists(cls) or os.path.getmtime(cls) < os.path.getmtime(src):
        r = subprocess.run(["javac", "-cp", cp, src], capture_output=True, text=True)
        if r.returncode != 0:
            return False
    return True


def labeled_parity(seed: int = 0, iters: int = 800, top_n: int = 6):
    """Compare topica.LabeledLDA to Java MALLET's LabeledLDA on a shared
    multi-label corpus. Topics correspond to labels, so they align by name.
    Returns a dict with mean cosine and per-label detail."""
    from topica import LabeledLDA

    if not _ensure_compiled("LabeledLDADriver"):
        raise RuntimeError("could not compile LabeledLDADriver")
    cp = _classpath()

    rng = np.random.default_rng(seed)
    vocab = {
        "sports": "game team score player coach win".split(),
        "politics": "election vote senate policy congress law".split(),
        "tech": "computer software code data network chip".split(),
    }
    docs, labels = [], []
    for _ in range(200):
        chosen = list(rng.choice(list(vocab), size=rng.integers(1, 3), replace=False))
        words = []
        for lab in chosen:
            words += list(rng.choice(vocab[lab], 8))
        docs.append(words)
        labels.append(chosen)

    d = tempfile.mkdtemp()
    try:
        inp, out = os.path.join(d, "in.txt"), os.path.join(d, "out.txt")
        with open(inp, "w") as f:
            for toks, labs in zip(docs, labels):
                f.write(f"{','.join(labs)}\t{' '.join(toks)}\n")
        subprocess.run(
            ["java", "-cp", f"{cp}:{HERE}", "LabeledLDADriver", inp,
             str(iters), "1", "0.1", "0.01", out],
            check=True, capture_output=True, text=True,
        )
        lines = open(out).read().splitlines()
        mal_labels = lines[0].split(",")
        counts = {}
        for ln in lines[1:]:
            p = ln.split()
            if p:
                counts[p[0]] = {int(x.split(":")[0]): int(x.split(":")[1]) for x in p[1:]}
    finally:
        shutil.rmtree(d, ignore_errors=True)

    K = len(mal_labels)
    tpt = np.zeros(K)
    for cc in counts.values():
        for t, c in cc.items():
            tpt[t] += c
    words = sorted(counts)
    W = len(words)
    beta = 0.01
    mal_phi = np.zeros((K, W))
    for j, w in enumerate(words):
        for t, c in counts[w].items():
            mal_phi[t, j] = (c + beta) / (tpt[t] + beta * W)

    model = LabeledLDA(alpha=0.1, beta=0.01, seed=1)
    model.fit(docs, labels, iters=iters, num_samples=5, sample_interval=25)
    oi = {w: i for i, w in enumerate(model.vocabulary)}
    olabels = model.labels
    our_phi = np.zeros((K, W))
    for t_mal, lab in enumerate(mal_labels):
        t_our = olabels.index(lab)
        for j, w in enumerate(words):
            if w in oi:
                our_phi[t_mal, j] = model.topic_word[t_our, oi[w]]

    cos = []
    for t in range(K):
        a, b = mal_phi[t], our_phi[t]
        cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    return {"labels": mal_labels, "mean_cosine": float(np.mean(cos)), "cosine": cos}


def planted_corpus(seed: int = 0, num_docs: int = 250, doc_len: int = 12):
    """A corpus with five disjoint-vocabulary topics; each document is drawn
    from a single topic, so both implementations should recover the five."""
    rng = np.random.default_rng(seed)
    topics = [
        "alpha bravo charlie delta echo foxtrot".split(),
        "lion tiger bear wolf fox otter".split(),
        "guitar piano violin drums trumpet cello".split(),
        "mercury venus earth mars jupiter saturn".split(),
        "python rust java haskell scala ocaml".split(),
    ]
    docs = [list(rng.choice(topics[rng.integers(0, len(topics))], doc_len)) for _ in range(num_docs)]
    return docs, len(topics)


def _mallet_phi(docs, k, iters, seed, beta=0.01, optimize_interval=0):
    """Run Java MALLET train-topics; return (phi (k, V), vocab).

    ``optimize_interval > 0`` turns on MALLET's Wallach alpha/beta hyperparameter
    optimization (its default cadence is 50, burn-in 200), matching topica's
    optimize_interval so the optimizer path is actually exercised."""
    mallet = shutil.which("mallet")
    d = tempfile.mkdtemp()
    try:
        txt = os.path.join(d, "tok.txt")
        with open(txt, "w") as f:
            for i, toks in enumerate(docs):
                f.write(f"doc{i}\t{' '.join(toks)}\n")
        mal = os.path.join(d, "tok.mallet")
        subprocess.run(
            [mallet, "import-file", "--input", txt, "--output", mal, "--keep-sequence",
             "--token-regex", r"\S+", "--line-regex", r"^(\S+)\t(.*)$",
             "--name", "1", "--data", "2", "--label", "0"],
            check=True, capture_output=True, text=True,
        )
        wtc = os.path.join(d, "wtc.txt")
        subprocess.run(
            [mallet, "train-topics", "--input", mal, "--num-topics", str(k),
             "--num-iterations", str(iters), "--random-seed", str(seed),
             "--optimize-interval", str(optimize_interval),
             "--word-topic-counts-file", wtc, "--num-top-words", "1"],
            check=True, capture_output=True, text=True,
        )
        counts = {}
        for line in open(wtc):
            parts = line.split()
            if len(parts) < 3:
                continue
            counts[parts[1]] = {int(p.split(":")[0]): int(p.split(":")[1]) for p in parts[2:]}
    finally:
        shutil.rmtree(d, ignore_errors=True)

    vocab = sorted(counts)
    tpt = np.zeros(k)
    for cc in counts.values():
        for t, c in cc.items():
            tpt[t] += c
    W = len(vocab)
    phi = np.zeros((k, W))
    for j, w in enumerate(vocab):
        for t, c in counts[w].items():
            phi[t, j] = (c + beta) / (tpt[t] + beta * W)
    return phi, vocab


def _align(a_phi, b_phi):
    """Greedily align rows of b_phi to rows of a_phi by cosine; return pairs."""
    def norm(m):
        return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
    sim = norm(a_phi) @ norm(b_phi).T
    used, pairs = set(), []
    for a in range(a_phi.shape[0]):
        for b in np.argsort(sim[a])[::-1]:
            if b not in used:
                used.add(int(b))
                pairs.append((a, int(b), float(sim[a, b])))
                break
    return pairs


def lda_parity(seed: int = 0, k: int | None = None, iters: int = 800, top_n: int = 6):
    """Compare topica LDA to Java MALLET on a shared planted corpus.

    Returns a dict with `mean_jaccard`, `mean_cosine`, and per-topic detail.
    """
    from topica import LDA

    docs, planted_k = planted_corpus(seed=seed)
    k = planted_k if k is None else k

    mal_phi, vocab = _mallet_phi(docs, k, iters, seed=1)
    vi = {w: j for j, w in enumerate(vocab)}

    model = LDA(num_topics=k, seed=1, optimize_interval=0)
    model.fit(docs, iters=iters, num_samples=5, sample_interval=25)
    ophi, ovocab = model.topic_word, model.vocabulary
    our_phi = np.zeros((k, len(vocab)))
    for j, w in enumerate(vocab):
        if w in {x: 1 for x in ovocab}:
            our_phi[:, j] = ophi[:, ovocab.index(w)]

    pairs = _align(mal_phi, our_phi)

    def topw(m, t):
        return set(vocab[j] for j in np.argsort(m[t])[::-1][:top_n])

    jacc = [len(topw(mal_phi, a) & topw(our_phi, b)) / len(topw(mal_phi, a) | topw(our_phi, b))
            for a, b, _ in pairs]
    return {
        "k": k,
        "mean_jaccard": float(np.mean(jacc)),
        "mean_cosine": float(np.mean([s for *_, s in pairs])),
        "jaccard": jacc,
        "cosine": [s for *_, s in pairs],
    }


def lda_parity_optimized(seed: int = 0, k: int | None = None, iters: int = 1000, top_n: int = 6):
    """Like :func:`lda_parity`, but with hyperparameter optimization ON in both
    engines and a *terminal* topica estimator (``num_samples=1``), so the check
    actually exercises the Wallach alpha/beta optimizer against MALLET rather than
    the optimize-off, snapshot-averaged path the committed gold uses (#713-#4).

    Both run ``--optimize-interval 50`` past a burn-in of 200. topica's asymmetric
    alpha step reproduces MALLET's ``learnParameters``; beta uses topica's
    corrected convergent estimator, so agreement is statistical (topic overlap),
    not byte parity."""
    from topica import LDA

    docs, planted_k = planted_corpus(seed=seed)
    k = planted_k if k is None else k

    mal_phi, vocab = _mallet_phi(docs, k, iters, seed=1, optimize_interval=50)

    # Terminal estimator (num_samples=1): the final Gibbs state, not an average of
    # snapshots, so it lines up with MALLET's single terminal draw.
    model = LDA(num_topics=k, seed=1, optimize_interval=50, burn_in=200)
    model.fit(docs, iters=iters, num_samples=1)
    ophi, ovocab = model.topic_word, model.vocabulary
    our_phi = np.zeros((k, len(vocab)))
    ov = {w: i for i, w in enumerate(ovocab)}
    for j, w in enumerate(vocab):
        if w in ov:
            our_phi[:, j] = ophi[:, ov[w]]

    pairs = _align(mal_phi, our_phi)

    def topw(m, t):
        return set(vocab[j] for j in np.argsort(m[t])[::-1][:top_n])

    jacc = [len(topw(mal_phi, a) & topw(our_phi, b)) / len(topw(mal_phi, a) | topw(our_phi, b))
            for a, b, _ in pairs]
    return {
        "k": k,
        "optimize_interval": 50,
        "mean_jaccard": float(np.mean(jacc)),
        "mean_cosine": float(np.mean([s for *_, s in pairs])),
        "alpha_sum": float(np.sum(model.alpha)),
        "beta": float(model.beta),
    }


def dmr_parity(seed: int = 0, iters: int = 800, num_docs: int = 160):
    """Compare topica.DMR to Java MALLET's DMRTopicModel. Because DMR fits
    feature weights with L-BFGS (which differs between implementations), this is
    a *statistical* check: do the topics align, and does the covariate's effect
    agree in sign? Returns topic cosine and the (space - animal) covariate effect
    from each implementation."""
    from topica import DMR

    if not _ensure_compiled("DMRDriver"):
        raise RuntimeError("could not compile DMRDriver")
    cp = _classpath()

    rng = np.random.default_rng(seed)
    animal = "cat dog fish bird horse cow".split()
    space = "planet star moon rocket comet galaxy".split()
    docs, cov = [], []
    for _ in range(num_docs):
        x = int(rng.integers(0, 2))
        docs.append(list(rng.choice(space if x else animal, 8)))
        cov.append(x)

    d = tempfile.mkdtemp()
    try:
        inp = os.path.join(d, "in.txt")
        oc, op = os.path.join(d, "c.txt"), os.path.join(d, "p.txt")
        with open(inp, "w") as f:
            for toks, x in zip(docs, cov):
                f.write(f"is_space={x}\t{' '.join(toks)}\n")
        subprocess.run(
            ["java", "-cp", f"{cp}:{HERE}", "DMRDriver", inp, "2", str(iters), "1", oc, op],
            check=True, capture_output=True, text=True,
        )
        import collections
        tw = collections.defaultdict(dict)
        wset = set()
        for ln in open(oc):
            p = ln.split("\t")
            if len(p) >= 3:
                tw[int(p[0])][p[1]] = float(p[2])
                wset.add(p[1])
        plines = open(op).read().splitlines()
        mal_lam = np.array([[float(x) for x in ln.split()] for ln in plines[1:]])  # (K,[icpt,is_space])
    finally:
        shutil.rmtree(d, ignore_errors=True)

    words = sorted(wset)
    K, W = 2, len(words)
    mal_phi = np.zeros((K, W))
    for t in range(K):
        for j, w in enumerate(words):
            mal_phi[t, j] = tw[t].get(w, 0.0)
        mal_phi[t] /= mal_phi[t].sum()

    model = DMR(num_topics=2, seed=1, optimize_interval=25, burn_in=50)
    model.fit(docs, np.array(cov, float)[:, None], feature_names=["is_space"],
              iters=iters, num_samples=5, sample_interval=25)
    oi = {w: i for i, w in enumerate(model.vocabulary)}
    ours = np.zeros((K, W))
    for j, w in enumerate(words):
        if w in oi:
            ours[:, j] = model.topic_word[:, oi[w]]
    ours /= ours.sum(axis=1, keepdims=True)

    def nm(m):
        return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
    cos = float((nm(mal_phi) @ nm(ours).T).max(axis=1).mean())

    def space_topic(phi):
        sc = [sum(phi[t, words.index(w)] for w in space if w in words) for t in range(K)]
        return int(np.argmax(sc))
    ms, os_ = space_topic(mal_phi), space_topic(ours)
    fe = model.feature_effects
    return {
        "topic_cosine": cos,
        "mallet_effect": float(mal_lam[ms, 1] - mal_lam[1 - ms, 1]),
        "topica_effect": float(fe[os_, 1] - fe[1 - os_, 1]),
    }


if __name__ == "__main__":
    if not mallet_available():
        raise SystemExit("mallet not found on PATH")
    r = lda_parity()
    print(f"LDA        vs MALLET: Jaccard={r['mean_jaccard']:.3f} cosine={r['mean_cosine']:.3f}")
    if java_drivers_available():
        rl = labeled_parity()
        print(f"LabeledLDA vs MALLET: cosine={rl['mean_cosine']:.3f}")
        rd = dmr_parity()
        print(f"DMR        vs MALLET: topic cosine={rd['topic_cosine']:.3f} "
              f"effect MALLET={rd['mallet_effect']:+.2f} ours={rd['topica_effect']:+.2f}")
    else:
        print("LabeledLDA / DMR vs MALLET: SKIPPED (needs a JDK with javac to "
              "compile the Java drivers)")
