"""InfoCTM: cross-lingual neural topic model (Wu et al. 2023).

Two ProdLDA models, one per language, aligned by a bilingual dictionary through the
Topic-Alignment Mutual-Information term. The headline behaviour is *cross-lingual
alignment*: topic ``k`` should denote the same theme in both languages.
"""

import numpy as np
import pytest

import topica

K = 3


def _bilingual(n_a=120, n_b=108, blocks=K, per=4, length=12, seed=0):
    """Two languages of ``blocks`` planted word-blocks. Lang A words look like
    ``a{block}_{i}``, lang B ``b{block}_{i}``; the dictionary links block-aligned
    words across languages. Returns (docs_a, docs_b, dictionary)."""
    rng = np.random.default_rng(seed)

    def corpus(prefix, n):
        out = []
        for d in range(n):
            b = d % blocks
            out.append([f"{prefix}{b}_{int(rng.integers(per))}" for _ in range(length)])
        return out

    docs_a = corpus("a", n_a)
    docs_b = corpus("b", n_b)
    dictionary = [
        (f"a{b}_{i}", f"b{b}_{j}")
        for b in range(blocks)
        for i in range(per)
        for j in range(per)
    ]
    return docs_a, docs_b, dictionary


def _fit(seed=1, **kw):
    a, b, d = _bilingual()
    m = topica.InfoCTM(num_topics=K, seed=seed, hidden_size=32, lr=0.01,
                       languages=("en", "zh"), **kw)
    m.fit(a, b, dictionary=d, iters=120, batch_size=40)
    return m, (a, b, d)


def _block_of(word):
    return word[1]  # "a{block}_{i}" / "b{block}_{i}"


# ---------------------------------------------------------------------------
# The headline behaviour: cross-lingual topic alignment
# ---------------------------------------------------------------------------

def test_topics_align_across_languages():
    m, _ = _fit()
    ta = m.top_words(3, lang="en", weights=True)
    tb = m.top_words(3, lang="zh", weights=True)
    aligned = sum(_block_of(ta[t][0][0]) == _block_of(tb[t][0][0]) for t in range(K))
    # The dictionary-seeded MI term should align all three planted topics.
    assert aligned == K


# ---------------------------------------------------------------------------
# Standard fitted surface, per language
# ---------------------------------------------------------------------------

def test_topic_word_and_doc_topic_valid():
    m, (a, b, _) = _fit()
    for lang, docs in (("en", a), ("zh", b)):
        tw = m.topic_word(lang=lang)
        assert tw.shape[0] == K
        assert np.allclose(tw.sum(axis=1), 1.0)
        dt = m.doc_topic(lang=lang)
        assert dt.shape == (len(docs), K)
        assert np.allclose(dt.sum(axis=1), 1.0)
        assert (dt >= 0).all()


def test_top_words_and_vocabulary():
    m, _ = _fit()
    tw = m.top_words(5, lang="en", weights=True)
    assert len(tw) == K and len(tw[0]) == 5
    assert all(isinstance(w, str) for w, _ in tw[0])
    assert len(m.vocabulary(lang="zh")) >= K


def test_transform_heldout():
    m, (a, _, _) = _fit()
    out = m.transform(a[:4], lang="en")
    assert out.shape == (4, K)
    assert np.allclose(out.sum(axis=1), 1.0)


def test_lang_selector_accepts_names_and_ab():
    m, _ = _fit()
    # "en"/"zh" (names), "a"/"b" (positional) select the same matrices.
    assert np.array_equal(m.topic_word(lang="en"), m.topic_word(lang="a"))
    assert np.array_equal(m.topic_word(lang="zh"), m.topic_word(lang="b"))


# ---------------------------------------------------------------------------
# Determinism + dictionary-only vs embedding-densified
# ---------------------------------------------------------------------------

def test_deterministic_under_seed():
    m1, _ = _fit(seed=7)
    m2, _ = _fit(seed=7)
    assert np.array_equal(m1.topic_word(lang="en"), m2.topic_word(lang="en"))
    assert np.array_equal(m1.doc_topic(lang="zh"), m2.doc_topic(lang="zh"))


def test_embeddings_optional_and_change_fit():
    a, b, d = _bilingual()
    rng = np.random.default_rng(3)
    # Embeddings where same-block words are close, so the mask densifies within block.
    def emb(prefix):
        out = {}
        for blk in range(K):
            center = rng.normal(size=8)
            for i in range(4):
                out[f"{prefix}{blk}_{i}"] = center + rng.normal(0, 0.05, 8)
        return out
    base = topica.InfoCTM(num_topics=K, seed=1, hidden_size=32, lr=0.01)
    base.fit(a, b, dictionary=d, iters=80, batch_size=40)
    dens = topica.InfoCTM(num_topics=K, seed=1, hidden_size=32, lr=0.01)
    dens.fit(a, b, dictionary=d, embeddings_a=emb("a"), embeddings_b=emb("b"),
             iters=80, batch_size=40)
    # The densified mask changes the alignment term, hence the fit.
    assert not np.allclose(base.topic_word(lang="a"), dens.topic_word(lang="a"))
    assert np.allclose(dens.topic_word(lang="a").sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# Validation / errors
# ---------------------------------------------------------------------------

def test_construction_validation():
    with pytest.raises(ValueError):
        topica.InfoCTM(num_topics=1)
    with pytest.raises(ValueError):
        topica.InfoCTM(num_topics=3, mi_temperature=0.0)
    with pytest.raises(ValueError):
        topica.InfoCTM(num_topics=3, dropout=1.0)
    # #517 guard-parity: convergence_tol must be finite and >= 0.
    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError):
            topica.InfoCTM(num_topics=3, convergence_tol=bad)


def test_unknown_lang_raises():
    m, _ = _fit()
    with pytest.raises(ValueError):
        m.topic_word(lang="fr")


def test_unfitted_access_raises():
    m = topica.InfoCTM(num_topics=3)
    with pytest.raises(RuntimeError):
        m.topic_word(lang="a")


def test_fit_history_records_epochs():
    m, _ = _fit()
    hist = m.fit_history
    assert len(hist) >= 1
    assert all(np.isfinite(hist))


def test_save_load_round_trip(tmp_path):
    """Both-language outputs survive a save/load exactly (#504)."""
    m, (a, b, _) = _fit(seed=1)
    path = str(tmp_path / "ictm.bin")
    m.save(path)
    ld = topica.InfoCTM.load(path)
    for lang, docs in (("en", a), ("zh", b)):
        assert np.array_equal(m.topic_word(lang=lang), ld.topic_word(lang=lang))
        assert np.array_equal(m.doc_topic(lang=lang), ld.doc_topic(lang=lang))
        assert m.vocabulary(lang=lang) == ld.vocabulary(lang=lang)
        # The retained per-language ProdLDA encoders reproduce held-out transform.
        assert np.allclose(m.transform(docs[:5], lang=lang), ld.transform(docs[:5], lang=lang))
    assert ld.fit_history == m.fit_history


def test_save_requires_fit(tmp_path):
    m = topica.InfoCTM(num_topics=3)
    with pytest.raises(RuntimeError):
        m.save(str(tmp_path / "x.bin"))
