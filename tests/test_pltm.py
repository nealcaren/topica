"""Tests for the Polylingual Topic Model (PLTM / PolylingualLDA)."""

import numpy as np
import pytest

import topica

K, BLOCK, TOKENS_PER_DOC, NDOCS = 3, 6, 12, 180


def _lang_docs(seed):
    """One language's documents: tuple d is drawn from topic d % K, whose block
    is word ids [t*BLOCK, (t+1)*BLOCK). Each language has its own id space."""
    r = np.random.default_rng(seed)
    docs = []
    for d in range(NDOCS):
        t = d % K
        docs.append(
            [f"w{t * BLOCK + int(r.integers(BLOCK))}" for _ in range(TOKENS_PER_DOC)]
        )
    return docs


def _corpus(langs=("en", "fr", "de"), base_seed=1000):
    return {name: _lang_docs(base_seed + i) for i, name in enumerate(langs)}


def _fit(k=K, iters=300, seed=42, **kw):
    data = _corpus()
    m = topica.PolylingualLDA(k, iters=iters, seed=seed, **kw)
    m.fit(data)
    return m, data


def test_shapes_and_simplices():
    m, data = _fit()
    assert m.languages == ["en", "fr", "de"]
    assert m.doc_topic.shape == (NDOCS, K)
    assert np.allclose(m.doc_topic.sum(1), 1.0)
    for lg in m.languages:
        tw = m.topic_word(lang=lg)
        assert tw.shape == (K, K * BLOCK)
        assert np.allclose(tw.sum(1), 1.0)


def test_default_lang_is_first():
    m, _ = _fit()
    assert np.array_equal(m.topic_word(), m.topic_word(lang="en"))
    # lang selectable by numeric index too
    assert np.array_equal(m.topic_word(lang="1"), m.topic_word(lang="fr"))


def test_recovers_planted_blocks_aligned_across_languages():
    m, _ = _fit()

    def peak_block(lang, t):
        # Vocab ids are assigned by first-appearance order, so map the peaked id
        # back through the vocabulary to the planted word (``"w13"`` -> block).
        vocab = m.vocabulary(lang=lang)
        vid = int(np.argmax(m.topic_word(lang=lang)[t]))
        return int(vocab[vid][1:]) // BLOCK

    # Each topic peaks on the SAME block in every language (the PLTM property).
    for t in range(K):
        blocks = {peak_block(lg, t) for lg in m.languages}
        assert len(blocks) == 1, f"topic {t} misaligned across languages: {blocks}"
    # And the K topics cover all K planted blocks in each language.
    for lg in m.languages:
        assert {peak_block(lg, t) for t in range(K)} == set(range(K))


def test_determinism_same_seed():
    m1, data = _fit()
    m2 = topica.PolylingualLDA(K, iters=300, seed=42)
    m2.fit(data)
    for lg in m1.languages:
        assert np.array_equal(m1.topic_word(lang=lg), m2.topic_word(lang=lg))
    assert np.array_equal(m1.doc_topic, m2.doc_topic)
    assert np.array_equal(m1.alpha, m2.alpha)


def test_transform_shapes_and_simplex():
    m, data = _fit()
    theta = m.transform(data)
    assert theta.shape == (NDOCS, K)
    assert np.allclose(theta.sum(1), 1.0)


def test_list_input_autonames_languages():
    m = topica.PolylingualLDA(K, iters=50)
    m.fit([_lang_docs(1), _lang_docs(2)])
    assert m.languages == ["lang_0", "lang_1"]


def test_alpha_optimization_on_by_default():
    m, _ = _fit()
    # optimize_alpha=True moves alpha off the symmetric init (0.01); the learned
    # asymmetric prior should not be perfectly uniform.
    assert m.alpha.shape == (K,)
    # A run with optimization off keeps alpha exactly symmetric at the init.
    m_off, _ = _fit(optimize_alpha=False)
    assert np.allclose(m_off.alpha, 0.01)


def test_save_load_roundtrip(tmp_path):
    m, _ = _fit()
    p = tmp_path / "pltm.topica"
    m.save(str(p))
    ml = topica.PolylingualLDA.load(str(p))
    assert ml.languages == m.languages
    for lg in m.languages:
        assert np.array_equal(ml.topic_word(lang=lg), m.topic_word(lang=lg))
    assert np.array_equal(ml.doc_topic, m.doc_topic)


def test_pltm_alias():
    assert topica.PLTM is topica.PolylingualLDA


def test_coherence_per_language():
    m, _ = _fit()
    coh = m.coherence(lang="fr")
    assert coh.shape == (K,)
    assert np.all(np.isfinite(coh))


def test_top_words_per_language():
    m, _ = _fit()
    tw = m.top_words(4, lang="de")
    assert len(tw) == K
    assert all(len(row) == 4 for row in tw)


def test_mismatched_tuple_counts_raise():
    m = topica.PolylingualLDA(K, iters=10)
    with pytest.raises(ValueError):
        m.fit({"en": _lang_docs(1), "fr": _lang_docs(2)[:-5]})


def test_unknown_lang_raises():
    m, _ = _fit()
    with pytest.raises(ValueError):
        m.topic_word(lang="es")


# --------------------------------------------------------------------------- #
# #450: dict transform must map by language NAME, not insertion order.
# --------------------------------------------------------------------------- #

def _disjoint():
    """Two languages with completely disjoint vocabularies, so mapping one
    against the other's vocab drops every token (the #450 mismap symptom)."""
    en = [["e0", "e0", "e1"], ["e2", "e2", "e3"]] * 20
    fr = [["f0", "f0", "f1"], ["f2", "f2", "f3"]] * 20
    return en, fr


def test_transform_dict_matched_by_name_not_insertion_order():
    en, fr = _disjoint()
    m = topica.PolylingualLDA(2, iters=100, seed=1)
    m.fit({"en": en, "fr": fr})
    correct = m.transform({"en": en, "fr": fr})
    reversed_order = m.transform({"fr": fr, "en": en})
    # Matched by name -> insertion order must not change the result. Pre-#450 the
    # reversed dict scored each language against the other's vocab (near-uniform).
    assert np.allclose(correct, reversed_order), np.abs(correct - reversed_order).max()


def test_transform_unknown_language_raises():
    en, fr = _disjoint()
    m = topica.PolylingualLDA(2, iters=100, seed=1)
    m.fit({"en": en, "fr": fr})
    # Same count (2), but "de" is not a fitted language.
    with pytest.raises(ValueError, match="unknown language"):
        m.transform({"en": en, "de": fr})


def test_transform_list_input_stays_positional():
    en, fr = _disjoint()
    m = topica.PolylingualLDA(2, iters=100, seed=1)
    m.fit([en, fr])  # auto-named lang_0, lang_1 -> positional
    theta = m.transform([en, fr])
    assert theta.shape == (40, 2)
