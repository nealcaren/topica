"""Python round-trip tests for issues #98 and #102.

Issue #98: SeededLDA save/load round-trip was broken.
  - num_topics became 0 after load (seed_names was reset to [] so
    num_topics_val() returned 0 + 0 = 0).
  - transform() panicked on the zero-topic model.

Issue #102a: Save format lacked versioning.
  - Loading a garbage file now gives a clear ValueError, not a bincode panic.
  - Loading a file saved by model X as model Y gives a clear error.

Issue #102b: LdaState dropped theta_draws and sampler flags.
  - A save/load round-trip now preserves theta_draws and the light/warp/cvb0
    sampler flags.
"""

import numpy as np
import pytest

import topica

# ---------------------------------------------------------------------------
# Shared fixture data
# ---------------------------------------------------------------------------

DOCS = [["cat", "dog", "pet"]] * 20 + [["star", "moon", "sky"]] * 20


# ---------------------------------------------------------------------------
# Issue #98: SeededLDA round-trip
# ---------------------------------------------------------------------------

def test_seededlda_roundtrip_num_topics_preserved(tmp_path):
    """num_topics must equal seed_topics + residual after a save/load cycle."""
    path = str(tmp_path / "seeded.tt")
    seed_words = {"animals": ["cat", "dog"], "space": ["star", "moon"]}
    m = topica.models.SeededLDA(seed_words, residual=1, seed=1)
    m.fit(DOCS, iters=200)

    expected_k = m.num_topics
    assert expected_k == 3  # 2 seeded + 1 residual

    m.save(path)
    loaded = topica.models.SeededLDA.load(path)

    assert loaded.num_topics == expected_k, (
        f"num_topics dropped from {expected_k} to {loaded.num_topics} after load"
    )


def test_seededlda_roundtrip_matrices_bit_identical(tmp_path):
    """topic_word and doc_topic must be bit-identical after a round-trip."""
    path = str(tmp_path / "seeded.tt")
    seed_words = {"politics": ["cat"], "nature": ["star", "moon"]}
    m = topica.models.SeededLDA(seed_words, residual=2, seed=7)
    m.fit(DOCS, iters=200)

    m.save(path)
    loaded = topica.models.SeededLDA.load(path)

    np.testing.assert_array_equal(
        m.topic_word, loaded.topic_word,
        err_msg="topic_word changed across a save/load round-trip"
    )
    np.testing.assert_array_equal(
        m.doc_topic, loaded.doc_topic,
        err_msg="doc_topic changed across a save/load round-trip"
    )


def test_seededlda_transform_works_after_load(tmp_path):
    """transform() must not panic or error after loading a saved SeededLDA."""
    path = str(tmp_path / "seeded.tt")
    seed_words = {"animals": ["cat", "dog"], "space": ["star", "moon"]}
    m = topica.models.SeededLDA(seed_words, residual=1, seed=2)
    m.fit(DOCS, iters=200)

    m.save(path)
    loaded = topica.models.SeededLDA.load(path)

    # This is the crash repro from issue #98: num_topics=0 made transform panic.
    new_docs = [["cat", "dog"], ["star", "moon"]]
    result = loaded.transform(new_docs)

    assert result.shape == (2, loaded.num_topics), (
        f"transform output shape {result.shape} does not match expected "
        f"(2, {loaded.num_topics})"
    )
    # Each row must be a valid probability simplex.
    np.testing.assert_allclose(
        result.sum(axis=1), np.ones(2), atol=1e-5,
        err_msg="transform rows do not sum to 1 after load"
    )


def test_seededlda_topic_names_preserved(tmp_path):
    """The named seed topics must come back with their original names after load."""
    path = str(tmp_path / "seeded.tt")
    seed_words = {"animals": ["cat", "dog"], "space": ["star", "moon"]}
    m = topica.models.SeededLDA(seed_words, residual=1, seed=3)
    m.fit(DOCS, iters=200)

    before = list(m.topic_names)
    m.save(path)
    loaded = topica.models.SeededLDA.load(path)

    assert list(loaded.topic_names) == before, (
        f"topic_names changed: {before!r} -> {list(loaded.topic_names)!r}"
    )


# ---------------------------------------------------------------------------
# Issue #102b: LDA theta_draws and sampler-flag persistence
# ---------------------------------------------------------------------------

def test_lda_theta_draws_survive_roundtrip(tmp_path):
    """theta_draws must be present and identical after a save/load cycle."""
    path = str(tmp_path / "lda.tt")
    m = topica.models.LDA(num_topics=2, seed=1)
    m.fit(DOCS, iters=200, keep_theta_draws=True, num_theta_draws=10)

    assert m.theta_draws is not None, "theta_draws should be populated after fit"
    before = np.asarray(m.theta_draws)

    m.save(path)
    loaded = topica.models.LDA.load(path)

    assert loaded.theta_draws is not None, (
        "theta_draws is None after load — drops draws (issue #102b)"
    )
    after = np.asarray(loaded.theta_draws)
    np.testing.assert_array_equal(
        before, after,
        err_msg="theta_draws changed across a save/load round-trip"
    )


def test_lda_no_theta_draws_roundtrip(tmp_path):
    """When fit with keep_theta_draws=False, loaded model also has theta_draws=None."""
    path = str(tmp_path / "lda_nodraws.tt")
    m = topica.models.LDA(num_topics=2, seed=1)
    m.fit(DOCS, iters=200, keep_theta_draws=False)

    assert m.theta_draws is None
    m.save(path)
    loaded = topica.models.LDA.load(path)
    assert loaded.theta_draws is None


# ---------------------------------------------------------------------------
# Issue #102a: Format versioning — bad/incompatible files give clear errors
# ---------------------------------------------------------------------------

def test_garbage_file_raises_value_error(tmp_path):
    """Loading a garbage file must raise ValueError, not crash with a bincode panic."""
    bad = str(tmp_path / "bad.tt")
    with open(bad, "wb") as f:
        f.write(b"not a model file at all")

    with pytest.raises(ValueError, match="not a topica model file"):
        topica.models.LDA.load(bad)


def test_wrong_model_tag_gives_clear_error(tmp_path):
    """Loading an LDA file as SeededLDA must name both models in the error message."""
    path = str(tmp_path / "lda.tt")
    m = topica.models.LDA(num_topics=2, seed=1)
    m.fit(DOCS, iters=100)
    m.save(path)

    with pytest.raises(ValueError) as exc_info:
        topica.models.SeededLDA.load(path)

    msg = str(exc_info.value)
    assert "LDA" in msg, f"error should mention the file's model (LDA): {msg}"
    assert "SeededLDA" in msg, f"error should mention the expected model (SeededLDA): {msg}"


def test_seededlda_file_cannot_load_as_lda(tmp_path):
    """SeededLDA file loaded as LDA must give a clear error naming both models."""
    path = str(tmp_path / "seeded.tt")
    m = topica.models.SeededLDA({"a": ["cat"], "b": ["star"]}, seed=1)
    m.fit(DOCS, iters=100)
    m.save(path)

    with pytest.raises(ValueError) as exc_info:
        topica.models.LDA.load(path)

    msg = str(exc_info.value)
    assert "SeededLDA" in msg, f"error should mention SeededLDA: {msg}"
    assert "LDA" in msg, f"error should mention LDA: {msg}"


# ---------------------------------------------------------------------------
# Issue #102b (extended): DMR/LabeledLDA/SAGE/KeyATM also persist theta_draws.
# Before the fix these four set theta_draws=None on load, so a saved-then-loaded
# model silently fell back to the Dirichlet approximation in composition_theta,
# changing estimate_effect / standard_errors uncertainty.
# ---------------------------------------------------------------------------

_TD_DOCS = [["a", "b", "c"], ["b", "c", "d"], ["a", "d", "e"],
            ["c", "e", "f"], ["a", "b", "f"]] * 8


def _fit_dmr():
    X = np.ones((len(_TD_DOCS), 1))
    m = topica.models.DMR(num_topics=3, seed=1)
    m.fit(_TD_DOCS, X, feature_names=["x"], iters=30)
    return m


def _fit_labeled():
    m = topica.models.LabeledLDA(seed=1)
    m.fit(_TD_DOCS, [["t0", "t1"]] * len(_TD_DOCS), iters=30)
    return m


def _fit_sage():
    groups = ["g0", "g1"] * (len(_TD_DOCS) // 2)
    m = topica.models.SAGE(num_topics=3, seed=1)
    m.fit(_TD_DOCS, groups, iters=30)
    return m


def _fit_keyatm():
    m = topica.models.KeyATM({"k0": ["a", "b"], "k1": ["c", "d"]}, seed=1)
    m.fit(_TD_DOCS, iters=30)
    return m


def _fit_seeded():
    m = topica.models.SeededLDA({"k0": ["a", "b"], "k1": ["c", "d"]}, residual=1, seed=1)
    m.fit(_TD_DOCS, iters=30)
    return m


@pytest.mark.parametrize(
    "fit_fn", [_fit_dmr, _fit_labeled, _fit_sage, _fit_keyatm, _fit_seeded]
)
def test_theta_draws_survive_save_load(fit_fn, tmp_path):
    m = fit_fn()
    before = m.theta_draws
    assert before is not None, "model should retain theta_draws by default"
    path = str(tmp_path / "m.bin")
    m.save(path)
    m2 = type(m).load(path)
    after = m2.theta_draws
    assert after is not None, "theta_draws must survive load (issue #102)"
    assert np.array_equal(before, after)
