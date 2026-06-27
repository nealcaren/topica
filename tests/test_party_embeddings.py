"""PartyEmbeddings: the corpus-trained word-embedding ideal-point scaler
(Rheault & Cochrane 2020).

A scaling model with no topic distribution, so it is exempt from the registry
topic-health suite; its validity is checked here: it recovers a planted party
ordering from a PV-DM fit, the toolkit (nearest words, guided projection,
polarization distance) behaves, the fit is deterministic, and it round-trips.
"""
import numpy as np
import pytest

import topica


def _planted(n_groups=10, docs_per=60, doc_len=18, seed=0):
    """Group g has a planted position in [-1, 1]; it emits right-marker words with
    probability rising in that position, left markers with the complement, plus
    neutral filler. The party language therefore encodes the ordering."""
    rng = np.random.default_rng(seed)
    pos = np.linspace(-1.0, 1.0, n_groups)
    left = [f"L{i}" for i in range(12)]
    right = [f"R{i}" for i in range(12)]
    filler = [f"f{i}" for i in range(40)]
    docs, groups, planted = [], [], {}
    for g in range(n_groups):
        label = f"P{g:02d}"
        planted[label] = float(pos[g])
        pr_right = (pos[g] + 1) / 2.0
        for _ in range(docs_per):
            doc = []
            for _ in range(doc_len):
                if rng.random() < 0.45:
                    pool = right if rng.random() < pr_right else left
                    doc.append(pool[rng.integers(len(pool))])
                else:
                    doc.append(filler[rng.integers(len(filler))])
            docs.append(doc)
            groups.append(label)
    return docs, groups, planted


def _fit(seed=0, **kw):
    docs, groups, planted = _planted(seed=seed)
    m = topica.PartyEmbeddings(
        num_dims=2, vector_size=48, window=5, min_count=1, negative=5,
        sample=1e-3, learning_rate=0.05, seed=seed,
    )
    m.fit(docs, group=groups, iters=40, **kw)
    return m, planted


def test_shapes():
    m, _ = _fit()
    g = m.num_authors
    assert m.author_positions.shape == (g, 2)
    assert m.author_vectors.shape == (g, 48)
    assert m.word_vectors.shape == (len(m.vocabulary), 48)
    assert len(m.author_names) == g


def test_recovers_planted_ordering():
    m, planted = _fit()
    plant = np.array([planted[n] for n in m.author_names])
    r = abs(np.corrcoef(m.author_positions[:, 0], plant)[0, 1])
    assert r > 0.85, f"planted recovery too low: |r| = {r:.3f}"


def test_anchors_orient_sign():
    docs, groups, planted = _planted(seed=1)
    m = topica.PartyEmbeddings(num_dims=1, vector_size=48, window=5, min_count=1,
                               negative=5, sample=1e-3, learning_rate=0.05, seed=1)
    # anchor the most-left group negative and the most-right group positive
    m.fit(docs, group=groups, anchors={"P00": -1.0, "P09": 1.0}, iters=40)
    names = m.author_names
    pos = m.author_positions[:, 0]
    i0, i9 = names.index("P00"), names.index("P09")
    assert pos[i0] < pos[i9], "anchors did not orient the axis as requested"


def test_nearest_words_are_on_the_right_pole():
    m, planted = _fit()
    right_party = max(m.author_names, key=lambda n: planted[n])
    near = {w for w, _ in m.nearest_words(right_party, 8)}
    # the right-most party's nearest words should be dominated by right markers
    n_right = sum(1 for w in near if w.startswith("R"))
    assert n_right >= 4, f"expected right-marker words near the right party, got {near}"


def test_guided_positions():
    m, planted = _fit()
    plant = np.array([planted[n] for n in m.author_names])
    gp = m.guided_positions(left=["L0", "L1", "L2", "L3"],
                            right=["R0", "R1", "R2", "R3"])
    assert gp.shape == (m.num_authors,)
    assert abs(np.corrcoef(gp, plant)[0, 1]) > 0.8


def test_distance_orders_polarization():
    m, planted = _fit()
    names = sorted(m.author_names, key=lambda n: planted[n])
    extremes = m.distance(names[0], names[-1])
    neighbors = m.distance(names[0], names[1])
    assert extremes > neighbors


def test_control_tag_runs():
    docs, groups, _ = _planted(seed=2)
    control = ["era_early" if i % 2 == 0 else "era_late" for i in range(len(docs))]
    m = topica.PartyEmbeddings(num_dims=1, vector_size=32, window=5, min_count=1,
                               negative=5, sample=1e-3, learning_rate=0.05, seed=2)
    m.fit(docs, group=groups, control=control, iters=10)
    assert m.author_positions.shape == (m.num_authors, 1)


def test_determinism():
    a, _ = _fit(seed=3)
    b, _ = _fit(seed=3)
    assert np.array_equal(a.author_vectors, b.author_vectors)
    assert np.array_equal(a.author_positions, b.author_positions)
    c, _ = _fit(seed=4)
    assert not np.array_equal(a.author_vectors, c.author_vectors)


def test_save_load_roundtrip(tmp_path):
    m, _ = _fit(seed=5)
    p = str(tmp_path / "pe.tt")
    m.save(p)
    loaded = topica.PartyEmbeddings.load(p)
    assert np.array_equal(m.author_positions, loaded.author_positions)
    assert np.array_equal(m.author_vectors, loaded.author_vectors)
    assert m.author_names == loaded.author_names
    assert loaded.nearest_words(m.author_names[0], 3)


def test_bad_params():
    with pytest.raises(ValueError):
        topica.PartyEmbeddings(num_dims=0)
    with pytest.raises(ValueError):
        topica.PartyEmbeddings(vector_size=1)
    with pytest.raises(ValueError):
        topica.PartyEmbeddings(negative=0)
    # group is required and must match num_docs
    m = topica.PartyEmbeddings(vector_size=16, min_count=1)
    with pytest.raises(Exception):
        m.fit([["a", "b"], ["c", "d"]], group=["x"])  # wrong length


def test_unfitted_raises():
    m = topica.PartyEmbeddings()
    with pytest.raises(Exception):
        _ = m.author_positions
