"""Exact contracts for SeededLDA and GSDMM.

These models do not have a convenient byte-identical external reference runner
in the way topica's MALLET-backed LDA path does. The reference contracts we can
test exactly are the advertised algorithmic formulas:

* SeededLDA's ``seed_prior="uniform"`` scheme: a seed word receives a flat
  ``weight * 100`` extra topic-word prior mass in its seed topic, and seeded
  tokens initialize into that topic. (The default ``seed_prior="frequency"``
  scales that mass by corpus frequency and initializes at random; its exact
  seed-mass construction is checked against R ``seededlda::tfm`` in
  ``parity/seededlda_r_compare.py``.)
* GSDMM follows the Movie Group Process equations for smoothed cluster-word
  distributions and in-sample soft document-cluster probabilities.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

import topica


def test_seededlda_zero_sweep_seed_prior_is_exact() -> None:
    docs = [["tax"], ["iraq"]]
    model = topica.SeededLDA(
        {"econ": ["tax"], "war": ["iraq"]},
        alpha=0.1,
        beta=0.01,
        weight=0.01,
        seed_prior="uniform",
        seed=123,
    )
    model.fit(docs, iters=0)

    # weight=0.01 means a 1.0 extra pseudocount on each seed word in its topic.
    # With one seeded token assigned to each seeded topic and V=2:
    #   phi(seed topic, seed word) = (1 + beta + 1.0) / (1 + 2*beta + 1.0)
    #   phi(seed topic, other word) = beta / (1 + 2*beta + 1.0)
    expected_phi = np.array(
        [
            [2.01 / 2.02, 0.01 / 2.02],
            [0.01 / 2.02, 2.01 / 2.02],
        ]
    )
    expected_theta = np.array(
        [
            [1.1 / 1.2, 0.1 / 1.2],
            [0.1 / 1.2, 1.1 / 1.2],
        ]
    )

    assert model.topic_names == ["econ", "war"]
    assert model.vocabulary == ["tax", "iraq"]
    np.testing.assert_allclose(model.topic_word, expected_phi, rtol=0, atol=1e-12)
    np.testing.assert_allclose(model.doc_topic, expected_theta, rtol=0, atol=1e-12)


def test_seededlda_weight_zero_reduces_to_symmetric_word_prior_at_initialization() -> None:
    docs = [["tax"], ["iraq"]]
    model = topica.SeededLDA(
        {"econ": ["tax"], "war": ["iraq"]},
        alpha=0.1,
        beta=0.01,
        weight=0.0,
        seed_prior="uniform",
        seed=123,
    )
    model.fit(docs, iters=0)

    # Seeded initialization still puts each seed token in its named topic, but
    # with weight=0 there is no extra seed-word pseudocount.
    expected_phi = np.array(
        [
            [1.01 / 1.02, 0.01 / 1.02],
            [0.01 / 1.02, 1.01 / 1.02],
        ]
    )
    np.testing.assert_allclose(model.topic_word, expected_phi, rtol=0, atol=1e-12)


def test_seededlda_frequency_seed_mass_is_exact() -> None:
    """The default seed_prior="frequency" builds each seed word's pseudocount as
    corpus-frequency(word) * weight * 100 (the seededlda::tfm construction),
    including repeated seed words with unequal corpus frequency. This is the exact
    seed-mass contract, checked without a reference toolchain via
    ``seed_prior_matrix``."""
    # 'tax' occurs 4x, 'budget' 2x, 'war' 3x across the corpus.
    docs = [
        ["tax", "tax", "budget"],
        ["tax", "war", "budget"],
        ["tax", "war", "war"],
        ["market", "market"],
    ]
    model = topica.SeededLDA(
        {"econ": ["tax", "budget"], "conflict": ["war"]},
        weight=0.02,
        seed_prior="frequency",
        seed=1,
    )
    model.fit(docs, iters=0)

    freq = Counter(w for d in docs for w in d)
    vi = {w: i for i, w in enumerate(model.vocabulary)}
    ti = {n: i for i, n in enumerate(model.topic_names)}
    m = np.asarray(model.seed_prior_matrix)

    assert m.shape == (2, len(model.vocabulary))
    for topic, words in {"econ": ["tax", "budget"], "conflict": ["war"]}.items():
        for w in words:
            expected = freq[w] * 0.02 * 100.0
            assert m[ti[topic], vi[w]] == expected, (topic, w, m[ti[topic], vi[w]], expected)
    # Non-seed cells (and the unrelated 'market') are zero.
    assert m[ti["econ"], vi["market"]] == 0.0
    assert m[ti["conflict"], vi["tax"]] == 0.0
    # 'war' (freq 3) carries more mass than 'budget' (freq 2): frequency scaling bites.
    assert m[ti["conflict"], vi["war"]] > m[ti["econ"], vi["budget"]]


# ---------------------------------------------------------------------------
# Dictionary seed matching (#456): quanteda-style fixed/glob/regex + case.
# All checked exactly via seed_prior_matrix, which reflects the seeds actually
# applied at fit — no reference toolchain required.
# ---------------------------------------------------------------------------

_DICT_DOCS = [
    ["tax", "taxes", "taxation", "revenue"],
    ["war", "conflict", "military"],
    ["tax", "revenue", "budget"],
    ["War", "Conflict", "defense"],
]


def _seeded_cells(model) -> set[tuple[str, str]]:
    """(topic_name, word) pairs carrying nonzero seed mass."""
    m = np.asarray(model.seed_prior_matrix)
    vocab = list(model.vocabulary)
    names = list(model.topic_names)
    return {
        (names[k], vocab[w])
        for k in range(m.shape[0])
        for w in range(m.shape[1])
        if m[k, w] != 0.0
    }


def test_fixed_matching_is_exact_and_default() -> None:
    """seed_match defaults to "fixed": a literal seed matches only that exact type,
    not morphological variants sharing a prefix."""
    model = topica.SeededLDA(
        {"econ": ["tax"], "mil": ["war"]}, seed_prior="uniform", seed=1
    )
    assert model.settings["seed_match"] == "fixed"
    assert model.settings["case_insensitive"] is False
    model.fit(_DICT_DOCS, iters=0)
    cells = _seeded_cells(model)
    assert ("econ", "tax") in cells
    # "taxes"/"taxation" are distinct types; a fixed "tax" seed must not reach them.
    assert ("econ", "taxes") not in cells
    assert ("econ", "taxation") not in cells
    # Case-sensitive by default: "War" is a different type from the "war" seed.
    assert ("mil", "War") not in cells


def test_glob_expands_to_matching_vocabulary() -> None:
    """seed_match="glob": "tax*" seeds every vocabulary type with that prefix."""
    model = topica.SeededLDA(
        {"econ": ["tax*"], "mil": ["war", "conflict"]},
        seed_prior="uniform",
        seed_match="glob",
        seed=1,
    )
    model.fit(_DICT_DOCS, iters=0)
    cells = _seeded_cells(model)
    assert ("econ", "tax") in cells
    assert ("econ", "taxes") in cells
    assert ("econ", "taxation") in cells
    # The glob is anchored, so "revenue" (no "tax" prefix) is not swept in.
    assert ("econ", "revenue") not in cells


def test_glob_case_insensitive_folds_case() -> None:
    """case_insensitive=True folds case, so a lowercase glob also seeds the
    capitalized surface forms (quanteda's dictionary default)."""
    model = topica.SeededLDA(
        {"econ": ["tax*"], "mil": ["war", "conflict"]},
        seed_prior="uniform",
        seed_match="glob",
        case_insensitive=True,
        seed=1,
    )
    model.fit(_DICT_DOCS, iters=0)
    cells = _seeded_cells(model)
    # Both "war"/"War" and "conflict"/"Conflict" are present in the vocab.
    assert ("mil", "war") in cells
    assert ("mil", "War") in cells
    assert ("mil", "conflict") in cells
    assert ("mil", "Conflict") in cells


def test_regex_matches_unanchored() -> None:
    """seed_match="regex" matches anywhere in the token (quanteda stri_detect_regex)."""
    model = topica.SeededLDA(
        {"econ": ["^tax"], "mil": ["war|conflict"]},
        seed_prior="uniform",
        seed_match="regex",
        seed=1,
    )
    model.fit(_DICT_DOCS, iters=0)
    cells = _seeded_cells(model)
    # "^tax" anchors the start: tax, taxes, taxation.
    assert {("econ", "tax"), ("econ", "taxes"), ("econ", "taxation")} <= cells
    # Alternation, matched unanchored.
    assert ("mil", "war") in cells
    assert ("mil", "conflict") in cells


def test_invalid_seed_match_rejected_at_construction() -> None:
    with np.testing.assert_raises(ValueError):
        topica.SeededLDA({"a": ["x"], "b": ["y"]}, seed_match="bogus")


def test_invalid_regex_pattern_rejected_at_fit() -> None:
    model = topica.SeededLDA(
        {"a": ["["], "b": ["y"]}, seed_match="regex", seed=1
    )
    with np.testing.assert_raises(ValueError):
        model.fit(_DICT_DOCS, iters=5)


def test_seed_match_survives_save_load(tmp_path) -> None:
    model = topica.SeededLDA(
        {"econ": ["tax*"], "mil": ["war"]},
        seed_prior="uniform",
        seed_match="glob",
        case_insensitive=True,
        seed=1,
    )
    model.fit(_DICT_DOCS, iters=0)
    before = _seeded_cells(model)
    path = str(tmp_path / "seeded_glob.topica")
    model.save(path)
    reloaded = topica.SeededLDA.load(path)
    assert reloaded.settings["seed_match"] == "glob"
    assert reloaded.settings["case_insensitive"] is True
    assert _seeded_cells(reloaded) == before


def _manual_gsdmm_counts(docs: list[list[str]], clusters: np.ndarray, vocab: list[str]):
    word_index = {w: i for i, w in enumerate(vocab)}
    k = int(clusters.max()) + 1
    m = np.zeros(k, dtype=float)
    n = np.zeros(k, dtype=float)
    nw = np.zeros((k, len(vocab)), dtype=float)
    encoded_docs: list[list[int]] = []
    for doc, cluster in zip(docs, clusters):
        kk = int(cluster)
        ids = [word_index[w] for w in doc]
        encoded_docs.append(ids)
        m[kk] += 1
        n[kk] += len(ids)
        for wid in ids:
            nw[kk, wid] += 1
    return encoded_docs, m, n, nw


def _manual_gsdmm_doc_topic(
    encoded_docs: list[list[int]],
    m: np.ndarray,
    n: np.ndarray,
    nw: np.ndarray,
    *,
    alpha: float,
    beta: float,
) -> np.ndarray:
    k, v = nw.shape
    out = np.zeros((len(encoded_docs), k), dtype=float)
    vbeta = v * beta
    for d, ids in enumerate(encoded_docs):
        counts = Counter(ids)
        logp = np.zeros(k, dtype=float)
        for kk in range(k):
            lp = np.log(m[kk] + alpha)
            for wid, count in counts.items():
                base = nw[kk, wid] + beta
                for j in range(count):
                    lp += np.log(base + j)
            for i in range(len(ids)):
                lp -= np.log(n[kk] + vbeta + i)
            logp[kk] = lp
        probs = np.exp(logp - logp.max())
        out[d] = probs / probs.sum()
    return out


def test_gsdmm_public_outputs_follow_movie_group_process_formulas() -> None:
    docs = [["cat", "cat"], ["dog", "dog"], ["cat", "dog"]]
    alpha = 0.1
    beta = 0.1
    model = topica.GSDMM(num_topics=3, alpha=alpha, beta=beta, seed=2)
    model.fit(docs, iters=0)

    clusters = np.asarray(model.doc_cluster)
    encoded_docs, m, n, nw = _manual_gsdmm_counts(docs, clusters, model.vocabulary)

    expected_phi = (nw + beta) / (n[:, None] + len(model.vocabulary) * beta)
    expected_theta = _manual_gsdmm_doc_topic(
        encoded_docs,
        m,
        n,
        nw,
        alpha=alpha,
        beta=beta,
    )

    np.testing.assert_allclose(model.topic_word, expected_phi, rtol=0, atol=1e-12)
    np.testing.assert_allclose(model.doc_topic, expected_theta, rtol=0, atol=1e-12)


def test_gsdmm_trace_records_effective_cluster_count_and_formula_likelihood() -> None:
    docs = [["cat", "cat"], ["dog", "dog"], ["cat", "dog"]]
    model = topica.GSDMM(num_topics=3, alpha=0.1, beta=0.1, seed=2)
    model.fit(docs, iters=1, progress_interval=1)

    clusters = np.asarray(model.doc_cluster)
    encoded_docs, _, n, nw = _manual_gsdmm_counts(docs, clusters, model.vocabulary)
    expected_ll = 0.0
    total_tokens = 0
    for ids, cluster in zip(encoded_docs, clusters):
        kk = int(cluster)
        denom = n[kk] + len(model.vocabulary) * 0.1
        for wid in ids:
            expected_ll += np.log((nw[kk, wid] + 0.1) / denom)
            total_tokens += 1
    expected_ll /= total_tokens

    assert model.cluster_count_history == [(1, model.num_topics)]
    assert len(model.log_likelihood_history) == 1
    iteration, observed_ll = model.log_likelihood_history[0]
    assert iteration == 1
    assert observed_ll == expected_ll

