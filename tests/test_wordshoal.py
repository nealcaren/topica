"""Wordshoal: two-stage multi-domain ideal-point scaling (Lauderdale & Herzog 2016).

Like Wordfish, Wordshoal has no topic distribution, so it is exempt from the
registry-driven topic-health suite; its validity is checked here — it recovers
planted actor positions from a two-stage generative corpus, is deterministic,
orients by anchors, round-trips through save/load, and guards its edge cases.
"""
import warnings

import numpy as np
import pytest

import topica


def _planted(n_authors=36, n_domains=10, n_words=50, seed=0):
    """A two-stage Wordshoal corpus: actor positions theta_i, domain loadings
    beta_j / intercepts alpha_j, and per-domain Wordfish words whose per-document
    latent position is z = alpha_j + beta_j * theta_i. Every actor speaks in every
    domain (a dense, connected design)."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(-1.5, 1.5, n_authors)
    beta = np.where(np.arange(n_domains) % 2 == 0, 1.0, -1.0) * rng.uniform(
        0.7, 1.3, n_domains
    )
    alpha = rng.normal(0.0, 0.3, n_domains)
    docs, speakers, domains = [], [], []
    for j in range(n_domains):
        bword = rng.uniform(-1.2, 1.2, n_words)
        pword = np.log(rng.uniform(3.0, 9.0, n_words))
        for i in range(n_authors):
            z = alpha[j] + beta[j] * theta[i]
            counts = rng.poisson(np.exp(pword + bword * z))
            doc = []
            for w, c in enumerate(counts):
                doc.extend([f"w{w}"] * int(c))
            if len(doc) < 2:
                doc = ["w0", "w1"]
            rng.shuffle(doc)
            docs.append(doc)
            speakers.append(f"s{i:02d}")
            domains.append(f"d{j:02d}")
    return docs, speakers, domains, theta


def test_recovers_actor_positions():
    docs, speakers, domains, theta = _planted(seed=1)
    m = topica.Wordshoal(seed=13).fit(docs, speakers=speakers, domains=domains)

    assert m.num_authors == 36
    assert m.num_domains == 10
    assert m.num_components == 1
    pos = dict(zip(m.author_names, m.author_positions[:, 0]))
    recovered = np.array([pos[f"s{i:02d}"] for i in range(36)])
    r = abs(np.corrcoef(recovered, theta)[0, 1])
    assert r > 0.9, f"actor-position recovery r={r:.3f}"
    # default orientation: sorted-first speaker below sorted-last
    assert recovered[0] < recovered[-1]
    # standard errors finite and positive, aligned to author_names
    assert m.position_se.shape == (36,)
    assert np.all(np.isfinite(m.position_se)) and np.all(m.position_se > 0)
    # domain_scales is (M, 2) = [alpha, beta]
    assert m.domain_scales.shape == (10, 2)


def test_deterministic():
    docs, speakers, domains, _ = _planted(seed=2)
    a = topica.Wordshoal(seed=13).fit(docs, speakers=speakers, domains=domains)
    b = topica.Wordshoal(seed=13).fit(docs, speakers=speakers, domains=domains)
    assert np.array_equal(a.author_positions, b.author_positions)
    assert np.array_equal(a.domain_scales, b.domain_scales)
    assert np.array_equal(a.position_se, b.position_se)


def test_anchors_orient_sign():
    docs, speakers, domains, _ = _planted(seed=3)
    # Force the opposite orientation via anchors and confirm it takes.
    m = topica.Wordshoal(seed=13).fit(
        docs, speakers=speakers, domains=domains,
        anchors={"s00": 1.0, "s35": -1.0},
    )
    pos = dict(zip(m.author_names, m.author_positions[:, 0]))
    assert pos["s00"] > pos["s35"], "anchors did not orient the axis"


def test_word_scores():
    docs, speakers, domains, _ = _planted(seed=4)
    m = topica.Wordshoal(seed=13).fit(docs, speakers=speakers, domains=domains)
    ws = m.word_scores("d00", n=5)
    assert len(ws) == 5
    assert all(isinstance(w, str) and isinstance(b, float) for w, b in ws)
    # sorted by descending discrimination
    betas = [b for _, b in ws]
    assert betas == sorted(betas, reverse=True)
    with pytest.raises(ValueError):
        m.word_scores("no-such-domain")


def test_save_load_round_trip(tmp_path):
    docs, speakers, domains, _ = _planted(seed=5)
    m = topica.Wordshoal(seed=13).fit(docs, speakers=speakers, domains=domains)
    p = tmp_path / "ws.topica"
    m.save(str(p))
    loaded = topica.Wordshoal.load(str(p))
    assert np.array_equal(loaded.author_positions, m.author_positions)
    assert np.array_equal(loaded.domain_scales, m.domain_scales)
    assert loaded.author_names == m.author_names
    assert loaded.domain_names == m.domain_names
    assert loaded.settings == m.settings


def test_single_doc_domain_errors():
    docs, speakers, domains, _ = _planted(seed=6)
    # Make domain "d00" have exactly one document by relabeling all but one.
    domains = list(domains)
    seen = False
    for k, d in enumerate(domains):
        if d == "d00":
            if seen:
                domains[k] = "d01"
            seen = True
    with pytest.raises(ValueError, match="fewer than 2 documents"):
        topica.Wordshoal(seed=13).fit(docs, speakers=speakers, domains=domains)


def test_disconnected_graph_warns():
    # Two blocks: speakers 0-9 only in domains 0-1, speakers 10-19 only in 2-3.
    docs, speakers, domains = [], [], []
    rng = np.random.default_rng(0)
    for block, (sp_lo, dm) in enumerate([(0, [0, 1]), (10, [2, 3])]):
        for j in dm:
            for i in range(sp_lo, sp_lo + 10):
                counts = rng.poisson(3.0, 8)
                doc = []
                for w, c in enumerate(counts):
                    doc.extend([f"w{w}"] * int(c))
                if len(doc) < 2:
                    doc = ["w0", "w1"]
                docs.append(doc)
                speakers.append(f"s{i:02d}")
                domains.append(f"d{j:02d}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = topica.Wordshoal(seed=13).fit(docs, speakers=speakers, domains=domains)
    assert m.num_components == 2
    assert any("disconnected" in str(w.message) for w in caught)
    # author_components segregates the two blocs (aligned to author_names/positions)
    comp = dict(zip(m.author_names, m.author_components))
    assert comp["s00"] != comp["s10"], "different blocs share a component label"
    assert comp["s00"] == comp["s09"], "same bloc split across components"
    assert len(set(m.author_components)) == 2


def test_connected_design_single_component():
    docs, speakers, domains, _ = _planted(seed=8)
    m = topica.Wordshoal(seed=13).fit(docs, speakers=speakers, domains=domains)
    assert m.num_components == 1
    assert set(m.author_components) == {0}
    assert m.author_components.shape == (m.num_authors,)


def test_non_string_labels_are_coerced():
    # A researcher naturally passes integer id/day columns; they must be accepted,
    # not rejected with an opaque TypeError (Gate-B sample-user T4-a).
    docs, speakers, domains, _ = _planted(seed=9)
    int_speakers = [int(s[1:]) for s in speakers]  # ints
    int_domains = [int(d[1:]) for d in domains]
    m = topica.Wordshoal(seed=13).fit(docs, speakers=int_speakers, domains=int_domains)
    assert m.num_authors == 36
    # labels are the str() of the ints, sorted lexicographically
    assert all(isinstance(a, str) for a in m.author_names)


def test_settings_keys():
    m = topica.Wordshoal(theta_prior_sd=1.0, loading_prior_sd=0.5)
    assert set(m.settings) == {
        "theta_prior_sd", "loading_prior_sd", "intercept_prior_sd",
        "tau_prior", "min_count", "convergence_tol", "seed",
    }
    assert m.settings["seed"] == 13


def test_length_mismatch_errors():
    docs, speakers, domains, _ = _planted(seed=7)
    with pytest.raises(ValueError, match="speakers must have length"):
        topica.Wordshoal(seed=13).fit(docs, speakers=speakers[:-1], domains=domains)
    with pytest.raises(ValueError, match="domains must have length"):
        topica.Wordshoal(seed=13).fit(docs, speakers=speakers, domains=domains[:-1])
