"""GSDMM UX follow-ups (#781): opt-in fit() progress + topic_crosstab helper."""

import warnings

import numpy as np
import pytest

import topica


def _corpus_with_party():
    docs = [["tax", "vote", "law"], ["health", "care", "clinic"]] * 60
    party = ["D", "R"] * 60
    import pandas as pd
    df = pd.DataFrame({"text": [" ".join(d) for d in docs], "party": party})
    return topica.from_dataframe(df, text_col="text", metadata_cols=["party"]), docs, party


# --- fit(verbose=) progress ------------------------------------------------

def test_verbose_prints_progress_to_stderr(capfd):
    docs = [["cat", "dog", "pet"], ["star", "moon", "sky"]] * 40
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        topica.GSDMM(6, seed=13).fit(docs, iters=10, progress_interval=2, verbose=True)
    err = capfd.readouterr().err
    assert "[GSDMM] sweep" in err
    # one line per recorded sweep (every 2 of 10, plus the final) — at least a few.
    assert err.count("[GSDMM] sweep") >= 3


def test_quiet_by_default(capfd):
    docs = [["cat", "dog", "pet"], ["star", "moon", "sky"]] * 40
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        topica.GSDMM(6, seed=13).fit(docs, iters=10, progress_interval=2)
    assert "[GSDMM] sweep" not in capfd.readouterr().err


def test_gsdmm_progress_callback_and_topica_progress(capfd):
    # #785: GSDMM.fit(progress=) calls (iteration, total, {"clusters", "ll"}) and
    # topica.progress() renders the live bar + cluster-count sparkline.
    docs = [["cat", "dog", "pet"], ["star", "moon", "sky"]] * 60
    calls = []
    topica.GSDMM(8, seed=13).fit(
        docs, iters=20, progress_interval=4,
        progress=lambda it, total, info: calls.append((it, total, dict(info))),
    )
    assert calls and all(t == 20 for _, t, _ in calls)
    assert all({"clusters", "ll"} <= set(info) for _, _, info in calls)
    # the shipped renderer draws a bar with the clusters sparkline
    topica.GSDMM(8, seed=13).fit(docs, iters=20, progress_interval=4,
                                 progress=topica.progress(metric="clusters", label="GSDMM"))
    err = capfd.readouterr().err
    assert "GSDMM |" in err and "clusters " in err and "100%" in err


def test_gsdmm_progress_does_not_change_the_fit():
    docs = [["cat", "dog", "pet"], ["star", "moon", "sky"], ["cat", "pet", "dog"]] * 20
    a = topica.GSDMM(8, seed=13).fit(docs, iters=25)
    b = topica.GSDMM(8, seed=13).fit(docs, iters=25,
                                     progress=lambda *xs: None, progress_interval=3)
    assert np.array_equal(np.asarray(a.doc_cluster), np.asarray(b.doc_cluster))


def test_verbose_does_not_change_the_fit():
    docs = [["cat", "dog", "pet"], ["star", "moon", "sky"], ["cat", "pet", "dog"]] * 30
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = topica.GSDMM(8, seed=13).fit(docs, iters=20)
        b = topica.GSDMM(8, seed=13).fit(docs, iters=20, verbose=True)
    assert np.array_equal(np.asarray(a.doc_cluster), np.asarray(b.doc_cluster))


# --- topic_crosstab --------------------------------------------------------

def test_topic_crosstab_hard_clustering_with_corpus():
    corpus, docs, party = _corpus_with_party()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = topica.GSDMM(6, seed=13).fit(corpus, iters=30)
    ct = topica.inspect.topic_crosstab(m, corpus, "party")
    # rows = discovered clusters, columns = the two parties, cells = doc counts.
    assert set(ct.columns) == {"D", "R"}
    assert ct.values.sum() == len(docs)
    assert ct.index.name == "cluster"


def test_topic_crosstab_mixture_model_uses_argmax():
    corpus, docs, party = _corpus_with_party()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = topica.LDA(2, seed=13).fit(corpus, iters=100)
    ct = topica.inspect.topic_crosstab(m, corpus, "party", normalize="index")
    assert ct.index.name == "topic"  # LDA has no doc_cluster -> dominant topic
    # row-normalized: each topic's split across parties sums to 1.
    np.testing.assert_allclose(ct.sum(axis=1).values, 1.0, atol=1e-9)


def test_topic_crosstab_accepts_bare_array_and_checks_length():
    corpus, docs, party = _corpus_with_party()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = topica.GSDMM(6, seed=13).fit(corpus, iters=30)
    ct = topica.inspect.topic_crosstab(m, np.array(party))
    assert ct.values.sum() == len(docs)
    with pytest.raises(ValueError, match="align the covariate"):
        topica.inspect.topic_crosstab(m, np.array(party[:-5]))


def test_topic_crosstab_keeps_missing_values_visible():
    # #784 review: NaN/None covariate values must NOT be silently dropped — the
    # table has to account for every document.
    corpus, docs, party = _corpus_with_party()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = topica.GSDMM(6, seed=13).fit(corpus, iters=30)
    cov = np.array(party, dtype=object)
    cov[:20] = None  # 20 missing
    ct = topica.inspect.topic_crosstab(m, cov)
    assert ct.values.sum() == len(docs)  # nothing dropped
    assert any(isinstance(c, float) and np.isnan(c) for c in ct.columns)  # NaN column present


def test_topic_crosstab_gives_every_topic_a_row():
    # #784 review: a mixture topic that is never the argmax winner still gets a
    # (zero) row, so the index is dense and lines up with topic_table.
    corpus, docs, party = _corpus_with_party()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lda = topica.LDA(6, seed=13).fit(corpus, iters=60)  # more topics than the 2 real ones
    ct = topica.inspect.topic_crosstab(lda, corpus, "party")
    assert list(ct.index) == list(range(6))


def test_topic_crosstab_directive_errors():
    # #784 review: wrong column and a 2-D covariate raise clean ValueErrors.
    corpus, docs, party = _corpus_with_party()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = topica.GSDMM(6, seed=13).fit(corpus, iters=30)
    with pytest.raises(ValueError, match="not found; available"):
        topica.inspect.topic_crosstab(m, corpus, "partee")
    with pytest.raises(ValueError, match="1-D"):
        topica.inspect.topic_crosstab(m, np.array(party).reshape(-1, 1))
