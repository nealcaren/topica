"""Tests for the per-model fit-statistics summary (issue #806).

Covers the three tiers of ``fit_summary`` (the printed block, the ``texts=`` quality
tier, and the ``heldout=`` held-out fit tier), the held-out guard from the sample-user
audit (issue #809: a bare corpus must not be reported as held-out), and the rollout
to every registered model.
"""

import numpy as np
import pytest

import topica
from topica import LDA, Corpus, make_heldout
from topica.fitsummary import FitSummary


@pytest.fixture(scope="module")
def fitted(toy_corpus):
    """A small fitted LDA on the toy corpus; seed 13 (topica's default)."""
    m = LDA(num_topics=4, seed=13)
    m.fit(toy_corpus, iters=150)
    return m


@pytest.fixture(scope="module")
def unseen_docs(toy_corpus):
    """Documents drawn from the training vocabulary but not the training set, to
    stand in for a genuinely external held-out corpus."""
    vocab = list(toy_corpus.vocabulary)
    rng = np.random.default_rng(99)
    return [list(rng.choice(vocab, size=8)) for _ in range(12)]


# --- binding / wiring -------------------------------------------------------


def test_lda_is_wired_at_import(fitted):
    """The feature is bound onto LDA at package import, not only via the helper."""
    assert hasattr(fitted, "fit_summary")
    assert isinstance(fitted.fit_summary(), FitSummary)


def test_unfitted_print_is_the_compact_repr():
    m = LDA(num_topics=3)
    assert str(m) == repr(m)
    assert "fitted=false" in repr(m)


def test_print_is_the_fit_block_and_repr_stays_one_line(fitted):
    text = str(fitted)
    assert text.startswith("LDA")
    assert "log-likelihood (in-sample)" in text
    assert "topics" in text and "documents" in text
    # repr is the native constructor form, so lists of models stay readable
    assert "\n" not in repr(fitted) and "fitted=true" in repr(fitted)


# --- tier boundaries --------------------------------------------------------


def test_free_tier_has_no_corpus_metrics(fitted):
    """Without texts= or heldout=, the corpus-dependent numbers stay n/a."""
    s = fitted.fit_summary()
    assert s.fitted is True
    assert s.coherence is None and s.exclusivity is None
    assert s.perplexity is None and s.heldout_loglik is None
    # cheap corpus-free health is still populated
    assert s.diversity is not None
    assert s.topic_significance is not None
    assert s.weak_topics is not None


def test_texts_tier_adds_quality_but_not_perplexity(fitted, toy_docs):
    """The key regression: coherence/exclusivity come from texts=, but perplexity
    must NOT be computed from the (training) texts; that was the mislabel bug."""
    s = fitted.fit_summary(texts=toy_docs)
    assert s.coherence is not None
    assert s.exclusivity is not None
    assert s.perplexity is None
    assert s.heldout_loglik is None


# --- held-out guard (issue #809, Tier 1) ------------------------------------


def test_heldout_training_corpus_raises(fitted, toy_corpus):
    """Passing the training corpus to heldout= must raise, not silently report a
    training-data perplexity under a held-out label."""
    with pytest.raises(ValueError, match="verifiably unseen"):
        fitted.fit_summary(heldout=toy_corpus)


def test_heldout_raw_docs_without_flag_raises(fitted, unseen_docs):
    """Even genuinely external docs raise without the explicit affirmation, because
    the summary cannot verify unseen-ness from here."""
    with pytest.raises(ValueError, match="assume_unseen"):
        fitted.fit_summary(heldout=unseen_docs)


def test_assume_unseen_yields_perplexity(fitted, unseen_docs):
    s = fitted.fit_summary(heldout=unseen_docs, assume_unseen=True)
    assert s.perplexity is not None
    assert np.isfinite(s.perplexity)
    assert s.heldout_loglik is None


def test_make_heldout_split_yields_loglik(toy_docs):
    """A make_heldout split is safe by construction (no flag needed) and routes to
    held-out log-likelihood, leaving perplexity n/a."""
    ho = make_heldout(toy_docs, seed=13)
    m = LDA(num_topics=4, seed=13)
    m.fit(Corpus.from_documents(ho.documents), iters=150)
    s = m.fit_summary(heldout=ho)
    assert s.heldout_loglik is not None
    assert np.isfinite(s.heldout_loglik)
    assert s.perplexity is None


# --- labeling (Tier 4 wins) -------------------------------------------------


def test_gibbs_converged_label(fitted):
    """LDA is a Gibbs sampler: a bare 'no' would misread as failure, so an
    unconverged Gibbs fit renders 'n/a (fixed sweeps)'."""
    assert fitted.fit_summary().sampler == "gibbs"
    rows = dict(fitted.fit_summary()._fit_rows())
    assert rows["converged"] == "n/a (fixed sweeps)"


def test_heldout_rows_carry_units(fitted):
    labels = [label for label, _ in fitted.fit_summary()._fit_rows()]
    assert "perplexity (held-out, document-completion)" in labels
    assert "heldout_loglik (mean per-doc)" in labels


def test_health_fields_are_documented():
    """help(FitSummary) must define the health metrics (they were undocumented)."""
    doc = FitSummary.__doc__ or ""
    for field in ("effective_topics", "diversity", "topic_redundancy",
                  "topic_significance", "weak_topics"):
        assert field in doc


def test_weak_topics_matches_significance(fitted):
    """weak_topics is sourced from evaluate.topic_significance, not an inline KL."""
    s = fitted.fit_summary()
    scores = np.asarray(
        topica.evaluate.topic_significance(fitted, kind="vacuous", per_topic=True),
        dtype=float,
    )
    expected = int((scores < 0.1 * scores.max()).sum())
    assert s.weak_topics == expected


# --- export -----------------------------------------------------------------


def test_to_markdown_is_paper_ready(fitted):
    md = fitted.fit_summary().to_markdown()
    assert "| statistic | value |" in md
    assert "LDA fit summary" in md


# --- roster-wide rollout -------------------------------------------------------


def test_every_registered_model_has_summary():
    for name in topica.REGISTRY:
        cls = getattr(topica, name, None)
        if cls is None:
            continue
        assert callable(getattr(cls, "fit_summary", None)), name
        assert callable(getattr(cls, "_repr_html_", None)), name


@pytest.mark.parametrize("build, label", [
    (lambda: topica.CTM(3), "variational bound"),
    (lambda: topica.NMF(3), "reconstruction error"),
    (lambda: topica.HDP(), "log-likelihood per token"),
    (lambda: topica.CorEx(3), "total correlation"),
    (lambda: topica.LSA(3), "objective"),
])
def test_summary_across_families(toy_corpus, build, label):
    """Each family reports its own objective under its own label; a model with no
    iterative trace (LSA) shows n/a instead of raising."""
    m = build()
    fitted_model = m.fit(toy_corpus) or m
    s = fitted_model.fit_summary()
    assert s.fitted
    assert s.objective_label == label
    assert s.num_docs == len(toy_corpus.documents())
    if label == "objective":
        assert s.objective is None
    else:
        assert np.isfinite(s.objective)
    assert f"{label} (in-sample)" in str(fitted_model)
    assert "<table" in fitted_model._repr_html_()


def test_models_without_fit_history_or_converged_still_summarize(toy_corpus):
    """BTM exposes neither fit_history nor converged; the summary shows n/a."""
    m = topica.BTM(3)
    fitted_model = m.fit(toy_corpus) or m
    s = fitted_model.fit_summary()
    assert s.fitted and s.converged is None and s.objective is None
    assert s.num_topics == 3


# every conformance-registry model, fitted with its own recipe, prints and summarizes
from test_convergence_interface import _PARAMS, _fit_model  # noqa: E402


@pytest.mark.parametrize("name,factory,family", _PARAMS, ids=[p[0] for p in _PARAMS])
def test_every_fitted_model_prints_its_summary(name, factory, family):
    model = _fit_model(name, factory)
    text = str(model)
    assert text.startswith(name)
    assert "\n" in text
    assert "<table" in model._repr_html_()
    assert model.fit_summary().fitted


def test_bare_float_trace_summarizes():
    """InfoCTM's fit_history is a bare per-epoch list, not (iteration, value) pairs;
    printing it used to raise TypeError."""
    from test_infoctm import _fit

    model, _ = _fit()
    s = model.fit_summary()
    assert s.objective_label == "ELBO per document"
    assert s.iterations == len(model.fit_history)
    assert np.isfinite(s.objective)
    assert "ELBO per document (in-sample)" in str(model)


def test_scaling_model_omits_the_texts_hint_and_rejects_texts():
    """Wordfish has no topic-word matrix: no coherence hint, and texts= says why."""
    from test_wordfish import _planted

    docs, group, _, _ = _planted(seed=1)
    m = topica.Wordfish(seed=1)
    m.fit(docs, group=group, anchors={"a0": -1.0, "a39": 1.0}, iters=50)
    text = str(m)
    assert "fit_summary(texts=" not in text
    assert "log-likelihood (up to a constant)" in text
    with pytest.raises(ValueError, match="no flat topic-word matrix"):
        m.fit_summary(texts=docs)
