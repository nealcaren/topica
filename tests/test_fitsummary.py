"""Tests for the per-model fit-statistics summary (issue #806).

Covers the three tiers of ``LDA.summary`` — the free repr, the ``texts=`` quality
tier, and the ``heldout=`` held-out fit tier — plus the held-out guard from the
sample-user audit (issue #809): a bare corpus must not be reported as held-out.
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
    assert hasattr(fitted, "summary")
    assert isinstance(fitted.summary(), FitSummary)


def test_unfitted_repr():
    assert repr(LDA(num_topics=3)) == "LDA (unfitted)"


def test_fitted_repr_is_the_fit_block(fitted):
    text = repr(fitted)
    assert text.startswith("LDA")
    assert "log_likelihood (in-sample)" in text
    assert "topics" in text and "documents" in text


# --- tier boundaries --------------------------------------------------------


def test_free_tier_has_no_corpus_metrics(fitted):
    """Without texts= or heldout=, the corpus-dependent numbers stay n/a."""
    s = fitted.summary()
    assert s.fitted is True
    assert s.coherence is None and s.exclusivity is None
    assert s.perplexity is None and s.heldout_loglik is None
    # cheap corpus-free health is still populated
    assert s.diversity is not None
    assert s.topic_significance is not None
    assert s.weak_topics is not None


def test_texts_tier_adds_quality_but_not_perplexity(fitted, toy_docs):
    """The key regression: coherence/exclusivity come from texts=, but perplexity
    must NOT be computed from the (training) texts — that was the mislabel bug."""
    s = fitted.summary(texts=toy_docs)
    assert s.coherence is not None
    assert s.exclusivity is not None
    assert s.perplexity is None
    assert s.heldout_loglik is None


# --- held-out guard (issue #809, Tier 1) ------------------------------------


def test_heldout_training_corpus_raises(fitted, toy_corpus):
    """Passing the training corpus to heldout= must raise, not silently report a
    training-data perplexity under a held-out label."""
    with pytest.raises(ValueError, match="verifiably unseen"):
        fitted.summary(heldout=toy_corpus)


def test_heldout_raw_docs_without_flag_raises(fitted, unseen_docs):
    """Even genuinely external docs raise without the explicit affirmation, because
    the summary cannot verify unseen-ness from here."""
    with pytest.raises(ValueError, match="assume_unseen"):
        fitted.summary(heldout=unseen_docs)


def test_assume_unseen_yields_perplexity(fitted, unseen_docs):
    s = fitted.summary(heldout=unseen_docs, assume_unseen=True)
    assert s.perplexity is not None
    assert np.isfinite(s.perplexity)
    assert s.heldout_loglik is None


def test_make_heldout_split_yields_loglik(toy_docs):
    """A make_heldout split is safe by construction (no flag needed) and routes to
    held-out log-likelihood, leaving perplexity n/a."""
    ho = make_heldout(toy_docs, seed=13)
    m = LDA(num_topics=4, seed=13)
    m.fit(Corpus.from_documents(ho.documents), iters=150)
    s = m.summary(heldout=ho)
    assert s.heldout_loglik is not None
    assert np.isfinite(s.heldout_loglik)
    assert s.perplexity is None


# --- labeling (Tier 4 wins) -------------------------------------------------


def test_gibbs_converged_label(fitted):
    """LDA is a Gibbs sampler: a bare 'no' would misread as failure, so an
    unconverged Gibbs fit renders 'n/a (fixed sweeps)'."""
    assert fitted.summary().sampler == "gibbs"
    rows = dict(fitted.summary()._fit_rows())
    assert rows["converged"] == "n/a (fixed sweeps)"


def test_heldout_rows_carry_units(fitted):
    labels = [label for label, _ in fitted.summary()._fit_rows()]
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
    s = fitted.summary()
    scores = np.asarray(
        topica.evaluate.topic_significance(fitted, kind="vacuous", per_topic=True),
        dtype=float,
    )
    expected = int((scores < 0.1 * scores.max()).sum())
    assert s.weak_topics == expected


# --- export -----------------------------------------------------------------


def test_to_markdown_is_paper_ready(fitted):
    md = fitted.summary().to_markdown()
    assert "| statistic | value |" in md
    assert "LDA fit summary" in md
