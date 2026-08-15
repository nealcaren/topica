"""#732: sample-user audit follow-ups — API-consistency and diagnostic ergonomics.

- bootstrap_stability accepts the library-wide num_topics= alias for k=
- coherence accepts metric= as an alias for coherence_type=
- from_dataframe raises a column-aware error when text_col= empties the corpus
- best_k's grid-edge warning cites the elbow/frontier K it can already compute
"""

import numpy as np
import pytest

import topica
from topica.validation import SearchKResult

DOCS = [["a", "b", "c", "a"], ["b", "c", "d"], ["a", "d", "e"], ["c", "e", "f"]] * 8


def test_bootstrap_stability_num_topics_alias():
    r = topica.bootstrap_stability(DOCS, num_topics=3, n_boot=2, seed=1)
    assert len(r["topic"]) == 3
    # k= still works and matches
    r2 = topica.bootstrap_stability(DOCS, k=3, n_boot=2, seed=1)
    assert np.array_equal(r["topic"], r2["topic"])


def test_bootstrap_stability_k_num_topics_conflict():
    with pytest.raises(ValueError, match="not both with different values"):
        topica.bootstrap_stability(DOCS, k=2, num_topics=3, n_boot=2)


def test_coherence_metric_alias():
    model = topica.LDA(num_topics=3, seed=13).fit(DOCS)
    by_alias = topica.coherence(model, DOCS, metric="u_mass")
    by_canon = topica.coherence(model, DOCS, coherence_type="u_mass")
    assert np.allclose(by_alias, by_canon)


def test_coherence_metric_coherence_type_conflict():
    model = topica.LDA(num_topics=3, seed=13).fit(DOCS)
    with pytest.raises(ValueError, match="not both with different values"):
        topica.coherence(model, DOCS, coherence_type="u_mass", metric="c_v")


def test_from_dataframe_wrong_text_col_names_the_column():
    pd = pytest.importorskip("pandas")
    # An "author"-like covariate: every value a distinct single token, so
    # min_doc_freq prunes the whole vocabulary and the corpus empties.
    authors = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
        "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
        "quebec", "romeo", "sierra", "tango",
    ]
    df = pd.DataFrame({"text": ["apple banana cherry", "date elder fig"] * 10,
                       "author": authors})
    with pytest.raises(ValueError) as exc:
        topica.from_dataframe(df, text_col="author", min_doc_freq=3)
    msg = str(exc.value)
    assert "text_col='author'" in msg
    assert "other columns are ['text']" in msg


def test_best_k_boundary_warning_cites_alternative_picks():
    # heldout_loglik improves monotonically (best -> largest K), but with
    # diminishing returns (an interior elbow) and an interior coherence/exclusivity
    # frontier. The boundary warning should name the concrete elbow/frontier K.
    ks = [10, 20, 30, 40]
    res = SearchKResult([
        {"k": 10, "heldout_loglik": -100.0, "coherence": -5.0, "exclusivity": 0.40},
        {"k": 20, "heldout_loglik": -50.0, "coherence": -4.0, "exclusivity": 0.60},
        {"k": 30, "heldout_loglik": -45.0, "coherence": -6.0, "exclusivity": 0.50},
        {"k": 40, "heldout_loglik": -44.0, "coherence": -8.0, "exclusivity": 0.45},
    ])
    with pytest.warns(UserWarning) as rec:
        pick = res.best_k("heldout_loglik")
    assert pick == 40  # the grid edge
    msg = " ".join(str(w.message) for w in rec)
    # at least one concrete alternative K is cited, not just a rule name
    assert "rule='elbow' gives K=" in msg or "metric='frontier' gives K=" in msg
    # and the cited K is an interior grid value, not the boundary pick
    assert any(f"K={k}" in msg for k in ks if k != 40)
