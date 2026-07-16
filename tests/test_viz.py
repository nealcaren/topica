"""The visualization toolkit: capability descriptor, panel contract, honest gating."""

import warnings

import numpy as np
import pytest

import topica

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
import topica.viz as viz  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Close matplotlib figures after each test so the suite does not leak them."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.fixture(scope="module")
def covariate_lda():
    rng = np.random.default_rng(0)
    n = 240
    x = rng.normal(size=n)
    docs = []
    for i in range(n):
        p0 = 1.0 / (1.0 + np.exp(-2.0 * x[i]))
        block = ["a0", "a1", "a2", "a3", "a4"] if rng.random() < p0 else \
                ["b0", "b1", "b2", "b3", "b4"]
        docs.append(list(rng.choice(block, size=12)))
    corpus = topica.Corpus.from_documents(docs)
    m = topica.models.LDA(2, seed=1)
    m.fit(corpus, iters=300)
    return m, corpus, x, [" ".join(d) for d in docs]


@pytest.fixture(scope="module")
def bertopic():
    rng = np.random.default_rng(1)
    docs = [["x0", "x1", "x2"]] * 40 + [["y0", "y1", "y2"]] * 40
    emb = np.vstack([rng.normal([0, 0], 0.4, (40, 2)), rng.normal([6, 0], 0.4, (40, 2))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = topica.models.BERTopic(min_cluster_size=15, seed=1)
        m.fit(docs, emb)
    return m


# --- capability descriptor -------------------------------------------------

def test_capabilities_generative(covariate_lda):
    m = covariate_lda[0]
    cap = viz.capabilities(m)
    assert cap.prob_simplex_words and cap.soft_theta
    assert cap.theta_posterior == "dirichlet"
    assert cap.word_weight_label == "P(w | topic)"
    assert "frex" in cap.word_modes


def test_capabilities_cluster(bertopic):
    cap = viz.capabilities(bertopic)
    assert not cap.prob_simplex_words and not cap.soft_theta
    assert cap.theta_posterior == "none"
    assert cap.word_weight_label == "c-TF-IDF weight"
    assert cap.word_modes == ["prob"]


# --- the .to_frame() / .to_png() contract ----------------------------------

def test_every_panel_frames_and_renders(covariate_lda):
    m, corpus, x, texts = covariate_lda
    panels = [
        viz.coherence_frontier(m, texts),
        viz.search_k(topica.search_k(
            [t.split() for t in texts], [2, 3], iters=80, num_samples=1)),
        viz.effect_plot(m, corpus, X=x[:, None], feature_names=["x"], nsims=15),
        viz.term_barchart(m, topic=0, mode="frex", n=6),
        viz.topic_similarity(m),
    ]
    for p in panels:
        df = p.to_frame()
        assert len(df) >= 1
        fig = p.to_png()  # returns a Figure, doesn't raise
        assert fig is not None


# --- honest gating ---------------------------------------------------------

def test_frex_mode_refused_on_ctfidf(bertopic):
    with pytest.raises(ValueError, match="not valid for BERTopic"):
        viz.term_barchart(bertopic, topic=0, mode="frex")
    # but prob mode works and is labeled c-TF-IDF
    tb = viz.term_barchart(bertopic, topic=0, mode="prob", n=4)
    assert tb.cap.word_weight_label == "c-TF-IDF weight"


def test_effect_plot_refuses_ci_without_posterior(bertopic):
    rng = np.random.default_rng(0)
    ep = viz.effect_plot(bertopic, X=rng.normal(size=(80, 1)), feature_names=["z"])
    assert ep.has_ci is False
    assert "no theta posterior" in ep.note
    ep.to_png()  # still draws point estimates


def test_topic_similarity_metric_switches(covariate_lda, bertopic):
    assert "Jensen-Shannon" in viz.topic_similarity(covariate_lda[0]).metric
    assert viz.topic_similarity(bertopic).metric == "cosine distance"


# --- dashboard -------------------------------------------------------------

def test_dashboard_picks_panels_and_exports(covariate_lda):
    m, corpus, x, texts = covariate_lda
    d = viz.dashboard(m, texts, corpus=corpus, X=x[:, None])
    # similarity + terms + health always; frontier (texts), effect (design),
    # correlation (soft theta) added by introspection.
    assert {"similarity", "terms", "health", "frontier", "effect", "correlation"} <= set(d.panels)
    frames = d.to_frame()
    assert set(frames) == set(d.panels)
    assert d.to_png() is not None


def test_term_overlay_only_in_prob_mode(covariate_lda):
    m = covariate_lda[0]
    prob = viz.term_barchart(m, topic=0, mode="prob", n=4).to_frame()
    frex = viz.term_barchart(m, topic=0, mode="frex", n=4).to_frame()
    assert "corpus_weight" in prob.columns          # the overlay is meaningful here
    assert "corpus_weight" not in frex.columns        # ...and dropped where it is not


def test_topic_similarity_is_a_valid_metric(covariate_lda):
    ts = viz.topic_similarity(covariate_lda[0])
    d = ts._dist
    assert np.allclose(d, d.T)                         # symmetric
    assert np.allclose(np.diag(d), 0.0)                # zero diagonal
    assert (d >= -1e-9).all()


def test_dashboard_png_is_vector_for_pdf(covariate_lda, tmp_path):
    m, corpus, x, texts = covariate_lda
    d = viz.dashboard(m, texts, corpus=corpus, X=x[:, None])
    p = tmp_path / "report.pdf"
    d.to_png(str(p))
    # a real vector PDF embeds fonts/text operators, not one giant image
    assert b"/Font" in p.read_bytes()


# --- deferred panels: health / groups / temporal / correlation -------------

@pytest.fixture(scope="module")
def lda_k4():
    rng = np.random.default_rng(3)
    n = 200
    blocks = [[f"t{g}w{j}" for j in range(4)] for g in range(4)]
    docs, group, year = [], [], []
    for i in range(n):
        g = int(rng.integers(0, 4))
        docs.append(list(rng.choice(blocks[g], size=10)))
        group.append(["north", "south"][g % 2])
        year.append(2000 + int(rng.integers(0, 3)))
    m = topica.models.LDA(4, seed=1)
    m.fit(docs, iters=300)
    return m, group, year, docs


def test_topic_health_flags_and_frame(lda_k4):
    m = lda_k4[0]
    h = viz.topic_health(m, min_mass_frac=0.05, dup_threshold=0.95)
    df = h.to_frame()
    assert {"mass_frac", "nearest_topic", "nearest_cosine", "flag"} <= set(df.columns)
    assert df["flag"].isin({"ok", "dead", "duplicate"}).all()
    assert h.to_png() is not None
    # an artificially low threshold flags nothing dead; a high one flags some
    assert (viz.topic_health(m, min_mass_frac=0.0).to_frame()["flag"] == "dead").sum() == 0


def test_prevalence_heatmap(lda_k4):
    m, group, _, _docs = lda_k4
    p = viz.prevalence_heatmap(m, group)
    df = p.to_frame()
    assert set(df["group"]) == {"north", "south"}
    assert {"prevalence", "ci_low", "ci_high"} <= set(df.columns)
    assert p.matrix().shape == (2, 4)
    assert p.to_png() is not None


def test_topics_over_time_small_multiples(lda_k4):
    m, _, year, docs = lda_k4
    t = viz.topics_over_time(m, year)
    df = t.to_frame()
    assert set(df["time"]) == set(year)
    assert t.to_png() is not None
    assert t.has_ci is False
    # with a corpus + nsims, CI ribbons appear in the frame
    tc = viz.topics_over_time(m, year, corpus=docs, nsims=10)
    assert tc.has_ci and "ci_low" in tc.to_frame().columns


def test_topic_correlation_methods_and_honesty(lda_k4):
    m = lda_k4[0]
    clr = viz.topic_correlation(m, method="clr").to_frame()
    assert clr.shape == (4, 4)
    assert np.allclose(np.diag(clr.values), 1.0)
    # partial correlation is a different matrix, still diagonal-1
    part = viz.topic_correlation(m, method="partial").to_frame()
    assert np.allclose(np.diag(part.values), 1.0)
    assert not np.allclose(clr.values, part.values)
    viz.topic_correlation(m, method="raw").to_png()  # available but labeled biased


def test_topic_correlation_refused_on_cluster(bertopic):
    with pytest.raises(ValueError, match="degenerate"):
        viz.topic_correlation(bertopic)


def test_topic_correlation_eta_uses_model_sigma():
    # method="eta" must read the model's fitted prior covariance Sigma (the K-1
    # reference-dropped logistic-normal covariance), not re-estimate from theta.
    rng = np.random.default_rng(0)
    docs = []
    for _ in range(150):
        base = ["a", "a", "b", "c"] if rng.random() < 0.5 else ["x", "y", "y", "z"]
        docs.append(list(rng.choice(base, size=10)))
    m = topica.models.CTM(3, seed=1)
    m.fit(docs, iters=15)
    cov = m.topic_covariance
    assert cov.shape == (2, 2)                      # K-1, reference dropped
    cc = viz.topic_correlation(m, method="eta")
    df = cc.to_frame()
    assert df.shape == (2, 2)
    assert np.allclose(np.diag(df.values), 1.0)
    assert cc.reference == 2
    # it is the cov-derived correlation, not corrcoef of eta_mean
    d = np.sqrt(np.diag(cov))
    expected = cov / np.outer(d, d)
    # to_frame is seriated; compare the unordered matrix the panel stored
    assert np.allclose(np.sort(cc._cor.ravel()), np.sort(expected.ravel()), atol=1e-9)


def test_dashboard_with_groups_and_time(lda_k4):
    m, group, year, _docs = lda_k4
    d = viz.dashboard(m, groups=group, timestamps=year)
    assert {"groups", "temporal", "health", "correlation"} <= set(d.panels)
    assert d.to_png() is not None


# --- document map (Rust projection) ----------------------------------------

def test_project_primitive():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal([0, 0, 0], 0.4, (30, 3)),
                   rng.normal([8, 8, 0], 0.4, (30, 3))])
    pca = topica.project(X, 2, method="pca", seed=0)
    assert pca.shape == (60, 2) and np.isfinite(pca).all()
    assert np.allclose(pca, topica.project(X, 2, method="pca", seed=0))  # deterministic
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert topica.project(X, 2, method="tsne", seed=0).shape == (60, 2)
        assert topica.project(X, 2, method="umap", seed=0).shape == (60, 2)
    with pytest.raises(ValueError, match="unknown method"):
        topica.project(X, 2, method="bogus")


def test_document_map_theta_path(lda_k4):
    m = lda_k4[0]
    dm = viz.document_map(m, method="pca")        # clr(theta) projection
    df = dm.to_frame()
    assert {"x", "y", "dominant_topic", "outlier", "sampled"} <= set(df.columns)
    assert len(df) == m.doc_topic.shape[0]
    assert dm.var_explained is not None            # PCA reports variance explained
    assert dm.to_png() is not None


def test_document_map_embedding_path_and_outliers(bertopic):
    rng = np.random.default_rng(0)
    emb = np.vstack([rng.normal([0, 0], 0.4, (40, 2)), rng.normal([6, 0], 0.4, (40, 2))])
    dm = viz.document_map(bertopic, emb, method="pca")
    df = dm.to_frame()
    assert len(df) == 80
    # BERTopic carries hard labels incl. possible -1 outliers in the frame
    assert df["outlier"].dtype == bool
    assert dm.to_png() is not None


def test_document_map_refused_without_embeddings(bertopic):
    with pytest.raises(ValueError, match="pass doc_embeddings"):
        viz.document_map(bertopic)


def test_document_map_subsamples(lda_k4):
    m = lda_k4[0]
    dm = viz.document_map(m, max_points=50)
    assert dm.sampled is True
    # stratified-proportional sampling lands near max_points (not exactly, per-group rounding)
    assert 0 < len(dm.to_frame()) < m.doc_topic.shape[0]
    assert "showing" in dm._caption()


def test_document_map_highlight(lda_k4):
    m = lda_k4[0]
    dm = viz.document_map(m, highlight_topic=1)
    assert dm.highlight_topic == 1
    assert dm.to_png() is not None


def test_document_map_interactive(lda_k4):
    pytest.importorskip("plotly")
    m = lda_k4[0]
    fig = viz.document_map(m).to_html()
    assert fig is not None and hasattr(fig, "to_html")


# --- document inspector ----------------------------------------------------

def test_document_inspector(covariate_lda):
    m, corpus, x, texts = covariate_lda
    di = viz.document_inspector(m, texts, doc=0)
    df = di.to_frame()
    assert {"word", "in_vocab", "dominant_topic", "p_topic"} <= set(df.columns)
    assert len(df) >= 1
    assert len(di.theta) == m.num_topics
    assert isinstance(di.neighbors, list) and len(di.neighbors) >= 1
    assert di.to_png() is not None


def test_document_inspector_refused_on_cluster(bertopic):
    with pytest.raises(ValueError, match="degenerate"):
        viz.document_inspector(bertopic, [""] * 80, doc=0)


# --- STM/SAGE content-covariate per-group view -----------------------------

@pytest.fixture(scope="module")
def sage_content():
    rng = np.random.default_rng(5)
    docs, groups = [], []
    for i in range(120):
        g = i % 2
        if rng.random() < 0.5:
            base = ["econ", "econ", "market"] if g == 0 else ["econ", "econ", "trade"]
        else:
            base = ["health", "health", "care"] if g == 0 else ["health", "health", "clinic"]
        docs.append(list(rng.choice(base, size=8)))
        groups.append(["north", "south"][g])
    m = topica.models.SAGE(2, seed=1)
    m.fit(docs, groups, iters=300, num_samples=2)
    return m, [" ".join(d) for d in docs]


def test_content_covariate(sage_content):
    m = sage_content[0]
    cc = viz.content_covariate(m, topic=0, n=5)
    df = cc.to_frame()
    assert set(df["group"]) == {"north", "south"}
    assert {"prob", "in_group_top"} <= set(df.columns)
    assert cc.matrix().shape == (len(cc._words), 2)
    assert cc.to_png() is not None


def test_content_covariate_refused_without_content(covariate_lda):
    with pytest.raises(ValueError, match="no content covariate"):
        viz.content_covariate(covariate_lda[0], topic=0)


def test_sage_all_term_modes(sage_content):
    # All five modes route through the 2-D marginal, so none crash on SAGE's 3-D phi.
    m = sage_content[0]
    for mode in ("prob", "frex", "lift", "relevance", "score"):
        df = viz.term_barchart(m, topic=0, mode=mode, n=3).to_frame()
        assert len(df) >= 1


def test_dashboard_content_and_inspector(sage_content):
    m, texts = sage_content
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = viz.dashboard(m, texts, inspect_doc=0)
    # SAGE is a content model, so the wording panel is auto-included; inspector
    # appears because inspect_doc was given. The generic panels survive SAGE's 3-D
    # phi via the group marginal (issue #27), including the coherence frontier.
    assert {"content", "inspector", "similarity", "terms", "health", "frontier"} <= set(d.panels)
    assert "frontier" not in d.skipped
    assert d.to_png() is not None


# --- interactive build (skips if plotly absent) ----------------------------

def test_interactive_browser(tmp_path):
    pytest.importorskip("plotly")
    rng = np.random.default_rng(0)
    docs = [["a", "b", "c"]] * 15 + [["x", "y", "z"]] * 15
    m = topica.models.LDA(2, seed=1)
    m.fit(docs, iters=100)
    fig = viz.term_topic_browser(m, n=5)
    html = fig.to_html(full_html=False)
    assert "plotly" in html.lower()
    # to_html(path) must write a self-contained page (regression for #135: a raw
    # Plotly Figure treats the first positional arg as config and writes nothing).
    out = tmp_path / "browser.html"
    ret = fig.to_html(str(out))
    assert out.exists() and out.stat().st_size > 0
    assert ret == str(out)
