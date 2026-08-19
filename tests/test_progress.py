"""topica.progress() live fit-progress callback + sparkline (#785)."""

import io

import topica
from topica.progress import _fmt_secs


def test_sparkline_basic_shapes():
    assert topica.sparkline([1, 2, 3, 4, 3, 2, 1]) == "▁▃▆█▆▃▁"
    assert topica.sparkline([5, 5, 5]) == "▁▁▁"      # flat -> low baseline
    assert topica.sparkline([]) == ""
    assert topica.sparkline([7]) == "▁"


def test_sparkline_ignores_non_finite_and_rolls():
    # NaN / inf are dropped, not rendered as garbage.
    s = topica.sparkline([1.0, float("nan"), 2.0, float("inf"), 3.0])
    assert set(s) <= set("▁▂▃▄▅▆▇█") and len(s) == 3
    # rolling window keeps a fixed width on a long series.
    assert len(topica.sparkline(list(range(1000)), width=40)) == 40


def test_fmt_secs():
    assert _fmt_secs(3.1) == "3.1s"
    assert _fmt_secs(90).endswith("s") and "m" in _fmt_secs(90)
    assert "h" in _fmt_secs(7200)
    assert _fmt_secs(float("nan")) == "?"


def test_progress_renders_bar_eta_and_sparkline():
    buf = io.StringIO()
    cb = topica.progress(metric="ll", stream=buf, label="GSDMM")
    lls = [-9.5, -9.0, -8.4, -8.0, -7.85, -7.82]
    ks = [50, 48, 42, 36, 31, 30]
    for i, (ll, k) in enumerate(zip(lls, ks), 1):
        cb(i, len(lls), {"ll": ll, "clusters": k})
    frames = [f for f in buf.getvalue().replace("\n", "").split("\r") if f.strip()]
    first, last = frames[0], frames[-1]
    assert first.startswith("GSDMM |") and "10" not in first[:8]
    assert "0%" in first or "16%" in first  # 1/6
    assert last.strip().endswith("clusters=30")
    assert "100%" in last and "6/6" in last
    # the ll sparkline accumulates across calls and climbs to convergence
    assert "ll " in last
    # a trailing newline closes the line once the final iteration reports
    assert buf.getvalue().endswith("\n")


def test_progress_accepts_scalar_and_none_metric():
    buf = io.StringIO()
    cb = topica.progress(stream=buf)
    cb(1, 2, -5.0)      # bare scalar metric
    cb(2, 2, None)      # no metric this step
    out = buf.getvalue()
    assert "fit |" in out and out.endswith("\n")


def test_progress_two_arg_legacy_callback_without_total():
    # Legacy (iteration, metric) callback (today's LDA/DMR): no total -> running
    # count + elapsed + sparkline, no bar/ETA, and no crash.
    buf = io.StringIO()
    cb = topica.progress(stream=buf, label="LDA")
    for i, ll in enumerate([-9.0, -8.5, -8.1, -8.0], 1):
        cb(i, ll)  # 2 positional args
    last = buf.getvalue().split("\r")[-1]
    assert last.startswith("LDA  4") and "%" not in last and "▁" in last


def test_progress_two_arg_with_total_gives_full_bar():
    buf = io.StringIO()
    cb = topica.progress(total=4, stream=buf, label="LDA")
    for i, ll in enumerate([-9.0, -8.5, -8.1, -8.0], 1):
        cb(i, ll)
    out = buf.getvalue()
    assert "100%" in out and "4/4" in out and "ETA" in out


def test_lda_drives_progress_with_3arg_contract(capfd):
    # #785: LDA's callback now passes (iteration, total, {"ll": ...}), so
    # topica.progress() renders the full bar + ETA + "ll" sparkline with no total=.
    import numpy as np
    docs = [["a", "b", "c"]] * 60 + [["x", "y", "z"]] * 60
    calls = []
    topica.LDA(2, seed=13).fit(
        docs, iters=40, progress_interval=8,
        progress=lambda it, total, info: calls.append((it, total, dict(info))),
    )
    assert calls and all(t == 40 for _, t, _ in calls)          # auto total
    assert all("ll" in info for _, _, info in calls)             # named metric
    # and the shipped renderer produces a bar labelled "ll"
    topica.LDA(2, seed=13).fit(docs, iters=40, progress_interval=8,
                               progress=topica.progress(label="LDA"))
    err = capfd.readouterr().err
    assert "LDA |" in err and "ll " in err and "100%" in err


def test_lda_progress_does_not_change_the_fit():
    import numpy as np
    docs = [["a", "b", "c"]] * 40 + [["x", "y", "z"]] * 40
    a = topica.LDA(2, seed=13).fit(docs, iters=50)
    b = topica.LDA(2, seed=13).fit(docs, iters=50,
                                   progress=lambda *xs: None, progress_interval=5)
    assert np.array_equal(np.asarray(a.doc_topic), np.asarray(b.doc_topic))


def test_keyatm_drives_progress_base_and_covariate(capfd):
    # #785: keyATM's base and covariate Gibbs paths call the 3-arg contract with
    # {"ll", "perplexity"} and topica.progress() renders the bar.
    import warnings

    import numpy as np
    docs = [["tax", "budget", "vote"]] * 80 + [["health", "care", "clinic"]] * 80
    keys = {"fiscal": ["tax", "budget"], "health": ["health", "care"]}
    for kwargs in ({}, dict(covariates=np.array([[0.0]] * 80 + [[1.0]] * 80),
                            feature_names=["grp"])):
        calls = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            topica.KeyATM(keys, num_topics=3, seed=13).fit(
                docs, iters=120, progress_interval=20,
                progress=lambda it, total, info: calls.append((it, total, dict(info))),
                **kwargs,
            )
        assert calls and all(t == 120 for _, t, _ in calls)
        assert all({"ll", "perplexity"} <= set(info) for _, _, info in calls)
    # the renderer draws the bar with an ll sparkline
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        topica.KeyATM(keys, num_topics=3, seed=13).fit(
            docs, iters=120, progress_interval=20,
            progress=topica.progress(label="keyATM"))
    err = capfd.readouterr().err
    assert "keyATM |" in err and "ll " in err and "100%" in err


def test_progress_metric_selection_defaults_to_first_key():
    buf = io.StringIO()
    cb = topica.progress(stream=buf)  # no metric= -> first key ("perplexity")
    cb(1, 1, {"perplexity": 120.0, "iters_run": 5})
    last = buf.getvalue().split("\r")[-1]
    assert "perplexity " in last and "iters_run=5" in last
