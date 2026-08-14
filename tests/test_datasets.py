"""Tests for topica.datasets (bundled + fetch-on-demand example data).

The vendored Gadarian dataset is exercised for real (it ships in the package).
The fetch path is exercised hermetically by monkeypatching urllib so the suite
never touches the network; the live URL/checksum is validated out of band.
"""

import hashlib
import io

import pytest

import topica
from topica import datasets


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------
# Vendored dataset: gadarian (ships in the wheel, loads offline)
# ---------------------------------------------------------------------------


def test_load_gadarian_dataframe():
    df = datasets.load_gadarian()
    assert len(df) == 341
    for col in ("MetaID", "treatment", "pid_rep", "open.ended.response"):
        assert col in df.columns


def test_load_gadarian_as_bunch_uniform_df():
    # as_bunch gives the uniform shape: a Bunch whose .df is the table (#686).
    import pandas as pd

    b = datasets.load_gadarian(as_bunch=True)
    assert isinstance(b, datasets.Bunch)
    assert isinstance(b.df, pd.DataFrame)
    assert len(b.df) == 341
    # default is still a bare DataFrame (non-breaking)
    assert isinstance(datasets.load_gadarian(), pd.DataFrame)


def test_load_gadarian_return_path():
    path = datasets.load_gadarian(return_path=True)
    assert path.exists()
    assert path.name == "gadarian.csv"
    # vendored checksum matches the registry record
    expected = datasets._REGISTRY["gadarian"]["sha256"]
    assert datasets._sha256(path) == expected


def test_gadarian_feeds_from_dataframe_and_fits():
    """The quickstart smoke test: dataset -> corpus -> model, fully offline."""
    df = datasets.load_gadarian()
    corpus = topica.from_dataframe(
        df,
        text_col="open.ended.response",
        stopwords=topica.ENGLISH_STOPWORDS,
        min_doc_freq=2,
    )
    model = topica.LDA(num_topics=4, seed=1)
    model.fit(corpus, iters=20)
    assert model.topic_word.shape[0] == 4


# ---------------------------------------------------------------------------
# Cache location
# ---------------------------------------------------------------------------


def test_data_home_respects_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TOPICA_DATA_HOME", str(tmp_path / "cache"))
    home = datasets.get_data_home()
    assert home == (tmp_path / "cache")
    assert home.is_dir()


# ---------------------------------------------------------------------------
# Fetch path (hermetic: no real network)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_remote(tmp_path, monkeypatch):
    """Point the cache at a temp dir and serve a known payload over a fake
    urlopen, with the registry checksum patched to match it."""
    monkeypatch.setenv("TOPICA_DATA_HOME", str(tmp_path / "cache"))
    payload = b"text,rating\nfoo bar baz,Liberal\nqux quux,Conservative\n"
    calls = {"n": 0}

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()
            return False

    def fake_urlopen(url, timeout=None):
        calls["n"] += 1
        return _Resp(payload)

    monkeypatch.setattr(datasets.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setitem(
        datasets._REGISTRY["poliblog"], "sha256", _sha256_bytes(payload)
    )
    return payload, calls


def test_fetch_downloads_then_caches(fake_remote):
    payload, calls = fake_remote
    path = datasets.load_poliblog(return_path=True)
    assert path.exists()
    assert path.read_bytes() == payload
    assert calls["n"] == 1
    # second call hits cache, no further download
    datasets.load_poliblog(return_path=True)
    assert calls["n"] == 1


def test_fetch_rejects_bad_checksum(tmp_path, monkeypatch):
    monkeypatch.setenv("TOPICA_DATA_HOME", str(tmp_path / "cache"))

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()
            return False

    monkeypatch.setattr(
        datasets.urllib.request, "urlopen", lambda url, timeout=None: _Resp(b"wrong")
    )
    monkeypatch.setitem(datasets._REGISTRY["poliblog"], "sha256", "0" * 64)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        datasets.load_poliblog(return_path=True)
    # nothing left behind in the cache
    assert not (tmp_path / "cache" / "poliblog.csv").exists()


def test_fetch_network_error_is_friendly(tmp_path, monkeypatch):
    monkeypatch.setenv("TOPICA_DATA_HOME", str(tmp_path / "cache"))

    def boom(url, timeout=None):
        raise datasets.urllib.error.URLError("no network")

    monkeypatch.setattr(datasets.urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="could not download"):
        datasets.load_dubois(return_path=True)


# ---------------------------------------------------------------------------
# Embeddings dataset: ng20_minilm (fetch-on-demand .npz -> Bunch)
# ---------------------------------------------------------------------------


def _fake_ng20_npz() -> bytes:
    import numpy as np

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        texts=np.array(["foo bar baz", "qux quux corge"], dtype=object),
        labels=np.array(["sci.space", "sci.med"], dtype=object),
        doc_embeddings=np.zeros((2, 4), dtype=np.float16),
        vocab=np.array(["foo", "bar", "baz", "qux"], dtype=object),
        word_embeddings=np.ones((4, 4), dtype=np.float16),
        meta=np.array("synthetic test payload", dtype=object),
    )
    return buf.getvalue()


def test_ng20_minilm_registered():
    assert "load_ng20_minilm" in datasets.__all__
    rec = datasets._REGISTRY["ng20_minilm"]
    assert rec["remote"].endswith("ng20_minilm.npz")
    assert len(rec["sha256"]) == 64


def test_congress_registered():
    assert "load_congress" in datasets.__all__
    rec = datasets._REGISTRY["congress"]
    assert rec["remote"].endswith("congress_press.csv")
    assert rec["text_col"] == "text"
    assert rec["n_docs"] == 3120
    assert len(rec["sha256"]) == 64
    # the summary advertises the covariates the STM example depends on
    assert "party" in rec["summary"] and "House" in rec["summary"]


def test_congress_committed_csv_matches_registry_sha():
    """When the repo tree is present (dev/CI checkout, not an installed wheel), the
    committed examples/congress_press.csv must hash to the registry sha256, so the
    real download the loader performs is guaranteed intact."""
    from pathlib import Path

    csv = Path(datasets.__file__).resolve().parents[3] / "examples" / "congress_press.csv"
    if not csv.exists():
        pytest.skip("examples/ not present (installed wheel); checked out of band")
    got = hashlib.sha256(csv.read_bytes()).hexdigest()
    assert got == datasets._REGISTRY["congress"]["sha256"]


def test_congress_fetch_hermetic(tmp_path, monkeypatch):
    monkeypatch.setenv("TOPICA_DATA_HOME", str(tmp_path / "cache"))
    payload = (
        b"text,date,year,party,state,bioguide_id,member,title\n"
        b"we support the bill,2015-06-01,2015,Democrat,CA,X000001,Rep A,Statement\n"
        b"we oppose the bill,2017-06-01,2017,Republican,TX,X000002,Rep B,Remarks\n"
    )

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()
            return False

    monkeypatch.setattr(datasets.urllib.request, "urlopen",
                        lambda url, timeout=None: _Resp(payload))
    monkeypatch.setitem(datasets._REGISTRY["congress"], "sha256", _sha256_bytes(payload))
    df = datasets.load_congress()
    assert list(df.columns) == ["text", "date", "year", "party", "state",
                                "bioguide_id", "member", "title"]
    assert set(df["party"]) == {"Democrat", "Republican"}


def test_bunch_attribute_access():
    b = datasets.Bunch(x=1, y=2)
    assert b.x == 1 and b["y"] == 2
    b.z = 3
    assert b["z"] == 3
    assert "x" in dir(b)  # keys surface for tab-completion
    with pytest.raises(AttributeError):
        _ = b.missing
    with pytest.raises(AttributeError):  # del of a missing attr, not KeyError
        del b.missing
    del b.z
    assert "z" not in b


def test_ng20_minilm_loads_bunch(tmp_path, monkeypatch):
    monkeypatch.setenv("TOPICA_DATA_HOME", str(tmp_path / "cache"))
    payload = _fake_ng20_npz()

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()
            return False

    monkeypatch.setattr(
        datasets.urllib.request, "urlopen", lambda url, timeout=None: _Resp(payload)
    )
    monkeypatch.setitem(
        datasets._REGISTRY["ng20_minilm"], "sha256", _sha256_bytes(payload)
    )

    b = datasets.load_ng20_minilm()
    assert isinstance(b, datasets.Bunch)
    assert b.texts == ["foo bar baz", "qux quux corge"]
    assert list(b.labels) == ["sci.space", "sci.med"]
    assert b.doc_embeddings.shape == (2, 4)
    assert b.vocab == ["foo", "bar", "baz", "qux"]
    assert b.word_embeddings.shape == (4, 4)
    assert "synthetic" in b.meta
    assert b["texts"] is b.texts  # attribute and item access are the same object
    # second call hits the cache; return_path yields the cached .npz
    p = datasets.load_ng20_minilm(return_path=True)
    assert p.name == "ng20_minilm.npz" and p.exists()
