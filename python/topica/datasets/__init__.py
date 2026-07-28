"""Bundled example datasets for quickstarts and worked examples.

Small datasets ship inside the wheel and load instantly, offline. Larger ones
are downloaded once from GitHub on first use and cached locally, so the wheel
stays lean. Every loader returns a :mod:`pandas` DataFrame ready for
:func:`topica.from_dataframe`::

    import topica

    df = topica.datasets.load_gadarian()           # vendored, instant, offline
    corpus = topica.from_dataframe(
        df, text_col="open.ended.response", stopwords=topica.ENGLISH_STOPWORDS
    )
    model = topica.STM(num_topics=10).fit(corpus, prevalence=corpus.metadata[["treatment"]])

Pass ``return_path=True`` to get the cached CSV path instead of a DataFrame
(no pandas required).

The cache lives under ``~/.cache/topica/datasets`` by default; set the
``TOPICA_DATA_HOME`` environment variable to relocate it. Downloads are pinned
to an immutable commit and verified against a SHA-256 checksum, so a given
topica version always fetches the same bytes.
"""

from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
from pathlib import Path

__all__ = [
    "load_gadarian",
    "load_poliblog",
    "load_dubois",
    "load_ng20_minilm",
    "get_data_home",
    "clear_cache",
]


class Bunch(dict):
    """A dict whose keys are also accessible as attributes (sklearn-style).

    Returned by loaders that carry more than a text table — e.g.
    :func:`load_ng20_minilm`, which bundles documents, labels, and precomputed
    embedding arrays. ``b["texts"]`` and ``b.texts`` are the same thing.
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - trivial
            raise AttributeError(key) from exc

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Bunch({', '.join(self.keys())})"

# Immutable commit that holds the dataset files. Pinned (not ``main``) so a
# given topica version always fetches the same bytes; override with the
# TOPICA_DATA_REF environment variable for local development against a branch.
_DATA_REF = "cbb6fe4adbfd19e4a2bca226f05e146c7a61080c"
_RAW_BASE = "https://raw.githubusercontent.com/nealcaren/topica"

# name -> dataset record. ``vendored`` files ship in the wheel; the rest carry a
# repo-relative ``remote`` path fetched on first use. ``sha256`` is verified for
# every fetched file.
_REGISTRY = {
    "gadarian": {
        "vendored": "gadarian.csv",
        "filename": "gadarian.csv",
        "sha256": "de30697c6f42a8d32fca4bb6798679e4004afe16e9f3dbd06e7409afedb474d2",
        "text_col": "open.ended.response",
        "n_docs": 341,
        "summary": (
            "Gadarian & Albertson (2014) immigration open-ended responses; the "
            "canonical stm prevalence example. Raw text in 'open.ended.response'; "
            "covariates 'treatment' (anxiety prime) and 'pid_rep' (Republican id)."
        ),
    },
    "poliblog": {
        "remote": "examples/poliblog.csv",
        "filename": "poliblog.csv",
        "sha256": "4c26bd96b5ca57ae99a7a72835f2aa1a549d55178e8779e079cc0fb9ac498ff9",
        "text_col": "text",
        "n_docs": 2000,
        "summary": (
            "CMU 2008 political blog corpus (a 2,000-document sample). Text is "
            "already tokenized and stemmed (space-separated). Covariates 'rating' "
            "(Liberal/Conservative), 'day', and 'blog'."
        ),
    },
    "dubois": {
        "remote": "examples/dubois_crisis.csv",
        "filename": "dubois_crisis.csv",
        "sha256": "b38b8b22e96fb2f6a07b7983db05c72e6bc1cffb1c0c23ab9ff006cdcce7513e",
        "text_col": "text",
        "n_docs": 704,
        "summary": (
            "Du Bois-era articles from The Crisis (1910-1922). Raw text in 'text'; "
            "covariates 'year', 'decade', 'volume', 'issue', 'author', 'subjects'."
        ),
    },
    "ng20_minilm": {
        "remote": "examples/ng20_minilm.npz",
        "filename": "ng20_minilm.npz",
        "sha256": "aa4afb4c88af8692b8639ae4ea62025139643b63e706de6412dace1df63a8518",
        "n_docs": 2594,
        "summary": (
            "20-Newsgroups (5 groups) with precomputed MiniLM sentence embeddings "
            "for both documents and vocabulary; the embedding-native example "
            "corpus for ProdLDA/FASTopic/BERTopic/Top2Vec. Bundles 'texts', "
            "'labels', 'doc_embeddings', 'vocab', 'word_embeddings'."
        ),
    },
}


def get_data_home() -> Path:
    """Return the directory where downloaded datasets are cached.

    Defaults to ``~/.cache/topica/datasets``; override with the
    ``TOPICA_DATA_HOME`` environment variable. The directory is created if it
    does not exist.
    """
    env = os.environ.get("TOPICA_DATA_HOME")
    home = Path(env).expanduser() if env else Path.home() / ".cache" / "topica" / "datasets"
    home.mkdir(parents=True, exist_ok=True)
    return home


def clear_cache() -> None:
    """Delete every cached (downloaded) dataset file. Vendored datasets are
    unaffected; the next ``load_*`` call re-downloads what it needs."""
    home = get_data_home()
    for record in _REGISTRY.values():
        if "remote" in record:
            (home / record["filename"]).unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _vendored_path(record: dict) -> Path:
    # importlib.resources keeps this working from a wheel, a zip, or the source
    # tree without assuming a filesystem layout.
    from importlib.resources import as_file, files

    resource = files(__name__).joinpath("_data", record["vendored"])
    with as_file(resource) as p:
        path = Path(p)
    if not path.exists():
        raise FileNotFoundError(
            f"vendored dataset {record['vendored']!r} is missing from the install"
        )
    return path


def _fetch(name: str, record: dict) -> Path:
    dest = get_data_home() / record["filename"]
    if dest.exists():
        if _sha256(dest) == record["sha256"]:
            return dest
        dest.unlink()  # corrupt/partial cache; re-download

    ref = os.environ.get("TOPICA_DATA_REF", _DATA_REF)
    url = f"{_RAW_BASE}/{ref}/{record['remote']}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    except (urllib.error.URLError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"could not download the {name!r} dataset from {url}\n"
            f"({exc}). Check your network, or download the file manually to "
            f"{dest} and retry. Set TOPICA_DATA_HOME to relocate the cache."
        ) from exc

    got = _sha256(tmp)
    if got != record["sha256"]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"checksum mismatch for {name!r}: expected {record['sha256']}, got "
            f"{got}. The download may be corrupt or the pinned data ref changed."
        )
    tmp.replace(dest)
    return dest


def _resolve(name: str) -> Path:
    record = _REGISTRY[name]
    if "vendored" in record:
        return _vendored_path(record)
    return _fetch(name, record)


def _read_csv(path: Path):
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised via error path
        raise ImportError(
            "loading a dataset as a DataFrame needs pandas (pip install pandas). "
            "Pass return_path=True to get the CSV path without pandas."
        ) from exc
    return pd.read_csv(path)


def _load(name: str, return_path: bool):
    path = _resolve(name)
    if return_path:
        return path
    return _read_csv(path)


def load_gadarian(*, return_path: bool = False):
    """Load the Gadarian & Albertson immigration experiment (341 documents).

    The canonical ``stm`` prevalence example. Open-ended survey responses with
    an anxiety-prime ``treatment`` and ``pid_rep`` (Republican identification).
    Raw, untokenized text lives in the ``open.ended.response`` column, so build a
    corpus with stopword removal::

        df = topica.datasets.load_gadarian()
        corpus = topica.from_dataframe(
            df, text_col="open.ended.response", stopwords=topica.ENGLISH_STOPWORDS
        )

    This dataset is bundled in the wheel and loads offline. Pass
    ``return_path=True`` for the CSV path instead of a DataFrame.
    """
    return _load("gadarian", return_path)


def load_poliblog(*, return_path: bool = False):
    """Load the CMU 2008 political blog corpus (a 2,000-document sample).

    The text in the ``text`` column is already tokenized and stemmed
    (space-separated), as in the ``stm`` ``poliblog`` vignette, so no stopword
    removal is needed::

        df = topica.datasets.load_poliblog()
        corpus = topica.from_dataframe(df, text_col="text")

    Covariates: ``rating`` (Liberal/Conservative), ``day``, ``blog``. Downloaded
    once and cached. Pass ``return_path=True`` for the CSV path.
    """
    return _load("poliblog", return_path)


def load_dubois(*, return_path: bool = False):
    """Load Du Bois-era articles from The Crisis, 1910-1922 (704 documents).

    Raw text in the ``text`` column; covariates ``year``, ``decade``,
    ``volume``, ``issue``, ``author``, ``subjects``. Build a corpus with
    stopword removal::

        df = topica.datasets.load_dubois()
        corpus = topica.from_dataframe(
            df, text_col="text", stopwords=topica.ENGLISH_STOPWORDS
        )

    Downloaded once and cached. Pass ``return_path=True`` for the CSV path.
    """
    return _load("dubois", return_path)


def load_ng20_minilm(*, return_path: bool = False):
    """Load 20-Newsgroups with precomputed MiniLM embeddings (5 groups).

    The embedding-native counterpart to the text datasets: the same corpus the
    ProdLDA/FASTopic/BERTopic/Top2Vec examples use, with
    ``sentence-transformers`` ``all-MiniLM-L6-v2`` vectors already computed for
    every document *and* every vocabulary term. This lets the embedding topic
    models run offline, with no ``sentence-transformers``/``torch`` install::

        b = topica.datasets.load_ng20_minilm()
        bt = topica.BERTopic(reducer="umap", n_components=5).fit(
            [t.split() for t in b.texts], b.doc_embeddings
        )
        tv = topica.Top2Vec(n_components=5).fit(
            [t.split() for t in b.texts], b.doc_embeddings,
            word_embeddings=b.word_embeddings, vocabulary=b.vocab,
        )

    Returns a :class:`Bunch` with attribute access to:

    - ``texts`` — list of documents (space-joined in-vocab tokens)
    - ``labels`` — newsgroup name per document (numpy object array)
    - ``doc_embeddings`` — ``(n_docs, 384)`` float16 MiniLM vectors
    - ``vocab`` — list of vocabulary terms
    - ``word_embeddings`` — ``(vocab, 384)`` float16 MiniLM vectors
    - ``meta`` — provenance string

    Embeddings are stored as float16 to keep the download small; cast to
    ``float32`` if a model needs it. Downloaded once and cached. Pass
    ``return_path=True`` for the cached ``.npz`` path instead of the Bunch.
    """
    path = _resolve("ng20_minilm")
    if return_path:
        return path
    import numpy as np

    with np.load(path, allow_pickle=True) as npz:
        return Bunch(
            texts=list(npz["texts"]),
            labels=npz["labels"],
            doc_embeddings=npz["doc_embeddings"],
            vocab=list(npz["vocab"]),
            word_embeddings=npz["word_embeddings"],
            meta=str(npz["meta"]),
        )
