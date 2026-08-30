"""Bundled example datasets for quickstarts and worked examples.

Small datasets ship inside the wheel and load instantly, offline. Larger ones
are downloaded once from GitHub on first use and cached locally, so the wheel
stays lean.

**Return shapes.** There are two, by dataset kind:

- **Text-table loaders** (:func:`load_gadarian`, :func:`load_poliblog`,
  :func:`load_dubois`, :func:`load_congress`, :func:`load_reviews`) return a
  :mod:`pandas` DataFrame by default.
- **The embedding loader** (:func:`load_ng20_minilm`) returns a :class:`Bunch`
  (attribute-access dict) because it carries embedding arrays alongside the text.

For a **uniform shape across the roster**, pass ``as_bunch=True`` to a text-table
loader: every loader then returns a :class:`Bunch`, and **every Bunch exposes a
``.df``** DataFrame view (plus any extra arrays). So ``load_X(as_bunch=True).df``
and ``load_ng20_minilm().df`` are the same idiom everywhere::

    import topica

    df = topica.datasets.load_gadarian()                    # DataFrame (default)
    b = topica.datasets.load_gadarian(as_bunch=True)         # Bunch; b.df is the table
    b = topica.datasets.load_ng20_minilm()                  # Bunch; b.df + b.doc_embeddings

    corpus = topica.from_dataframe(
        df, text_col="open.ended.response", stopwords=topica.ENGLISH_STOPWORDS
    )
    model = topica.STM(num_topics=10).fit(corpus, prevalence=corpus.metadata[["treatment"]])

Pass ``return_path=True`` to get the cached file path instead (no pandas required).

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
    "load_congress",
    "load_reviews",
    "load_ng20_minilm",
    "load_threads",
    "get_data_home",
    "clear_cache",
]


class Bunch(dict):
    """A dict whose keys are also accessible as attributes (sklearn-style).

    Returned by loaders that carry more than a text table — e.g.
    :func:`load_ng20_minilm`, which bundles documents, labels, and precomputed
    embedding arrays. ``b["texts"]`` and ``b.texts`` are the same thing.

    Every loader can return a Bunch (``as_bunch=True`` for the text-table
    loaders; :func:`load_ng20_minilm` always does), and every Bunch exposes a
    ``.df`` DataFrame view for a uniform shape across the roster.
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - trivial
            raise AttributeError(key) from exc

    __setattr__ = dict.__setitem__

    def __delattr__(self, key):  # `del b.missing` must raise AttributeError
        try:
            del self[key]
        except KeyError as exc:  # pragma: no cover - trivial
            raise AttributeError(key) from exc

    def __dir__(self):  # surface the keys for REPL/Jupyter tab-completion
        return list(self.keys()) + list(super().__dir__())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Bunch({', '.join(self.keys())})"

# Immutable commit that holds the dataset files. Pinned (not ``main``) so a
# given topica version always fetches the same bytes; override with the
# TOPICA_DATA_REF environment variable for local development against a branch.
_DATA_REF = "548b09228e8641422557d79c0b84aace1cdd97b3"
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
    "reviews": {
        "vendored": "reviews.csv",
        "filename": "reviews.csv",
        "sha256": "c94b9c16afaab55d72d11b88a76b6ce30e3f3b379a0fe2314213b5fee37825a8",
        "text_col": "text",
        "n_docs": 1500,
        "summary": (
            "Yelp business reviews, 1,500 short reviews balanced across the 1-5 "
            "star scale (300 each). Raw text in 'text'; the ordinal covariate "
            "'stars' (1-5) is the sentiment signal. Ships in the wheel; loads "
            "offline. Derived from the Yelp Open Dataset."
        ),
    },
    "congress": {
        "remote": "examples/congress_press.csv",
        "filename": "congress_press.csv",
        "sha256": "ec0918d80b31f808dbb25c95dc8bd9a61812de32a1dff86d86f9be91d6029ffd",
        "text_col": "text",
        "n_docs": 3120,
        "summary": (
            "U.S. House press releases, 2013-2024 (260 per year, balanced by "
            "party). Raw text in 'text'; covariates 'party' (Democrat/Republican), "
            "'year' and 'date' (time), 'state', 'member', 'bioguide_id', 'title'. "
            "The canonical STM party + time example. Source: Derek Willis's "
            "congress-press (MIT)."
        ),
    },
    "threads": {
        "remote": "examples/reddit_threads.csv",
        "filename": "reddit_threads.csv",
        "sha256": "318206f6b9c64fddcedc42954c32abcf72b1b865d0e521237b8e4800cf94398f",
        "text_col": "text",
        "n_docs": 5042,
        "summary": (
            "Two-subreddit threaded Reddit corpus (5,042 comments in 171 reply "
            "trees): 'askscience' (technical Q&A, replies answer their parent) and "
            "'pokemontrades' (the deepest trees in the source, but replies "
            "coordinate trades rather than respond on-topic). The ThreadTM reply-"
            "tree vignette. Columns 'doc_id', 'thread_root', 'parent' (0-based row "
            "index of the comment replied to, -1 for a root), 'subreddit', "
            "'timestamp', raw 'text'. Source: ConvoKit reddit-corpus-small "
            "(Chang et al. 2020)."
        ),
    },
    "ng20_minilm": {
        "remote": "examples/ng20_minilm.npz",
        "filename": "ng20_minilm.npz",
        "sha256": "e1c1971fab4d1af1f6693577a9d7b7c37049f57aa0f81757b044208102787a86",
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


def _load(name: str, return_path: bool, as_bunch: bool = False):
    path = _resolve(name)
    if return_path:
        return path
    df = _read_csv(path)
    if as_bunch:
        return Bunch(df=df)
    return df


def load_gadarian(*, return_path: bool = False, as_bunch: bool = False):
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
    ``return_path=True`` for the CSV path instead of a DataFrame, or
    ``as_bunch=True`` for a :class:`Bunch` whose ``.df`` is this table (the
    uniform shape shared with :func:`load_ng20_minilm`).
    """
    return _load("gadarian", return_path, as_bunch)


def load_poliblog(*, return_path: bool = False, as_bunch: bool = False):
    """Load the CMU 2008 political blog corpus (a 2,000-document sample).

    The text in the ``text`` column is already tokenized and stemmed
    (space-separated), as in the ``stm`` ``poliblog`` vignette, so no stopword
    removal is needed::

        df = topica.datasets.load_poliblog()
        corpus = topica.from_dataframe(df, text_col="text")

    Covariates: ``rating`` (Liberal/Conservative), ``day``, ``blog``. Downloaded
    once and cached. Pass ``return_path=True`` for the CSV path, or
    ``as_bunch=True`` for a :class:`Bunch` whose ``.df`` is this table.
    """
    return _load("poliblog", return_path, as_bunch)


def load_dubois(*, return_path: bool = False, as_bunch: bool = False):
    """Load Du Bois-era articles from The Crisis, 1910-1934 (704 documents).

    Raw text in the ``text`` column; covariates ``year``, ``decade``,
    ``volume``, ``issue``, ``author``, ``subjects``. Build a corpus with
    stopword removal::

        df = topica.datasets.load_dubois()
        corpus = topica.from_dataframe(
            df, text_col="text", stopwords=topica.ENGLISH_STOPWORDS
        )

    The corpus holds a few (3) exact-duplicate articles reprinted across issues;
    drop them with ``df.drop_duplicates("text")`` if a fit should not double-count
    them. Downloaded once and cached. Pass ``return_path=True`` for the CSV path.

    Note on ``author`` for :class:`~topica.AuthorTopic`: this field is dominated by
    Du Bois (about 675 of 704 articles) and contains delimited composites
    (``"Du Bois; Gruening"``) and name/initial variants (``"Du Bois"`` vs
    ``"Du Bois, W.E.B."``). Split composites (``[s.split("; ") for s in df.author]``)
    and normalize variants before using it as an author-topic input, or a co-authored
    article becomes a phantom author and one person splits across several rows.

    Pass ``return_path=True`` for the CSV path, or ``as_bunch=True`` for a
    :class:`Bunch` whose ``.df`` is this table (the uniform shape shared with
    :func:`load_ng20_minilm`).
    """
    return _load("dubois", return_path, as_bunch)


def load_congress(*, return_path: bool = False, as_bunch: bool = False):
    """Load U.S. House press releases, 2013-2024 (3,120 documents).

    A balanced sample of 260 releases per year across twelve years (1,560
    Democratic, 1,560 Republican), so the ``party`` and ``year`` covariates are
    both well supported. This is the canonical Structural Topic Model example
    where prevalence depends on a group *and* on time::

        import numpy as np
        df = topica.datasets.load_congress()
        corpus = topica.from_dataframe(
            df, text_col="text", strip_html=True,
            stopwords=topica.ENGLISH_STOPWORDS, min_doc_freq=10,
        )
        # party (contrast vs Democrat) + a smooth trend in year
        X, names = topica.design_matrix(
            "~ party + spline(year, df=4)", corpus.metadata
        )
        model = topica.STM(num_topics=20, seed=13)
        model.fit(corpus, prevalence=X, prevalence_names=names)

    Columns: raw ``text`` (press-release body — pass ``strip_html=True`` to
    :func:`topica.from_dataframe`, since some releases carry markup); ``date``
    (YYYY-MM-DD) and ``year`` for the time covariate; ``party``
    (``Democrat``/``Republican``); and ``state``, ``member``, ``bioguide_id``,
    ``title`` for reference. Chamber is not a column: the sample is House-only, so
    ``party`` and ``year`` are the covariates that vary. (Chamber is taken from the
    source's member metadata, which mislabels a handful of members, so treat the
    House scope as approximate.) Some releases are in Spanish; a bilingual member's
    output can surface as its own topic.

    Downloaded once and cached. Pass ``return_path=True`` for the CSV path, or
    ``as_bunch=True`` for a :class:`Bunch` whose ``.df`` is this table.

    Source: Derek Willis's `congress-press
    <https://github.com/dwillis/congress-press>`_ (MIT licensed); the underlying
    press releases are U.S. government works. For the full multi-year archive
    (raw JSONL) start from that repository — the ``examples/congress_tutorial.py``
    script walks the raw-to-STM pipeline on it.
    """
    return _load("congress", return_path, as_bunch)


def load_reviews(*, return_path: bool = False, as_bunch: bool = False):
    """Load Yelp business reviews, balanced across the star scale (1,500 documents).

    Three hundred reviews at each of the five star ratings, so the ordinal
    ``stars`` covariate (1–5) is the sentiment signal. Ships inside the wheel and
    loads offline. The canonical example for a covariate topic model whose outcome
    is *valence*::

        df = topica.datasets.load_reviews()
        corpus = topica.from_dataframe(
            df, text_col="text",
            stopwords=topica.SENTIMENT_STOPWORDS,   # keep 'not'/'no'/'very'
        )
        X = (df["stars"].to_numpy(float) - 3.0).reshape(-1, 1)   # centered ordinal
        model = topica.DMR(num_topics=12, seed=13).fit(corpus, X, feature_names=["stars"])

    Pass ``stopwords=topica.SENTIMENT_STOPWORDS`` rather than the default: the
    default ``ENGLISH_STOPWORDS`` strips ``not``/``no``/``very``, which would
    collapse "not clean" into "clean" in exactly the study whose outcome is
    sentiment.

    Raw review text in ``text``; the ordinal covariate in ``stars``. Pass
    ``return_path=True`` for the CSV path, or ``as_bunch=True`` for a :class:`Bunch`
    whose ``.df`` is this table. Derived from the Yelp Open Dataset.
    """
    return _load("reviews", return_path, as_bunch)


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

    Embeddings are stored as float16 to keep the download small. topica's own
    models accept them directly (inputs are coerced internally); cast to
    ``float32`` only for an external tool that needs it. Downloaded once and
    cached. Pass ``return_path=True`` for the cached ``.npz`` path instead of the
    Bunch.
    """
    path = _resolve("ng20_minilm")
    if return_path:
        return path
    import numpy as np

    with np.load(path, allow_pickle=True) as npz:
        texts = list(npz["texts"])
        labels = npz["labels"]
        bunch = Bunch(
            texts=texts,
            labels=labels,
            doc_embeddings=npz["doc_embeddings"],
            vocab=list(npz["vocab"]),
            word_embeddings=npz["word_embeddings"],
            meta=str(npz["meta"]),
        )
    # `.df` gives the uniform DataFrame view every loader's Bunch exposes (the
    # per-document text table; the embedding arrays stay as attributes).
    try:
        import pandas as pd

        bunch["df"] = pd.DataFrame({"text": texts, "label": labels})
    except ImportError:  # pandas optional; the arrays are still available
        pass
    return bunch


def load_threads(*, return_path: bool = False):
    """Load the two-subreddit threaded Reddit corpus (5,042 comments, 171 trees).

    The :class:`~topica.ThreadTM` reply-tree vignette. Two subreddits, chosen to
    make the model's point honestly:

    - ``askscience`` — technical Q&A; replies genuinely answer their parent, so
      the reply tree carries topic structure and persistence is *identifiable*.
    - ``pokemontrades`` — the deepest reply trees in the source corpus, yet its
      replies coordinate trades ("added you on DS") rather than respond on-topic,
      so persistence is *not* identifiable. Tree depth is not persistence.

    Unlike the flat text datasets, threaded data cannot go through
    :func:`topica.from_dataframe` (that discards the reply tree), so this returns
    a :class:`Bunch` whose rows stay aligned to the ``parents`` index. Fit is
    turnkey::

        b = topica.datasets.load_threads()
        topica.enable_experimental()  # ThreadTM is experimental
        model = topica.ThreadTM(8, coupling="parent").fit(
            b.documents, parents=b.parents, covariates=b.subreddit
        )
        model.persistence()   # read `reliability` before claiming persistence

    The Bunch carries:

    - ``documents`` — token lists (lowercased, letters-only, min length 3, English
      stopwords removed), one per row and in row order (empty rows are kept so
      ``parents`` stays valid).
    - ``texts`` — the raw, untokenized comment text (retokenize this yourself for
      a different vocabulary; keep every row to preserve the ``parents`` index).
    - ``parents`` — list of ints: the 0-based row index of the comment each row
      replies to, or ``-1`` for a thread root. A parent's index is always smaller
      than its child's, so the array is safe to pass straight to ``fit``.
    - ``subreddit`` — the per-row subreddit, the prevalence/content covariate.
    - ``thread_root`` — the root comment id shared by every row in a tree.
    - ``timestamp`` — unix seconds (may be null).
    - ``df`` — the full table as a DataFrame.

    Source: ConvoKit ``reddit-corpus-small`` (Chang et al. 2020); see
    ``examples/build_reddit_threads.py`` to regenerate. Downloaded once and
    cached. Pass ``return_path=True`` for the cached CSV path instead of the Bunch.
    """
    path = _resolve("threads")
    if return_path:
        return path
    from .. import ENGLISH_STOPWORDS, tokenize

    df = _read_csv(path)
    texts = [str(t) for t in df["text"].tolist()]
    stop = set(ENGLISH_STOPWORDS)
    # Letters-only so curly-apostrophe contractions ("i’d", "it’s") don't survive
    # as tokens; min length 3 drops the short residue. Raw text is in `texts` for
    # anyone who wants a different vocabulary.
    documents = [
        tokenize(t, stopwords=stop, token_regex=r"[A-Za-z]+", min_length=3)
        for t in texts
    ]
    parents = [int(p) for p in df["parent"].tolist()]
    return Bunch(
        df=df,
        documents=documents,
        texts=texts,
        parents=parents,
        subreddit=df["subreddit"].tolist(),
        thread_root=df["thread_root"].tolist(),
        timestamp=df["timestamp"].tolist(),
    )
