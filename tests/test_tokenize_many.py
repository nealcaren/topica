"""Parallel batch tokenizer (#786): fast, byte-identical fast path for
from_dataframe, with opt-in build progress.

The single-doc `tokenize` recompiled the regex and rebuilt the stopword set on
every call, so a Python loop over it dominated corpus-build time on large
corpora. `tokenize_many` hoists both out and runs multi-core; these tests pin the
equivalence guarantee (byte-identical output) and the progress plumbing.
"""

import io

import pytest

import topica
from topica._topica import tokenize_many

pd = pytest.importorskip("pandas")

STOP = ["the", "a", "and", "of", "to", "today"]
TEXTS = [
    "The Senate passed a budget bill today, funding health and education.",
    "Health care clinics and doctors and nurses treat patients daily.",
    "",  # empty doc must survive as an empty token list
    "URLs http://example.com and <a href>markup</a> stay literal here.",
    "MiXeD CaSe Words Should Lowercase By Default.",
]


def test_batch_is_byte_identical_to_per_doc():
    for kw in (
        dict(),
        dict(stopwords=STOP),
        dict(stopwords=STOP, min_length=3),
        dict(lowercase=False),
        dict(min_length=4, stopwords=STOP, lowercase=False),
    ):
        per_doc = [topica.tokenize(t, **kw) for t in TEXTS]
        batch = tokenize_many(TEXTS, **kw)
        assert batch == per_doc, kw


def test_batch_handles_empty_and_single():
    assert tokenize_many([]) == []
    assert tokenize_many([""]) == [[]]
    assert tokenize_many(["one two three"]) == [topica.tokenize("one two three")]


def test_batch_matches_across_chunk_boundary():
    # More docs than one chunk (chunk floor is 4096) to exercise the chunked
    # parallel loop and its accumulation order.
    big = [f"word{i % 7} shared common term" for i in range(9000)]
    assert tokenize_many(big, stopwords=STOP) == [
        topica.tokenize(t, stopwords=STOP) for t in big
    ]


def test_batch_reports_progress_reaching_total():
    seen = []
    tokenize_many(
        TEXTS * 3,
        stopwords=STOP,
        progress=lambda done, total, info: seen.append((done, total)),
    )
    assert seen, "progress callback never fired"
    assert seen[-1][0] == seen[-1][1] == len(TEXTS) * 3  # ends at 100%


def _df():
    return pd.DataFrame(
        {
            "text": ["the senate passed a budget bill today"] * 30
            + ["health care clinic doctor patient nurse"] * 30,
            "party": ["D"] * 30 + ["R"] * 30,
        }
    )


def test_from_dataframe_matches_explicit_per_doc_tokenizer():
    # The default (batch) path must build exactly the corpus an explicit per-doc
    # tokenizer would, so nothing about pruning / alignment changes.
    df = _df()
    sw = list(STOP)
    fast = topica.from_dataframe(df, text_col="text", stopwords=sw, min_doc_freq=2)
    slow = topica.from_dataframe(
        df,
        text_col="text",
        tokenizer=lambda s: topica.tokenize(s, stopwords=sw),
        min_doc_freq=2,
    )
    assert fast.num_docs == slow.num_docs
    assert list(fast.vocabulary) == list(slow.vocabulary)
    assert fast.kept_indices == slow.kept_indices


def test_from_dataframe_verbose_false_is_silent(capfd):
    topica.from_dataframe(_df(), text_col="text", verbose=False, min_doc_freq=2)
    assert capfd.readouterr().err == ""


def test_from_dataframe_default_is_silent_off_tty(capfd):
    # capfd's stderr is not a tty, so verbose=None must not print.
    topica.from_dataframe(_df(), text_col="text", min_doc_freq=2)
    assert capfd.readouterr().err == ""


def test_custom_tokenizer_progress_is_throttled(capfd):
    # The custom-tokenizer path can't parallelize, but it must still throttle its
    # progress frames (~200 max), not emit one per document, or a redirected
    # stderr on a large corpus collects megabytes of bar frames (#786).
    big = pd.DataFrame({"text": ["the senate passed a budget bill today"] * 20000})
    topica.from_dataframe(
        big, text_col="text", tokenizer=lambda s: s.split(), verbose=True
    )
    err = capfd.readouterr().err
    assert err.count("\r") <= 205  # not ~20000
    assert "100%" in err and err.endswith("\n")  # still completes and closes
