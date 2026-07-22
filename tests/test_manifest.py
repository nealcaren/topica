"""Analysis manifest / provenance record (issue #394).

Covers the privacy contract (no leakage), deterministic serialization, the
fingerprint-v1 canonicalization, change detection in verify(), and schema
forward-compatibility.
"""
import json

import numpy as np
import pytest

import topica
from topica.manifest import (
    BUILTIN_DIAGNOSTICS,
    BUNDLE_VERSION,
    FINGERPRINT_SPEC,
    SCHEMA,
    SCHEMA_VERSION,
    AnalysisManifest,
    ManifestDiff,
    fingerprint_array,
    fingerprint_corpus,
    record_fit,
)

# Distinctive tokens that cannot collide with hyperparameter names (alpha/beta/…)
# recorded as legitimate model settings.
DOCS = [
    ["wordone", "wordtwo", "wordthree", "wordone"],
    ["wordone", "wordtwo", "wordtwo", "wordfour"],
    ["wordthree", "wordthree", "wordfour", "wordfive"],
    ["wordfour", "wordfive", "wordone", "wordtwo"],
    ["wordfive", "wordsix", "wordone", "wordthree"],
    ["wordtwo", "wordthree", "wordfour", "wordfive"],
]
X = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.5], [0.0, 0.0], [1.0, 0.2]])


@pytest.fixture
def fitted():
    corpus = topica.Corpus.from_documents(DOCS)
    model = topica.LDA(3, seed=7)
    model.fit(corpus, iters=50)
    return corpus, model


# -- privacy / leakage -----------------------------------------------------


def test_minimal_default_leaks_no_content(fitted):
    corpus, model = fitted
    rec = record_fit(model, corpus, prevalence=X, prevalence_names=["a", "b"], iters=50)
    js = rec.to_json()
    assert rec.corpus["privacy"] == "minimal"
    # Coarse counts only; no vocabulary, length distribution, preprocessing,
    # content fingerprint, or any raw token.
    assert set(rec.corpus) == {"privacy", "num_docs", "total_tokens"}
    for token in {t for doc in DOCS for t in doc}:
        assert token not in js
    for leaked in ("description", "vocab_size", "doc_length_summary", "fingerprint"):
        assert f'"{leaked}"' not in js
    # A design matrix is fingerprinted, not embedded: the prevalence entry is a
    # fingerprint dict (spec/algo/digest/columns/…), never the raw values.
    prev = rec.inputs["prevalence"]
    assert prev["spec"] == FINGERPRINT_SPEC and prev["digest"]
    assert set(prev) <= {"spec", "algo", "digest", "keyed", "kind", "n_rows", "columns"}


def test_no_raw_document_ids_or_metadata(fitted):
    corpus, model = fitted
    rec = record_fit(model, corpus, iters=50)
    # V1 records no metadata values, document IDs, or paths at any privacy level.
    assert "metadata" not in rec.corpus
    for block in (rec.corpus, rec.model, rec.inputs):
        assert "path" not in block and "doc_names" not in block


def test_aggregate_opts_in_description_but_still_no_tokens(fitted):
    corpus, model = fitted
    rec = record_fit(model, corpus, privacy="aggregate", iters=50)
    desc = rec.corpus["description"]
    assert desc["vocab_size"] == corpus.num_words
    assert desc["doc_length_summary"]["count"] == corpus.num_docs
    js = rec.to_json()
    for token in {t for doc in DOCS for t in doc}:
        assert token not in js


def test_full_privacy_rejected(fitted):
    corpus, model = fitted
    with pytest.raises(ValueError, match="not available in V1|minimal"):
        record_fit(model, corpus, privacy="full")


def test_content_fingerprint_is_opt_in(fitted):
    corpus, model = fitted
    assert record_fit(model, corpus).corpus.get("fingerprint") is None
    rec = record_fit(model, corpus, content_fingerprint=True)
    assert rec.corpus["fingerprint"]["spec"] == FINGERPRINT_SPEC


# -- deterministic serialization + round trip ------------------------------


def test_serialization_is_deterministic(fitted):
    corpus, model = fitted
    a = record_fit(model, corpus, iters=50).to_json()
    b = record_fit(model, corpus, iters=50).to_json()
    assert a == b  # no wall-clock, sorted keys


def test_round_trip(tmp_path, fitted):
    corpus, model = fitted
    rec = record_fit(model, corpus, prevalence=X, iters=50)
    rec.add_decision("K", "chose 3")
    rec.add_diagnostic("coh", float(np.mean(model.coherence(5))))
    p = tmp_path / "a.topica.json"
    rec.save(str(p))
    back = AnalysisManifest.load(str(p))
    assert back.to_json() == rec.to_json()
    assert back.decisions[0]["key"] == "K"
    assert back.diagnostics[0]["kind"] == "computed_evidence"


# -- fingerprint-v1 canonicalization ---------------------------------------


def test_fingerprint_array_order_sensitive():
    a = fingerprint_array(np.array([1.0, 2.0, 3.0]))
    b = fingerprint_array(np.array([3.0, 2.0, 1.0]))
    assert a["digest"] != b["digest"]
    assert a["spec"] == FINGERPRINT_SPEC and a["shape"] == [3]


def test_fingerprint_normalises_negative_zero_and_nan():
    assert (fingerprint_array(np.array([0.0, 1.0]))["digest"]
            == fingerprint_array(np.array([-0.0, 1.0]))["digest"])
    # Distinct NaN bit patterns must hash identically (no bit-pattern leak).
    n1 = np.array([np.nan], dtype="<f8")
    n2 = np.frombuffer(np.uint64(0x7FF8000000000001).tobytes(), dtype="<f8").copy()
    assert (fingerprint_array(n1)["digest"] == fingerprint_array(n2)["digest"])


def test_corpus_fingerprint_survives_reindex_changes_on_edit():
    c1 = topica.Corpus.from_documents(DOCS)
    c2 = topica.Corpus.from_documents(DOCS)  # same content
    assert fingerprint_corpus(c1)["digest"] == fingerprint_corpus(c2)["digest"]
    reordered = [DOCS[0][::-1]] + DOCS[1:]   # change one document's token order
    c3 = topica.Corpus.from_documents(reordered)
    assert fingerprint_corpus(c3)["digest"] != fingerprint_corpus(c1)["digest"]


def test_keyed_fingerprint_is_flagged_and_differs():
    plain = fingerprint_array(X)
    keyed = fingerprint_array(X, key=b"secret")
    assert "keyed" not in plain
    assert keyed["keyed"] is True
    assert keyed["digest"] != plain["digest"]


# -- verify: graded outcomes, change detection -----------------------------


def test_verify_same_inputs_matches(fitted):
    corpus, model = fitted
    rec = record_fit(model, corpus, content_fingerprint=True, iters=50)
    res = rec.verify(corpus, model)
    assert res.fields["corpus_counts"] == "exact"
    assert res.fields["corpus_fingerprint"] == "exact"
    assert res.fields["model_topic_word"] == "exact"
    assert res.fields["model_doc_topic"] == "exact"


def test_verify_detects_changed_corpus(fitted):
    corpus, model = fitted
    rec = record_fit(model, corpus, content_fingerprint=True, iters=50)
    changed = topica.Corpus.from_documents([DOCS[0][::-1]] + DOCS[1:])
    res = rec.verify(changed, model)
    assert res.fields["corpus_fingerprint"] == "input_changed"
    assert not res.ok


def test_verify_detects_changed_model(fitted):
    corpus, model = fitted
    rec = record_fit(model, corpus, iters=50)
    other = topica.LDA(3, seed=999)
    other.fit(corpus, iters=50)
    res = rec.verify(corpus, other)
    assert res.fields["model_topic_word"] == "artifact_changed"


def test_verify_unrecorded_fingerprint_is_unverifiable_not_pass(fitted):
    corpus, model = fitted
    rec = record_fit(model, corpus, iters=50)  # no content fingerprint
    res = rec.verify(corpus, model)
    # Absence of a recorded fingerprint must never read as a pass.
    assert res.fields["corpus_fingerprint"] == "unverifiable"


def test_verify_result_never_a_bare_bool(fitted):
    corpus, model = fitted
    res = record_fit(model, corpus, iters=50).verify(corpus, model)
    assert isinstance(res.fields, dict) and len(res.fields) >= 3
    assert "exact" in res.summary()


# -- schema + fingerprint-spec forward compatibility -----------------------


def test_unknown_schema_version_rejected(fitted):
    corpus, model = fitted
    d = record_fit(model, corpus).to_dict()
    d["schema_version"] = 999
    with pytest.raises(ValueError, match="schema_version"):
        AnalysisManifest.from_dict(d)


def test_non_topica_manifest_rejected():
    with pytest.raises(ValueError, match="not a topica manifest"):
        AnalysisManifest.from_dict({"schema": "something.else"})


def test_unknown_fingerprint_spec_is_unverifiable_not_reinterpreted(fitted):
    corpus, model = fitted
    rec = record_fit(model, corpus, content_fingerprint=True, iters=50)
    rec.corpus["fingerprint"]["spec"] = "fp99"  # a future spec this build can't read
    res = rec.verify(corpus, model)
    assert res.fields["corpus_fingerprint"] == "unverifiable"


def test_schema_constants_are_stable():
    # These are the durable contract; changing them is a breaking change.
    assert SCHEMA == "topica.manifest"
    assert SCHEMA_VERSION == 1
    assert FINGERPRINT_SPEC == "fp1"


def test_manifest_is_valid_json(fitted):
    corpus, model = fitted
    d = json.loads(record_fit(model, corpus, iters=50).to_json())
    assert d["schema"] == SCHEMA and d["fingerprint_spec"] == FINGERPRINT_SPEC


# -- content-addressed bundle (V2) -----------------------------------------


def test_bundle_layout_is_content_addressed(tmp_path, fitted):
    import json
    import zipfile

    corpus, model = fitted
    rec = record_fit(model, corpus, iters=50)
    p = tmp_path / "a.zip"
    rec.bundle(str(p), model=model)
    with zipfile.ZipFile(p) as zf:
        names = zf.namelist()
        assert "manifest.json" in names
        art = [n for n in names if n.startswith("artifacts/")]
        assert len(art) == 1
        d = json.loads(zf.read("manifest.json"))
        assert d["bundle"]["version"] == BUNDLE_VERSION
        ref = d["model"]["artifact"]
        # The artifact's file name is its digest.
        assert ref["path"] == f"artifacts/{ref['digest']}.tt"


def test_load_bundle_round_trips_and_verifies(tmp_path, fitted):
    corpus, model = fitted
    rec = record_fit(model, corpus, content_fingerprint=True, iters=50)
    p = tmp_path / "a.zip"
    rec.bundle(str(p), model=model, corpus=corpus, include_corpus=True)
    back = AnalysisManifest.load_bundle(str(p))
    assert back.model["class"] == "LDA"
    assert back.verify(corpus, model).ok


def test_bundle_detects_tampering(tmp_path, fitted):
    import zipfile

    corpus, model = fitted
    p = tmp_path / "a.zip"
    record_fit(model, corpus, iters=50).bundle(str(p), model=model)
    # Rewrite the zip with one artifact byte flipped.
    with zipfile.ZipFile(p) as zf:
        data = {n: zf.read(n) for n in zf.namelist()}
    art = next(n for n in data if n.startswith("artifacts/"))
    data[art] = data[art][:-1] + bytes([data[art][-1] ^ 1])
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(tampered, "w") as zf:
        for n, b in data.items():
            zf.writestr(n, b)
    with pytest.raises(ValueError, match="digest mismatch"):
        AnalysisManifest.load_bundle(str(tampered))


def test_bundle_corpus_is_opt_in_and_flagged_sensitive(tmp_path, fitted):
    corpus, model = fitted
    rec = record_fit(model, corpus, iters=50)
    default = tmp_path / "default.zip"
    rec.bundle(str(default), model=model)
    assert AnalysisManifest.load_bundle(str(default)).corpus.get("artifact") is None
    withc = tmp_path / "withc.zip"
    rec.bundle(str(withc), model=model, corpus=corpus, include_corpus=True)
    loaded = AnalysisManifest.load_bundle(str(withc))
    assert loaded.corpus["artifact"]["sensitive"] is True


def test_bundle_requires_the_object(tmp_path, fitted):
    corpus, model = fitted
    rec = record_fit(model, corpus, iters=50)
    with pytest.raises(ValueError, match="requires the model="):
        rec.bundle(str(tmp_path / "x.zip"))  # include_model default True, no model=


def test_extract_bundle_reloads_artifacts(tmp_path, fitted):
    corpus, model = fitted
    p = tmp_path / "a.zip"
    record_fit(model, corpus, iters=50).bundle(
        str(p), model=model, corpus=corpus, include_corpus=True)
    dest = tmp_path / "out"
    dest.mkdir()
    paths = AnalysisManifest.extract_bundle(str(p), str(dest))
    reloaded = topica.LDA.load(paths["model"])
    assert np.array_equal(reloaded.topic_word, model.topic_word)
    assert topica.Corpus.load(paths["corpus"]).num_docs == corpus.num_docs


def test_bundle_file_digest_distinct_from_content_fingerprint(tmp_path, fitted):
    import json
    import zipfile

    corpus, model = fitted
    rec = record_fit(model, corpus, content_fingerprint=True, iters=50)
    p = tmp_path / "a.zip"
    rec.bundle(str(p), model=model, corpus=corpus, include_corpus=True)
    with zipfile.ZipFile(p) as zf:
        d = json.loads(zf.read("manifest.json"))
    # The saved-file digest and the content fingerprint are different notions.
    assert d["corpus"]["artifact"]["digest"] != d["corpus"]["fingerprint"]["digest"]


def test_unknown_bundle_version_rejected(tmp_path, fitted):
    import json
    import zipfile

    corpus, model = fitted
    p = tmp_path / "a.zip"
    record_fit(model, corpus, iters=50).bundle(str(p), model=model)
    with zipfile.ZipFile(p) as zf:
        data = {n: zf.read(n) for n in zf.namelist()}
    d = json.loads(data["manifest.json"])
    d["bundle"]["version"] = 999
    data["manifest.json"] = json.dumps(d).encode("utf-8")
    future = tmp_path / "future.zip"
    with zipfile.ZipFile(future, "w") as zf:
        for n, b in data.items():
            zf.writestr(n, b)
    with pytest.raises(ValueError, match="bundle version"):
        AnalysisManifest.load_bundle(str(future))


# -- built-in diagnostic capture (V2) --------------------------------------


def test_record_fit_captures_builtin_diagnostics(fitted):
    corpus, model = fitted
    rec = record_fit(model, corpus, diagnostics=["coherence", "exclusivity"],
                     diagnostics_n=5, iters=50)
    names = {d["name"]: d for d in rec.diagnostics}
    assert set(names) == {"coherence", "exclusivity"}
    for d in rec.diagnostics:
        assert d["kind"] == "computed_evidence"
        assert d["params"] == {"n": 5}
        assert isinstance(d["value"], float)


def test_unknown_diagnostic_rejected(fitted):
    corpus, model = fitted
    with pytest.raises(ValueError, match="unknown diagnostic"):
        record_fit(model, corpus, diagnostics=["not_a_diagnostic"])


def test_builtin_diagnostics_is_stable():
    assert BUILTIN_DIAGNOSTICS == ("coherence", "exclusivity")


def test_diagnostic_unavailable_is_recorded_not_dropped(fitted):
    corpus, model = fitted

    class _NoCoherence:
        # A stand-in whose coherence() raises, like a cluster/neural model.
        num_topics = 3
        topic_word = model.topic_word
        doc_topic = model.doc_topic

        def coherence(self, n):
            raise NotImplementedError("no coherence for this model")

    rec = record_fit(_NoCoherence(), corpus, diagnostics=["coherence"], iters=50)
    entry = rec.diagnostics[0]
    assert entry["name"] == "coherence"
    assert entry["value"] is None
    assert "unavailable for this model" in entry["note"]


def test_manual_add_diagnostic_still_works(fitted):
    corpus, model = fitted
    rec = record_fit(model, corpus, iters=50)
    rec.add_diagnostic("custom", 0.42)
    assert rec.diagnostics[-1] == {"name": "custom", "value": 0.42,
                                   "kind": "computed_evidence"}


# -- comparison: two manifests (V2) ----------------------------------------


def test_compare_identical_runs(fitted):
    corpus, model = fitted
    a = record_fit(model, corpus, prevalence=X, content_fingerprint=True, iters=50)
    b = record_fit(model, corpus, prevalence=X, content_fingerprint=True, iters=50)
    diff = a.compare(b)
    assert isinstance(diff, ManifestDiff)
    assert diff.same
    assert all(v == "same" for v in diff.fields.values())


def test_compare_localizes_a_changed_model(fitted):
    corpus, model = fitted
    a = record_fit(model, corpus, iters=50)
    other = topica.LDA(4, seed=7)   # different K
    other.fit(corpus, iters=50)
    b = record_fit(other, corpus, iters=50)
    diff = a.compare(b)
    assert not diff.same
    assert diff.fields["num_topics"] == "changed"
    assert diff.fields["model_topic_word"] == "changed"
    assert diff.fields["corpus_counts"] == "same"     # same corpus


def test_compare_localizes_a_changed_input(fitted):
    corpus, model = fitted
    a = record_fit(model, corpus, prevalence=X, iters=50)
    b = record_fit(model, corpus, prevalence=X * 2 + 1, iters=50)  # different design matrix
    diff = a.compare(b)
    assert diff.fields["input_prevalence"] == "changed"
    assert diff.fields["model_topic_word"] == "same"


def test_compare_only_in_one(fitted):
    corpus, model = fitted
    with_fp = record_fit(model, corpus, content_fingerprint=True, iters=50)
    without = record_fit(model, corpus, iters=50)
    assert with_fp.compare(without).fields["corpus_fingerprint"] == "only_in_a"
    assert without.compare(with_fp).fields["corpus_fingerprint"] == "only_in_b"


def test_compare_incomparable_across_specs(fitted):
    corpus, model = fitted
    a = record_fit(model, corpus, content_fingerprint=True, iters=50)
    b = record_fit(model, corpus, content_fingerprint=True, iters=50)
    b.corpus["fingerprint"]["spec"] = "fp99"   # a future spec
    assert a.compare(b).fields["corpus_fingerprint"] == "incomparable"


def test_compare_survives_round_trip(tmp_path, fitted):
    corpus, model = fitted
    a = record_fit(model, corpus, prevalence=X, content_fingerprint=True, iters=50)
    p = tmp_path / "a.json"
    a.save(str(p))
    reloaded = AnalysisManifest.load(str(p))
    assert a.compare(reloaded).same


def test_compare_summary_is_not_a_bare_bool(fitted):
    corpus, model = fitted
    a = record_fit(model, corpus, iters=50)
    diff = a.compare(a)
    assert isinstance(diff.fields, dict) and "same" in diff.summary()


# -- rendering: the analysis card (V2) -------------------------------------


def test_render_html_is_self_contained(tmp_path, fitted):
    corpus, model = fitted
    rec = record_fit(model, corpus, iters=50)
    html = rec.render()
    assert html.startswith("<!doctype html>") and html.rstrip().endswith("</html>")
    # No external resources (CSP-free, offline).
    head = html.split("</style>")[0]
    assert "http://" not in head and "https://" not in head and "src=" not in html
    p = tmp_path / "card.html"
    rec.render(str(p))
    assert p.read_text(encoding="utf-8") == html


def test_render_escapes_user_text(fitted):
    corpus, model = fitted
    rec = record_fit(model, corpus, iters=50)
    rec.add_decision("x", "<script>alert(1)</script>")
    html = rec.render()
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_render_is_privacy_aware(fitted):
    corpus, model = fitted
    minimal = record_fit(model, corpus, iters=50).render()
    assert "content fingerprint" not in minimal
    assert "vocabulary size" not in minimal
    rich = record_fit(model, corpus, privacy="aggregate",
                      content_fingerprint=True, iters=50).render()
    assert "content fingerprint" in rich and "vocabulary size" in rich


def test_render_labels_decisions_as_authored(fitted):
    corpus, model = fitted
    rec = record_fit(model, corpus, iters=50)
    rec.add_decision("K", "chose 3")
    html = rec.render()
    assert "Researcher decisions" in html
    assert "recorded, not verified" in html  # never presented as tool-vouched


def test_render_includes_graded_verification(fitted):
    corpus, model = fitted
    rec = record_fit(model, corpus, content_fingerprint=True, iters=50)
    other = topica.LDA(3, seed=999)
    other.fit(corpus, iters=50)
    ver = rec.verify(corpus, other)
    html = rec.render(verification=ver)
    assert "Verification" in html
    # The graded statuses appear, not a single badge.
    assert "artifact changed" in html and "exact" in html


def test_to_markdown_has_sections_and_verification(fitted):
    corpus, model = fitted
    rec = record_fit(model, corpus, iters=50)
    rec.add_decision("K", "chose 3")
    md = rec.to_markdown(verification=rec.verify(corpus, model))
    for section in ("# Analysis card", "## Fit", "## Corpus", "## Researcher decisions",
                    "## Verification"):
        assert section in md
