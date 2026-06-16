"""LLM topic labeling as plumbing: prompt assembly, the BYO-callable path, and
the optional `llm` backend. No network or real model is used; the model is a
deterministic stub callable."""

import importlib.util

import numpy as np
import pytest

import topica


@pytest.fixture(scope="module")
def model_and_texts():
    pets = ["cat dog pet cat dog fur"]
    sky = ["star moon sky star moon night"]
    docs = [d.split() for d in (pets * 20 + sky * 20)]
    texts = pets * 20 + sky * 20
    m = topica.LDA(num_topics=2, seed=1)
    m.fit(docs, iters=300)
    return m, texts


def test_prompts_carry_words_and_docs(model_and_texts):
    m, texts = model_and_texts
    prompts = topica.topic_label_prompts(m, texts, n_words=5, n_docs=2)
    assert len(prompts) == 2
    # Every topic's prompt names its top words and shows representative documents.
    joined = " ".join(prompts).lower()
    assert "top words:" in joined
    assert "representative documents:" in joined
    # The two themes' words appear somewhere across the prompts.
    assert "cat" in joined and "moon" in joined


def test_prompts_without_texts_have_no_doc_block(model_and_texts):
    m, _ = model_and_texts
    prompts = topica.topic_label_prompts(m, n_words=5)
    assert all("Representative documents" not in p for p in prompts)
    assert all("Top words:" in p for p in prompts)


def test_byo_callable_labels_each_topic(model_and_texts):
    m, texts = model_and_texts
    calls = []

    def fake(prompt):
        calls.append(prompt)
        # A deterministic "label": the first top word after the "Top words:" line.
        line = next(ln for ln in prompt.splitlines() if ln.startswith("Top words:"))
        return line.split(":", 1)[1].split(",")[0].strip()

    labels = topica.llm_topic_labels(m, texts, backend=fake, n_words=4)
    assert len(labels) == 2 == len(calls)
    assert all(isinstance(x, str) and x for x in labels)


def test_set_labels_flows_into_topic_info(model_and_texts):
    m, texts = model_and_texts
    labels = topica.llm_topic_labels(
        m, texts, backend=lambda p: "THEME", set_labels=True
    )
    assert labels == ["THEME", "THEME"]
    info = topica.topic_info(m)
    assert all(row["label"] == "THEME" for row in info if row["topic"] >= 0)
    assert topica.topic_labels(m) == ["THEME", "THEME"]
    # reset so the module-scoped model does not leak custom labels to other tests
    topica.set_topic_labels(m, {})


def test_instructions_override(model_and_texts):
    m, _ = model_and_texts
    prompts = topica.topic_label_prompts(m, instructions="LABEL THIS THING")
    assert all(p.startswith("LABEL THIS THING") for p in prompts)


def test_llm_backend_requires_llm():
    # When the optional `llm` package is absent, llm_backend raises a clear error;
    # when present we cannot call a real model in tests, so just check it builds.
    if importlib.util.find_spec("llm") is None:
        with pytest.raises(ImportError, match="llm"):
            topica.llm_backend("gpt-4o-mini")
    else:
        assert callable(topica.llm_backend("gpt-4o-mini"))


# --- llm_embed: embeddings via the llm library --------------------------------


def test_llm_embed_with_a_fake_llm(monkeypatch):
    import sys
    import types

    class _FakeEmbModel:
        def embed_multi(self, items):
            return ([float(len(t)), 1.0, 2.0] for t in items)

        def embed(self, t):
            return [float(len(t)), 1.0, 2.0]

    fake = types.ModuleType("llm")
    fake.get_embedding_model = lambda name: _FakeEmbModel()
    monkeypatch.setitem(sys.modules, "llm", fake)

    texts = ["aa", "bbbb", "c"]
    arr = topica.llm_embed(texts, model="whatever")
    assert arr.shape == (3, 3)
    assert list(arr[:, 0]) == [2.0, 4.0, 1.0]  # encodes len(text) in our fake
    # batch=False path takes the same shape.
    arr2 = topica.llm_embed(texts, model="whatever", batch=False)
    assert arr2.shape == (3, 3)


def test_llm_embed_requires_llm():
    if importlib.util.find_spec("llm") is None:
        with pytest.raises(ImportError, match="llm"):
            topica.llm_embed(["a", "b"])
    else:
        pytest.skip("llm is installed; cannot exercise the missing-package path")


# --- save/load embeddings and the llm_embed cache -----------------------------


def test_save_load_embeddings_roundtrip(tmp_path):
    emb = np.arange(12, dtype=float).reshape(4, 3)
    p = topica.save_embeddings(tmp_path / "emb", emb, texts=["a", "b", "c", "d"], model="m1")
    assert p.endswith(".npz")
    loaded = topica.load_embeddings(tmp_path / "emb")  # suffix added automatically
    assert np.array_equal(loaded, emb)
    arr, meta = topica.load_embeddings(p, with_meta=True)
    assert np.array_equal(arr, emb)
    assert meta["model"] == "m1"
    assert "texts_hash" in meta


def _counting_fake_llm(monkeypatch):
    import sys
    import types

    calls = {"n": 0}

    class _Emb:
        def embed_multi(self, items):
            return ([float(len(t)), 0.5] for t in items)

    fake = types.ModuleType("llm")

    def get_embedding_model(name):
        calls["n"] += 1
        return _Emb()

    fake.get_embedding_model = get_embedding_model
    monkeypatch.setitem(sys.modules, "llm", fake)
    return calls


def test_llm_embed_cache_embeds_once(tmp_path, monkeypatch):
    calls = _counting_fake_llm(monkeypatch)
    texts = ["alpha", "beta", "gamma"]
    cache = tmp_path / "cache"

    a = topica.llm_embed(texts, model="x", cache=cache)
    assert calls["n"] == 1
    assert (tmp_path / "cache.npz").exists()

    # Second call with the same texts loads from cache; no model is called.
    b = topica.llm_embed(texts, model="x", cache=cache)
    assert calls["n"] == 1
    assert np.array_equal(a, b)

    # Different texts miss the cache and recompute.
    topica.llm_embed(["alpha", "beta", "delta"], model="x", cache=cache)
    assert calls["n"] == 2


def test_llm_backend_passes_explicit_key(monkeypatch):
    import sys
    import types

    seen = {}

    class _Resp:
        def text(self):
            return "LABEL"

    class _Model:
        def __init__(self):
            self.key = None

        def prompt(self, prompt, **kw):
            seen["key_at_call"] = self.key
            return _Resp()

    obj = _Model()
    fake = types.ModuleType("llm")
    fake.get_model = lambda name: obj
    monkeypatch.setitem(sys.modules, "llm", fake)

    # Explicit key is set on the model; default (None) leaves llm to resolve it.
    call = topica.llm_backend("gpt-4o-mini", key="sk-test-123")
    assert obj.key == "sk-test-123"
    assert call("hi") == "LABEL"
    assert seen["key_at_call"] == "sk-test-123"

    obj.key = None
    topica.llm_backend("gpt-4o-mini")  # no key -> untouched, llm resolves from env
    assert obj.key is None


def test_unknown_model_gives_actionable_error(monkeypatch):
    import sys
    import types

    class UnknownModelError(Exception):
        pass

    class _M:
        model_id = "gpt-4o-mini"

    fake = types.ModuleType("llm")
    fake.UnknownModelError = UnknownModelError

    def get_model(name):
        raise UnknownModelError(f"Unknown model: {name}")

    fake.get_model = get_model
    fake.get_models = lambda: [_M()]
    fake.get_embedding_models = lambda: [_M()]
    fake.get_embedding_model = lambda name: (_ for _ in ()).throw(UnknownModelError(name))
    monkeypatch.setitem(sys.modules, "llm", fake)

    with pytest.raises(ValueError) as exc:
        topica.llm_backend("gemma4:e2b")
    msg = str(exc.value)
    assert "gemma4:e2b" in msg and "ollama" in msg and "OPENAI_API_KEY" in msg

    with pytest.raises(ValueError) as exc2:
        topica.llm_embed(["a"], model="mystery-embedder")
    assert "mystery-embedder" in str(exc2.value) and "sentence-transformers" in str(exc2.value)
