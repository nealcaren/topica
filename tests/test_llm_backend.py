"""The provider-dispatched LLM backends: topica.llm_backend (chat) and
topica.llm_embed (embeddings) pick a lightweight SDK from the model name. No
network is used; fake `openai`, `anthropic`, and `google.genai` modules stand in
for the clients so we exercise dispatch, message/config assembly, key handling,
the base_url OpenAI-compatible path, and the embedding cache."""

import sys
import types

import numpy as np
import pytest

import topica
from topica import labeling


# --- fake SDK modules ---------------------------------------------------------


def _fake_openai(cap):
    class _Msg:
        def __init__(self, content):
            self.message = types.SimpleNamespace(content=content)

    class _Chat:
        def create(self, *, model, messages, **opts):
            cap["openai"] = {"model": model, "messages": messages, "opts": opts}
            user = [m for m in messages if m["role"] == "user"][-1]["content"]
            return types.SimpleNamespace(choices=[_Msg(user.splitlines()[0])])

    class _Emb:
        def create(self, *, model, input):
            cap["openai_embed"] = {"model": model, "input": list(input)}
            data = [types.SimpleNamespace(embedding=[float(len(t)), 1.0]) for t in input]
            return types.SimpleNamespace(data=data)

    class _Client:
        def __init__(self, *, base_url=None, api_key=None):
            cap["openai_init"] = {"base_url": base_url, "api_key": api_key}
            self.chat = types.SimpleNamespace(completions=_Chat())
            self.embeddings = _Emb()

    mod = types.ModuleType("openai")
    mod.OpenAI = _Client
    return mod


def _fake_anthropic(cap):
    class _Block:
        def __init__(self, text):
            self.type = "text"
            self.text = text

    class _Messages:
        def create(self, *, model, max_tokens, messages, system=None, **opts):
            cap["anthropic"] = {
                "model": model, "max_tokens": max_tokens, "system": system,
                "messages": messages, "opts": opts,
            }
            user = messages[-1]["content"]
            return types.SimpleNamespace(content=[_Block(user.splitlines()[0])])

    class _Client:
        def __init__(self, *, api_key=None):
            cap["anthropic_init"] = {"api_key": api_key}
            self.messages = _Messages()

    mod = types.ModuleType("anthropic")
    mod.Anthropic = _Client
    return mod


def _fake_genai(cap):
    # google.genai plus its .types submodule.
    genai = types.ModuleType("google.genai")
    gtypes = types.ModuleType("google.genai.types")

    class _Config:
        def __init__(self, *, system_instruction=None, **kw):
            self.system_instruction = system_instruction
            self.kw = kw

    class _Models:
        def generate_content(self, *, model, contents, config=None):
            cap["gemini"] = {
                "model": model, "contents": contents,
                "system": getattr(config, "system_instruction", None),
                "opts": getattr(config, "kw", {}),
            }
            return types.SimpleNamespace(text=str(contents).splitlines()[0])

        def embed_content(self, *, model, contents):
            cap["gemini_embed"] = {"model": model, "contents": list(contents)}
            embs = [types.SimpleNamespace(values=[float(len(t)), 2.0, 3.0]) for t in contents]
            return types.SimpleNamespace(embeddings=embs)

    class _Client:
        def __init__(self, *, api_key=None):
            cap["gemini_init"] = {"api_key": api_key}
            self.models = _Models()

    genai.Client = _Client
    gtypes.GenerateContentConfig = _Config
    google_pkg = sys.modules.get("google") or types.ModuleType("google")
    return {"google": google_pkg, "google.genai": genai, "google.genai.types": gtypes}


def _install(monkeypatch, cap):
    monkeypatch.setitem(sys.modules, "openai", _fake_openai(cap))
    monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic(cap))
    for name, mod in _fake_genai(cap).items():
        monkeypatch.setitem(sys.modules, name, mod)


# --- provider detection -------------------------------------------------------


@pytest.mark.parametrize("model,expected", [
    ("gpt-4o-mini", "openai"),
    ("o3-mini", "openai"),
    ("ft:gpt-4o:acme", "openai"),
    ("claude-3-5-haiku-latest", "anthropic"),
    ("anthropic/claude-3-5-sonnet", "anthropic"),
    ("gemini-2.5-flash", "gemini"),
    ("models/gemini-2.5-pro", "gemini"),
])
def test_detect_provider(model, expected):
    assert labeling._detect_provider(model, None) == expected


def test_base_url_forces_openai():
    assert labeling._detect_provider("claude-3-5-haiku-latest", "http://x/v1") == "openai"


# --- chat dispatch ------------------------------------------------------------


def test_openai_chat(monkeypatch):
    cap = {}
    _install(monkeypatch, cap)
    call = topica.llm_backend("gpt-4o-mini", key="sk-1", temperature=0)
    assert call("Label me:\nx") == "Label me:"
    assert cap["openai"]["opts"] == {"temperature": 0}
    assert cap["openai_init"]["api_key"] == "sk-1"
    assert [m["role"] for m in cap["openai"]["messages"]] == ["user"]


def test_openai_system_and_env_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    cap = {}
    _install(monkeypatch, cap)
    topica.llm_backend("gpt-4o-mini", system="Be terse.")("hi")
    assert cap["openai_init"]["api_key"] == "sk-env"
    assert cap["openai"]["messages"][0] == {"role": "system", "content": "Be terse."}


def test_base_url_local_placeholder_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cap = {}
    _install(monkeypatch, cap)
    topica.llm_backend("llama3", base_url="http://localhost:11434/v1")("hi")
    assert cap["openai_init"]["base_url"] == "http://localhost:11434/v1"
    assert cap["openai_init"]["api_key"] == "not-needed"


def test_anthropic_chat(monkeypatch):
    cap = {}
    _install(monkeypatch, cap)
    call = topica.llm_backend("claude-3-5-haiku-latest", key="sk-a",
                              system="Be terse.", max_tokens=64, temperature=0)
    assert call("Topic:\nmore") == "Topic:"
    a = cap["anthropic"]
    assert a["model"] == "claude-3-5-haiku-latest"
    assert a["max_tokens"] == 64          # pulled out of options
    assert a["system"] == "Be terse."
    assert a["opts"] == {"temperature": 0}
    assert cap["anthropic_init"]["api_key"] == "sk-a"


def test_anthropic_default_max_tokens(monkeypatch):
    cap = {}
    _install(monkeypatch, cap)
    topica.llm_backend("claude-3-5-haiku-latest")("hi")
    assert cap["anthropic"]["max_tokens"] == 1024


def test_gemini_chat(monkeypatch):
    cap = {}
    _install(monkeypatch, cap)
    call = topica.llm_backend("gemini-2.5-flash", key="sk-g",
                              system="Be terse.", temperature=0)
    assert call("Theme:\nrest") == "Theme:"
    g = cap["gemini"]
    assert g["model"] == "gemini-2.5-flash"
    assert g["system"] == "Be terse."
    assert g["opts"] == {"temperature": 0}
    assert cap["gemini_init"]["api_key"] == "sk-g"


def test_provider_override(monkeypatch):
    # A base_url-less custom name forced onto anthropic.
    cap = {}
    _install(monkeypatch, cap)
    topica.llm_backend("my-tuned-claude", provider="anthropic")("hi")
    assert "anthropic" in cap


def test_backend_drives_llm_topic_labels(monkeypatch):
    cap = {}
    _install(monkeypatch, cap)
    docs = [d.split() for d in (["cat dog pet"] * 20 + ["star moon sky"] * 20)]
    m = topica.LDA(num_topics=2, seed=1)
    m.fit(docs, iters=200)
    labels = topica.llm_topic_labels(m, backend=topica.llm_backend("claude-3-5-haiku-latest"))
    assert len(labels) == 2 and all(isinstance(x, str) and x for x in labels)


# --- embeddings dispatch ------------------------------------------------------


def test_openai_embed(monkeypatch):
    cap = {}
    _install(monkeypatch, cap)
    arr = topica.llm_embed(["aa", "bbbb", "c"], model="text-embedding-3-small")
    assert arr.shape == (3, 2)
    assert list(arr[:, 0]) == [2.0, 4.0, 1.0]
    assert cap["openai_embed"]["model"] == "text-embedding-3-small"


def test_gemini_embed(monkeypatch):
    cap = {}
    _install(monkeypatch, cap)
    arr = topica.llm_embed(["aa", "bbbb"], model="gemini-embedding-001")
    assert arr.shape == (2, 3)
    assert cap["gemini_embed"]["model"] == "gemini-embedding-001"


def test_embed_cache_embeds_once(tmp_path, monkeypatch):
    cap = {}
    _install(monkeypatch, cap)
    calls = {"n": 0}
    fake = sys.modules["openai"]
    orig = fake.OpenAI

    class _Counting(orig):
        def __init__(self, **kw):
            super().__init__(**kw)
            inner = self.embeddings.create

            def create(**kwargs):
                calls["n"] += 1
                return inner(**kwargs)

            self.embeddings.create = create

    monkeypatch.setattr(fake, "OpenAI", _Counting)
    cache = tmp_path / "cache"
    a = topica.llm_embed(["alpha", "beta"], model="text-embedding-3-small", cache=cache)
    assert calls["n"] == 1 and (tmp_path / "cache.npz").exists()
    b = topica.llm_embed(["alpha", "beta"], model="text-embedding-3-small", cache=cache)
    assert calls["n"] == 1  # served from cache
    assert np.array_equal(a, b)


# --- missing-package errors ---------------------------------------------------


def test_chat_missing_sdk_message(monkeypatch):
    # None in sys.modules makes import raise ImportError, simulating absence.
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(ImportError, match="anthropic"):
        topica.llm_backend("claude-3-5-haiku-latest")


def test_embed_missing_sdk_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(ImportError, match="openai"):
        topica.llm_embed(["a", "b"], model="text-embedding-3-small")
