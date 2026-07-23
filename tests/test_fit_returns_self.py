"""#402 — `fit()` returns `self`, matching the estimator contract (`-> Self`).

Every model's `fit` returns the bound model so calls chain
(`topica.LDA(3, seed=1).fit(docs)`), and the returned object *is* the same
instance (not a copy). Callers that ignore the return value are unaffected.
"""

from __future__ import annotations

import pytest

import topica

topica.enable_experimental()

_DOCS = [
    ["a", "b", "c", "d"],
    ["b", "c", "d", "e"],
    ["a", "d", "e", "f"],
    ["c", "e", "f", "a"],
    ["b", "f", "a", "d"],
] * 4

# (name, construct, fit) — fit(model) calls fit on _DOCS with a light budget and
# returns whatever fit returned. Covers a Gibbs, variational, neural, matrix,
# short-text, nonparametric, and guided model plus a covariate model.
_CASES = {
    "LDA": (lambda: topica.LDA(3), lambda m: m.fit(_DOCS, iters=5)),
    "CTM": (lambda: topica.CTM(3), lambda m: m.fit(_DOCS)),
    "NMF": (lambda: topica.NMF(3), lambda m: m.fit(_DOCS)),
    "LSA": (lambda: topica.LSA(3), lambda m: m.fit(_DOCS)),
    "HDP": (lambda: topica.HDP(), lambda m: m.fit(_DOCS, iters=5)),
    "GSDMM": (lambda: topica.GSDMM(3), lambda m: m.fit(_DOCS, iters=5)),
    "BTM": (lambda: topica.BTM(3), lambda m: m.fit(_DOCS, iters=5)),
    "ProdLDA": (lambda: topica.ProdLDA(3), lambda m: m.fit(_DOCS)),
    "SeededLDA": (
        lambda: topica.SeededLDA({"t0": ["a", "b"], "t1": ["e", "f"]}),
        lambda m: m.fit(_DOCS, iters=5),
    ),
    "LabeledLDA": (
        lambda: topica.LabeledLDA(),
        lambda m: m.fit(_DOCS, [["x"], ["y"], ["x"], ["y"], ["x"]] * 4, iters=5),
    ),
}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_fit_returns_same_instance(name):
    construct, do_fit = _CASES[name]
    model = construct()
    returned = do_fit(model)
    assert returned is model, f"{name}.fit() returned {returned!r}, not self"


def test_chaining_one_liner():
    model = topica.LDA(3, seed=1).fit(_DOCS, iters=5)
    assert model is not None
    assert model.topic_word.shape[0] == 3


def test_stub_fit_returns_not_none():
    """The .pyi must not regress `fit` back to `-> None` while the runtime chains."""
    import ast
    import pathlib

    stub = pathlib.Path(__file__).parent.parent / "python" / "topica" / "_topica.pyi"
    tree = ast.parse(stub.read_text())
    offenders = []
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        for m in cls.body:
            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name == "fit":
                r = m.returns
                if isinstance(r, ast.Constant) and r.value is None:
                    offenders.append(cls.name)
    assert not offenders, f"fit stubs still typed -> None: {offenders}"
