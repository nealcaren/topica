"""Documentation-coverage lint for the PyO3 bindings (issues #267, #271).

Every parameter in a ``#[pyo3(signature = (...))]`` (constructors and methods
alike) must be named in the preceding ``///`` doc-comment, so the runtime
``help()`` text and the mkdocstrings API site never silently drift behind the
signatures again. This is the tested-artifact guard #271 item 7 asks for: it
turns "a parameter has no documented meaning" from an invisible gap into a test
failure. Constructors are covered here precisely because the *runtime* can't see
them (PyO3 exposes ``__init__`` as ``*args, **kwargs``).
"""

import re
from pathlib import Path

MOD_RS = Path(__file__).resolve().parent.parent / "src" / "python" / "mod.rs"

# Receiver / framework params that never need documenting.
SKIP = {"self", "py", "data", "_py"}


def _undocumented_params():
    src = MOD_RS.read_text(encoding="utf-8").splitlines()
    n = len(src)
    findings = []
    i = 0
    while i < n:
        if "#[pyo3(signature" not in src[i]:
            i += 1
            continue
        # Gather the (possibly multi-line) signature attribute.
        j = i
        sig_lines = [src[i]]
        while ")]" not in src[j]:
            j += 1
            sig_lines.append(src[j])
        sig_text = " ".join(sig_lines)
        m = re.search(r"signature\s*=\s*\((.*)\)\s*\)\]", sig_text, re.S)
        params = []
        if m:
            for tok in m.group(1).split(","):
                tok = tok.strip()
                if not tok or tok.startswith("*"):
                    continue
                name = tok.split("=")[0].strip().lstrip("*").strip()
                if name and name not in SKIP:
                    params.append(name)
        # Walk back over attribute / blank lines to the doc-comment block.
        k = i - 1
        while k >= 0 and (src[k].strip().startswith("#[") or src[k].strip() == ""):
            k -= 1
        doc = []
        while k >= 0 and src[k].strip().startswith("///"):
            doc.append(src[k].strip()[3:])
            k -= 1
        doc_text = "\n".join(reversed(doc))
        fn = "?"
        for look in range(j, min(j + 6, n)):
            fm = re.search(r"fn\s+(\w+)", src[look])
            if fm:
                fn = fm.group(1)
                break
        missing = [p for p in params
                   if not re.search(rf"(?<![\w]){re.escape(p)}(?![\w])", doc_text)]
        if missing:
            findings.append((i + 1, fn, missing))
        i = j + 1
    return findings


def test_every_binding_param_is_documented():
    assert MOD_RS.exists(), f"bindings not found at {MOD_RS}"
    findings = _undocumented_params()
    if findings:
        lines = [f"  src/python/mod.rs:{ln}  {fn}(): {', '.join(ps)}"
                 for ln, fn, ps in findings]
        raise AssertionError(
            "Undocumented binding parameters (add them to the `///` docstring):\n"
            + "\n".join(lines)
        )
