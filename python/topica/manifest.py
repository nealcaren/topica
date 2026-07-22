"""Analysis manifest (issue #394): a portable, privacy-aware record of one fit.

An :class:`AnalysisManifest` records the inputs and decisions needed to inspect
or replay a single fit, plus a verifier that reports *what actually holds* rather
than a single green check. It is deliberately **not** a second model
serialization: it composes with ``Corpus.save`` / ``model.save`` and references
them by path + digest rather than embedding them.

    record = topica.record_fit(model, corpus, prevalence=X, iters=1000)
    record.add_decision("K", "Chose 20 after inspecting 15/20/25.")
    record.save("analysis.topica.json")

    later = topica.AnalysisManifest.load("analysis.topica.json")
    print(later.verify(corpus, model).summary())

Scope: the manifest, the fingerprint-v1 contract below, ``record_fit`` for the
count-based and structural models, ``verify`` with a textual report, and (V2) an
HTML / Markdown *analysis card* via :meth:`AnalysisManifest.render` /
:meth:`~AnalysisManifest.to_markdown`. Generic diagnostic auto-capture and
manifest comparison remain future work.

Fingerprint-v1 contract ("fp1")
-------------------------------
A fingerprint is ``{"spec": "fp1", "algo": "blake2b-256", "digest": <hex>, ...}``.
The contract is frozen: a manifest carries ``spec`` forever, and a verifier that
does not recognise the spec reports ``unverifiable`` -- it never silently
reinterprets it.

- **Algorithm**: BLAKE2b, 32-byte digest, optional ``key`` for keyed/salted
  private verification (a keyed digest is NOT portable and is marked as such).
- **Encoding**: every field is domain-tagged and length-prefixed (u64
  little-endian counts), so no two distinct inputs share an encoding. Integers and
  array bytes are little-endian regardless of host.
- **Arrays**: cast to ``float64``, C-contiguous, little-endian; ``-0.0`` is
  normalised to ``0.0`` and every ``NaN`` to the canonical quiet-NaN
  ``0x7ff8000000000000`` before hashing (bit patterns must not leak).
- **Corpus**: hashed as token *strings* in document order (robust to vocabulary
  re-indexing). This is the sensitive content fingerprint -- opt-in only.

Privacy
-------
Default ``privacy="minimal"``: model + environment + coarse corpus counts
(document count, total tokens) only -- no length distribution, vocabulary size,
or preprocessing detail. ``privacy="aggregate"`` opts those in. A content
fingerprint is a separate opt-in (``content_fingerprint=True``) and is documented
as sensitive: a hash is not anonymisation, and a small or public corpus can be
brute-forced. ``privacy="full"`` (raw values) is intentionally not in V1.
"""
from __future__ import annotations

import hashlib
import json
import platform
import struct
from dataclasses import dataclass, field
from html import escape as _esc
from typing import Any

import numpy as np

SCHEMA = "topica.manifest"
SCHEMA_VERSION = 1
FINGERPRINT_SPEC = "fp1"
HASH_ALGORITHM = "blake2b-256"
BUNDLE_VERSION = 1
_DIGEST_SIZE = 32
_CANONICAL_NAN = 0x7FF8000000000000

# Model settings captured by the per-model adapters (allowlist, not
# serialize-everything). Everything else is omitted and marked as such.
_SETTINGS_ALLOWLIST: dict[str, tuple[str, ...]] = {
    "LDA": ("num_topics", "alpha", "beta"),
    "STM": ("num_topics", "beta", "sigma_prior"),
}


# --------------------------------------------------------------------------
# Fingerprint-v1 primitives
# --------------------------------------------------------------------------


class _Canon:
    """Incremental BLAKE2b over a length-prefixed, domain-tagged byte stream."""

    def __init__(self, key: bytes | None = None):
        self._h = hashlib.blake2b(digest_size=_DIGEST_SIZE, key=key or b"")

    def tag(self, name: bytes) -> "_Canon":
        self._h.update(name + b"\x00")
        return self

    def u64(self, n: int) -> "_Canon":
        self._h.update(struct.pack("<Q", n))
        return self

    def blob(self, b: bytes) -> "_Canon":
        self._h.update(struct.pack("<Q", len(b)))
        self._h.update(b)
        return self

    def text(self, s: str) -> "_Canon":
        return self.blob(s.encode("utf-8"))

    def hexdigest(self) -> str:
        return self._h.hexdigest()


def _canonical_array_bytes(arr: np.ndarray) -> bytes:
    """float64, C-contiguous, little-endian, with -0.0 and NaN normalised."""
    a = np.ascontiguousarray(arr, dtype="<f8").copy()
    # Normalise -0.0 -> 0.0 without touching other values.
    a[a == 0.0] = 0.0
    if a.size:
        nan_mask = np.isnan(a)
        if nan_mask.any():
            canon = np.frombuffer(struct.pack("<Q", _CANONICAL_NAN), dtype="<f8")[0]
            a[nan_mask] = canon
    return a.tobytes(order="C")


def _fp(digest: str, *, keyed: bool, **extra: Any) -> dict[str, Any]:
    fp = {"spec": FINGERPRINT_SPEC, "algo": HASH_ALGORITHM, "digest": digest}
    if keyed:
        # A keyed digest verifies only against the same key; it does not support
        # public replay. Flag it so a consumer never mistakes it for portable.
        fp["keyed"] = True
    fp.update(extra)
    return fp


def fingerprint_array(arr, *, key: bytes | None = None) -> dict[str, Any]:
    """Order-sensitive fingerprint of a numeric array (embeddings, vectors)."""
    a = np.asarray(arr)
    c = _Canon(key).tag(b"ndarray").u64(a.ndim)
    for dim in a.shape:
        c.u64(int(dim))
    c.blob(_canonical_array_bytes(a))
    return _fp(c.hexdigest(), keyed=key is not None, shape=list(a.shape))


def fingerprint_design(matrix, names=None, *, key: bytes | None = None) -> dict[str, Any]:
    """Fingerprint of a numeric design matrix, binding column names to values."""
    a = np.asarray(matrix)
    if a.ndim != 2:
        a = a.reshape(a.shape[0], -1)
    cols = list(names) if names is not None else [f"x{i}" for i in range(a.shape[1])]
    c = _Canon(key).tag(b"design").u64(a.shape[0]).u64(a.shape[1])
    for name in cols:
        c.text(str(name))
    c.blob(_canonical_array_bytes(a))
    return _fp(
        c.hexdigest(), keyed=key is not None, n_rows=a.shape[0], columns=cols
    )


def fingerprint_corpus(corpus, *, key: bytes | None = None) -> dict[str, Any]:
    """Sensitive, order-sensitive fingerprint of a corpus's token content.

    Hashes token strings in document order, so it survives a vocabulary
    re-indexing but changes if any document's tokens or the document order
    change. A hash is not anonymisation; treat this as sensitive.
    """
    docs = corpus.documents()
    c = _Canon(key).tag(b"corpus").u64(len(docs))
    for doc in docs:
        c.u64(len(doc))
        for tok in doc:
            c.text(tok)
    return _fp(c.hexdigest(), keyed=key is not None, num_docs=len(docs))


# --------------------------------------------------------------------------
# Environment (shared notion with the paper's provenance report)
# --------------------------------------------------------------------------


def _environment(thread_count: int | None = None) -> dict[str, Any]:
    import topica

    return {
        "topica_version": getattr(topica, "__version__", None),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "cpu_count": _cpu_count(),
        "thread_count": thread_count,
    }


def _cpu_count() -> int | None:
    import os

    try:
        return os.cpu_count()
    except Exception:  # pragma: no cover - platform dependent
        return None


# --------------------------------------------------------------------------
# The manifest
# --------------------------------------------------------------------------


@dataclass
class VerifyResult:
    """The outcome of :meth:`AnalysisManifest.verify`, per field.

    ``fields`` maps each checked field to one of: ``exact`` (recomputed value
    matches), ``input_changed``, ``artifact_changed``, ``environment_changed``
    (differs but the model's determinism class says it may still replay),
    ``unverifiable`` (nothing to check against, or a bounded/unknown component).
    ``ok`` is a convenience for "every field exact"; the summary always shows the
    full breakdown -- the statuses are never collapsed into one flag.
    """

    fields: dict[str, str]

    @property
    def ok(self) -> bool:
        return all(v == "exact" for v in self.fields.values())

    def summary(self) -> str:
        width = max((len(k) for k in self.fields), default=0)
        lines = [f"verify: {'all exact' if self.ok else 'differences found'}"]
        for name, status in self.fields.items():
            lines.append(f"  {name.ljust(width)}  {status}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"VerifyResult(ok={self.ok}, fields={self.fields})"


@dataclass
class ManifestDiff:
    """The outcome of :meth:`AnalysisManifest.compare`, per field.

    Compares two manifests directly (no corpus or model needed), so it answers
    "did these two runs differ, and where?". ``fields`` maps each field to
    ``same``, ``changed``, ``only_in_a`` / ``only_in_b`` (recorded in one but not
    the other), or ``incomparable`` (e.g. fingerprints from different specs, or a
    keyed digest). ``same`` is a convenience for "every field same"; the summary
    always shows the breakdown.
    """

    fields: dict[str, str]

    @property
    def same(self) -> bool:
        return all(v == "same" for v in self.fields.values())

    def summary(self) -> str:
        width = max((len(k) for k in self.fields), default=0)
        lines = [f"compare: {'identical' if self.same else 'differences found'}"]
        for name, status in self.fields.items():
            lines.append(f"  {name.ljust(width)}  {status}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ManifestDiff(same={self.same}, fields={self.fields})"


@dataclass
class AnalysisManifest:
    """A versioned, privacy-aware record of one fit. See the module docstring."""

    topica_version: str
    environment: dict[str, Any]
    model: dict[str, Any]
    corpus: dict[str, Any]
    inputs: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION
    created: str | None = None

    # -- authoring ---------------------------------------------------------

    def add_decision(self, key: str, note: str, *, author: str | None = None,
                     time: str | None = None) -> "AnalysisManifest":
        """Record a researcher-authored decision (K choice, labels, …).

        Always stored as visibly researcher-authored, never as a substantive
        conclusion the tool is vouching for.
        """
        self.decisions.append(
            {"key": key, "note": note, "author": author, "time": time})
        return self

    def add_diagnostic(self, name: str, value: Any) -> "AnalysisManifest":
        """Record a diagnostic as computed *evidence* (not a conclusion)."""
        self.diagnostics.append(
            {"name": name, "value": _jsonable(value), "kind": "computed_evidence"})
        return self

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "schema_version": self.schema_version,
            "created": self.created,
            "topica_version": self.topica_version,
            "fingerprint_spec": FINGERPRINT_SPEC,
            "hash_algorithm": HASH_ALGORITHM,
            "environment": self.environment,
            "model": self.model,
            "corpus": self.corpus,
            "inputs": self.inputs,
            "diagnostics": self.diagnostics,
            "decisions": self.decisions,
        }

    def to_json(self) -> str:
        # sort_keys => deterministic bytes for a given manifest.
        return json.dumps(self.to_dict(), indent=2, sort_keys=True,
                          ensure_ascii=False)

    def save(self, path: str) -> None:
        from pathlib import Path

        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AnalysisManifest":
        if d.get("schema") != SCHEMA:
            raise ValueError(f"not a topica manifest (schema={d.get('schema')!r})")
        ver = d.get("schema_version")
        if ver != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported manifest schema_version {ver!r} "
                f"(this build reads {SCHEMA_VERSION}); upgrade topica or migrate")
        return cls(
            topica_version=d["topica_version"],
            environment=d["environment"],
            model=d["model"],
            corpus=d["corpus"],
            inputs=d.get("inputs", {}),
            diagnostics=d.get("diagnostics", []),
            decisions=d.get("decisions", []),
            schema_version=ver,
            created=d.get("created"),
        )

    @classmethod
    def load(cls, path: str) -> "AnalysisManifest":
        from pathlib import Path

        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # -- verification ------------------------------------------------------

    def verify(self, corpus=None, model=None) -> VerifyResult:
        """Re-derive fingerprints from the supplied corpus/model and report,
        per field, whether the fit's identity and replay conditions still hold.

        Never collapses the fields into one boolean. Uses the recorded
        determinism class to decide whether an environment difference is
        potentially replayable rather than a mismatch.
        """
        fields: dict[str, str] = {}
        fields["environment"] = self._verify_environment()
        if corpus is not None:
            fields.update(self._verify_corpus(corpus))
        if model is not None:
            fields.update(self._verify_model(model))
        return VerifyResult(fields)

    def _verify_environment(self) -> str:
        now = _environment(self.environment.get("thread_count"))
        if now == self.environment:
            return "exact"
        determinism = self.model.get("determinism")
        threads_match = now.get("thread_count") == self.environment.get("thread_count")
        if determinism == "bit-exact":
            return "environment_changed"  # bit-exact: env change should not matter
        if determinism == "seed-reproducible" and threads_match:
            return "environment_changed"  # replayable under the same thread count
        return "unverifiable"

    def _verify_corpus(self, corpus) -> dict[str, str]:
        rec = self.corpus
        # Coarse counts are always present.
        counts_ok = (rec.get("num_docs") == corpus.num_docs
                     and rec.get("total_tokens") == corpus.total_tokens)
        out = {"corpus_counts": "exact" if counts_ok else "input_changed"}
        stored = rec.get("fingerprint")
        if stored is None:
            out["corpus_fingerprint"] = "unverifiable"  # not recorded (privacy)
        elif stored.get("spec") != FINGERPRINT_SPEC:
            out["corpus_fingerprint"] = "unverifiable"  # unknown spec, do not reinterpret
        else:
            key = None  # keyed fingerprints require the caller's key; unsupported here
            if stored.get("keyed"):
                out["corpus_fingerprint"] = "unverifiable"
            else:
                now = fingerprint_corpus(corpus, key=key)
                out["corpus_fingerprint"] = (
                    "exact" if now["digest"] == stored["digest"] else "input_changed")
        return out

    def _verify_model(self, model) -> dict[str, str]:
        stored = self.model.get("output_fingerprints", {})
        determinism = self.model.get("determinism")
        out: dict[str, str] = {}
        for name in ("topic_word", "doc_topic"):
            want = stored.get(name)
            have = getattr(model, name, None)
            if want is None or have is None:
                out[f"model_{name}"] = "unverifiable"
                continue
            if want.get("spec") != FINGERPRINT_SPEC:
                out[f"model_{name}"] = "unverifiable"
                continue
            now = fingerprint_array(have)
            if now["digest"] == want["digest"]:
                out[f"model_{name}"] = "exact"
            elif determinism == "llm-bounded":
                out[f"model_{name}"] = "unverifiable"  # bounded nondeterminism
            else:
                out[f"model_{name}"] = "artifact_changed"
        return out

    # -- comparison (V2): two manifests, no corpus/model needed ------------

    def compare(self, other: "AnalysisManifest") -> ManifestDiff:
        """Compare this manifest with ``other`` field by field.

        Uses only what each manifest recorded (fingerprints, counts, settings),
        so it needs neither corpus nor model. Answers "did these two runs use the
        same corpus / model / inputs, or did something change, and what?".
        """
        f: dict[str, str] = {}
        f["environment"] = _cmp_value(self.environment, other.environment)
        f["model_class"] = _cmp_value(self.model.get("class"), other.model.get("class"))
        f["num_topics"] = _cmp_value(self.model.get("num_topics"), other.model.get("num_topics"))
        f["model_settings"] = _cmp_value(self.model.get("settings"), other.model.get("settings"))
        f["fit_settings"] = _cmp_value(self.model.get("fit_settings"), other.model.get("fit_settings"))

        a_out = self.model.get("output_fingerprints", {})
        b_out = other.model.get("output_fingerprints", {})
        for name in sorted(set(a_out) | set(b_out)):
            f[f"model_{name}"] = _cmp_fp(a_out.get(name), b_out.get(name))

        f["corpus_counts"] = _cmp_value(
            (self.corpus.get("num_docs"), self.corpus.get("total_tokens")),
            (other.corpus.get("num_docs"), other.corpus.get("total_tokens")))
        f["corpus_fingerprint"] = _cmp_fp(
            self.corpus.get("fingerprint"), other.corpus.get("fingerprint"))

        for name in sorted(set(self.inputs) | set(other.inputs)):
            f[f"input_{name}"] = _cmp_fp(self.inputs.get(name), other.inputs.get(name))
        return ManifestDiff(f)

    # -- rendering (V2): a human-facing "analysis card" ---------------------

    def render(self, path: str | None = None, *, title: str | None = None,
               verification: "VerifyResult | None" = None) -> str:
        """Render the manifest as a self-contained HTML *analysis card*.

        Returns the HTML string, and writes it to ``path`` if given. The card
        shows only what the manifest actually recorded, so it cannot over-claim:
        researcher decisions are labelled as authored (not tool-verified),
        diagnostics as computed evidence, and fingerprints as verifiable identity
        rather than content. Pass a :class:`VerifyResult` (from :meth:`verify`) to
        include the graded verification table.
        """
        doc = _render_html(self, title=title, verification=verification)
        if path is not None:
            from pathlib import Path

            Path(path).write_text(doc, encoding="utf-8")
        return doc

    def to_markdown(self, *, verification: "VerifyResult | None" = None) -> str:
        """Render the manifest as a Markdown analysis card (Quarto/notebook)."""
        return _render_markdown(self, verification=verification)

    # -- content-addressed bundle (V2) ------------------------------------

    def bundle(self, path: str, *, model=None, corpus=None,
               include_model: bool = True, include_corpus: bool = False) -> str:
        """Write a self-contained, content-addressed bundle (a ``.zip``).

        The bundle packages this manifest with the saved artifacts as one
        shareable, self-verifying unit::

            manifest.json            this manifest, plus artifact references
            artifacts/<digest>.tt    model.save() / corpus.save(), named by hash

        Each artifact's file name is its BLAKE2b digest, and the manifest's
        ``model`` / ``corpus`` blocks gain an ``artifact`` reference, so
        :meth:`load_bundle` can prove nothing was corrupted or swapped.

        ``include_corpus`` is opt-in and **sensitive**: a bundled corpus embeds
        the raw tokens (a hash is not anonymisation). Pass the fitted ``model`` /
        ``corpus`` objects to bundle them (the manifest itself holds only
        fingerprints, not the objects). The *file* digest here proves "this exact
        saved file"; it is distinct from the corpus *content* fingerprint, which
        proves corpus identity across save-format versions.
        """
        import copy
        import zipfile

        d = copy.deepcopy(self.to_dict())
        d["bundle"] = {"version": BUNDLE_VERSION, "hash_algorithm": HASH_ALGORITHM}
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            if include_model:
                d["model"]["artifact"] = _add_artifact(zf, model, "model")
            if include_corpus:
                ref = _add_artifact(zf, corpus, "corpus")
                ref["sensitive"] = True
                d["corpus"]["artifact"] = ref
            zf.writestr("manifest.json",
                        json.dumps(d, indent=2, sort_keys=True, ensure_ascii=False))
        return path

    @classmethod
    def load_bundle(cls, path: str) -> "AnalysisManifest":
        """Load a bundle written by :meth:`bundle`, verifying artifact integrity.

        Recomputes each bundled artifact's digest and checks it against both the
        content-addressed file name and the manifest reference; a mismatch (a
        corrupt or tampered bundle) raises ``ValueError``. Returns the manifest;
        use :meth:`extract_bundle` to recover the artifacts for reloading.
        """
        import zipfile

        with zipfile.ZipFile(path) as zf:
            d = json.loads(zf.read("manifest.json").decode("utf-8"))
            version = d.get("bundle", {}).get("version")
            if version != BUNDLE_VERSION:
                raise ValueError(
                    f"unsupported bundle version {version!r} "
                    f"(this build reads {BUNDLE_VERSION})")
            for role in ("model", "corpus"):
                art = d.get(role, {}).get("artifact")
                if art:
                    _verify_artifact(zf.read(art["path"]), art)
        return cls.from_dict(d)

    @staticmethod
    def extract_bundle(path: str, dest_dir: str) -> dict:
        """Extract a bundle's artifacts to ``dest_dir`` (verifying integrity).

        Returns ``{"model": <path or None>, "corpus": <path or None>}``; reload
        each with the matching class, e.g. ``topica.LDA.load(paths["model"])``.
        """
        import zipfile
        from pathlib import Path

        out = {"model": None, "corpus": None}
        with zipfile.ZipFile(path) as zf:
            d = json.loads(zf.read("manifest.json").decode("utf-8"))
            for role in ("model", "corpus"):
                art = d.get(role, {}).get("artifact")
                if not art:
                    continue
                data = zf.read(art["path"])
                _verify_artifact(data, art)
                target = Path(dest_dir) / f"{role}.tt"
                target.write_bytes(data)
                out[role] = str(target)
        return out

    def __repr__(self) -> str:
        return (f"AnalysisManifest(model={self.model.get('class')!r}, "
                f"privacy={self.corpus.get('privacy')!r}, "
                f"decisions={len(self.decisions)}, "
                f"diagnostics={len(self.diagnostics)})")


# --------------------------------------------------------------------------
# comparison helpers
# --------------------------------------------------------------------------


def _cmp_value(a, b) -> str:
    a_has, b_has = a is not None, b is not None
    if not a_has and not b_has:
        return "same"
    if a_has and not b_has:
        return "only_in_a"
    if b_has and not a_has:
        return "only_in_b"
    return "same" if a == b else "changed"


def _hash_bytes(data: bytes) -> str:
    """BLAKE2b-256 of raw bytes -- the file digest used for bundle artifacts.

    Distinct from the fingerprint-v1 encoding (which is length-prefixed and
    domain-tagged for structured inputs); a saved artifact is just its bytes.
    """
    return hashlib.blake2b(data, digest_size=_DIGEST_SIZE).hexdigest()


def _save_to_bytes(obj, role: str) -> bytes:
    import os
    import tempfile

    if obj is None:
        raise ValueError(f"include_{role}=True requires the {role}= object to bundle it")
    fd, tmp = tempfile.mkstemp(suffix=".tt")
    os.close(fd)
    try:
        obj.save(tmp)
        with open(tmp, "rb") as fh:
            return fh.read()
    finally:
        os.unlink(tmp)


def _add_artifact(zf, obj, role: str) -> dict:
    data = _save_to_bytes(obj, role)
    digest = _hash_bytes(data)
    arc = f"artifacts/{digest}.tt"
    if arc not in zf.namelist():  # dedupe if model and corpus somehow coincide
        zf.writestr(arc, data)
    return {"path": arc, "digest": digest, "algo": HASH_ALGORITHM}


def _verify_artifact(data: bytes, art: dict) -> None:
    digest = _hash_bytes(data)
    if digest != art.get("digest") or not art.get("path", "").endswith(f"{digest}.tt"):
        raise ValueError(
            f"bundle artifact digest mismatch for {art.get('path')!r} "
            f"(corrupt or tampered bundle)")


def _cmp_fp(a: dict | None, b: dict | None) -> str:
    if a is None and b is None:
        return "same"
    if a is None:
        return "only_in_b"
    if b is None:
        return "only_in_a"
    # Different fingerprint specs, or a keyed digest, cannot be compared for
    # equality of content -- do not guess.
    if a.get("spec") != b.get("spec") or a.get("keyed") or b.get("keyed"):
        return "incomparable"
    return "same" if a.get("digest") == b.get("digest") else "changed"


# --------------------------------------------------------------------------
# record_fit
# --------------------------------------------------------------------------


def record_fit(model, corpus, *, prevalence=None, prevalence_names=None,
               privacy: str = "minimal", content_fingerprint: bool = False,
               fingerprint_key: bytes | None = None, thread_count: int | None = None,
               diagnostics=None, diagnostics_n: int = 10,
               **fit_settings) -> AnalysisManifest:
    """Record a fitted ``model`` on ``corpus`` as an :class:`AnalysisManifest`.

    The explicit call is deliberate: it cannot honestly see preprocessing done
    before it, so it records only what it can observe and marks the rest.

    Parameters
    ----------
    privacy : ``"minimal"`` (default) or ``"aggregate"``. ``"full"`` is not in V1.
    content_fingerprint : opt-in, **sensitive**. Adds an order-sensitive hash of
        the corpus tokens so ``verify`` can prove corpus identity. A hash is not
        anonymisation.
    fingerprint_key : optional key for keyed/salted fingerprints (private
        verification only; keyed digests are marked non-portable).
    diagnostics : optional list of built-in diagnostics to compute and record as
        evidence: any of :data:`BUILTIN_DIAGNOSTICS` (``"coherence"``,
        ``"exclusivity"``). Each is recorded as computed evidence (its mean over
        topics), or ``value=None`` with a note if it is not defined for this model.
    diagnostics_n : the top-N words each diagnostic uses (default 10).
    fit_settings : the fit arguments you passed (``iters=``, …), recorded as
        provenance. Only JSON-serialisable values are kept; others are dropped
        with a note.
    """
    if privacy not in ("minimal", "aggregate"):
        raise ValueError(
            f"privacy must be 'minimal' or 'aggregate' (got {privacy!r}); "
            f"'full' (raw values) is intentionally not available in V1")
    for name in diagnostics or ():
        if name not in BUILTIN_DIAGNOSTICS:
            raise ValueError(
                f"unknown diagnostic {name!r}; choose from {sorted(BUILTIN_DIAGNOSTICS)}")

    import topica

    cls = type(model).__name__
    determinism = None
    reg = getattr(topica, "registry", None)
    if reg is not None and cls in reg.REGISTRY:
        determinism = reg.REGISTRY[cls].determinism

    model_block: dict[str, Any] = {
        "class": cls,
        "determinism": determinism,
        "num_topics": getattr(model, "num_topics", None),
        "seed": getattr(model, "seed", None),  # most models do not expose it -> None
        "settings": _capture_settings(model, cls),
        "settings_coverage": f"adapter:{cls}" if cls in _SETTINGS_ALLOWLIST else "generic",
        "fit_settings": _allowlist_kwargs(fit_settings),
        "output_fingerprints": _output_fingerprints(model),
    }

    corpus_block = _corpus_block(corpus, privacy, content_fingerprint, fingerprint_key)

    inputs: dict[str, Any] = {}
    if prevalence is not None:
        inputs["prevalence"] = {
            "kind": "design_matrix",
            **fingerprint_design(prevalence, prevalence_names, key=fingerprint_key),
        }

    manifest = AnalysisManifest(
        topica_version=getattr(topica, "__version__", None),
        environment=_environment(thread_count),
        model=model_block,
        corpus=corpus_block,
        inputs=inputs,
    )
    for name in diagnostics or ():
        manifest.diagnostics.append(_capture_diagnostic(name, model, corpus, diagnostics_n))
    return manifest


# Built-in diagnostics record_fit can compute from the model alone (mean over
# topics). Both are the classic STM-style topic-quality metrics.
BUILTIN_DIAGNOSTICS = ("coherence", "exclusivity")


def _capture_diagnostic(name: str, model, corpus, n: int) -> dict[str, Any]:
    import numpy as _np
    import topica

    entry = {"name": name, "kind": "computed_evidence", "params": {"n": n}}
    try:
        if name == "coherence":
            value = float(_np.mean(model.coherence(n)))
        elif name == "exclusivity":
            value = float(_np.mean(topica.exclusivity(model, n=n)))
        else:  # pragma: no cover - guarded by BUILTIN_DIAGNOSTICS
            raise KeyError(name)
        entry["value"] = value
    except Exception as exc:
        # Honestly record the attempt and why it could not be computed rather
        # than silently dropping it (e.g. coherence on a cluster/neural model).
        entry["value"] = None
        entry["note"] = f"unavailable for this model: {type(exc).__name__}"
    return entry


def _capture_settings(model, cls: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in _SETTINGS_ALLOWLIST.get(cls, ()):
        val = getattr(model, name, None)
        if val is not None:
            out[name] = _jsonable(val)
    return out


def _output_fingerprints(model) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("topic_word", "doc_topic"):
        arr = getattr(model, name, None)
        if arr is not None:
            out[name] = fingerprint_array(arr)
    return out


def _corpus_block(corpus, privacy, content_fingerprint, key) -> dict[str, Any]:
    block: dict[str, Any] = {
        "privacy": privacy,
        "num_docs": corpus.num_docs,
        "total_tokens": corpus.total_tokens,
    }
    if privacy == "aggregate":
        lengths = np.asarray(corpus.doc_lengths, dtype=float)
        block["description"] = {
            "vocab_size": corpus.num_words,
            "doc_length_summary": _length_summary(lengths),
            "preprocessing": _preprocessing(corpus),
        }
    if content_fingerprint:
        block["fingerprint"] = fingerprint_corpus(corpus, key=key)
    return block


def _length_summary(lengths: np.ndarray) -> dict[str, Any]:
    if lengths.size == 0:
        return {"count": 0}
    return {
        "count": int(lengths.size),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
    }


def _preprocessing(corpus) -> dict[str, Any]:
    # Only parameters Topica itself applied and can observe. Preprocessing done
    # outside Topica is not visible here (a fingerprint proves identity, not
    # provenance).
    out: dict[str, Any] = {}
    for name in ("min_doc_freq", "max_doc_fraction", "min_cf", "rm_top"):
        val = getattr(corpus, name, None)
        if val is not None:
            out[name] = _jsonable(val)
    return out


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

_JSON_SCALARS = (str, int, float, bool, type(None))


def _allowlist_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep JSON-serialisable fit settings; drop the rest with a marker."""
    out: dict[str, Any] = {}
    dropped: list[str] = []
    for k, v in kwargs.items():
        j = _jsonable(v, _allow_containers=False)
        if j is _UNSERIALISABLE:
            dropped.append(k)
        else:
            out[k] = j
    if dropped:
        out["_dropped"] = sorted(dropped)
    return out


_UNSERIALISABLE = object()


def _jsonable(value: Any, _allow_containers: bool = True):
    if isinstance(value, _JSON_SCALARS):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist() if _allow_containers else _UNSERIALISABLE
    if _allow_containers and isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if _allow_containers and isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return _UNSERIALISABLE if not _allow_containers else str(value)


# --------------------------------------------------------------------------
# Rendering: the analysis card (V2)
# --------------------------------------------------------------------------

# Status -> (label, background colour). The verification table is colour-coded so
# the exact/changed/unverifiable distinction reads at a glance and is never
# collapsed into a single badge.
_STATUS_STYLE = {
    "exact": ("exact", "#e6f4ea", "#137333"),
    "input_changed": ("input changed", "#fce8e6", "#c5221f"),
    "artifact_changed": ("artifact changed", "#fce8e6", "#c5221f"),
    "environment_changed": ("environment changed", "#fef7e0", "#b06000"),
    "unverifiable": ("unverifiable", "#f1f3f4", "#5f6368"),
}


def _short(fp: dict | None) -> str:
    if not fp or "digest" not in fp:
        return "—"
    d = fp["digest"]
    keyed = " (keyed)" if fp.get("keyed") else ""
    return f"{d[:12]}…{keyed}"


def _kv_rows(pairs) -> str:
    out = []
    for k, v in pairs:
        if v is None or v == {} or v == []:
            continue
        out.append(
            f"<tr><th>{_esc(str(k))}</th><td>{_esc(str(v))}</td></tr>")
    return "".join(out)


def _render_html(m: "AnalysisManifest", *, title, verification) -> str:
    model = m.model
    corpus = m.corpus
    title = title or f"Analysis card — {model.get('class', 'model')}"
    parts: list[str] = []

    # Header.
    sub = f"K = {model.get('num_topics')} · topica {m.topica_version}"
    if m.created:
        sub += f" · {m.created}"
    parts.append(f"<h1>{_esc(title)}</h1><p class='sub'>{_esc(sub)}</p>")
    parts.append("<p class='note'>Generated from a topica analysis manifest. "
                 "It shows only what the fit recorded: fingerprints and settings, "
                 "not raw text.</p>")

    # Fit.
    fit_rows = _kv_rows([
        ("model", model.get("class")),
        ("topics (K)", model.get("num_topics")),
        ("determinism", model.get("determinism")),
        ("seed", model.get("seed")),
    ])
    fit_rows += _kv_rows(model.get("settings", {}).items())
    fit_rows += _kv_rows(model.get("fit_settings", {}).items())
    parts.append(f"<h2>Fit</h2><table>{fit_rows}</table>")

    # Corpus (privacy-aware: only what is recorded).
    c_rows = _kv_rows([
        ("privacy", corpus.get("privacy")),
        ("documents", corpus.get("num_docs")),
        ("total tokens", corpus.get("total_tokens")),
    ])
    desc = corpus.get("description")
    if desc:
        c_rows += _kv_rows([("vocabulary size", desc.get("vocab_size"))])
        length = desc.get("doc_length_summary", {})
        if length:
            c_rows += _kv_rows([("doc length (min/median/max)",
                                 f"{length.get('min')} / {length.get('median')} / {length.get('max')}")])
        c_rows += _kv_rows(desc.get("preprocessing", {}).items())
    if corpus.get("fingerprint"):
        c_rows += (f"<tr><th>content fingerprint</th><td class='fp'>"
                   f"{_esc(_short(corpus['fingerprint']))}</td></tr>")
    parts.append(f"<h2>Corpus</h2><table>{c_rows}</table>")

    # Inputs (design matrices etc.) as fingerprints.
    if m.inputs:
        rows = []
        for name, info in m.inputs.items():
            cols = ", ".join(info.get("columns", [])) if isinstance(info, dict) else ""
            rows.append(f"<tr><th>{_esc(name)}</th><td class='fp'>{_esc(_short(info))}</td>"
                        f"<td>{_esc(cols)}</td></tr>")
        parts.append("<h2>Inputs</h2><table>" + "".join(rows) + "</table>")

    # Model output fingerprints.
    ofp = model.get("output_fingerprints", {})
    if ofp:
        rows = []
        for name, fp in ofp.items():
            shape = "×".join(str(s) for s in fp.get("shape", []))
            rows.append(f"<tr><th>{_esc(name)}</th><td class='fp'>{_esc(_short(fp))}</td>"
                        f"<td>{_esc(shape)}</td></tr>")
        parts.append("<h2>Model outputs</h2>"
                     "<p class='note'>Fingerprints of the fitted matrices; "
                     "verifiable identity, not the values.</p>"
                     "<table>" + "".join(rows) + "</table>")

    # Researcher decisions -- prominent, and clearly authored not verified.
    if m.decisions:
        items = []
        for d in m.decisions:
            who = f" — {_esc(str(d['author']))}" if d.get("author") else ""
            items.append(f"<li><b>{_esc(str(d.get('key','')))}</b>: "
                         f"{_esc(str(d.get('note','')))}{who}</li>")
        parts.append("<h2>Researcher decisions</h2>"
                     "<p class='note'>Authored by the researcher; recorded, not "
                     "verified by topica.</p><ul>" + "".join(items) + "</ul>")

    # Diagnostics -- computed evidence.
    if m.diagnostics:
        rows = _kv_rows([(d.get("name"), d.get("value")) for d in m.diagnostics])
        parts.append("<h2>Computed evidence</h2>"
                     "<p class='note'>Diagnostics recorded as evidence, not as "
                     "substantive conclusions.</p><table>" + rows + "</table>")

    # Verification (only if supplied).
    if verification is not None:
        rows = []
        for fname, status in verification.fields.items():
            label, bg, fg = _STATUS_STYLE.get(status, (status, "#f1f3f4", "#5f6368"))
            rows.append(
                f"<tr><th>{_esc(fname)}</th>"
                f"<td><span class='pill' style='background:{bg};color:{fg}'>"
                f"{_esc(label)}</span></td></tr>")
        parts.append("<h2>Verification</h2>"
                     "<p class='note'>Re-derived from a supplied corpus and model. "
                     "Each field is graded on its own.</p>"
                     "<table>" + "".join(rows) + "</table>")

    # Environment.
    parts.append("<h2>Environment</h2><table>"
                 + _kv_rows(m.environment.items()) + "</table>")

    parts.append(f"<p class='foot'>manifest schema {m.schema_version} · "
                 f"fingerprints {FINGERPRINT_SPEC} ({HASH_ALGORITHM})</p>")

    style = (
        "body{font-family:system-ui,-apple-system,sans-serif;max-width:820px;"
        "margin:2rem auto;padding:0 1rem;color:#202124;line-height:1.5;}"
        "h1{margin-bottom:0;}h2{margin-top:1.8rem;border-bottom:1px solid #eee;"
        "padding-bottom:.2rem;}.sub{color:#5f6368;margin-top:.2rem;}"
        ".note{color:#5f6368;font-size:.9em;}"
        "table{border-collapse:collapse;width:100%;}"
        "th{text-align:left;padding:.25rem .6rem .25rem 0;color:#5f6368;"
        "font-weight:500;vertical-align:top;white-space:nowrap;}"
        "td{padding:.25rem 0;}.fp{font-family:ui-monospace,Menlo,monospace;}"
        ".pill{padding:.1rem .5rem;border-radius:1rem;font-size:.85em;}"
        ".foot{margin-top:2rem;color:#9aa0a6;font-size:.8em;}"
    )
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{_esc(title)}</title><style>{style}</style></head>"
            f"<body>{''.join(parts)}</body></html>")


def _render_markdown(m: "AnalysisManifest", *, verification) -> str:
    model = m.model
    corpus = m.corpus
    L = [f"# Analysis card — {model.get('class', 'model')}",
         "",
         f"K = {model.get('num_topics')} · topica {m.topica_version}"
         + (f" · {m.created}" if m.created else ""),
         "",
         "_Generated from a topica analysis manifest: fingerprints and settings, "
         "not raw text._", ""]

    L += ["## Fit", ""]
    for k, v in [("model", model.get("class")), ("topics (K)", model.get("num_topics")),
                 ("determinism", model.get("determinism")), ("seed", model.get("seed"))]:
        if v is not None:
            L.append(f"- **{k}**: {v}")
    for k, v in {**model.get("settings", {}), **model.get("fit_settings", {})}.items():
        L.append(f"- {k}: {v}")

    L += ["", "## Corpus", "",
          f"- privacy: {corpus.get('privacy')}",
          f"- documents: {corpus.get('num_docs')}",
          f"- total tokens: {corpus.get('total_tokens')}"]
    if corpus.get("fingerprint"):
        L.append(f"- content fingerprint: `{_short(corpus['fingerprint'])}`")

    ofp = model.get("output_fingerprints", {})
    if ofp:
        L += ["", "## Model outputs", ""]
        for name, fp in ofp.items():
            L.append(f"- {name}: `{_short(fp)}` ({'×'.join(str(s) for s in fp.get('shape', []))})")

    if m.decisions:
        L += ["", "## Researcher decisions", "",
              "_Authored by the researcher; recorded, not verified._", ""]
        for d in m.decisions:
            L.append(f"- **{d.get('key','')}**: {d.get('note','')}")

    if m.diagnostics:
        L += ["", "## Computed evidence", ""]
        for d in m.diagnostics:
            L.append(f"- {d.get('name')}: {d.get('value')}")

    if verification is not None:
        L += ["", "## Verification", ""]
        for fname, status in verification.fields.items():
            L.append(f"- {fname}: **{status}**")

    L += ["", f"_manifest schema {m.schema_version} · fingerprints "
          f"{FINGERPRINT_SPEC} ({HASH_ALGORITHM})_"]
    return "\n".join(L)
