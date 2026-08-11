"""Coverage gate (issue #271, "Wave 0").

Every model in ``topica.list_models()`` must have *some* form of validity
coverage. A model is covered if it is one of:

  (a) present in ``FIT_ADAPTERS`` (the registry-driven invariant suite fits it),
  (b) shipped with a committed gold fixture ``parity/<lower>_gold.npz``, or
  (c) listed in the explicit, documented ``INVARIANT_EXEMPT`` set.

So a newly added model with no validity coverage FAILS this test, and the only
way to pass is to add an adapter, a gold fixture, or a documented exemption with
a reason. This is the structural guard that the suite cannot silently fall behind
the model roster.
"""
from __future__ import annotations

from pathlib import Path

import topica
from test_model_invariants import FIT_ADAPTERS

# Models that cannot be exercised by a reference-free fit and are deliberately
# left uncovered, each with a documented reason.
INVARIANT_EXEMPT = {
    "TopicGPT": "needs an LLM / external API",
    "Wordfish": "a pure ideal-point scaler with no topic/doc-topic distribution; "
    "the topic-health invariants do not apply (covered by tests/test_wordfish.py)",
    "Wordshoal": "a two-stage ideal-point scaler with no topic/doc-topic "
    "distribution; the topic-health invariants do not apply (covered by "
    "tests/test_wordshoal.py and parity/wordshoal_r_compare.py)",
    "PartyEmbeddings": "a learned-embedding ideal-point scaler with no "
    "topic/doc-topic distribution; the topic-health invariants do not apply "
    "(covered by tests/test_party_embeddings.py and "
    "parity/party_embeddings_compare.py)",
}

_PARITY_DIR = Path(__file__).resolve().parents[1] / "parity"


def _has_gold_fixture(name: str) -> bool:
    return (_PARITY_DIR / f"{name.lower()}_gold.npz").exists()


def test_every_model_has_validity_coverage():
    offenders = []
    for info in topica.list_models():
        name = info.name
        covered = (
            name in FIT_ADAPTERS
            or _has_gold_fixture(name)
            or name in INVARIANT_EXEMPT
        )
        if not covered:
            offenders.append(name)

    assert not offenders, (
        "models with no validity coverage (add a FIT_ADAPTER, a "
        f"parity/<lower>_gold.npz, or an INVARIANT_EXEMPT entry): {offenders}"
    )


def test_no_stale_adapters_or_exemptions():
    """Adapters and exemptions must name real, registered models (no drift)."""
    registered = {m.name for m in topica.list_models()}
    stale_adapters = sorted(set(FIT_ADAPTERS) - registered)
    stale_exempt = sorted(set(INVARIANT_EXEMPT) - registered)
    assert not stale_adapters, f"FIT_ADAPTERS names not in registry: {stale_adapters}"
    assert not stale_exempt, f"INVARIANT_EXEMPT names not in registry: {stale_exempt}"


def test_exempt_models_have_reasons():
    for name, reason in INVARIANT_EXEMPT.items():
        assert isinstance(reason, str) and reason.strip(), (
            f"INVARIANT_EXEMPT[{name!r}] must give a non-empty reason"
        )
