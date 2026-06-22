"""Reference-free statistical-validity invariants for every topica model
(issue #271, "Wave 0").

The historical test suite checked *mechanics* — a fit ran, the shapes were
right, rows summed to one. None of that catches a *degenerate* fit. Issue #270
is the canonical example: a covariate keyATM collapsed every document's topic
proportions onto a single topic, yet every shape/sum test still passed because a
one-hot row is a perfectly valid simplex.

This module supplies the validity checks the suite was missing. They are
universal (no reference implementation, no gold fixture) and they are written so
that a *degenerate* fit FAILS:

- ``effective_topics`` measures how many topics actually carry mass.
- ``assert_healthy_theta`` rejects a θ where one topic hoovers up the mass or
  where the effective topic count has collapsed.
- the metamorphic helpers (``assert_more_iters_not_worse``,
  ``assert_seed_reproducible``) check relations *between* fits rather than a
  single fit in isolation, which is where #270 actually showed up (more
  iterations made the collapse worse, not better).

A self-test (:func:`_degenerate_theta_must_raise`, exercised as a real test in
``tests/test_model_invariants.py``) proves the assertions are not vacuous: a
deliberately collapsed θ must raise.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "effective_topics",
    "assert_finite",
    "assert_simplex",
    "assert_healthy_theta",
    "assert_more_iters_not_worse",
    "assert_seed_reproducible",
    "_degenerate_theta_must_raise",
]


def effective_topics(theta) -> float:
    """The effective number of topics carrying mass: ``exp(H(mean_d theta_d))``.

    ``theta`` is a ``(D, K)`` document-topic matrix. We average over documents to
    get the corpus-level topic distribution, then return the exponential of its
    Shannon entropy (the "perplexity" of the topic distribution). A perfectly
    flat distribution over ``K`` topics gives ``K``; a distribution collapsed
    onto one topic gives ``1``. This is the single number that #270 would have
    tripped on (it collapsed to ~1).
    """
    theta = np.asarray(theta, dtype=float)
    mean_t = theta.mean(axis=0)
    total = mean_t.sum()
    if total <= 0:
        raise AssertionError("mean theta has non-positive mass; cannot be a topic distribution")
    mean_t = mean_t / total
    # entropy with a tiny floor so log(0) is well-behaved for empty topics
    h = -(mean_t * np.log(mean_t + 1e-12)).sum()
    return float(np.exp(h))


def assert_finite(*arrays, model: str = "") -> None:
    """Assert every supplied array is all-finite (no NaN, no Inf)."""
    tag = f"[{model}] " if model else ""
    for i, a in enumerate(arrays):
        if a is None:
            continue
        arr = np.asarray(a, dtype=float)
        if not np.isfinite(arr).all():
            n_bad = int((~np.isfinite(arr)).sum())
            raise AssertionError(
                f"{tag}array #{i} has {n_bad} non-finite value(s) "
                f"(NaN or Inf) out of {arr.size}"
            )


def assert_simplex(matrix, axis: int = 1, *, atol: float = 1e-5, model: str = "") -> None:
    """Assert the rows (``axis=1``) or columns (``axis=0``) form valid simplices:
    non-negative and summing to one."""
    tag = f"[{model}] " if model else ""
    m = np.asarray(matrix, dtype=float)
    if (m < -atol).any():
        raise AssertionError(f"{tag}simplex has a negative entry (min={m.min():.3e})")
    sums = m.sum(axis=axis)
    if not np.allclose(sums, 1.0, atol=atol):
        worst = float(np.abs(sums - 1.0).max())
        raise AssertionError(
            f"{tag}simplex rows do not sum to 1 along axis {axis} "
            f"(max |sum - 1| = {worst:.3e})"
        )


def assert_healthy_theta(
    theta,
    n_topics: int,
    *,
    max_mass: float = 0.8,
    min_eff_frac: float = 0.25,
    model: str = "",
) -> None:
    """Assert a document-topic matrix ``theta`` describes a *healthy*, non-degenerate
    fit. Three conditions, each of which #270's collapse would have violated:

    1. Every row is a valid simplex (non-negative, sums to ~1).
    2. No single topic holds more than ``max_mass`` of the corpus-mean mass.
       (A collapse parks ~all the mass on one topic.)
    3. The effective topic count exceeds ``min_eff_frac * n_topics``.
       (A collapse drives effective_topics toward 1.)

    Raises ``AssertionError`` naming the model and the offending number.
    """
    tag = f"[{model}] " if model else ""
    theta = np.asarray(theta, dtype=float)

    # 1) valid simplex per document
    assert_finite(theta, model=model)
    assert_simplex(theta, axis=1, model=model)

    # 2) no single topic dominates the mean mass
    mean_mass = theta.mean(axis=0)
    mean_mass = mean_mass / mean_mass.sum()
    top_mass = float(mean_mass.max())
    if top_mass > max_mass:
        raise AssertionError(
            f"{tag}theta collapsed: one topic holds {top_mass:.3f} of the mean "
            f"mass (> max_mass={max_mass}); fit is degenerate"
        )

    # 3) enough effective topics
    eff = effective_topics(theta)
    floor = min_eff_frac * n_topics
    if eff <= floor:
        raise AssertionError(
            f"{tag}theta collapsed: effective_topics={eff:.2f} <= "
            f"{floor:.2f} (= {min_eff_frac} * {n_topics}); fit carries too few topics"
        )


# ---------------------------------------------------------------------------
# Metamorphic helpers — relations between fits, not a single fit in isolation.
# ---------------------------------------------------------------------------

def assert_more_iters_not_worse(
    fit_fn,
    *,
    low: int = 25,
    high: int = 150,
    retain_frac: float = 0.6,
    n_topics: int | None = None,
    model: str = "",
) -> None:
    """Fit at ``low`` and ``high`` iterations and require the longer fit not be
    MORE degenerate than the shorter one.

    ``fit_fn(iters) -> theta`` returns the document-topic matrix for that
    iteration count. A longer fit may legitimately *sharpen* (a Gibbs/EM model
    concentrates mass as it settles, so effective_topics drifts down somewhat),
    so we allow a sharpening but reject a *collapse*: the high-iters effective
    topic count must retain at least ``retain_frac`` of the low-iters value. This
    is the #270 failure mode -- training longer drove the covariate prior to dump
    everything on one topic, taking effective_topics from ~K down to ~1 (a
    retention far below ``retain_frac``).
    """
    tag = f"[{model}] " if model else ""
    theta_low = np.asarray(fit_fn(low), dtype=float)
    theta_high = np.asarray(fit_fn(high), dtype=float)
    assert_finite(theta_low, theta_high, model=model)

    eff_low = effective_topics(theta_low)
    eff_high = effective_topics(theta_high)

    # The high-iters fit may sharpen but must not collapse: keep a fraction of the
    # low-iters effective topic count.
    if eff_high < retain_frac * eff_low:
        raise AssertionError(
            f"{tag}more iterations made the fit MORE degenerate: "
            f"effective_topics fell from {eff_low:.2f} (@{low} iters) to "
            f"{eff_high:.2f} (@{high} iters), below {retain_frac:.0%} retention; "
            f"collapse-on-training (cf. #270)"
        )

    # And it must not be outright collapsed onto a single topic.
    k = n_topics if n_topics is not None else theta_high.shape[1]
    if eff_high <= 1.0 + 1e-6 and k > 1:
        raise AssertionError(
            f"{tag}high-iters fit collapsed onto one topic "
            f"(effective_topics={eff_high:.2f})"
        )


def assert_seed_reproducible(fit_fn, *, atol: float = 1e-6, model: str = "") -> None:
    """Two fits with the same seed/threads must give ~identical θ.

    ``fit_fn() -> theta``. For seed-reproducible / bit-exact models this is a
    cheap guard that the seeding actually plumbs through. Allclose (not exact) so
    a model with benign floating-point reordering still passes.
    """
    tag = f"[{model}] " if model else ""
    a = np.asarray(fit_fn(), dtype=float)
    b = np.asarray(fit_fn(), dtype=float)
    if a.shape != b.shape:
        raise AssertionError(f"{tag}same-seed fits differ in shape: {a.shape} vs {b.shape}")
    if not np.allclose(a, b, atol=atol):
        worst = float(np.abs(a - b).max())
        raise AssertionError(
            f"{tag}same-seed fits are not reproducible (max |Δθ| = {worst:.3e})"
        )


# ---------------------------------------------------------------------------
# Non-vacuity self-test: a deliberately degenerate theta MUST raise. This is the
# proof that the invariant has teeth (run as test_invariants_catch_degenerate_theta).
# ---------------------------------------------------------------------------

def _degenerate_theta_must_raise() -> None:
    """A θ with all mass on one topic must be rejected by every health check.

    If any of these does NOT raise, the invariant is vacuous and the whole wave
    is worthless — so this self-test is the load-bearing one.
    """
    K = 5
    D = 40

    # Fully collapsed: every document is one-hot on topic 0.
    collapsed = np.zeros((D, K))
    collapsed[:, 0] = 1.0

    # effective_topics of a collapse is ~1.
    eff = effective_topics(collapsed)
    assert eff < 1.01, f"effective_topics failed to detect collapse: {eff}"

    raised = False
    try:
        assert_healthy_theta(collapsed, K, model="self-test")
    except AssertionError:
        raised = True
    assert raised, "assert_healthy_theta accepted a fully collapsed theta (VACUOUS!)"

    # A near-collapse (95% on one topic, the rest spread thin) must also fail the
    # max_mass guard.
    near = np.full((D, K), 0.05 / (K - 1))
    near[:, 0] = 0.95
    raised = False
    try:
        assert_healthy_theta(near, K, model="self-test")
    except AssertionError:
        raised = True
    assert raised, "assert_healthy_theta accepted a 95%-on-one-topic theta (VACUOUS!)"

    # A NaN theta must be rejected by assert_finite / assert_healthy_theta.
    nan_theta = np.full((D, K), 1.0 / K)
    nan_theta[0, 0] = np.nan
    raised = False
    try:
        assert_healthy_theta(nan_theta, K, model="self-test")
    except AssertionError:
        raised = True
    assert raised, "assert_healthy_theta accepted a theta with a NaN (VACUOUS!)"

    # Sanity in the other direction: a healthy (flat) theta must PASS, otherwise
    # the check is so strict it rejects everything.
    healthy = np.full((D, K), 1.0 / K)
    assert_healthy_theta(healthy, K, model="self-test")  # must not raise

    # The metamorphic helper must catch a collapse-on-training: low-iters is
    # healthy (flat), high-iters is collapsed onto one topic.
    def _collapsing(iters):
        return collapsed if iters >= 100 else healthy

    raised = False
    try:
        assert_more_iters_not_worse(_collapsing, low=25, high=150, model="self-test")
    except AssertionError:
        raised = True
    assert raised, "assert_more_iters_not_worse accepted a collapse-on-training (VACUOUS!)"

    # ...but a benign sharpening (flat -> moderately concentrated) must PASS.
    sharper = np.full((D, K), 0.5 / (K - 1))
    sharper[:, 0] = 0.5

    def _sharpening(iters):
        return sharper if iters >= 100 else healthy

    assert_more_iters_not_worse(_sharpening, low=25, high=150, model="self-test")
