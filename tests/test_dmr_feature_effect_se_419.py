"""#419: DMR feature_effect_se was computed from the post-sampling topic counts,
which drift away from the counts lambda was optimized against (the extra
num_samples*sample_interval sweeps mutate them). It is now computed from the last
converged optimization's counts, so it is J^-1 at the optimum for the returned
lambda -- and therefore invariant to the sampling controls. It is also None when
lambda was never optimized to a stationary point."""
import numpy as np
import pytest
import topica


def _corpus(seed=0, n=120, k=3, block=6):
    # Deliberately ambiguous: each doc is only ~60% from its block and ~40% drawn
    # uniformly over the whole vocabulary, with short docs, so the Gibbs sampler
    # keeps reassigning tokens and the per-doc counts genuinely fluctuate across
    # sweeps (needed for the drift witness below to be meaningful).
    rng = np.random.default_rng(seed)
    v = k * block
    docs, feats = [], []
    for d in range(n):
        b = d % k
        toks = []
        for _ in range(10):
            if rng.random() < 0.6:
                toks.append(f"w{b * block + int(rng.integers(block))}")
            else:
                toks.append(f"w{int(rng.integers(v))}")
        docs.append(toks)
        feats.append([float(b == j) for j in range(k)])  # one-hot-ish covariates
    return docs, np.asarray(feats)


def _fit(sampler, num_samples, sample_interval, **kw):
    docs, feats = _corpus()
    m = topica.DMR(num_topics=3, seed=42, optimize_interval=10, burn_in=20,
                   sampler=sampler, **kw)
    m.fit(docs, features=feats, iters=80, num_samples=num_samples,
          sample_interval=sample_interval, keep_theta_draws=False)
    return m


# Only the stochastic samplers run a post-training sampling phase (the issue notes
# these are where the drift is worst); CVB0 is deterministic and ignores num_samples.
@pytest.mark.parametrize("sampler", ["sparse", "warp"])
def test_se_is_invariant_to_the_sampling_controls(sampler):
    # The training loop (and thus lambda + the counts it was optimized at) is
    # identical; only the post-training sampling phase differs. The SE must not
    # move. Pre-#419 it was read from the drifted post-sampling counts and did.
    a = _fit(sampler, num_samples=1, sample_interval=5)
    b = _fit(sampler, num_samples=8, sample_interval=30)
    se_a = a.feature_effect_se
    se_b = b.feature_effect_se
    assert se_a is not None and se_b is not None
    assert np.allclose(se_a, se_b, atol=1e-10), np.abs(se_a - se_b).max()
    # lambda itself is likewise unaffected by the sampling controls.
    assert np.allclose(a.feature_effects, b.feature_effects, atol=1e-10)
    # Witness that the invariance is non-trivial: the sampling phase genuinely
    # diverges, so the post-sampling counts differ between the two runs. The old
    # code fed exactly those counts to the SE, so it would NOT have matched here.
    assert not np.allclose(a.doc_topic, b.doc_topic)


def test_cvb0_se_is_invariant_to_trailing_non_optimize_sweeps():
    # optimize_interval=10, burn_in=20 -> the last optimization is at iter 50 for
    # both runs; the second run just does 8 more (non-optimizing) CVB0 sweeps, which
    # drift the expected counts. The SE, taken from the iter-50 counts, must match.
    # Pre-#419 it was read from the drifted final expected counts and would not.
    docs, feats = _corpus()

    def fit(iters):
        m = topica.DMR(num_topics=3, seed=42, optimize_interval=10, burn_in=20,
                       sampler="cvb0")
        m.fit(docs, features=feats, iters=iters, keep_theta_draws=False)
        return m

    a, b = fit(50), fit(58)
    assert a.feature_effect_se is not None and b.feature_effect_se is not None
    assert np.allclose(a.feature_effect_se, b.feature_effect_se, atol=1e-10)
    assert np.allclose(a.feature_effects, b.feature_effects, atol=1e-10)


def test_normal_fit_has_finite_positive_se():
    m = _fit("sparse", num_samples=3, sample_interval=10)
    se = m.feature_effect_se
    assert se is not None
    assert np.all(np.isfinite(se)) and np.all(se > 0.0)


def test_no_optimization_yields_no_se():
    # optimize_interval=0 disables lambda optimization entirely: an observed-
    # information SE would be meaningless, so it must be None rather than fabricated.
    docs, feats = _corpus()
    m = topica.DMR(num_topics=3, seed=42, optimize_interval=0, burn_in=20)
    m.fit(docs, features=feats, iters=60, num_samples=2, sample_interval=10,
          keep_theta_draws=False)
    assert m.feature_effect_se is None


def test_zero_lbfgs_iters_yields_no_se():
    # lbfgs_iters=0 means lambda can never reach a stationary point.
    docs, feats = _corpus()
    m = topica.DMR(num_topics=3, seed=42, optimize_interval=10, burn_in=20, lbfgs_iters=0)
    m.fit(docs, features=feats, iters=60, num_samples=2, sample_interval=10,
          keep_theta_draws=False)
    assert m.feature_effect_se is None
