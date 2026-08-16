"""MCMC-native convergence diagnostics for the collapsed-Gibbs models.

For the Gibbs samplers (``LDA``, ``KeyATM``, ``DMR``, ``SAGE``, ``PA``,
``SeededLDA``, ``LabeledLDA``, ...), the ``convergence_tol`` early-stop is a
pragmatic heuristic on the log-likelihood trace, *not* a measure of MCMC
convergence: a flat log-likelihood says the chain found a mode, not that it has
mixed. This module adds the chain diagnostics a Bayesian workflow expects,
computed from traces the models already retain -- the log-likelihood history and
the thinned ``theta_draws``.

- :func:`autocorrelation` -- the autocorrelation function of a 1-D trace.
- :func:`integrated_autocorr_time` -- the integrated autocorrelation time
  ``tau`` (Geyer 1992 initial-positive-sequence estimator).
- :func:`effective_sample_size` -- ``ESS = N / tau`` for one chain or, columnwise,
  for a ``(draws, params)`` matrix.
- :func:`mcmc_diagnostics` -- read the log-likelihood trace and ``theta_draws``
  off a fitted Gibbs model and summarize both.

Run the same model at several seeds and the multi-chain diagnostics compare the
chains against each other -- the question a single chain cannot answer:

- :func:`rhat` -- the rank-normalized split-R-hat (Gelman-Rubin, in the improved
  form of Vehtari et al. 2021) for a set of chains. ``R-hat`` near 1 means the
  chains agree; a value above ~1.01 means they have not converged to a common
  distribution.
- :func:`multichain_diagnostics` -- take several fitted Gibbs models (same corpus,
  different seeds) and report R-hat and multi-chain ESS on the permutation-invariant
  log-likelihood trace, plus per-topic R-hat on each topic's prevalence after the
  topics are aligned across chains (Hungarian match on the topic-word matrix).

.. note::
   Topic indices are label-switched across seeds -- topic 3 in one chain need not
   be topic 3 in another -- so :func:`multichain_diagnostics` aligns the topics
   before comparing them. The alignment quality is reported per topic; a topic
   that did not align well makes its R-hat meaningless, so read the two together.

.. note::
   ``theta_draws`` are *thinned* MCMC draws (``num_theta_draws`` samples spaced
   ``sample_interval`` sweeps apart), so their ESS measures the effective size of
   the retained draws. Raise ``num_theta_draws`` on ``fit`` for a finer estimate.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .registry import REGISTRY


def _autocovariance(x: np.ndarray) -> np.ndarray:
    """Biased autocovariance of a 1-D series at lags 0..N-1, via FFT."""
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    x = x - x.mean()
    # zero-pad to at least 2N-1 and up to a power of two for a fast transform
    size = 1 << (2 * n - 1).bit_length()
    freq = np.fft.rfft(x, n=size)
    acov = np.fft.irfft(freq * np.conjugate(freq), n=size)[:n]
    return acov / n


def autocorrelation(x, max_lag: int | None = None) -> np.ndarray:
    """Autocorrelation function of a 1-D trace at lags ``0..max_lag``.

    Parameters
    ----------
    x : 1-D sequence
        The trace (e.g. a log-likelihood history or a single scalar chain).
    max_lag : int, optional
        Highest lag to return. Defaults to ``len(x) - 1``. Lag 0 is always 1.

    Returns
    -------
    numpy.ndarray
        ``rho[0..max_lag]`` with ``rho[0] == 1``. A constant trace (zero
        variance) returns all zeros except ``rho[0]``.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 2:
        return np.ones(1)
    acov = _autocovariance(x)
    var = acov[0]
    lag = n - 1 if max_lag is None else min(max_lag, n - 1)
    if var <= 0:
        rho = np.zeros(lag + 1)
        rho[0] = 1.0
        return rho
    return acov[: lag + 1] / var


def integrated_autocorr_time(x) -> float:
    """Integrated autocorrelation time ``tau = 1 + 2 * sum_{t>=1} rho_t``.

    Uses Geyer's (1992) initial-positive-sequence estimator: the pair sums
    ``Gamma_m = rho_{2m} + rho_{2m+1}`` are summed until the first non-positive
    pair, which truncates the noisy tail of the empirical autocorrelation.
    Floored at 1 (an independent chain), so ``ESS`` never exceeds ``N``.

    A degenerate (constant) chain returns ``inf`` -- there is no information to
    resample from, so its effective sample size is zero.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 2:
        return 1.0
    acov = _autocovariance(x)
    var = acov[0]
    if var <= 0:
        return float("inf")
    rho = acov / var
    gamma_sum = 0.0
    for m in range(0, (n - 1) // 2 + 1):
        i, j = 2 * m, 2 * m + 1
        pair = rho[i] + (rho[j] if j < n else 0.0)
        if pair <= 0:
            break
        gamma_sum += pair
    tau = 2.0 * gamma_sum - 1.0
    return max(tau, 1.0)


def effective_sample_size(chain):
    """Effective sample size ``ESS = N / tau`` of a chain.

    Parameters
    ----------
    chain : 1-D or 2-D array
        A single scalar chain of length ``N``, or an ``(N, params)`` matrix of
        ``params`` independent chains sharing the ``N`` draws (e.g.
        ``theta_draws`` flattened over docs and topics).

    Returns
    -------
    float or numpy.ndarray
        A scalar for a 1-D chain, or a ``(params,)`` array of per-column ESS for
        a 2-D chain. A degenerate (constant) column yields ``0.0``.
    """
    chain = np.asarray(chain, dtype=float)
    if chain.ndim == 1:
        return chain.shape[0] / integrated_autocorr_time(chain)
    if chain.ndim == 2:
        n = chain.shape[0]
        out = np.empty(chain.shape[1])
        for j in range(chain.shape[1]):
            out[j] = n / integrated_autocorr_time(chain[:, j])
        return out
    raise ValueError(f"chain must be 1-D or 2-D, got {chain.ndim}-D")


@dataclass
class McmcDiagnostics:
    """Single-chain MCMC diagnostics for a fitted Gibbs model.

    Attributes
    ----------
    model : str
        The model class name.
    inference : str or None
        The model's inference engine from the registry (``"gibbs"`` for the
        samplers these diagnostics are meant for).
    n_draws : int
        Number of retained ``theta_draws``.
    loglik_autocorr : numpy.ndarray or None
        Autocorrelation of the log-likelihood trace, or ``None`` when the model
        recorded no trace (e.g. the WarpLDA / CVB0 sampler paths).
    loglik_tau : float or None
        Integrated autocorrelation time of the log-likelihood trace.
    loglik_ess : float or None
        Effective sample size of the log-likelihood trace (``len(trace) / tau``).
    theta_ess : numpy.ndarray
        Per-element effective sample size of ``theta_draws``, shaped
        ``(num_docs, num_topics)``.
    """

    model: str
    inference: str | None
    n_draws: int
    loglik_autocorr: np.ndarray | None
    loglik_tau: float | None
    loglik_ess: float | None
    theta_ess: np.ndarray

    @property
    def theta_ess_min(self) -> float:
        return float(np.min(self.theta_ess))

    @property
    def theta_ess_median(self) -> float:
        return float(np.median(self.theta_ess))

    @property
    def theta_ess_mean(self) -> float:
        return float(np.mean(self.theta_ess))

    def summary(self) -> str:
        """A short human-readable table of the diagnostics."""
        lines = [
            f"MCMC diagnostics for {self.model} (inference={self.inference})",
            f"  retained draws          : {self.n_draws}",
        ]
        if self.loglik_tau is not None:
            lines.append(f"  log-likelihood tau      : {self.loglik_tau:.2f}")
            ess = "n/a" if self.loglik_ess is None else f"{self.loglik_ess:.1f}"
            lines.append(f"  log-likelihood ESS      : {ess}")
        else:
            lines.append("  log-likelihood trace    : none recorded")
        lines.append(
            f"  theta ESS (min/median)  : {self.theta_ess_min:.1f} / "
            f"{self.theta_ess_median:.1f} (of {self.n_draws} draws)"
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def mcmc_diagnostics(model, *, warn: bool = True) -> McmcDiagnostics:
    """Single-chain MCMC diagnostics from a fitted Gibbs model's retained traces.

    Reads the model's log-likelihood history and thinned ``theta_draws`` and
    reports the autocorrelation and effective sample size of each -- the honest
    "has the chain mixed?" companion to the ``convergence_tol`` plateau check.

    Parameters
    ----------
    model : a fitted topica model
        Must expose ``theta_draws`` (fit with ``keep_theta_draws=True``, the
        default). The log-likelihood diagnostics also need a non-empty
        ``log_likelihood_history`` / ``fit_history``.
    warn : bool, default True
        Warn when the model is not a Gibbs sampler. The variational models
        (STM, CTM, ...) converge a bound and have no MCMC chain; these
        diagnostics do not apply to them.

    Returns
    -------
    McmcDiagnostics

    Raises
    ------
    ValueError
        If the model retained no ``theta_draws``.
    """
    name = type(model).__name__
    info = REGISTRY.get(name)
    inference = info.inference if info is not None else None
    if warn and inference is not None and inference != "gibbs":
        warnings.warn(
            f"{name} uses {inference!r} inference, not Gibbs sampling; MCMC "
            "chain diagnostics (autocorrelation / ESS) do not apply to it. Its "
            "convergence is measured by the variational bound.",
            stacklevel=2,
        )

    draws = getattr(model, "theta_draws", None)
    if draws is None:
        raise ValueError(
            f"{name} retained no theta_draws; refit with keep_theta_draws=True "
            "to compute MCMC diagnostics."
        )
    draws = np.asarray(draws, dtype=float)  # (num_draws, num_docs, num_topics)
    n_draws, n_docs, n_topics = draws.shape
    flat = draws.reshape(n_draws, n_docs * n_topics)
    theta_ess = effective_sample_size(flat).reshape(n_docs, n_topics)

    ll_hist = getattr(model, "log_likelihood_history", None)
    if not ll_hist:
        ll_hist = getattr(model, "fit_history", None)
    ll_acf = None
    ll_tau = None
    ll_ess = None
    if ll_hist:
        ll_vals = np.array([float(v) for _, v in ll_hist], dtype=float)
        if ll_vals.size >= 2:
            ll_acf = autocorrelation(ll_vals)
            ll_tau = integrated_autocorr_time(ll_vals)
            ll_ess = None if np.isinf(ll_tau) else ll_vals.size / ll_tau

    return McmcDiagnostics(
        model=name,
        inference=inference,
        n_draws=n_draws,
        loglik_autocorr=ll_acf,
        loglik_tau=ll_tau,
        loglik_ess=ll_ess,
        theta_ess=theta_ess,
    )


# ---------------------------------------------------------------------------
# Multi-chain diagnostics: R-hat (Gelman-Rubin) and cross-chain ESS.
# ---------------------------------------------------------------------------


def _ndtri(p: np.ndarray) -> np.ndarray:
    """Inverse of the standard-normal CDF (probit), Acklam's rational approximation.

    Pure numpy so the R-hat primitive carries no SciPy dependency. Accurate to
    ~1e-9 on ``(0, 1)``; the boundaries map to +-inf. Used only for the
    rank-normalization of R-hat, where inputs sit safely inside the open interval.
    """
    p = np.asarray(p, dtype=float)
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    out = np.empty_like(p)
    lo, hi = 0.02425, 1.0 - 0.02425
    lower = p < lo
    upper = p > hi
    middle = ~(lower | upper)
    # lower tail
    if np.any(lower):
        q = np.sqrt(-2.0 * np.log(p[lower]))
        out[lower] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    # upper tail
    if np.any(upper):
        q = np.sqrt(-2.0 * np.log(1.0 - p[upper]))
        out[upper] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    # central region
    if np.any(middle):
        q = p[middle] - 0.5
        r = q * q
        out[middle] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    return out


def _rankdata_average(a: np.ndarray) -> np.ndarray:
    """Ranks of a 1-D array with ties averaged (SciPy's ``rankdata`` default)."""
    a = np.asarray(a, dtype=float).ravel()
    n = a.size
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(n, dtype=int)
    inv[sorter] = np.arange(n)
    a_sorted = a[sorter]
    obs = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))
    dense = np.cumsum(obs)[inv]
    # positions where each distinct value's run starts (plus the end sentinel)
    starts = np.concatenate((np.nonzero(obs)[0], [n]))
    return 0.5 * (starts[dense] + starts[dense - 1] + 1)


def _as_chain_matrix(chains) -> np.ndarray:
    """Coerce chains to a ``(n_chains, n_draws)`` float matrix.

    Accepts a 2-D array or a sequence of 1-D chains; unequal-length chains are
    truncated to the shortest so every chain contributes the same number of draws.
    """
    if isinstance(chains, np.ndarray) and chains.ndim == 2:
        return np.asarray(chains, dtype=float)
    rows = [np.asarray(c, dtype=float).ravel() for c in chains]
    if not rows:
        raise ValueError("no chains supplied")
    n = min(r.size for r in rows)
    return np.array([r[:n] for r in rows], dtype=float)


def _split_chains(chains: np.ndarray) -> np.ndarray:
    """Split each chain into two non-overlapping halves (split-R-hat).

    Splitting turns ``m`` chains of length ``n`` into ``2m`` chains of length
    ``n // 2``, which lets R-hat detect within-chain non-stationarity (a chain
    still drifting looks like two disagreeing half-chains). Chains too short to
    split are returned unchanged.
    """
    m, n = chains.shape
    half = n // 2
    if half < 2:
        return chains
    return np.concatenate((chains[:, :half], chains[:, n - half:]), axis=0)


def _rank_normalize(chains: np.ndarray) -> np.ndarray:
    """Rank-normalize pooled draws to standard-normal scores (Vehtari 2021).

    Pooling all draws, ranking, and mapping the ranks through the normal quantile
    makes R-hat robust to heavy tails and non-normality -- the improvement over
    the classical Gelman-Rubin statistic.
    """
    shape = chains.shape
    ranks = _rankdata_average(chains.ravel())
    total = ranks.size
    quantile = (ranks - 0.375) / (total + 0.25)
    return _ndtri(quantile).reshape(shape)


def _classic_rhat(chains: np.ndarray) -> float:
    """Gelman-Rubin potential scale reduction on ``(m, n)`` chains (no splitting)."""
    m, n = chains.shape
    if n < 2:
        return float("nan")
    chain_means = chains.mean(axis=1)
    within = chains.var(axis=1, ddof=1).mean()          # W
    if within <= 0:
        # every chain is constant: converged iff they share the same constant
        return 1.0 if np.ptp(chain_means) == 0 else float("inf")
    between = n * chain_means.var(ddof=1)               # B
    var_plus = (n - 1) / n * within + between / n
    return float(np.sqrt(var_plus / within))


def rhat(chains, *, split: bool = True, rank_normalize: bool = True) -> float:
    """Gelman-Rubin R-hat (potential scale reduction) across MCMC chains.

    Compares the variance *between* chains to the variance *within* each chain.
    At convergence the chains are draws from one distribution and ``R-hat`` -> 1;
    a value above roughly ``1.01`` means the chains have not mixed to a common
    target and the run needs more sweeps (Vehtari et al. 2021).

    Parameters
    ----------
    chains : 2-D array or sequence of 1-D chains
        ``(n_chains, n_draws)``, or a list of per-chain traces. Unequal-length
        chains are truncated to the shortest. At least two chains are required.
    split : bool, default True
        Split each chain in half before comparing (split-R-hat), so a single
        chain that has not stopped drifting is caught as two disagreeing halves.
    rank_normalize : bool, default True
        Rank-normalize the pooled draws first (the improved, tail-robust R-hat).
        Set ``False`` for the classical Gelman-Rubin statistic on the raw scale.

    Returns
    -------
    float
        The R-hat statistic. ``inf`` if the chains are constant but disagree;
        ``1.0`` if they share a single constant value.
    """
    mat = _as_chain_matrix(chains)
    if mat.shape[0] < 2:
        raise ValueError("R-hat needs at least two chains")
    work = _split_chains(mat) if split else mat
    if rank_normalize:
        work = _rank_normalize(work)
    return _classic_rhat(work)


def _multichain_ess(chains: np.ndarray) -> float:
    """Cross-chain effective sample size on ``(m, n)`` chains (Vehtari 2021).

    Combines the within- and between-chain autocorrelation into a single ESS for
    the pooled draws, using Geyer's initial-monotone-sequence truncation. Mirrors
    the ``posterior`` package's bulk-ESS on already rank-normalized, split chains.
    """
    m, n = chains.shape
    if n < 4 or m < 1:
        return float("nan")
    chain_means = chains.mean(axis=1)
    # unbiased within-chain variance and the combined variance estimate var_plus
    within = chains.var(axis=1, ddof=1).mean()
    if within <= 0:
        return float("nan")
    between = chain_means.var(ddof=1) if m > 1 else 0.0
    var_plus = (n - 1) / n * within + between
    # mean over chains of each chain's (biased, /n) autocovariance
    acov = np.mean([_autocovariance(chains[i]) for i in range(m)], axis=0)
    # rescale the biased /n autocovariance to the /(n-1) within-chain variance
    rho = 1.0 - (within - acov * n / (n - 1)) / var_plus
    rho[0] = 1.0
    # Geyer initial-monotone-sequence: sum positive, non-increasing paired sums
    tau = 1.0
    prev_pair = np.inf
    for t in range(1, n - 1, 2):
        pair = rho[t] + rho[t + 1]
        if pair <= 0:
            break
        pair = min(pair, prev_pair)      # enforce monotone non-increasing
        tau += 2.0 * pair
        prev_pair = pair
    tau = max(tau, 1.0)
    return float(m * n / tau)


@dataclass
class MultiChainDiagnostics:
    """Multi-chain (Gelman-Rubin) diagnostics for a set of fitted Gibbs models.

    Attributes
    ----------
    model : str
        The model class name (all chains must share it).
    inference : str or None
        The model's inference engine from the registry.
    n_chains : int
        Number of chains compared.
    n_draws : int
        Retained ``theta_draws`` per chain used for the topic-level statistics.
    loglik_rhat : float or None
        R-hat of the (post-warmup) log-likelihood trace across chains -- the
        permutation-invariant "did the chains agree?" headline. ``None`` when the
        chains recorded no usable log-likelihood trace.
    loglik_ess : float or None
        Cross-chain effective sample size of the log-likelihood trace.
    loglik_n : int or None
        Post-warmup trace length per chain used for the log-likelihood statistics.
    topic_rhat : numpy.ndarray or None
        Per-topic R-hat of each aligned topic's per-draw prevalence, shape
        ``(num_topics,)``. ``None`` when the chains retained no ``theta_draws``.
    topic_ess : numpy.ndarray or None
        Per-topic cross-chain ESS of the aligned topic prevalence.
    topic_alignment : numpy.ndarray or None
        Per-topic alignment quality: the minimum top-word Jaccard of the topic to
        its reference-chain match across the other chains. Low values flag topics
        whose R-hat compares topics that did not line up.
    reference : int or None
        Index of the chain used as the alignment reference.
    """

    model: str
    inference: str | None
    n_chains: int
    n_draws: int
    loglik_rhat: float | None
    loglik_ess: float | None
    loglik_n: int | None
    topic_rhat: np.ndarray | None
    topic_ess: np.ndarray | None
    topic_alignment: np.ndarray | None
    reference: int | None

    @property
    def topic_rhat_max(self) -> float | None:
        return None if self.topic_rhat is None else float(np.max(self.topic_rhat))

    @property
    def topic_rhat_median(self) -> float | None:
        return None if self.topic_rhat is None else float(np.median(self.topic_rhat))

    @property
    def converged(self) -> bool:
        """Whether every reported R-hat clears the conventional 1.01 threshold."""
        rhats = []
        if self.loglik_rhat is not None:
            rhats.append(self.loglik_rhat)
        if self.topic_rhat is not None:
            rhats.append(float(np.max(self.topic_rhat)))
        return bool(rhats) and all(r <= 1.01 for r in rhats)

    def summary(self) -> str:
        """A short human-readable table of the multi-chain diagnostics."""
        lines = [
            f"Multi-chain diagnostics for {self.model} "
            f"({self.n_chains} chains, inference={self.inference})",
        ]
        if self.loglik_rhat is not None:
            ess = "n/a" if self.loglik_ess is None else f"{self.loglik_ess:.0f}"
            lines.append(
                f"  log-likelihood R-hat    : {self.loglik_rhat:.3f} "
                f"(ESS {ess}, n={self.loglik_n})"
            )
        else:
            lines.append("  log-likelihood trace    : none recorded")
        if self.topic_rhat is not None:
            lines.append(
                f"  topic-prevalence R-hat  : max {self.topic_rhat_max:.3f} / "
                f"median {self.topic_rhat_median:.3f} over {self.topic_rhat.size} "
                f"aligned topics"
            )
            worst_align = float(np.min(self.topic_alignment))
            lines.append(
                f"  topic alignment (Jaccard): min {worst_align:.2f} "
                f"(low -> that topic's R-hat is not comparable)"
            )
        else:
            lines.append("  topic-prevalence R-hat  : no theta_draws retained")
        verdict = "chains mixed" if self.converged else "not converged (R-hat > 1.01)"
        lines.append(f"  -> {verdict}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _loglik_values(model) -> np.ndarray | None:
    hist = getattr(model, "log_likelihood_history", None)
    if not hist:
        hist = getattr(model, "fit_history", None)
    if not hist:
        return None
    return np.array([float(v) for _, v in hist], dtype=float)


def multichain_diagnostics(
    chains,
    *,
    warmup: float = 0.5,
    metric: str = "cosine",
    reference: int = 0,
    warn: bool = True,
) -> MultiChainDiagnostics:
    """Gelman-Rubin diagnostics across several fitted Gibbs models.

    Fit the same model at several seeds on the same corpus, pass the fitted
    models here, and this reports whether the chains agree. Two views are
    computed: R-hat and cross-chain ESS on the permutation-invariant
    log-likelihood trace, and per-topic R-hat on each topic's prevalence after
    the topics are aligned across chains (topic indices are label-switched across
    seeds, so alignment comes first).

    Parameters
    ----------
    chains : sequence of fitted topica models
        At least two fits of the *same* model class on the *same* corpus at
        different seeds. The topic-level statistics need ``theta_draws`` on every
        chain (fit with ``keep_theta_draws=True``, the default).
    warmup : float, default 0.5
        Fraction of each log-likelihood trace to discard from the front as
        burn-in before computing R-hat. The ``theta_draws`` are already the last
        post-warmup thinned samples, so the warmup fraction applies to the
        log-likelihood trace only.
    metric : str, default "cosine"
        Topic-word distance metric for the cross-chain alignment (passed to
        :func:`topica.evaluate.align_topics`).
    reference : int, default 0
        Index of the chain whose topic order the others are aligned to.
    warn : bool, default True
        Warn when the chains are not Gibbs samplers, disagree on class, or lack
        the traces a statistic needs.

    Returns
    -------
    MultiChainDiagnostics

    Raises
    ------
    ValueError
        If fewer than two chains are supplied.
    """
    chains = list(chains)
    if len(chains) < 2:
        raise ValueError(
            "multichain_diagnostics needs at least two chains "
            "(the same model fit at different seeds)."
        )
    names = {type(m).__name__ for m in chains}
    name = type(chains[0]).__name__
    if warn and len(names) > 1:
        warnings.warn(
            f"chains are not all the same model class ({sorted(names)}); "
            "R-hat compares them as if they share a target distribution.",
            stacklevel=2,
        )
    info = REGISTRY.get(name)
    inference = info.inference if info is not None else None
    if warn and inference is not None and inference != "gibbs":
        warnings.warn(
            f"{name} uses {inference!r} inference, not Gibbs sampling; multi-chain "
            "R-hat is a diagnostic for MCMC samplers.",
            stacklevel=2,
        )
    if not 0 <= reference < len(chains):
        raise ValueError(f"reference index {reference} out of range for {len(chains)} chains")
    if not 0.0 <= warmup < 1.0:
        raise ValueError("warmup must be in [0, 1)")

    # --- log-likelihood R-hat (permutation-invariant) --------------------
    ll_rhat = ll_ess = ll_n = None
    ll_traces = [_loglik_values(m) for m in chains]
    if all(t is not None and t.size >= 4 for t in ll_traces):
        common = min(t.size for t in ll_traces)
        start = int(common * warmup)
        post = np.array([t[start:common] for t in ll_traces], dtype=float)
        if post.shape[1] >= 4:
            ll_rhat = rhat(post)
            ll_ess = _multichain_ess(_rank_normalize(_split_chains(post)))
            ll_n = int(post.shape[1])
    elif warn:
        warnings.warn(
            "not every chain recorded a usable log-likelihood trace; skipping the "
            "log-likelihood R-hat. Fit with check_every > 0 to record one.",
            stacklevel=2,
        )

    # --- per-topic R-hat on aligned topic prevalence ---------------------
    topic_rhat = topic_ess = topic_alignment = None
    ref_idx = reference
    draws = [getattr(m, "theta_draws", None) for m in chains]
    if all(d is not None for d in draws):
        from .validation import align_topics

        arrs = [np.asarray(d, dtype=float) for d in draws]  # (n_draws, n_docs, K)
        n_draws = min(a.shape[0] for a in arrs)
        K = arrs[ref_idx].shape[2]
        ref_model = chains[ref_idx]
        # prevalence trace per (chain, topic): mean over docs of each draw's theta
        prevalence = np.empty((len(chains), K, n_draws), dtype=float)
        alignment = np.ones((len(chains), K), dtype=float)
        ref_sets = _topic_word_sets(ref_model)
        for c, model in enumerate(chains):
            if c == ref_idx:
                perm = list(range(K))
            else:
                pairs = align_topics(ref_model, model, metric=metric)
                perm = list(range(K))
                for i, j, _ in pairs:
                    perm[i] = j
                alignment[c] = _alignment_jaccard(ref_sets, _topic_word_sets(model), perm)
            aligned = arrs[c][:n_draws][:, :, perm]         # (n_draws, n_docs, K)
            prevalence[c] = aligned.mean(axis=1).T          # (K, n_draws)
        topic_rhat = np.array([rhat(prevalence[:, k, :]) for k in range(K)])
        topic_ess = np.array([
            _multichain_ess(_rank_normalize(_split_chains(prevalence[:, k, :])))
            for k in range(K)
        ])
        topic_alignment = alignment.min(axis=0)
    elif warn:
        warnings.warn(
            f"{name} chains retained no theta_draws; skipping per-topic R-hat. "
            "Refit with keep_theta_draws=True.",
            stacklevel=2,
        )

    n_draws_reported = 0 if topic_rhat is None else int(min(a.shape[0] for a in arrs))
    return MultiChainDiagnostics(
        model=name,
        inference=inference,
        n_chains=len(chains),
        n_draws=n_draws_reported,
        loglik_rhat=ll_rhat,
        loglik_ess=ll_ess,
        loglik_n=ll_n,
        topic_rhat=topic_rhat,
        topic_ess=topic_ess,
        topic_alignment=topic_alignment,
        reference=ref_idx if topic_rhat is not None else None,
    )


def _topic_word_sets(model, topn: int = 10):
    """Per-topic set of top-``topn`` term indices from a model's topic-word matrix."""
    beta = np.asarray(model.topic_word, dtype=float)
    return [set(np.argsort(beta[t])[::-1][:topn].tolist()) for t in range(beta.shape[0])]


def _alignment_jaccard(ref_sets, other_sets, perm) -> np.ndarray:
    """Top-word Jaccard of each reference topic to its matched topic under ``perm``."""
    out = np.zeros(len(ref_sets))
    for i, rs in enumerate(ref_sets):
        os = other_sets[perm[i]]
        union = rs | os
        out[i] = (len(rs & os) / len(union)) if union else 0.0
    return out
