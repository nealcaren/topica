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

Single-chain autocorrelation and ESS are the cheap diagnostics that land here.
Multi-chain R-hat (Gelman-Rubin) needs a multi-chain runner and is tracked
separately (issue #269).

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
