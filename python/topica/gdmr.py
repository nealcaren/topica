"""Generalized DMR (g-DMR) topic model.

GDMR (Lee & Song 2020) extends DMR by replacing the raw document features with
a Legendre-polynomial tensor-product basis over one or more *continuous* metadata
variables, plus a decay prior that progressively shrinks higher-order basis terms.
The result is a smooth topic-distribution function (TDF) over the continuous
metadata domain.

We implement GDMR as a Python wrapper around the compiled ``topica.DMR`` engine.
The Legendre basis is constructed in NumPy and passed to DMR as its feature
matrix; the heterogeneous per-column Gaussian prior is realized via column scaling
(the "scaling trick"), matching tomotopy's ``GDMRModel`` prior (``sigma0`` on the
constant term, ``sigma`` on the non-constant terms, per-dimension ``decay``)
without any changes to the Rust core.

References
----------
Lee, M., & Song, M. (2020). Incorporating citation impact into analysis of
research trends. *Scientometrics* 124, 1191-1224.
Reference implementations (both by the paper's first author, ``bab2min``):
tomotopy ``GDMRModel`` and the ``bab2min/g-dmr`` experiment repo.

Note: the constant term's prior is centered at ``log(alpha)`` in both reference
implementations. g-DMR realizes that centering with an explicit ``log(alpha)``
predictor offset (see ``fit``), which is why the inner ``DMR`` is constructed with
``alpha=1.0`` (zero-mean intercept prior) — DMR's own ``log(alpha)`` centering
(added in 0.54, #563) would otherwise double-apply. The ``sigma``/``sigma0`` roles
and the per-dimension decay implemented here match the references.
"""

from __future__ import annotations

import pickle
from itertools import product as _iproduct
from math import prod as _prod
from pathlib import Path
from typing import Sequence

import numpy as np

from ._topica import DMR, Corpus


# ---------------------------------------------------------------------------
# Legendre-basis helpers
# ---------------------------------------------------------------------------

def _legendre_polys(t: np.ndarray, max_deg: int) -> np.ndarray:
    """Evaluate Legendre polynomials P_0 ... P_{max_deg} at each point in t.

    Parameters
    ----------
    t:
        1-D array of values in [-1, 1].
    max_deg:
        Highest polynomial degree to include.

    Returns
    -------
    Array of shape ``(len(t), max_deg + 1)``.
    """
    n = len(t)
    P = np.empty((n, max_deg + 1), dtype=np.float64)
    P[:, 0] = 1.0
    if max_deg >= 1:
        P[:, 1] = t
    for k in range(2, max_deg + 1):
        # Bonnet's recursion: (k)*P_k = (2k-1)*t*P_{k-1} - (k-1)*P_{k-2}
        P[:, k] = ((2 * k - 1) * t * P[:, k - 1] - (k - 1) * P[:, k - 2]) / k
    return P


def _build_basis(metadata: np.ndarray, degrees: list[int],
                 metadata_range: list[tuple[float, float]]) -> np.ndarray:
    """Build the Legendre tensor-product basis matrix.

    Each continuous dimension d is mapped from ``metadata_range[d]`` to
    ``[-1, 1]``, clipped, then its Legendre polynomials through degree
    ``degrees[d]`` are evaluated.  The full basis is the Cartesian product of
    all per-dimension polynomial vectors, giving
    ``prod(deg + 1 for deg in degrees)`` columns.

    Parameters
    ----------
    metadata:
        Array of shape ``(num_docs, D)``.
    degrees:
        Per-dimension maximum Legendre degree.
    metadata_range:
        Per-dimension ``(lo, hi)`` bounds used for the [-1, 1] mapping.

    Returns
    -------
    Array of shape ``(num_docs, num_basis)`` with **no** intercept prepended
    (the all-constant column is the order-0 Legendre product and is already
    present).
    """
    num_docs, D = metadata.shape
    # Collect per-dimension polynomial matrices
    per_dim: list[np.ndarray] = []
    for d in range(D):
        lo, hi = metadata_range[d]
        span = hi - lo if hi != lo else 1.0
        t = np.clip(2.0 * (metadata[:, d] - lo) / span - 1.0, -1.0, 1.0)
        per_dim.append(_legendre_polys(t, degrees[d]))  # (num_docs, deg+1)

    # Tensor product over dimensions
    # For D=1 this is just per_dim[0].
    # For D>1 we iterate over all multi-index tuples.
    degree_ranges = [range(deg + 1) for deg in degrees]
    multi_indices = list(_iproduct(*degree_ranges))  # all (p0, p1, ...) combos
    num_basis = len(multi_indices)
    basis = np.empty((num_docs, num_basis), dtype=np.float64)
    for col, idx in enumerate(multi_indices):
        col_vals = np.ones(num_docs, dtype=np.float64)
        for d, p in enumerate(idx):
            col_vals *= per_dim[d][:, p]
        basis[:, col] = col_vals
    return basis


def _basis_order(degrees: list[int]) -> np.ndarray:
    """Return the polynomial order (sum of per-dim degrees) for each basis column."""
    degree_ranges = [range(deg + 1) for deg in degrees]
    return np.array([sum(idx) for idx in _iproduct(*degree_ranges)], dtype=np.float64)


def _basis_multi_indices(degrees: list[int]) -> list[tuple[int, ...]]:
    """Per-dimension Legendre degree tuple ``(p_0, .., p_{D-1})`` for each basis
    column, in the same column order as the design matrix and feature names."""
    return list(_iproduct(*[range(deg + 1) for deg in degrees]))


def _basis_feature_names(degrees: list[int], metadata_names: list[str]) -> list[str]:
    """Readable label for each basis column, aligned with ``feature_effects``.

    The all-zeros term is ``"intercept"``; a term with per-dim Legendre degrees
    ``idx`` is the ``:``-joined ``"{name}^{k}"`` over the dimensions where
    ``idx[d] > 0`` (e.g. ``"year^2"``, ``"year^1:citations^1"``). These are
    Legendre basis terms, so ``^k`` denotes the degree-``k`` term, not a raw power.
    """
    degree_ranges = [range(deg + 1) for deg in degrees]
    names = []
    for idx in _iproduct(*degree_ranges):
        parts = [f"{metadata_names[d]}^{k}" for d, k in enumerate(idx) if k > 0]
        names.append("intercept" if not parts else ":".join(parts))
    return names


def _column_scales(degrees: list[int], sigma: float, sigma0: float,
                   decay: float) -> np.ndarray:
    """Column scaling factors ``c_j = sqrt(v_j / v0)`` that reparameterize the
    uniform DMR prior ``N(0, v0)`` into GDMR's heterogeneous prior, matching
    tomotopy's ``GDMRModel`` (Lee & Song; ``bab2min`` is the paper author).

    With ``v0 = sigma0**2`` (the DMR ``prior_variance``), each column's target
    prior variance is:

    - constant (order-0) term: ``sigma0**2``            -> ``c = 1``
    - a non-constant term with per-dimension Legendre degrees ``(p_0..p_{D-1})``:
      ``sigma**2 / prod_d (p_d + 1)**(2*decay)``         ->
      ``c = (sigma / sigma0) / prod_d (p_d + 1)**decay``

    So ``sigma0`` is the std of the *constant* term and ``sigma`` the std of the
    *non-constant* terms (tomotopy's roles), and any ``decay > 0`` shrinks
    higher-order terms per dimension. ``decay == 0`` gives a uniform ``sigma``
    over every non-constant term — the original paper's decay-free prior.
    """
    v0 = sigma0 ** 2
    indices = _basis_multi_indices(degrees)
    scales = np.empty(len(indices), dtype=np.float64)
    for j, idx in enumerate(indices):
        if sum(idx) == 0:
            v_j = sigma0 ** 2
        else:
            denom = 1.0
            for p in idx:
                denom *= float(p + 1) ** (2.0 * decay)
            v_j = sigma ** 2 / denom
        scales[j] = np.sqrt(v_j / v0)
    return scales


def _resolve_covariates(features, covariates, metadata, *, where, required):
    """Resolve the covariate argument from its three accepted spellings.

    GDMR follows DMR's vocabulary: ``features`` is canonical, ``covariates`` is
    a no-deprecation alias, and ``metadata`` is an alias for users porting from
    tomotopy's ``GDMRModel``. Exactly one may be supplied.
    """
    supplied = [(n, v) for n, v in
                (("features", features), ("covariates", covariates),
                 ("metadata", metadata)) if v is not None]
    if len(supplied) > 1:
        names = ", ".join(n for n, _ in supplied)
        raise ValueError(
            f"{where}: pass only one of features=/covariates=/metadata= "
            f"(got {names})"
        )
    if not supplied:
        if required:
            raise ValueError(
                f"{where}: continuous covariates are required "
                f"(pass features=, or its aliases covariates=/metadata=)"
            )
        return None
    return supplied[0][1]


# ---------------------------------------------------------------------------
# GDMR class
# ---------------------------------------------------------------------------

class GDMR:
    """Generalized DMR topic model (g-DMR; Lee & Song 2020).

    GDMR replaces the raw document covariates of DMR with a Legendre
    tensor-product polynomial basis over one or more continuous metadata
    variables.  A decay prior progressively shrinks the higher-order basis
    terms, producing a smooth topic-distribution function (TDF) over the
    continuous metadata domain.

    We implement GDMR as a thin wrapper around the compiled ``topica.DMR``
    engine.  The Legendre basis is realized in NumPy and passed to DMR as its
    feature matrix; the decay prior is realized via column scaling (the
    "scaling trick"), so no changes to the Rust core are required.

    Parameters
    ----------
    num_topics:
        Number of topics K.
    degrees:
        Per continuous-metadata-dimension maximum Legendre degree.  Length
        must equal the number of metadata dimensions D.  ``degrees=[3]``
        gives a cubic TDF over a single continuous covariate.
    beta:
        Dirichlet word smoothing parameter (passed through to DMR).
    optimize_interval:
        How often (in Gibbs sweeps) to run the L-BFGS lambda-optimization.
    burn_in:
        Sweeps before optimization begins.
    seed:
        RNG seed.
    sigma:
        Prior std on the **non-constant** (order >= 1) basis terms, matching
        tomotopy's ``GDMRModel.sigma``.
    sigma0:
        Prior std on the **constant** (order-0 / intercept) term, matching
        tomotopy's ``GDMRModel.sigma0``.
    decay:
        Per-dimension shrinkage of higher-order terms (tomotopy
        ``GDMRModel.decay``): a non-constant term with per-dimension Legendre
        degrees ``(p_0..p_{D-1})`` has prior variance
        ``sigma**2 / prod_d (p_d + 1)**(2*decay)``.  Any ``decay > 0`` shrinks;
        ``decay == 0`` gives a uniform ``sigma`` over all non-constant terms (the
        original paper's decay-free prior).
    alpha:
        Baseline Dirichlet concentration (tomotopy ``GDMRModel.alpha``, default
        0.1).  Applied as a constant ``log(alpha)`` offset in the DMR predictor, so
        it sets the baseline topic concentration (smaller ``alpha`` -> sparser
        per-document topic mixtures) and centers the intercept prior at
        ``log(alpha)``, matching both reference implementations.  ``alpha = 1``
        reproduces a zero-mean intercept prior (topica's pre-#426 behavior).
    metadata_range:
        Per-dimension ``(lo, hi)`` bounds for the [-1, 1] mapping.  If None,
        we infer from the training data at fit time.
    lbfgs_iters:
        L-BFGS step cap per optimization round.
    sampler:
        Gibbs sampler variant: ``"sparse"`` (default), ``"warp"``, or
        ``"cvb0"``.  See ``topica.DMR`` for details.
    num_threads:
        Worker count for the inner ``DMR`` sparse Gibbs fit.  ``> 1`` enables
        MALLET-style approximate-parallel sampling (deterministic for a fixed
        ``num_threads`` + ``seed``); ``1`` is the exact serial path.  Ignored by
        the ``"warp"``/``"cvb0"`` backends.
    """

    def __init__(
        self,
        num_topics: int,
        *,
        degrees: list[int],
        beta: float = 0.01,
        optimize_interval: int = 50,
        burn_in: int = 200,
        seed: int = 13,
        sigma: float = 1.0,
        sigma0: float = 3.0,
        decay: float = 0.0,
        alpha: float = 0.1,
        metadata_range: list[tuple[float, float]] | None = None,
        lbfgs_iters: int = 20,
        sampler: str = "sparse",
        num_threads: int = 1,
    ) -> None:
        if num_topics < 1:
            raise ValueError("num_topics must be >= 1")
        if beta <= 0:
            raise ValueError("beta must be > 0")
        if not degrees:
            raise ValueError("degrees must be a non-empty list of ints")
        if any(d < 0 for d in degrees):
            raise ValueError("each element of degrees must be >= 0")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        if sigma0 <= 0:
            raise ValueError("sigma0 must be > 0")
        if decay < 0:
            raise ValueError("decay must be >= 0")
        if alpha <= 0 or not np.isfinite(alpha):
            raise ValueError("alpha must be finite and > 0")

        self._num_topics = num_topics
        self._degrees = list(degrees)
        self._beta = beta
        self._optimize_interval = optimize_interval
        self._burn_in = burn_in
        self._seed = seed
        self._sigma = sigma
        self._sigma0 = sigma0
        self._decay = decay
        self._alpha = alpha
        self._metadata_range: list[tuple[float, float]] | None = (
            [tuple(r) for r in metadata_range] if metadata_range is not None else None
        )
        self._lbfgs_iters = lbfgs_iters
        self._sampler = sampler
        self._num_threads = num_threads
        # names of the D continuous metadata dimensions; set at fit
        self._metadata_names: list[str] | None = None

        # the inner DMR engine
        self._dmr: DMR | None = None
        self._fitted = False

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        data: "Corpus | Sequence[Sequence[str]]",
        features=None,
        *,
        metadata_names=None,
        iters: int = 1000,
        num_samples: int = 5,
        sample_interval: int = 25,
        keep_theta_draws: bool = True,
        convergence_tol: float = 0.0,
        check_every: int = 10,
        covariates=None,
        metadata=None,
    ) -> None:
        """Fit GDMR by collapsed Gibbs with the Legendre-basis DMR prior.

        We construct the Legendre tensor-product basis from the continuous
        covariates, apply column scaling to realize the decay prior, then hand
        off to the compiled ``topica.DMR`` engine for sampling and L-BFGS
        optimization. After fitting we recover the true lambda coefficients
        (``feature_effects``) by undoing the column scaling.

        Parameters
        ----------
        data:
            A ``topica.Corpus`` or a list of token lists.
        features:
            Array-like of shape ``(num_docs, D)`` of continuous covariate values
            where D equals ``len(degrees)``. Values outside ``metadata_range``
            are clipped to [-1, 1] in Legendre space. As in :class:`DMR`,
            ``covariates=`` is an accepted alias; ``metadata=`` is also accepted
            for users porting from tomotopy's ``GDMRModel``. Pass exactly one.
        metadata_names:
            Optional human-readable labels for the ``D`` covariate columns,
            surfaced in the Legendre-basis ``feature_names``. Defaults to
            ``x0, x1, ...``.
        iters:
            Total Gibbs sweeps.
        num_samples:
            Number of topic-word phi snapshots to average.
        sample_interval:
            Sweeps between phi snapshots.
        keep_theta_draws:
            Whether to retain thinned MCMC theta draws.
        convergence_tol:
            Relative-change early-stop threshold (0 disables).
        check_every:
            Sweeps between convergence checks.
        covariates:
            Alias for ``features`` (topica's DMR vocabulary).
        metadata:
            Alias for ``features`` (tomotopy ``GDMRModel`` vocabulary).
        """
        features = _resolve_covariates(
            features, covariates, metadata, where="GDMR.fit", required=True
        )
        meta = np.asarray(features, dtype=np.float64)
        if meta.ndim == 1:
            meta = meta[:, np.newaxis]
        if meta.ndim != 2:
            raise ValueError("covariates must be 1-D (single dim) or 2-D (num_docs, D)")
        if not np.all(np.isfinite(meta)):
            raise ValueError("covariates contain non-finite values (NaN or inf)")
        num_docs, D = meta.shape
        if D != len(self._degrees):
            raise ValueError(
                f"covariates have {D} dimensions but degrees has {len(self._degrees)}"
            )

        if metadata_names is None:
            self._metadata_names = [f"x{d}" for d in range(D)]
        else:
            self._metadata_names = [str(n) for n in metadata_names]
            if len(self._metadata_names) != D:
                raise ValueError(
                    f"metadata_names has {len(self._metadata_names)} entries but "
                    f"covariates have {D} dimensions"
                )

        # Infer metadata_range from data if not provided
        if self._metadata_range is None:
            self._metadata_range = [
                (float(meta[:, d].min()), float(meta[:, d].max()))
                for d in range(D)
            ]
        else:
            if len(self._metadata_range) != D:
                raise ValueError(
                    f"metadata_range has {len(self._metadata_range)} entries but "
                    f"metadata has {D} dimensions"
                )

        # Build Legendre basis: shape (num_docs, num_basis)
        # Note: no intercept prepended — the order-0 Legendre product IS the intercept.
        # DMR.fit prepends its own intercept column, so we must NOT include the
        # order-0 constant column ourselves; instead we pass the remaining basis
        # columns and let DMR prepend its intercept.
        #
        # But: the contract defines num_basis = prod(deg+1), which INCLUDES the
        # order-0 column. DMR also prepends an intercept. We therefore build the
        # full Legendre basis (all columns including the constant), then pass
        # the columns EXCLUDING the order-0 constant to DMR.fit so that after
        # DMR prepends its own intercept the total columns match the full Legendre
        # basis (intercept at index 0, then the order >= 1 Legendre columns).
        full_basis = _build_basis(meta, self._degrees, self._metadata_range)
        # full_basis[:, 0] is the all-ones intercept (order-0 Legendre product)
        # DMR will prepend its own intercept, so we pass only the order >= 1 columns.
        non_intercept_basis = full_basis[:, 1:]  # (num_docs, num_basis - 1)

        # Reparameterize the heterogeneous GDMR prior onto the uniform-prior DMR by
        # column scaling (tomotopy's roles: sigma0 = constant-term std, sigma =
        # non-constant std, per-dimension decay). We choose the DMR prior_variance
        # to be the constant term's variance sigma0**2, so the DMR intercept column
        # already carries the correct prior (c_0 == 1) and only the non-intercept
        # columns are rescaled by c_j.
        col_scales = _column_scales(self._degrees, self._sigma, self._sigma0, self._decay)
        prior_variance = self._sigma0 ** 2
        non_intercept_scales = col_scales[1:]
        scaled_features = non_intercept_basis * non_intercept_scales[np.newaxis, :]

        # Readable basis-term labels (intercept first); the inner DMR prepends its
        # own intercept, so hand it the non-intercept names only.
        feature_names = _basis_feature_names(self._degrees, self._metadata_names)[1:]

        # Build inner DMR. Since 0.54 DMR itself centers the intercept prior at
        # log(alpha) (#563), but g-DMR realizes that centering via its own `offset`
        # below (so the constant-term prior interacts correctly with the Legendre
        # basis / decay scaling). Pin the inner DMR to alpha=1.0 (log(1)=0, zero-mean
        # intercept prior) so log(alpha) is applied exactly once, here, and NOT
        # doubled by DMR's native centering.
        self._dmr = DMR(
            self._num_topics,
            beta=self._beta,
            optimize_interval=self._optimize_interval,
            burn_in=self._burn_in,
            seed=self._seed,
            alpha=1.0,
            prior_variance=prior_variance,
            lbfgs_iters=self._lbfgs_iters,
            sampler=self._sampler,
            num_threads=self._num_threads,
        )

        # Center the intercept (constant-term) prior at log(alpha) by adding a
        # constant offset to the DMR predictor, matching tomotopy's GDMRModel (and
        # the paper's g-dmr): alpha_{d,t} = exp(lambda_t . x_d + log(alpha)). This
        # sets the baseline Dirichlet concentration (alpha=0.1 => sparser theta) and
        # shifts the intercept prior mean; the DMR prior on the fitted coefficients
        # stays zero-mean, so the effective intercept is centered at log(alpha).
        num_docs = scaled_features.shape[0]
        offset = np.full((num_docs, self._num_topics), np.log(self._alpha),
                         dtype=np.float64)
        self._dmr.fit(
            data,
            scaled_features,
            feature_names=feature_names,
            iters=iters,
            num_samples=num_samples,
            sample_interval=sample_interval,
            keep_theta_draws=keep_theta_draws,
            convergence_tol=convergence_tol,
            check_every=check_every,
            offset=offset,
        )

        # Recover true coefficients as lambda_j = dmr_lambda_j * c_j. c_0 == 1 by
        # construction (prior_variance == the constant term's variance sigma0**2).
        self._recover_scales = col_scales
        self._fitted = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_fitted(self):
        if not self._fitted or self._dmr is None:
            raise RuntimeError("model is not fitted yet; call fit() first")

    def _get_true_feature_effects(self) -> np.ndarray:
        """Return true lambda (num_topics, num_basis) after undoing column scaling.

        Column 0 is the constant term. The inner DMR fits it under a zero-mean
        prior with a fixed ``log(alpha)`` offset applied to the predictor, so the
        effective constant coefficient is ``dmr_lambda_0 + log(alpha)`` — that
        offset is folded back in here, so ``feature_effects`` and ``tdf`` are on
        tomotopy's ``log(alpha)``-centered scale. (A uniform shift of the constant
        across topics leaves the normalized ``tdf`` unchanged; it matters for the
        reported coefficients and the ``alpha`` baseline.)
        """
        dmr_fe = self._dmr.feature_effects  # (num_topics, num_basis)
        fe = dmr_fe * self._recover_scales[np.newaxis, :]
        fe[:, 0] = fe[:, 0] + np.log(self._alpha)
        return fe

    def _build_basis_for_eval(self, metadata: np.ndarray) -> np.ndarray:
        """Build Legendre basis for evaluation, shape (P, num_basis)."""
        self._require_fitted()
        meta = np.asarray(metadata, dtype=np.float64)
        single_point = meta.ndim == 1
        if single_point:
            meta = meta[np.newaxis, :]
        if meta.ndim != 2:
            raise ValueError("metadata must be 1-D (single point) or 2-D (P, D)")
        if meta.shape[1] != len(self._degrees):
            raise ValueError(
                f"metadata has {meta.shape[1]} dimensions but model has "
                f"{len(self._degrees)}"
            )
        basis = _build_basis(meta, self._degrees, self._metadata_range)
        return basis, single_point

    # ------------------------------------------------------------------
    # tdf
    # ------------------------------------------------------------------

    def tdf(self, metadata, *, normalize: bool = True) -> np.ndarray:
        """Topic-distribution function at one or more metadata points.

        Evaluates the fitted surface at ``metadata`` and returns topic
        prevalences implied by the Legendre-basis DMR prior.

        Parameters
        ----------
        metadata:
            Array-like of shape ``(D,)`` for a single point or ``(P, D)`` for
            P points, in original metadata units (mapped internally via
            ``metadata_range``).
        normalize:
            If True (default), normalize each row so topic prevalences sum to 1.
            If False, return the raw alpha = exp(lambda @ phi(metadata)).

        Returns
        -------
        Array of shape ``(num_topics,)`` for a single point, or
        ``(P, num_topics)`` for P points.
        """
        self._require_fitted()
        basis, single_point = self._build_basis_for_eval(metadata)
        fe = self._get_true_feature_effects()  # (K, num_basis)
        logalpha = basis @ fe.T  # (P, K)
        if normalize:
            # Stable softmax over topics: subtract the per-row max before exp so an
            # extreme predictor cannot overflow to inf/inf (matches tomotopy's TDF).
            shifted = logalpha - logalpha.max(axis=1, keepdims=True)
            exp = np.exp(shifted)
            alpha = exp / exp.sum(axis=1, keepdims=True)
        else:
            alpha = np.exp(logalpha)
        if single_point:
            return alpha[0]
        return alpha

    # ------------------------------------------------------------------
    # tdf_linspace
    # ------------------------------------------------------------------

    def tdf_linspace(
        self,
        start,
        stop,
        num: int,
        *,
        endpoint: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """Evaluate the TDF on a regular grid over the metadata domain.

        For D == 1: ``start`` and ``stop`` are scalars (or length-1 arrays).
        Returns an array of shape ``(num, num_topics)``.

        For D > 1: ``start`` and ``stop`` have length D.  Returns a tensor
        grid of shape ``(num, ..., num, num_topics)`` with D leading axes of
        size ``num``.  The 1-D case is the primary tested path.

        Parameters
        ----------
        start:
            Lower bound of the evaluation range.  Scalar for D == 1, or
            length-D for D > 1.
        stop:
            Upper bound of the evaluation range.
        num:
            Number of grid points per dimension.
        endpoint:
            Include ``stop`` in the grid (default True, matching np.linspace).
        normalize:
            See :meth:`tdf`.

        Returns
        -------
        Array of shape ``(num, num_topics)`` for D == 1, or
        ``(num, ..., num, num_topics)`` for D > 1.
        """
        self._require_fitted()
        D = len(self._degrees)
        start_arr = np.atleast_1d(np.asarray(start, dtype=np.float64))
        stop_arr = np.atleast_1d(np.asarray(stop, dtype=np.float64))
        if start_arr.shape != (D,) or stop_arr.shape != (D,):
            raise ValueError(
                f"start and stop must be scalars (D=1) or length-{D} arrays (D={D})"
            )

        if D == 1:
            pts = np.linspace(start_arr[0], stop_arr[0], num, endpoint=endpoint)
            meta_grid = pts[:, np.newaxis]  # (num, 1)
            result = self.tdf(meta_grid, normalize=normalize)  # (num, K)
            return result
        else:
            # Build D-dimensional meshgrid
            axes = [
                np.linspace(start_arr[d], stop_arr[d], num, endpoint=endpoint)
                for d in range(D)
            ]
            grids = np.meshgrid(*axes, indexing="ij")  # each (num, ..., num)
            flat_pts = np.stack([g.ravel() for g in grids], axis=1)  # (num^D, D)
            flat_result = self.tdf(flat_pts, normalize=normalize)  # (num^D, K)
            K = flat_result.shape[1]
            new_shape = [num] * D + [K]
            return flat_result.reshape(new_shape)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict, keyword-named
        to match ``__init__`` (issue #400)."""
        return {
            "num_topics": self._num_topics,
            "degrees": list(self._degrees),
            "beta": self._beta,
            "optimize_interval": self._optimize_interval,
            "burn_in": self._burn_in,
            "seed": self._seed,
            "sigma": self._sigma,
            "sigma0": self._sigma0,
            "decay": self._decay,
            "alpha": self._alpha,
            "metadata_range": (
                [list(t) for t in self._metadata_range]
                if self._metadata_range is not None
                else None
            ),
            "lbfgs_iters": self._lbfgs_iters,
            "sampler": self._sampler,
            "num_threads": self._num_threads,
        }

    @property
    def topic_word(self) -> np.ndarray:
        """Topic-word matrix phi, shape ``(num_topics, num_words)``, rows sum to 1."""
        self._require_fitted()
        return self._dmr.topic_word

    @property
    def doc_topic(self) -> np.ndarray:
        """Document-topic matrix theta, shape ``(num_docs, num_topics)``, rows sum to 1."""
        self._require_fitted()
        return self._dmr.doc_topic

    @property
    def alpha(self) -> np.ndarray:
        """Intercept baseline topic prevalence, shape ``(num_topics,)``.

        Equals ``exp(lambda_intercept)`` — the per-topic prior with every
        non-constant Legendre column set to zero. This is a *formal* baseline, not
        the TDF at the metadata midpoint: at the midpoint (mapped to Legendre
        ``t = 0``) the even-degree columns are non-zero (``P_2(0) = -1/2``,
        ``P_4(0) = 3/8``, ...), so for any degree >= 2 the midpoint TDF differs from
        this intercept-only value. Use :meth:`tdf` at the midpoint for the latter.
        """
        self._require_fitted()
        fe = self._get_true_feature_effects()
        return np.exp(fe[:, 0])

    @property
    def feature_effects(self) -> np.ndarray:
        """Learned lambda over the Legendre basis, shape ``(num_topics, num_basis)``.

        Column 0 is the intercept (order-0 Legendre product).  Subsequent
        columns correspond to the remaining basis terms in the tensor-product
        enumeration order.
        """
        self._require_fitted()
        return self._get_true_feature_effects()

    @property
    def feature_effect_se(self) -> np.ndarray | None:
        """Standard error of :attr:`feature_effects`, shape ``(num_topics,
        num_basis)``, from the underlying DMR's observed-information SE rescaled by
        the same per-column factor that undoes the basis standardization. Aligned to
        ``feature_effects``; ``None`` for models saved before this was added.
        """
        self._require_fitted()
        dmr_se = self._dmr.feature_effect_se
        if dmr_se is None:
            return None
        # The basis rescaling is a per-column linear map, so the SE scales by the
        # same (non-negative) factor as the effect itself.
        return dmr_se * np.abs(self._recover_scales)[np.newaxis, :]

    @property
    def degrees(self) -> list[int]:
        """Maximum Legendre degree per metadata dimension (read-only)."""
        return list(self._degrees)

    @property
    def metadata_names(self) -> list[str]:
        """Names of the D continuous metadata dimensions (the model's inputs).

        These are distinct from :attr:`feature_names`: a metadata dimension (say
        ``"year"``) expands into several Legendre basis terms (``year^1``,
        ``year^2``, ...), and it is those basis terms that :attr:`feature_effects`
        is indexed by. Set via ``metadata_names=`` on :meth:`fit`; defaults to
        ``["x0", "x1", ...]``.
        """
        self._require_fitted()
        return list(self._metadata_names)

    @property
    def feature_names(self) -> list[str]:
        """Labels for the Legendre basis terms, aligned with the columns of
        :attr:`feature_effects`.

        Column 0 is ``"intercept"``; the rest are ``:``-joined ``"{name}^{k}"``
        terms over the metadata dimensions (e.g. ``"year^2"``,
        ``"year^1:citations^1"``), using :attr:`metadata_names`. The ``^k`` marks
        the degree-``k`` Legendre term, not a raw power. Because a continuous
        covariate's per-degree coefficients are rarely interpretable on their own,
        read the fitted surface with :meth:`tdf` / :meth:`tdf_linspace` rather
        than the individual basis coefficients.
        """
        self._require_fitted()
        return _basis_feature_names(self._degrees, self._metadata_names)

    @property
    def metadata_range(self) -> list[tuple[float, float]]:
        """Per-dimension ``(lo, hi)`` bounds used for the [-1, 1] mapping."""
        self._require_fitted()
        return list(self._metadata_range)

    @property
    def sigma(self) -> float:
        """Prior std on the order-0 (intercept) basis term."""
        return self._sigma

    @property
    def sigma0(self) -> float:
        """Prior std on order >= 1 basis terms (before decay scaling)."""
        return self._sigma0

    @property
    def decay(self) -> float:
        """Decay exponent for higher-order prior shrinkage (0 disables decay)."""
        return self._decay

    @property
    def vocabulary(self) -> list[str]:
        """Word vocabulary in corpus token-ID order."""
        self._require_fitted()
        return self._dmr.vocabulary

    @property
    def doc_names(self) -> list[str]:
        """Per-document identifiers in corpus order."""
        self._require_fitted()
        return self._dmr.doc_names

    @property
    def num_topics(self) -> int:
        """Number of topics K."""
        return self._num_topics

    @property
    def topic_names(self) -> list[str]:
        """Per-topic labels. Defaults to ``["topic_0", ...]`` after fit."""
        self._require_fitted()
        return self._dmr.topic_names

    @topic_names.setter
    def topic_names(self, value: list[str]) -> None:
        self._require_fitted()
        self._dmr.topic_names = value

    @property
    def doc_lengths(self) -> list[int]:
        """Number of tokens in each training document."""
        self._require_fitted()
        return self._dmr.doc_lengths

    @property
    def theta_draws(self) -> "np.ndarray | None":
        """Thinned MCMC theta draws ``(num_draws, num_docs, num_topics)`` or None."""
        self._require_fitted()
        return self._dmr.theta_draws

    @property
    def fit_history(self) -> list[tuple[int, float]]:
        """Per-iteration ``(iteration, objective)`` trace."""
        self._require_fitted()
        return self._dmr.fit_history

    @property
    def converged(self) -> bool:
        """True if fit early-stopped due to convergence.

        :func:`topica.stop_reason` turns this flag into a plain-language summary of
        why the fit stopped."""
        self._require_fitted()
        return self._dmr.converged

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def top_words(self, n: int = 10, *, topic=None, weights=False):
        """Top n words per topic. Returns bare word strings; pass
        ``weights=True`` for ``(word, probability)`` pairs.

        Parameters
        ----------
        n:
            Number of top words to return per topic.
        topic:
            If given, return only the list for that topic index.
            If None (default), return a list of lists (one per topic).
        weights:
            If True, return ``(word, probability)`` pairs instead of bare words.
        """
        self._require_fitted()
        return self._dmr.top_words(n, topic=topic, weights=weights)

    def coherence(self, n: int = 10) -> np.ndarray:
        """UMass topic coherence per topic, shape ``(num_topics,)``."""
        self._require_fitted()
        return self._dmr.coherence(n)

    def transform(
        self,
        data,
        features=None,
        *,
        iters: int = 100,
        burn_in: int = 10,
        num_samples: int = 10,
        sample_interval: int = 5,
        seed=None,
        covariates=None,
        metadata=None,
    ) -> np.ndarray:
        """Infer document-topic theta for new documents.

        Parameters
        ----------
        data:
            A ``topica.Corpus`` or list of token lists.
        features:
            Optional continuous covariate array-like of shape ``(num_new_docs, D)``.
            If provided, the Legendre-basis DMR prior is used for each document;
            if None, the intercept-only baseline is used. ``covariates=`` and
            ``metadata=`` are accepted aliases (see :meth:`fit`).
        iters:
            Inference sweeps.
        burn_in:
            Sweeps before sampling begins.
        num_samples:
            Number of theta snapshots to average.
        sample_interval:
            Sweeps between snapshots.
        seed:
            Optional RNG seed override.
        covariates:
            Alias for ``features`` (topica's DMR vocabulary).
        metadata:
            Alias for ``features`` (tomotopy ``GDMRModel`` vocabulary).

        Returns
        -------
        Array of shape ``(num_new_docs, num_topics)``.
        """
        self._require_fitted()
        features = _resolve_covariates(
            features, covariates, metadata, where="GDMR.transform", required=False
        )
        if features is not None:
            meta = np.asarray(features, dtype=np.float64)
            if meta.ndim == 1:
                meta = meta[:, np.newaxis]
            # Build Legendre basis for new docs
            full_basis = _build_basis(meta, self._degrees, self._metadata_range)
            non_intercept_basis = full_basis[:, 1:]
            # Apply same non-intercept scales
            non_intercept_scales = self._recover_scales[1:]
            scaled_features = non_intercept_basis * non_intercept_scales[np.newaxis, :]
            return self._dmr.transform(
                data,
                scaled_features,
                iters=iters,
                burn_in=burn_in,
                num_samples=num_samples,
                sample_interval=sample_interval,
                seed=seed,
            )
        else:
            return self._dmr.transform(
                data,
                iters=iters,
                burn_in=burn_in,
                num_samples=num_samples,
                sample_interval=sample_interval,
                seed=seed,
            )

    def save(self, path: str) -> None:
        """Persist the fitted GDMR model to ``path``.

        We save the GDMR wrapper state (degrees, metadata_range, sigma, sigma0,
        decay, recover_scales, constructor parameters) alongside the inner DMR
        model, using Python pickle for the wrapper envelope and the DMR native
        save format.  Reload with :meth:`GDMR.load`.
        """
        self._require_fitted()
        p = Path(path)
        dmr_path = str(p) + "._inner_dmr"
        self._dmr.save(dmr_path)
        # The sidecar is resolved relative to the wrapper file at load time (from
        # the load path, not this stored one), so moving the pair together works.
        state = {
            "version": 1,
            "num_topics": self._num_topics,
            "degrees": self._degrees,
            "beta": self._beta,
            "optimize_interval": self._optimize_interval,
            "burn_in": self._burn_in,
            "seed": self._seed,
            "sigma": self._sigma,
            "sigma0": self._sigma0,
            "decay": self._decay,
            "alpha": self._alpha,
            "metadata_range": self._metadata_range,
            "metadata_names": self._metadata_names,
            "lbfgs_iters": self._lbfgs_iters,
            "sampler": self._sampler,
            "num_threads": self._num_threads,
            "recover_scales": self._recover_scales,
            "fitted": self._fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @staticmethod
    def load(path: str) -> "GDMR":
        """Load a GDMR model previously written by :meth:`save`.

        Parameters
        ----------
        path:
            File path passed to :meth:`save`.

        Returns
        -------
        A fitted ``GDMR`` instance.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)
        m = GDMR(
            state["num_topics"],
            degrees=state["degrees"],
            beta=state["beta"],
            optimize_interval=state["optimize_interval"],
            burn_in=state["burn_in"],
            seed=state["seed"],
            sigma=state["sigma"],
            sigma0=state["sigma0"],
            decay=state["decay"],
            alpha=state.get("alpha", 1.0),
            metadata_range=state["metadata_range"],
            lbfgs_iters=state["lbfgs_iters"],
            sampler=state["sampler"],
            num_threads=state.get("num_threads", 1),
        )
        # Resolve the sidecar next to the wrapper file we are loading, so a moved
        # (wrapper + sidecar) pair loads from the new location rather than the
        # absolute path recorded at save time.
        m._dmr = DMR.load(str(Path(path)) + "._inner_dmr")
        m._recover_scales = state["recover_scales"]
        m._metadata_names = state.get("metadata_names")
        m._fitted = state["fitted"]
        return m

    def __repr__(self) -> str:
        return (
            f"GDMR(num_topics={self._num_topics}, "
            f"degrees={self._degrees}, "
            f"fitted={self._fitted})"
        )
