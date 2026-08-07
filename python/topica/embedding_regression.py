"""Embedding regression: context-specific description and inference.

A faithful port of the *à la carte on text* (conText) embedding regression of
Rodriguez, Spirling and Stewart (2023, "Embedding Regression: Models for
Context-Specific Description and Inference", *American Political Science Review*),
which builds on the à la carte (ALC) embeddings of Khodak et al. (2018).

The question this answers is different from the covariate topic models here (STM,
DMR, Scholar). Those ask how a covariate shifts topic *prevalence* -- how *much* a
group discusses a theme. Embedding regression asks how a covariate shifts *meaning*
-- *how* a group uses a word or theme, in a pretrained embedding space where
semantic and framing differences live that word counts cannot see. "Do Republicans
and Democrats mean different things by 'immigration'?" is its native question.

The pipeline is entirely numpy-native and mirrors ``conText``:

1. :func:`compute_transform` learns the ALC transform matrix ``A`` from a corpus and
   a set of pretrained word embeddings (or you supply a precomputed ``A``).
2. :func:`alc_embeddings` turns each document -- or each context around a focal term
   -- into a single ALC embedding: the (count-weighted) mean of its in-vocabulary
   context words' pretrained vectors, mapped through ``A``.
3. :func:`embedding_regression` regresses those ``N x D`` embeddings on a covariate
   design matrix (multivariate OLS), reports the Euclidean norm of each covariate's
   ``D``-dimensional coefficient as its effect size, and gets a permutation p-value
   and bootstrap confidence interval for that norm.
4. :meth:`EmbeddingRegression.nearest_neighbors` and
   :meth:`EmbeddingRegression.nns_ratio` describe *what* the shift means, by ranking
   pretrained vocabulary words near the predicted embedding at chosen covariate
   values.

You bring the pretrained embeddings (GloVe, word2vec, a local model) as a
``(word -> vector)`` mapping, following topica's convention that the library does
not call an embedding model for you.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "compute_transform",
    "alc_embeddings",
    "embedding_regression",
    "EmbeddingRegression",
]


# ---------------------------------------------------------------------------
# Pretrained embedding lookup
# ---------------------------------------------------------------------------
def _as_lookup(pre_trained):
    """Accept a dict ``{word: vector}`` or a ``(matrix, vocab)`` pair and return
    ``(words, matrix, index)`` with a float64 ``(V, D)`` matrix."""
    if isinstance(pre_trained, dict):
        words = list(pre_trained.keys())
        matrix = np.asarray([pre_trained[w] for w in words], dtype=np.float64)
    else:
        matrix, vocab = pre_trained
        words = list(vocab)
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape[0] != len(words):
            raise ValueError("pre_trained matrix rows must match the vocabulary length")
    if matrix.ndim != 2:
        raise ValueError("pretrained embeddings must be 2-d (vocab, dim)")
    index = {w: i for i, w in enumerate(words)}
    return words, matrix, index


def _context_windows(tokens, target, window, case_insensitive):
    """Yield the context token lists (focal token excluded) around each occurrence
    of ``target`` in ``tokens``. ``target`` may be a string or a set of strings."""
    targets = {target} if isinstance(target, str) else set(target)
    if case_insensitive:
        targets = {t.lower() for t in targets}
    for i, tok in enumerate(tokens):
        hit = tok.lower() if case_insensitive else tok
        if hit in targets:
            lo = max(0, i - window)
            hi = min(len(tokens), i + window + 1)
            yield [tokens[j] for j in range(lo, hi) if j != i]


# ---------------------------------------------------------------------------
# à la carte transform matrix A
# ---------------------------------------------------------------------------
def compute_transform(docs, pre_trained, *, window=6, min_count=100, case_insensitive=True):
    """Estimate the à la carte transform matrix ``A`` (``D x D``) from a corpus.

    Following Khodak et al. (2018): for every vocabulary word with at least
    ``min_count`` occurrences, form ``u_w``, the average of the pretrained vectors
    of the words appearing in its context windows, and regress the word's own
    pretrained vector ``v_w`` on ``u_w`` across words. The least-squares solution is
    ``A = (U'U)^-1 U'V`` where ``U`` stacks the ``u_w`` and ``V`` the ``v_w``. The
    resulting ``A`` corrects a raw additive context embedding toward the target
    word's own representation, so the ALC embedding ``u @ A`` lives in the pretrained
    space.

    Parameters
    ----------
    docs : sequence of token lists
        The corpus, tokenized. Word order matters (context windows are used).
    pre_trained : dict or (matrix, vocab)
        Pretrained word embeddings.
    window : int, default 6
        Context window each side of a word. The paper recommends >= 5.
    min_count : int, default 100
        Minimum corpus frequency for a word to enter the transform regression.
    case_insensitive : bool, default True
        Lowercase tokens before matching them to the pretrained vocabulary.

    Returns
    -------
    numpy.ndarray
        The ``(D, D)`` transform matrix ``A``.
    """
    _, matrix, index = _as_lookup(pre_trained)
    D = matrix.shape[1]
    lower = case_insensitive

    # Accumulate, per word, the sum of context vectors and the context count.
    ctx_sum: dict[int, np.ndarray] = {}
    ctx_cnt: dict[int, int] = {}
    for tokens in docs:
        ids = [index.get(t.lower() if lower else t) for t in tokens]
        n = len(ids)
        for i, wi in enumerate(ids):
            if wi is None:
                continue
            lo = max(0, i - window)
            hi = min(n, i + window + 1)
            for j in range(lo, hi):
                if j == i or ids[j] is None:
                    continue
                if wi not in ctx_sum:
                    ctx_sum[wi] = np.zeros(D)
                    ctx_cnt[wi] = 0
                ctx_sum[wi] += matrix[ids[j]]
                ctx_cnt[wi] += 1

    rows_u, rows_v = [], []
    for wi, cnt in ctx_cnt.items():
        if cnt >= min_count:
            rows_u.append(ctx_sum[wi] / cnt)
            rows_v.append(matrix[wi])
    if len(rows_u) < D:
        raise ValueError(
            f"only {len(rows_u)} words meet min_count={min_count}; need at least "
            f"D={D} to estimate the transform. Lower min_count or use transform='additive'."
        )
    U = np.asarray(rows_u)
    V = np.asarray(rows_v)
    # A = (U'U)^-1 U'V  (least squares of V on U, no intercept)
    A, *_ = np.linalg.lstsq(U, V, rcond=None)
    return A


# ---------------------------------------------------------------------------
# ALC document / focal-term embeddings (dem)
# ---------------------------------------------------------------------------
def alc_embeddings(docs, pre_trained, *, transform=None, target=None, window=6,
                   case_insensitive=True):
    """Compute one à la carte embedding per document.

    Each document's embedding is the count-weighted mean of its in-vocabulary
    context words' pretrained vectors, mapped through the transform ``A``. With
    ``target=None`` the context is the whole document; with a focal ``target`` the
    context is the tokens within ``window`` of each occurrence of the term (the
    focal token excluded), pooled over the document.

    Out-of-vocabulary tokens are dropped and the mean taken over the remaining
    in-vocabulary occurrences. A document with no in-vocabulary context is dropped.

    Parameters
    ----------
    docs : sequence of token lists
    pre_trained : dict or (matrix, vocab)
    transform : numpy.ndarray or None
        The ``(D, D)`` matrix ``A``. ``None`` uses the identity ("additive"
        embeddings), a valid, simpler baseline.
    target : str, sequence of str, or None
        Focal term(s). ``None`` embeds the whole document.
    window : int, default 6
    case_insensitive : bool, default True

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``embeddings`` of shape ``(n_kept, D)`` and ``kept`` -- the integer indices
        of the documents that produced an embedding (in ``docs`` order).
    """
    _, matrix, index = _as_lookup(pre_trained)
    D = matrix.shape[1]
    A = np.eye(D) if transform is None else np.asarray(transform, dtype=np.float64)
    if A.shape != (D, D):
        raise ValueError(f"transform must be ({D}, {D}) to match the embedding dim")
    lower = case_insensitive

    embs, kept = [], []
    for d, tokens in enumerate(docs):
        if target is None:
            contexts = [tokens]
        else:
            contexts = list(_context_windows(tokens, target, window, case_insensitive))
        vec = np.zeros(D)
        n = 0
        for ctx in contexts:
            for t in ctx:
                wi = index.get(t.lower() if lower else t)
                if wi is not None:
                    vec += matrix[wi]
                    n += 1
        if n == 0:
            continue
        embs.append((vec / n) @ A)
        kept.append(d)
    if not embs:
        raise ValueError("no document had any in-vocabulary context tokens")
    return np.asarray(embs), np.asarray(kept, dtype=int)


# ---------------------------------------------------------------------------
# Embedding regression
# ---------------------------------------------------------------------------
@dataclass
class EmbeddingRegression:
    """Result of :func:`embedding_regression`.

    Attributes
    ----------
    coefficients : numpy.ndarray
        ``(n_covariates, D)`` -- each covariate's D-dimensional coefficient
        (intercept excluded). Individual dimensions are not interpretable; use the
        norm and the nearest-neighbor methods.
    names : list of str
        Covariate names, aligned with ``coefficients`` rows.
    normed_estimate : numpy.ndarray
        ``(n_covariates,)`` effect size per covariate (see ``statistic``).
    p_value : numpy.ndarray
        ``(n_covariates,)`` permutation p-values for the effect size.
    normed_ci : numpy.ndarray
        ``(n_covariates, 2)`` bootstrap percentile CI for the effect size.
    intercept : numpy.ndarray
        The ``(D,)`` intercept (the reference-level ALC embedding).
    statistic : str
        Which effect size was reported: ``"norm"``, ``"squared"`` or
        ``"squared_deflated"``.
    """

    coefficients: np.ndarray
    names: list
    normed_estimate: np.ndarray
    p_value: np.ndarray
    normed_ci: np.ndarray
    intercept: np.ndarray
    statistic: str
    confidence_level: float
    _design_names: list = field(default=None, repr=False)
    _pretrained_words: list = field(default=None, repr=False)
    _pretrained_matrix: np.ndarray = field(default=None, repr=False)
    _beta_full: np.ndarray = field(default=None, repr=False)  # (p+1, D) incl intercept

    # -- interpretation -----------------------------------------------------
    def predict(self, profile) -> np.ndarray:
        """The predicted ALC embedding at a covariate profile.

        ``profile`` is a mapping ``{covariate_name: value}`` or a length-``p`` array
        in ``names`` order; unspecified covariates default to 0 (their reference
        level). The intercept is added automatically.
        """
        p = len(self.names)
        x = np.zeros(p)
        if isinstance(profile, dict):
            for k, v in profile.items():
                if k not in self.names:
                    raise KeyError(f"unknown covariate {k!r}; have {self.names}")
                x[self.names.index(k)] = v
        else:
            x = np.asarray(profile, dtype=np.float64).ravel()
            if x.size != p:
                raise ValueError(f"profile must have {p} entries (one per covariate)")
        return self.intercept + x @ self.coefficients

    def nearest_neighbors(self, profile, *, n=10, candidates=None):
        """Pretrained words most similar to the predicted embedding at ``profile``.

        Returns a list of ``(word, cosine)`` pairs, highest first. ``candidates``, if
        given, restricts the ranking to that set of words (the paper notes this cuts
        noise from rare/garbage vocabulary).
        """
        if self._pretrained_matrix is None:
            raise RuntimeError("nearest_neighbors needs the pretrained embeddings kept at fit")
        yhat = self.predict(profile)
        return _cosine_topn(yhat, self._pretrained_words, self._pretrained_matrix, n, candidates)

    def nns_ratio(self, profile_a, profile_b, *, n=10, candidates=None):
        """Contrast two covariate profiles by cosine ratio.

        For each candidate word ``q`` (the union of the two profiles' top-``n``
        neighbors, or ``candidates`` if given), returns ``(word, ratio)`` with
        ``ratio = cos(y_a, q) / cos(y_b, q)``; ``ratio > 1`` means ``q`` sits closer
        to profile ``a``. Sorted by ratio, descending.
        """
        if self._pretrained_matrix is None:
            raise RuntimeError("nns_ratio needs the pretrained embeddings kept at fit")
        ya, yb = self.predict(profile_a), self.predict(profile_b)
        words, M = self._pretrained_words, self._pretrained_matrix
        if candidates is None:
            top_a = {w for w, _ in _cosine_topn(ya, words, M, n, None)}
            top_b = {w for w, _ in _cosine_topn(yb, words, M, n, None)}
            candidates = top_a | top_b
        idx = [i for i, w in enumerate(words) if w in candidates]
        Mn = M[idx] / (np.linalg.norm(M[idx], axis=1, keepdims=True) + 1e-12)
        ca = Mn @ (ya / (np.linalg.norm(ya) + 1e-12))
        cb = Mn @ (yb / (np.linalg.norm(yb) + 1e-12))
        ratio = ca / np.where(np.abs(cb) < 1e-12, np.nan, cb)
        order = np.argsort(-ratio)
        return [(words[idx[i]], float(ratio[i])) for i in order]

    def summary(self):
        """A compact text table of covariate, effect size, CI and p-value."""
        lines = [f"Embedding regression ({self.statistic}, "
                 f"{int(self.confidence_level * 100)}% CI)"]
        lines.append(f"{'covariate':<20s} {'estimate':>10s} {'ci_low':>9s} "
                     f"{'ci_high':>9s} {'p':>7s}")
        for i, nm in enumerate(self.names):
            lines.append(f"{nm:<20s} {self.normed_estimate[i]:10.4f} "
                         f"{self.normed_ci[i, 0]:9.4f} {self.normed_ci[i, 1]:9.4f} "
                         f"{self.p_value[i]:7.3f}")
        return "\n".join(lines)


def _cosine_topn(vec, words, matrix, n, candidates):
    if candidates is not None:
        idx = np.array([i for i, w in enumerate(words) if w in set(candidates)])
        if idx.size == 0:
            return []
    else:
        idx = np.arange(len(words))
    M = matrix[idx]
    sims = (M @ vec) / ((np.linalg.norm(M, axis=1) * np.linalg.norm(vec)) + 1e-12)
    order = np.argsort(-sims)[:n]
    return [(words[idx[i]], float(sims[i])) for i in order]


def _ols(X, Y):
    """Multivariate OLS ``B = (X'X)^-1 X'Y``; returns ``B`` and ``(X'X)^-1``."""
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    B = XtX_inv @ (X.T @ Y)
    return B, XtX_inv


def _effect_size(B, X, Y, XtX_inv, statistic):
    """Per-covariate effect size from coefficient block ``B[1:]`` (intercept at 0)."""
    beta = B[1:]  # (p, D)
    sq = np.sum(beta ** 2, axis=1)  # squared norm per covariate
    if statistic == "norm":
        return np.sqrt(sq)
    if statistic == "squared":
        return sq
    if statistic == "squared_deflated":
        resid = Y - X @ B
        dof = max(X.shape[0] - X.shape[1], 1)
        sigma2_tr = np.sum(resid ** 2) / dof  # trace(Sigma_eps) estimate
        bias = np.diag(XtX_inv)[1:] * sigma2_tr
        return sq - bias
    raise ValueError("statistic must be 'norm', 'squared' or 'squared_deflated'")


def embedding_regression(
    docs,
    covariates,
    pre_trained,
    *,
    names=None,
    transform="additive",
    target=None,
    window=6,
    statistic="norm",
    permutations=100,
    bootstrap=100,
    confidence_level=0.95,
    case_insensitive=True,
    seed=0,
    keep_pretrained=True,
):
    """Regress à la carte document embeddings on covariates (conText).

    Parameters
    ----------
    docs : sequence of token lists
        The tokenized corpus (word order matters).
    covariates : array-like (N, p) or (N,)
        The covariate design, one row per document, **without** an intercept
        (added internally). Rows must align with ``docs`` before ALC dropping.
    pre_trained : dict or (matrix, vocab)
        Pretrained word embeddings.
    names : sequence of str, optional
        Covariate names; defaults to ``x0, x1, ...``.
    transform : {"additive", "estimate"}, numpy.ndarray, or None, default "additive"
        The ALC transform ``A``. ``"additive"``/``None`` uses the identity;
        ``"estimate"`` calls :func:`compute_transform` on ``docs``; or pass a matrix.
    target : str, sequence of str, or None
        Focal term(s) whose context is embedded; ``None`` embeds whole documents.
    window : int, default 6
    statistic : {"norm", "squared", "squared_deflated"}, default "norm"
        Effect-size functional of each covariate coefficient. ``"norm"`` is the
        paper's Euclidean norm; ``"squared_deflated"`` applies a small-sample bias
        correction to the squared norm.
    permutations : int, default 100
        Covariate permutations for the p-value (paper uses 100). 0 skips.
    bootstrap : int, default 100
        Document bootstrap resamples for the CI. 0 skips.
    confidence_level : float, default 0.95
    case_insensitive : bool, default True
    seed : int, default 0
    keep_pretrained : bool, default True
        Retain the pretrained matrix on the result so ``nearest_neighbors`` works.

    Returns
    -------
    EmbeddingRegression
    """
    rng = np.random.default_rng(seed)
    words, matrix, _ = _as_lookup(pre_trained)

    if transform == "estimate":
        A = compute_transform(docs, (matrix, words), window=window,
                              case_insensitive=case_insensitive)
    elif transform == "additive" or transform is None:
        A = None
    else:
        A = np.asarray(transform, dtype=np.float64)

    Y, kept = alc_embeddings(docs, (matrix, words), transform=A, target=target,
                             window=window, case_insensitive=case_insensitive)

    Xraw = np.asarray(covariates, dtype=np.float64)
    if Xraw.ndim == 1:
        Xraw = Xraw[:, None]
    if Xraw.shape[0] != len(docs):
        raise ValueError("covariates must have one row per document")
    Xraw = Xraw[kept]  # align to documents that produced an embedding
    p = Xraw.shape[1]
    if names is None:
        names = [f"x{i}" for i in range(p)]
    names = list(names)
    if len(names) != p:
        raise ValueError(f"names has {len(names)} entries but there are {p} covariates")

    N = Xraw.shape[0]
    X = np.column_stack([np.ones(N), Xraw])  # intercept + covariates
    B, XtX_inv = _ols(X, Y)
    est = _effect_size(B, X, Y, XtX_inv, statistic)

    # permutation p-value: shuffle covariate rows, refit, recompute effect size.
    if permutations and permutations > 0:
        ge = np.zeros(p)
        for _ in range(permutations):
            perm = rng.permutation(N)
            Xp = np.column_stack([np.ones(N), Xraw[perm]])
            Bp, inv_p = _ols(Xp, Y)
            ep = _effect_size(Bp, Xp, Y, inv_p, statistic)
            ge += (ep >= est)
        p_value = ge / permutations
    else:
        p_value = np.full(p, np.nan)

    # bootstrap CI on the effect size: resample documents with replacement.
    if bootstrap and bootstrap > 0:
        boot = np.empty((bootstrap, p))
        for b in range(bootstrap):
            idx = rng.integers(0, N, size=N)
            Xb = np.column_stack([np.ones(N), Xraw[idx]])
            Bb, inv_b = _ols(Xb, Y[idx])
            boot[b] = _effect_size(Bb, Xb, Y[idx], inv_b, statistic)
        alpha = (1 - confidence_level) / 2
        ci = np.column_stack([np.quantile(boot, alpha, axis=0),
                              np.quantile(boot, 1 - alpha, axis=0)])
    else:
        ci = np.full((p, 2), np.nan)

    return EmbeddingRegression(
        coefficients=B[1:],
        names=names,
        normed_estimate=est,
        p_value=p_value,
        normed_ci=ci,
        intercept=B[0],
        statistic=statistic,
        confidence_level=confidence_level,
        _design_names=["(intercept)"] + names,
        _pretrained_words=words if keep_pretrained else None,
        _pretrained_matrix=matrix if keep_pretrained else None,
        _beta_full=B,
    )
