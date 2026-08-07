"""Embedding regression: context-specific description and inference.

A validated port of the *à la carte on text* (conText) embedding regression of
Rodriguez, Spirling and Stewart (2023, "Embedding Regression: Models for
Context-Specific Description and Inference", *American Political Science Review*),
which builds on the à la carte (ALC) embeddings of Khodak et al. (2018). It is
checked bit-for-bit against the R ``conText`` package on its own bundled data
(``parity/embedding_regression_context.py``): identical ALC embeddings, squared
coefficient norm, and HC1-deflated norm.

This is a **text-as-data analysis tool, not a topic model.** The covariate topic
models here (STM, DMR, Scholar) ask how a covariate shifts topic *prevalence* --
how *much* a group discusses a theme. Embedding regression asks how a covariate
shifts *meaning* -- *how* a group uses a word or theme, in a pretrained embedding
space where semantic and framing differences live that word counts cannot see.
"Do Republicans and Democrats mean different things by 'immigration'?" is its
native question. It is a regression on embeddings, so it needs no
``enable_experimental`` and produces no topics.

The pipeline is entirely numpy-native and mirrors ``conText``:

1. :func:`compute_transform` learns the ALC transform matrix ``A`` from a corpus and
   pretrained word embeddings (or you supply a precomputed ``A``).
2. :func:`alc_embeddings` turns each document -- or each context around a focal term
   -- into an ALC embedding: the count-weighted mean of its in-vocabulary context
   words' pretrained vectors, mapped through ``A``.
3. :func:`embedding_regression` regresses those embeddings on a covariate design
   (multivariate OLS), reports each covariate coefficient's squared Euclidean norm
   (with an HC1 small-sample-deflated variant, conText's headline), and gets a
   permutation p-value and bootstrap confidence interval.
4. :meth:`EmbeddingRegression.nearest_neighbors` / :meth:`EmbeddingRegression.nns_ratio`
   describe *what* the shift means by ranking pretrained vocabulary words near the
   predicted embedding at chosen covariate values.

You bring the pretrained embeddings (GloVe, word2vec, a local model) as a
``{word: vector}`` mapping or ``(matrix, vocab)`` pair, per topica's convention that
the library does not call an embedding model for you.

Note on transform orientation: ``conText``'s published transform matrices (e.g.
``cr_transform``) are the transpose of what :func:`compute_transform` returns here.
Pass a ``conText`` matrix as ``transform=cr_transform.T``.
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
def _as_lookup(pre_trained, case_insensitive):
    """Accept a dict ``{word: vector}`` or a ``(matrix, vocab)`` pair and return
    ``(words, matrix, index)`` with a float64 ``(V, D)`` matrix. When
    ``case_insensitive`` the lookup index is keyed on lowercased words so cased
    pretrained vocabularies (e.g. word2vec) still match lowercased tokens."""
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
    if not np.isfinite(matrix).all():
        raise ValueError("pretrained embeddings contain non-finite values (NaN or inf)")
    if case_insensitive:
        index = {}
        for i, w in enumerate(words):
            index.setdefault(w.lower(), i)
    else:
        index = {w: i for i, w in enumerate(words)}
    return words, matrix, index


def _lookup(index, token, case_insensitive):
    return index.get(token.lower() if case_insensitive else token)


def _contexts_in_doc(tokens, target, window, case_insensitive):
    """Yield the context token lists (focal token excluded) around each occurrence
    of ``target`` in one document. ``target`` may be a string or a set of strings."""
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

    Following Khodak et al. (2018): for every vocabulary word occurring at least
    ``min_count`` times, form ``u_w``, the average of the pretrained vectors of the
    words in its context windows, and regress the word's own pretrained vector
    ``v_w`` on ``u_w`` across words. The least-squares solution ``A = (U'U)^-1 U'V``
    maps a raw additive context embedding toward the target word's representation, so
    ``u @ A`` lives in the pretrained space.

    ``conText``'s published transform matrices are the transpose of this ``A``; apply
    an external ``conText`` matrix as ``transform=matrix.T``.

    Parameters
    ----------
    docs : sequence of token lists
    pre_trained : dict or (matrix, vocab)
    window : int, default 6
        Context window each side of a word (paper recommends >= 5).
    min_count : int, default 100
        Minimum number of *occurrences* of a word for it to enter the regression.
    case_insensitive : bool, default True
    """
    import warnings

    _, matrix, index = _as_lookup(pre_trained, case_insensitive)
    D = matrix.shape[1]

    ctx_sum: dict[int, np.ndarray] = {}
    ctx_cnt: dict[int, int] = {}   # context-token count (denominator of u_w)
    word_cnt: dict[int, int] = {}  # word occurrence count (for min_count)
    for tokens in docs:
        ids = [_lookup(index, t, case_insensitive) for t in tokens]
        n = len(ids)
        for i, wi in enumerate(ids):
            if wi is None:
                continue
            word_cnt[wi] = word_cnt.get(wi, 0) + 1
            lo, hi = max(0, i - window), min(n, i + window + 1)
            for j in range(lo, hi):
                if j == i or ids[j] is None:
                    continue
                if wi not in ctx_sum:
                    ctx_sum[wi] = np.zeros(D)
                    ctx_cnt[wi] = 0
                ctx_sum[wi] += matrix[ids[j]]
                ctx_cnt[wi] += 1

    rows_u, rows_v = [], []
    for wi, wc in word_cnt.items():
        if wc >= min_count and ctx_cnt.get(wi, 0) > 0:
            rows_u.append(ctx_sum[wi] / ctx_cnt[wi])
            rows_v.append(matrix[wi])
    if len(rows_u) < D:
        raise ValueError(
            f"only {len(rows_u)} words occur at least min_count={min_count} times; need "
            f"at least D={D} to estimate the transform. Lower min_count or use "
            f"transform='additive'."
        )
    U = np.asarray(rows_u)
    V = np.asarray(rows_v)
    A, _resid, rank, _sv = np.linalg.lstsq(U, V, rcond=None)
    if rank < D:
        warnings.warn(
            f"transform design is rank-deficient (rank {rank} < {D}); the estimated A "
            "may be unstable. Use more data or transform='additive'.",
            stacklevel=2,
        )
    fro = float(np.linalg.norm(A))
    if fro > 20 * np.sqrt(D):
        warnings.warn(
            f"estimated transform has large norm (||A||_F={fro:.1f}); results may be "
            "distorted. Check that the corpus is large enough for min_count.",
            stacklevel=2,
        )
    return A


# ---------------------------------------------------------------------------
# ALC document / focal-term embeddings (dem)
# ---------------------------------------------------------------------------
def alc_embeddings(docs, pre_trained, *, transform=None, target=None, window=6,
                   case_insensitive=True, aggregate="document"):
    """Compute à la carte embeddings.

    Each embedding is the count-weighted mean of its in-vocabulary context words'
    pretrained vectors, mapped through the transform ``A``. Out-of-vocabulary tokens
    are dropped and the mean taken over the remaining in-vocabulary occurrences; a
    context with no in-vocabulary token is dropped.

    With ``target=None`` the context is the whole document (one row per document).
    With a focal ``target``:

    - ``aggregate="instance"`` (conText's unit of analysis): one row per *occurrence*
      of the term, the window around that occurrence (focal token excluded).
    - ``aggregate="document"``: one row per document, pooling all of its occurrences'
      contexts.

    Parameters
    ----------
    docs : sequence of token lists
    pre_trained : dict or (matrix, vocab)
    transform : numpy.ndarray or None
        The ``(D, D)`` matrix ``A``; ``None`` uses the identity ("additive"). A
        ``conText`` matrix must be passed transposed.
    target : str, sequence of str, or None
    window : int, default 6
    case_insensitive : bool, default True
    aggregate : {"document", "instance"}, default "document"
        Ignored when ``target`` is None (always one row per document).

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``embeddings`` of shape ``(n_rows, D)`` and ``doc_ids`` -- the source
        document index for each row, used to align covariates to the surviving rows.
        In document mode there is one row per surviving document; in instance mode
        one row per surviving mention (each mention is an independent observation --
        within-document clustering is not corrected for, matching conText's default).
    """
    if aggregate not in ("document", "instance"):
        raise ValueError("aggregate must be 'document' or 'instance'")
    if target is not None and window < 1:
        raise ValueError("window must be >= 1 when a target is given")
    # A raw string is iterable, so a document passed as text (rather than a token
    # list) would silently be treated as a sequence of single characters -- none in
    # vocabulary -- and surface later as the opaque "no context" error. Catch it
    # here with the actual fix.
    docs = list(docs)
    if docs and isinstance(docs[0], str):
        raise ValueError(
            "docs must be a sequence of token lists, not raw strings; tokenize "
            "first (e.g. [t.split() for t in docs])"
        )
    _, matrix, index = _as_lookup(pre_trained, case_insensitive)
    D = matrix.shape[1]
    A = np.eye(D) if transform is None else np.asarray(transform, dtype=np.float64)
    if A.shape != (D, D):
        raise ValueError(f"transform must be ({D}, {D}) to match the embedding dim")

    def embed(ctx_tokens):
        vec = np.zeros(D)
        n = 0
        for t in ctx_tokens:
            wi = _lookup(index, t, case_insensitive)
            if wi is not None:
                vec += matrix[wi]
                n += 1
        return (vec / n) @ A if n else None

    embs, doc_ids = [], []
    for d, tokens in enumerate(docs):
        if target is None:
            v = embed(tokens)
            if v is not None:
                embs.append(v)
                doc_ids.append(d)
        else:
            contexts = list(_contexts_in_doc(tokens, target, window, case_insensitive))
            if aggregate == "instance":
                for ctx in contexts:
                    v = embed(ctx)
                    if v is not None:
                        embs.append(v)
                        doc_ids.append(d)
            else:  # pool all of the document's contexts into one row
                v = embed([t for ctx in contexts for t in ctx])
                if v is not None:
                    embs.append(v)
                    doc_ids.append(d)
    if not embs:
        if target is not None:
            raise ValueError(
                f"no context produced an embedding for target {target!r}: it does "
                "not occur in the corpus, or every window around it is entirely "
                "out-of-vocabulary. Check the term is present and lowercased to "
                "match the pretrained vocabulary."
            )
        raise ValueError(
            "no document produced an embedding: every document is entirely "
            "out-of-vocabulary against the pretrained embeddings"
        )
    return np.asarray(embs), np.asarray(doc_ids, dtype=int)


# ---------------------------------------------------------------------------
# Design helpers
# ---------------------------------------------------------------------------
def _build_design(covariates, names):
    """Return ``(Xraw (N,p), names)``. A 1-D non-numeric covariate is dummy-coded
    (sorted categories, first dropped as reference, columns named ``col_level``)."""
    arr = np.asarray(covariates, dtype=object) if _is_categorical(covariates) else None
    if arr is not None:
        cats = sorted(set(arr.tolist()))
        if len(cats) < 2:
            raise ValueError(
                f"categorical covariate has a single level ({cats[0]!r}); a "
                "regression needs at least two groups to contrast"
            )
        ref, levels = cats[0], cats[1:]
        Xraw = np.array([[1.0 if v == lv else 0.0 for lv in levels] for v in arr])
        base = names[0] if names else "x"
        return Xraw, [f"{base}_{lv}" for lv in levels], ref
    X = np.asarray(covariates, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    p = X.shape[1]
    if names is None:
        names = [f"x{i}" for i in range(p)]
    names = list(names)
    if len(names) != p:
        raise ValueError(f"names has {len(names)} entries but there are {p} covariates")
    return X, names, None


def _is_categorical(covariates):
    if isinstance(covariates, np.ndarray) and covariates.dtype.kind in "OUS":
        return covariates.ndim == 1
    if isinstance(covariates, (list, tuple)) and covariates and isinstance(covariates[0], str):
        return True
    return False


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
        ``(n_covariates,)`` permutation p-values.
    normed_ci : numpy.ndarray
        ``(n_covariates, 2)`` confidence interval for the effect size: a
        leave-one-out jackknife t-interval under ``inference="context"`` (the
        default), a bootstrap percentile interval under ``inference="paper"``.
    intercept : numpy.ndarray
        The ``(D,)`` intercept (reference-level ALC embedding).
    statistic : str
        ``"norm"``, ``"squared"`` or ``"squared_deflated"``.
    """

    coefficients: np.ndarray
    names: list
    normed_estimate: np.ndarray
    p_value: np.ndarray
    normed_ci: np.ndarray
    intercept: np.ndarray
    statistic: str
    confidence_level: float
    n_obs: int
    reference_level: object = None
    _pretrained_words: list = field(default=None, repr=False)
    _pretrained_matrix: np.ndarray = field(default=None, repr=False)

    # -- interpretation -----------------------------------------------------
    def predict(self, profile) -> np.ndarray:
        """Predicted ALC embedding at a covariate profile (``{name: value}`` or a
        length-``p`` array in ``names`` order; unset covariates default to 0)."""
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
        """Pretrained words most similar to the predicted embedding at ``profile``;
        list of ``(word, cosine)``, highest first. ``candidates`` restricts the pool."""
        if self._pretrained_matrix is None:
            raise RuntimeError("nearest_neighbors needs the pretrained embeddings kept at fit")
        return _cosine_topn(self.predict(profile), self._pretrained_words,
                            self._pretrained_matrix, n, candidates)

    def nns_ratio(self, profile_a, profile_b, *, n=10, candidates=None):
        """Contrast two profiles: for each candidate word, ``cos(y_a, q)/cos(y_b, q)``.
        Only words with a positive cosine to *both* profiles are ranked (the ratio is
        meaningless when a cosine is <= 0); ``ratio > 1`` means closer to ``a``."""
        if self._pretrained_matrix is None:
            raise RuntimeError("nns_ratio needs the pretrained embeddings kept at fit")
        ya, yb = self.predict(profile_a), self.predict(profile_b)
        words, M = self._pretrained_words, self._pretrained_matrix
        if candidates is None:
            top = {w for w, _ in _cosine_topn(ya, words, M, n, None)}
            top |= {w for w, _ in _cosine_topn(yb, words, M, n, None)}
            candidates = top
        idx = [i for i, w in enumerate(words) if w in candidates]
        if not idx:
            return []
        Mn = M[idx] / (np.linalg.norm(M[idx], axis=1, keepdims=True) + 1e-12)
        ca = Mn @ (ya / (np.linalg.norm(ya) + 1e-12))
        cb = Mn @ (yb / (np.linalg.norm(yb) + 1e-12))
        out = [(words[idx[i]], float(ca[i] / cb[i]))
               for i in range(len(idx)) if ca[i] > 0 and cb[i] > 0]
        out.sort(key=lambda t: -t[1])
        return out

    def summary(self):
        """A compact text table of covariate, effect size, CI and p-value."""
        lines = [f"Embedding regression ({self.statistic}, "
                 f"{int(self.confidence_level * 100)}% CI, n={self.n_obs})"]
        if self.reference_level is not None:
            lines.append(f"reference level: {self.reference_level}")
        if self.statistic == "squared_deflated":
            lines.append("(deflated = squared norm - HC1 bias; can be <=0 under the null)")
        # Widen the covariate column to the longest name so dummy names like
        # ``party_rec.sport.baseball`` do not push the numeric columns out of line.
        w = max(20, *(len(n) for n in self.names)) if self.names else 20
        lines.append(f"{'covariate':<{w}s} {'estimate':>10s} {'ci_low':>9s} "
                     f"{'ci_high':>9s} {'p':>7s}")
        for i, nm in enumerate(self.names):
            lines.append(f"{nm:<{w}s} {self.normed_estimate[i]:10.4f} "
                         f"{self.normed_ci[i, 0]:9.4f} {self.normed_ci[i, 1]:9.4f} "
                         f"{self.p_value[i]:7.3f}")
        return "\n".join(lines)


def _cosine_topn(vec, words, matrix, n, candidates):
    if candidates is not None:
        cset = set(candidates)
        idx = np.array([i for i, w in enumerate(words) if w in cset])
        if idx.size == 0:
            return []
    else:
        idx = np.arange(len(words))
    M = matrix[idx]
    sims = (M @ vec) / ((np.linalg.norm(M, axis=1) * np.linalg.norm(vec)) + 1e-12)
    order = np.argsort(-sims)[:n]
    return [(words[idx[i]], float(sims[i])) for i in order]


def _ols(X, Y):
    """Multivariate OLS ``B = (X'X)^-1 X'Y``; returns ``B`` and ``(X'X)^-1``.
    Warns if the design is rank-deficient (collinear/constant covariates), which
    makes the min-norm ``pinv`` solution and the intercept unreliable."""
    import warnings

    if np.linalg.matrix_rank(X) < X.shape[1]:
        warnings.warn(
            "covariate design is rank-deficient (constant or collinear covariate); "
            "coefficients and the intercept are not uniquely identified.",
            stacklevel=3,
        )
    XtX_inv = np.linalg.pinv(X.T @ X)
    B = XtX_inv @ (X.T @ Y)
    return B, XtX_inv


def _effect_size(B, X, Y, XtX_inv, statistic):
    """Per-covariate effect size from the coefficient block ``B[1:]`` (intercept at 0).
    ``squared_deflated`` subtracts the HC1 robust bias ``sum_d Var_hc1(beta_jd)`` to
    match conText's ``normed.estimate.deflated``."""
    beta = B[1:]
    sq = np.sum(beta ** 2, axis=1)
    if statistic == "norm":
        return np.sqrt(sq)
    if statistic == "squared":
        return sq
    if statistic == "squared_deflated":
        N, k = X.shape
        resid = Y - X @ B
        XtXi_Xt = XtX_inv @ X.T                      # (k, N)
        scale = N / max(N - k, 1)                     # HC1 finite-sample correction
        bias = np.zeros(B.shape[0] - 1)
        for j in range(1, B.shape[0]):
            hj = XtXi_Xt[j]                            # (N,)
            # sum_d Var(beta_jd) = sum_d sum_i hj_i^2 e_id^2  (HC0) then HC1-scaled
            bias[j - 1] = scale * np.sum((hj ** 2)[:, None] * resid ** 2)
        return sq - bias
    raise ValueError("statistic must be 'norm', 'squared' or 'squared_deflated'")


def _residual_permutation_p(X, Y, B, est, statistic, n_perm, rng):
    """Freedman-Lane residual permutation p-value (conText's procedure): permute the
    rows of the fitted residuals, refit, and compare the effect size to ``est``."""
    resid = Y - X @ B
    N = X.shape[0]
    p = B.shape[0] - 1
    ge = np.zeros(p)
    for _ in range(n_perm):
        rp = resid[rng.permutation(N)]
        Bp, inv_p = _ols(X, rp)
        ge += (_effect_size(Bp, X, rp, inv_p, statistic) >= est)
    return (1.0 + ge) / (1.0 + n_perm)


def _jackknife_ci(X, Y, statistic, est, confidence_level):
    """Leave-one-out jackknife SE and t-interval for the effect size (conText's
    default CI). Returns ``(ci (p,2), se (p,))``."""
    from scipy.stats import t as _t

    N, k = X.shape
    p = k - 1
    thetas = np.empty((N, p))
    keep = np.ones(N, dtype=bool)
    for i in range(N):
        keep[i] = False
        Xi, Yi = X[keep], Y[keep]
        Bi, inv_i = _ols(Xi, Yi)
        thetas[i] = _effect_size(Bi, Xi, Yi, inv_i, statistic)
        keep[i] = True
    theta_bar = thetas.mean(axis=0)
    se = np.sqrt((N - 1) / N * np.sum((thetas - theta_bar) ** 2, axis=0))
    tcrit = _t.ppf(1 - (1 - confidence_level) / 2, N - 1)
    ci = np.column_stack([est - tcrit * se, est + tcrit * se])
    return ci, se


def embedding_regression(
    docs,
    covariates,
    pre_trained,
    *,
    names=None,
    transform="additive",
    target=None,
    window=6,
    aggregate=None,
    statistic="squared_deflated",
    inference="context",
    permutations=100,
    bootstrap=100,
    confidence_level=0.95,
    case_insensitive=True,
    seed=0,
    keep_pretrained=True,
):
    """Regress à la carte embeddings on covariates (conText embedding regression).

    Parameters
    ----------
    docs : sequence of token lists
    covariates : array-like
        ``(N,)`` or ``(N, p)`` numeric design (no intercept), or a ``(N,)`` sequence
        of category labels (dummy-coded, first level dropped as reference).
    pre_trained : dict or (matrix, vocab)
    names : sequence of str, optional
        Covariate names (for a categorical covariate, the single source name).
    transform : {"additive", "estimate"}, numpy.ndarray, or None, default "additive"
        ALC transform ``A``. ``"additive"``/``None`` = identity; ``"estimate"`` calls
        :func:`compute_transform`; or pass a matrix (a ``conText`` matrix must be
        transposed first).
    target : str, sequence of str, or None
        Focal term(s); ``None`` embeds whole documents.
    window : int, default 6
    aggregate : {"instance", "document"}, optional
        Unit of analysis for a focal term. Defaults to ``"instance"`` (one row per
        mention, as conText) when ``target`` is given, else ``"document"``.
    statistic : {"squared_deflated", "squared", "norm"}, default "squared_deflated"
        Effect size. ``"squared_deflated"`` is conText's headline (squared norm minus
        HC1 bias; centered at ~0 under the null). ``"norm"`` is the paper's Euclidean
        norm (biased upward). ``"squared"`` is the raw squared norm.
    inference : {"context", "paper"}, default "context"
        How the p-value and confidence interval are computed. ``"context"`` follows
        the current R ``conText`` package's procedure: a Freedman-Lane
        residual-permutation p-value and a leave-one-out jackknife t-interval (the
        interval is centered on the estimate and matches conText's to numerical
        precision). The permutation p-value uses the selected ``statistic`` and the
        validity-preserving ``(1 + #ge) / (1 + permutations)`` smoothing, so it will
        not be bit-identical to conText's unsmoothed count (conText also permutes the
        uncorrected norm); the effect-size estimates and the jackknife interval are.
        ``"paper"`` is the original article's method: a covariate permutation p-value
        and a bootstrap percentile interval over the resampled rows (the norm's
        upward bias makes that interval sit above the point estimate).
    permutations : int, default 100
        Permutations for the p-value; 0 skips. p is smoothed as
        ``(1 + #{perm >= obs}) / (1 + permutations)``.
    bootstrap : int, default 100
        Bootstrap resamples for the ``"paper"`` CI; 0 skips. Ignored for ``"context"``
        (which uses the jackknife).
    confidence_level : float, default 0.95
    case_insensitive : bool, default True
    seed : int, default 0
    keep_pretrained : bool, default True

    Returns
    -------
    EmbeddingRegression
    """
    rng = np.random.default_rng(seed)
    words, matrix, _ = _as_lookup(pre_trained, case_insensitive)

    if isinstance(transform, str):
        if transform == "estimate":
            A = compute_transform(docs, (matrix, words), window=window,
                                  case_insensitive=case_insensitive)
        elif transform == "additive":
            A = None
        else:
            raise ValueError("transform string must be 'additive' or 'estimate'")
    elif transform is None:
        A = None
    else:
        A = np.asarray(transform, dtype=np.float64)

    if aggregate is None:
        aggregate = "instance" if target is not None else "document"

    Y, doc_ids = alc_embeddings(docs, (matrix, words), transform=A, target=target,
                                window=window, case_insensitive=case_insensitive,
                                aggregate=aggregate)

    Xraw_full, names, reference = _build_design(covariates, names)
    if Xraw_full.shape[0] != len(docs):
        raise ValueError("covariates must have one row per document")
    Xraw = Xraw_full[doc_ids]  # expand/align to the surviving rows (instances or docs)
    p = Xraw.shape[1]
    N = Xraw.shape[0]
    X = np.column_stack([np.ones(N), Xraw])
    if inference not in ("context", "paper"):
        raise ValueError("inference must be 'context' or 'paper'")
    B, XtX_inv = _ols(X, Y)
    est = _effect_size(B, X, Y, XtX_inv, statistic)

    # p-value
    if not permutations or permutations <= 0:
        p_value = np.full(p, np.nan)
    elif inference == "context":
        p_value = _residual_permutation_p(X, Y, B, est, statistic, permutations, rng)
    else:  # paper: permute the covariate
        ge = np.zeros(p)
        for _ in range(permutations):
            perm = rng.permutation(N)
            Xp = np.column_stack([np.ones(N), Xraw[perm]])
            Bp, inv_p = _ols(Xp, Y)
            ge += (_effect_size(Bp, Xp, Y, inv_p, statistic) >= est)
        p_value = (1.0 + ge) / (1.0 + permutations)

    # confidence interval
    if inference == "context":
        ci, _se = _jackknife_ci(X, Y, statistic, est, confidence_level)
    elif bootstrap and bootstrap > 0:
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
        n_obs=N,
        reference_level=reference,
        _pretrained_words=words if keep_pretrained else None,
        _pretrained_matrix=matrix if keep_pretrained else None,
    )
