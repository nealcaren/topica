from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Union, overload
import numpy
import numpy.typing

DEFAULT_TOKEN_REGEX: str
__version__: str


def set_experimental(enabled: bool) -> None: ...
def experimental_is_enabled() -> bool: ...


def tokenize(
    text: str,
    *,
    lowercase: bool = True,
    stopwords: Iterable[str] | None = None,
    token_regex: str | None = None,
    min_length: int = 1,
) -> list[str]:
    """Tokenize a string with the corpus loader's regex; lowercase, drop short
    tokens and stopwords. `stopwords` is any iterable of strings (list, set, or
    `topica.ENGLISH_STOPWORDS`). Convenience for building list[list[str]] input."""
    ...


def project(
    data: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]],
    n_components: int = 2,
    *,
    method: str = "pca",
    n_neighbors: int = 15,
    perplexity: float = 30.0,
    seed: int = 0,
) -> numpy.typing.NDArray[numpy.float64]:
    """Project a high-dimensional array to `n_components` for plotting or clustering.

    `method` is "pca" (default, deterministic, distance-faithful), "umap", or
    "tsne". UMAP and t-SNE preserve local neighborhoods but distort global geometry
    (between-cluster distances and cluster sizes are not meaningful) and are not
    reproducible across runs (a warning is issued); PCA is the honest default.
    `data` is a 2D float array or a list of float lists. Returns an
    `(n_rows, n_components)` array.
    """
    ...


def inspect_frex_scores(
    beta: list[list[float]],
    word_counts: list[int],
    w: float = 0.5,
) -> list[list[float]]:
    """stm-faithful FREX score matrix (K x V) from topica-core's `inspect` (internal).

    `word_counts` (length V) enables stm's James-Stein exclusivity shrinkage when
    non-empty; pass [] to skip it. Backs the cross-language FREX parity check
    against the pure-Python topica.frex.
    """
    ...

def inspect_lift_scores(
    beta: list[list[float]],
    word_counts: list[int],
) -> list[list[float]]:
    """stm-faithful lift matrix (K x V): log(beta) - log(empirical word freq) (internal)."""
    ...

def inspect_score_scores(beta: list[list[float]]) -> list[list[float]]:
    """stm-faithful score matrix (K x V): beta * (log beta - mean_k log beta) (internal)."""
    ...

def inspect_exclusivity(
    beta: list[list[float]],
    m: int,
    frexw: float = 0.7,
) -> list[float]:
    """stm-faithful per-topic exclusivity (FREX summary over top-m words) (internal)."""
    ...

def inspect_semantic_coherence(
    beta: list[list[float]],
    docs: list[list[int]],
    m: int,
) -> list[float]:
    """stm-faithful per-topic semantic coherence (semCoh1beta) over top-m words (internal)."""
    ...

def window_cooccurrence(
    docs: list[list[int]],
    num_relevant: int,
    pairs: list[tuple[int, int]],
    window: int,
) -> tuple[list[float], list[float], float]:
    """Window/document co-occurrence counts for coherence scoring (internal).

    docs holds relevant-word ids per token, 4294967295 marks a non-relevant
    token; pairs are (a, b) with a < b; window=0 requests document-level
    co-occurrence. Returns (occ, co, n_windows). Used by topica.coherence.
    """
    ...


class Corpus:
    """A preprocessed token corpus for LDA training."""

    @staticmethod
    def from_documents(
        documents: list[list[str]],
        *,
        doc_names: list[str] | None = None,
        doc_labels: list[str] | None = None,
        stopwords: list[str] | None = None,
        min_doc_freq: int = 1,
        max_doc_fraction: float = 1.0,
        min_cf: int = 0,
        rm_top: int = 0,
    ) -> Corpus:
        """Build a Corpus from a list of token lists.

        A document left with no tokens by pruning is dropped, so ``num_docs`` can
        be smaller than ``len(documents)``; the surviving original indices are in
        ``kept_indices`` (realign external covariates with ``X[corpus.kept_indices]``).
        ``min_cf`` drops words whose corpus frequency is below the threshold.
        ``rm_top`` removes the top-N most frequent words before any other pruning.
        """
        ...

    @staticmethod
    def from_text_file(
        path: str,
        *,
        format: str = "plain",
        id_field: bool = False,
        id_column: int = 0,
        label_column: int | None = 1,
        text_column: int = 2,
        token_regex: str | None = None,
        stopwords: list[str] | None = None,
        min_doc_freq: int = 1,
        max_doc_fraction: float = 1.0,
    ) -> Corpus:
        """Build a Corpus by reading and tokenizing a text file."""
        ...

    @staticmethod
    def load(path: str) -> Corpus:
        """Load a binary corpus previously saved by .save() or the preprocess CLI."""
        ...

    def save(self, path: str) -> None:
        """Serialize the corpus to a binary file."""
        ...

    @property
    def num_docs(self) -> int:
        """Number of documents in the corpus."""
        ...

    @property
    def num_words(self) -> int:
        """Vocabulary size (number of unique word types)."""
        ...

    @property
    def total_tokens(self) -> int:
        """Total number of tokens across all documents."""
        ...

    @property
    def doc_lengths(self) -> list[int]:
        """Tokens per document in the pruned vocabulary, parallel to a model's
        ``doc_topic`` rows. The N_d that ``dirichlet_theta_samples`` needs."""
        ...

    @property
    def word_counts(self) -> list[int]:
        """Total occurrences of each vocabulary term across all documents, parallel
        to ``vocabulary``. The empirical P(w) for stm's lift / FREX shrinkage."""
        ...

    @property
    def vocabulary(self) -> list[str]:
        """Ordered list of vocabulary terms."""
        ...
    def documents(self) -> list[list[str]]:
        """The corpus as token lists (one per document), the inverse of
        from_documents."""
        ...

    @property
    def kept_indices(self) -> list[int]:
        """Original document indices that survived pruning, parallel to the rows
        of this corpus. Use to realign an external covariate array/DataFrame:
        ``X = X[corpus.kept_indices]`` (see :func:`topica.align`)."""
        ...

    metadata: object | None
    """Optional per-document metadata aligned to the surviving rows (a pandas
    DataFrame, set by :func:`topica.from_dataframe`, or assigned directly)."""

    @property
    def doc_names(self) -> list[str]:
        """Document identifiers, one per document."""
        ...

    @property
    def doc_labels(self) -> list[str]:
        """Document labels, one per document."""
        ...

    def __repr__(self) -> str: ...


class DMR:
    """Dirichlet-Multinomial Regression topic model (Mimno & McCallum 2008).

    Like LDA, but the per-document topic prior is log-linear in document
    features: alpha_{d,t} = exp(lambda_t . x_d). After fitting, the learned
    weights are in `feature_effects`.
    """

    def __init__(
        self,
        num_topics: int,
        *,
        beta: float = 0.01,
        optimize_interval: int = 50,
        burn_in: int = 200,
        seed: int = 42,
        prior_variance: float = 1.0,
        lbfgs_iters: int = 20,
        sampler: str = "sparse",
    ) -> None:
        """Create an unfitted DMR model. prior_variance is the Gaussian prior
        variance on the feature weights; lbfgs_iters caps L-BFGS steps per round.

        sampler selects the inference backend: "sparse" (default) is the
        SparseLDA collapsed-Gibbs sweep with the per-document DMR prior; "warp"
        is the WarpLDA cache-efficient sampler (per-document-α doc phase), whose
        per-sweep cost is flat in K; "cvb0" is deterministic collapsed
        variational Bayes (per-document α; the soft expected counts feed the λ
        optimizer directly), the quality choice at the cost of O(K)-per-token
        compute. As with plain LDA, prefer "sparse" up to ~K=200, "warp" for
        large-K (K >= ~500); the "warp"/"cvb0" paths record no convergence
        trace, so convergence_tol has no effect there."""
        ...

    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        features: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]] | None = None,
        *,
        feature_names: list[str] | None = None,
        iters: int = 1000,
        num_samples: int = 5,
        sample_interval: int = 25,
        progress: object | None = None,
        progress_interval: int = 50,
        keep_theta_draws: bool = True,
        num_theta_draws: int = 25,
        convergence_tol: float = 0.0,
        check_every: int = 10,
        covariates: Optional[numpy.typing.NDArray[numpy.float64]] = None,
    ) -> None:
        """Fit by collapsed Gibbs with the per-document Dirichlet prior
        alpha_{d,t} = exp(lambda_t . x_d). `features` (or `covariates`, a
        symmetric alias) is required: an (num_docs, F) covariate matrix (no
        intercept column — one is prepended), with feature_names naming the F
        columns. The L-BFGS optimization of lambda runs every optimize_interval
        sweeps after burn_in; topic-word phi is averaged over num_samples samples
        taken every sample_interval sweeps."""
        ...

    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]:
        """phi matrix of shape (num_topics, num_words)."""
        ...

    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]:
        """theta matrix of shape (num_docs, num_topics); rows sum to 1."""
        ...

    @property
    def alpha(self) -> numpy.typing.NDArray[numpy.float64]:
        """The baseline document-topic Dirichlet prior alpha, shape (num_topics,):
        exp(lambda_intercept), the per-topic prior at covariates = 0."""
        ...

    @property
    def theta_draws(self) -> numpy.typing.NDArray[numpy.float32] | None:
        """Thinned MCMC theta draws, shape (num_draws, num_docs, num_topics), float32,
        or None when keep_theta_draws=False."""
        ...

    @property
    def doc_lengths(self) -> list[int]:
        """Number of tokens in each training document."""
        ...

    @property
    def feature_effects(self) -> numpy.typing.NDArray[numpy.float64]:
        """Learned feature weights lambda, shape (num_topics, num_features). Column
        0 is the intercept; positive entries raise that topic's prevalence as the
        feature increases."""
        ...

    @property
    def feature_names(self) -> list[str]:
        """Feature names aligned with feature_effects columns ('intercept' first)."""
        ...

    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    @property
    def num_topics(self) -> int: ...
    @property
    def topic_names(self) -> list[str]: ...
    @topic_names.setter
    def topic_names(self, value: list[str]) -> None: ...

    @overload
    def top_words(self, n: int = ..., *, topic: int) -> list[tuple[str, float]]: ...
    @overload
    def top_words(self, n: int = ..., *, topic: None = ...) -> list[list[tuple[str, float]]]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]:
        """Top n (word, probability) pairs for one or all topics."""
        ...

    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]:
        """UMass topic coherence per topic, shape (num_topics,)."""
        ...

    def transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        features: numpy.typing.NDArray[numpy.float64] | None = None,
        *,
        iters: int = 100,
        burn_in: int = 10,
        num_samples: int = 10,
        sample_interval: int = 5,
        seed: int | None = None,
        iterations: int | None = None,
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Infer document-topic theta for new documents by collapsed Gibbs
        against the fitted topic-word matrix. `features` (optional, no intercept)
        sets each document's Dirichlet prior alpha_d = exp(Xgamma); if omitted
        the intercept-only baseline is used. Shape (num_new_docs, num_topics).
        `iterations` is deprecated; use `iters` instead."""
        ...

    def save(self, path: str) -> None:
        """Persist the fitted model to path. Reload with DMR.load."""
        ...

    @staticmethod
    def load(path: str) -> "DMR":
        """Load a model previously written by save."""
        ...

    @property
    def fit_history(self) -> list[tuple[int, float]]:
        """Per-iteration ``(iteration, objective)`` trace recorded every
        ``check_every`` sweeps during :meth:`fit` (empty when ``check_every=0``)."""
        ...
    @property
    def converged(self) -> bool:
        """True if fit early-stopped because the relative change in the objective
        fell below ``convergence_tol``; False when the full ``iters`` ran (the
        default, opt-in early stopping)."""
        ...
    def __repr__(self) -> str: ...


class CTM:
    """Correlated Topic Model (Blei & Lafferty; STM's logistic-normal core).
    Topics drawn from a logistic-normal prior with full covariance, so they can
    correlate (unlike LDA's Dirichlet). Fit by variational EM (STM's Laplace
    E-step)."""

    def __init__(
        self,
        num_topics: int,
        *,
        sigma_shrink: float = 0.0,
        seed: int = 42,
        init: str = "spectral",
        variational: str = "laplace",
    ) -> None:
        """num_topics >= 2. sigma_shrink in [0,1] shrinks topic covariance toward
        diagonal. init is "spectral" (default; deterministic anchor-word init,
        matching STM's default — seed is then irrelevant) or "random" (seeded).
        variational is "laplace" (default; full posterior covariance nu = H^-1)
        or "diagonal" (mean-field nu = diag(1/H_ii); faster at high K, drops the
        off-diagonal posterior covariance)."""
        ...

    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iters: int = 500,
        convergence_tol: float = 1e-5,
        inference: str = "batch",
        batch_size: int = 256,
        tau: float = 64.0,
        kappa: float = 0.7,
        beta_init: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]] | None = None,
        em_tol: Optional[float] = None,
        keep_eta_cov: bool = True,
        num_threads: Optional[int] = None,
    ) -> None:
        """EM stops once the relative change in the variational bound falls below
        convergence_tol or after iters iterations, whichever comes first. Pass
        convergence_tol=0 to always run iters steps. Check converged and bound
        afterward.

        inference selects the backend: "batch" (default, full variational EM) or
        "svi" (stochastic/online VB, Hoffman et al. 2013) for very large corpora.
        Under "svi", iters is the number of epochs (passes over the data), and the
        global parameters update from minibatches of batch_size docs with a Robbins-
        Monro step rho_t = (tau + t)^(-kappa). tau (>= 0) and kappa in (0.5, 1] set
        the learning-rate schedule; convergence_tol is ignored. SVI does not retain a
        per-iteration bound trace.

        beta_init (K x num_words) overrides the spectral/random topic-word
        initialization with a caller-supplied base beta -- the warm-start hook for
        reproducing an external fit (e.g. R stm's exact spectral beta). Batch only
        (not supported with inference="svi").

        keep_eta_cov=False skips storing the per-document variational covariance (nu),
        saving O(N*K^2) memory. The fit is bit-identical. Use _recompute_eta_cov() or
        posterior_theta_samples() (which falls back automatically) to regenerate nu.

        em_tol is a deprecated alias for convergence_tol."""
        ...

    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def bound(self) -> float:
        """Final variational bound (approximate ELBO) at convergence."""
        ...
    @property
    def bound_history(self) -> list[float]:
        """Variational bound after each EM iteration (length = iterations run)."""
        ...
    @property
    def converged(self) -> bool:
        """True if EM met em_tol; False if it hit the iters cap."""
        ...
    @property
    def variational(self) -> str:
        """Variational-covariance mode: "laplace" (full nu = H^-1) or "diagonal"
        (mean-field nu = diag(1/H_ii))."""
        ...
    @property
    def fit_history(self) -> list[tuple[int, float]]:
        """Per-iteration variational bound trace: list of (iteration, bound) pairs."""
        ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def topic_correlation(self) -> numpy.typing.NDArray[numpy.float64]:
        """Topic-correlation matrix (num_topics, num_topics) from theta across docs."""
        ...
    @property
    def eta_mean(self) -> numpy.typing.NDArray[numpy.float64]:
        """Variational posterior means lambda, shape (num_docs, num_topics-1)."""
        ...
    @property
    def eta_cov(self) -> numpy.typing.NDArray[numpy.float32]:
        """Variational posterior covariances nu, shape (num_docs, K-1, K-1).
        Stored as float32 to halve memory; cast with np.asarray(model.eta_cov, dtype=np.float64) if needed.
        Raises RuntimeError if the model was fit with keep_eta_cov=False."""
        ...
    def _recompute_eta_cov(self) -> numpy.typing.NDArray[numpy.float32]:
        """Recompute the per-document variational covariance nu on demand.
        Use when the model was fit with keep_eta_cov=False. Returns the same
        (num_docs, K-1, K-1) float32 array as eta_cov."""
        ...
    @property
    def topic_covariance(self) -> numpy.typing.NDArray[numpy.float64]:
        """The fitted logistic-normal prior covariance Sigma over eta, shape
        (K-1, K-1); the last topic is the softmax reference. The model's own topic
        covariance (cf. topic_correlation, an across-document theta correlation)."""
        ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    @property
    def num_topics(self) -> int: ...
    @property
    def topic_names(self) -> list[str]: ...
    @topic_names.setter
    def topic_names(self, value: list[str]) -> None: ...

    @overload
    def top_words(self, n: int = ..., *, topic: int) -> list[tuple[str, float]]: ...
    @overload
    def top_words(self, n: int = ..., *, topic: None = ...) -> list[list[tuple[str, float]]]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...

    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def transform(
        self, data: Corpus | Sequence[Sequence[str]]
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Infer document-topic theta for new documents by the variational
        E-step against the fitted globals. Shape (num_new_docs, num_topics)."""
        ...

    def save(self, path: str) -> None:
        """Persist the fitted model to path. Reload with CTM.load."""
        ...

    @staticmethod
    def load(path: str) -> "CTM":
        """Load a model previously written by save."""
        ...

    def __repr__(self) -> str: ...


class STM:
    """Structural Topic Model (Roberts, Stewart & Tingley): the correlated-topic
    core (CTM) plus prevalence covariates — the prior topic mean is a regression
    on document covariates (mu_d = X_d gamma)."""

    def __init__(
        self,
        num_topics: int,
        *,
        sigma_shrink: float = 0.0,
        seed: int = 42,
        init: str = "spectral",
        variational: str = "laplace",
    ) -> None:
        """init is "spectral" (default; deterministic anchor-word init matching
        STM's default) or "random" (seeded). With a content model the per-group
        beta is always random. variational is "laplace" (default; full posterior
        covariance nu = H^-1) or "diagonal" (mean-field nu = diag(1/H_ii); faster
        at high K, drops the off-diagonal posterior covariance)."""
        ...

    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        prevalence: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]] | None = None,
        *,
        prevalence_names: list[str] | None = None,
        content: Sequence[str] | Sequence[int] | None = None,
        content_names: list[str] | None = None,
        iters: int = 500,
        convergence_tol: float = 1e-5,
        gamma_prior: str = "pooled",
        gamma_enet: float = 1.0,
        beta_init: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]] | None = None,
        em_tol: Optional[float] = None,
        covariates: Optional[numpy.typing.NDArray[numpy.float64]] = None,
        keep_eta_cov: bool = True,
        num_threads: Optional[int] = None,
    ) -> None:
        """Fit. prevalence (or covariates, a symmetric alias) is (num_docs, F)
        covariates driving topic proportions (mu_d = X_d gamma; intercept
        prepended). content is one group label per document, making topic-word
        distributions vary by group (SAGE). At least one of
        prevalence/content must be given.

        EM stops once the relative change in the variational bound falls below
        convergence_tol (R stm's emtol) or after iters iterations, whichever
        comes first. Pass convergence_tol=0 to always run iters steps. Check
        converged and bound afterward.

        gamma_prior controls the prevalence-coefficient regression in the M-step:
        "pooled" (default) uses ridge regression; "l1" uses an elastic-net path
        with AIC-selected penalty, recommended for high-dimensional prevalence
        designs. gamma_enet is the elastic-net mix (1.0 = pure lasso, values in
        (0,1) add ridge; R stm's gamma.enet). Ignored when gamma_prior="pooled".

        beta_init (K x num_words) overrides the spectral/random topic-word
        initialization with a caller-supplied base beta -- the warm-start hook for
        reproducing an external fit (e.g. R stm's exact spectral beta).

        keep_eta_cov=False skips storing the per-document variational covariance (nu),
        saving O(N*K^2) memory. The fit is bit-identical. Use _recompute_eta_cov() or
        posterior_theta_samples() (which falls back automatically) to regenerate nu.

        em_tol is a deprecated alias for convergence_tol."""
        ...

    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def bound(self) -> float:
        """Final variational bound (approximate ELBO) at convergence — R stm's
        convergence$bound."""
        ...
    @property
    def bound_history(self) -> list[float]:
        """Variational bound after each EM iteration (length = iterations run)."""
        ...
    @property
    def converged(self) -> bool:
        """True if EM met em_tol; False if it hit the iters cap."""
        ...
    @property
    def variational(self) -> str:
        """Variational-covariance mode: "laplace" (full nu = H^-1) or "diagonal"
        (mean-field nu = diag(1/H_ii))."""
        ...
    @property
    def fit_history(self) -> list[tuple[int, float]]:
        """Per-iteration variational bound trace: list of (iteration, bound) pairs."""
        ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def topic_correlation(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def eta_mean(self) -> numpy.typing.NDArray[numpy.float64]:
        """Variational posterior means lambda, shape (num_docs, num_topics-1).
        With eta_cov, the logistic-normal posterior for method-of-composition."""
        ...
    @property
    def eta_cov(self) -> numpy.typing.NDArray[numpy.float32]:
        """Variational posterior covariances nu, shape (num_docs, K-1, K-1).
        Stored as float32 to halve memory; cast with np.asarray(model.eta_cov, dtype=np.float64) if needed.
        Raises RuntimeError if the model was fit with keep_eta_cov=False."""
        ...
    def _recompute_eta_cov(self) -> numpy.typing.NDArray[numpy.float32]:
        """Recompute the per-document variational covariance nu on demand.
        Use when the model was fit with keep_eta_cov=False. Returns the same
        (num_docs, K-1, K-1) float32 array as eta_cov."""
        ...
    @property
    def topic_covariance(self) -> numpy.typing.NDArray[numpy.float64]:
        """The fitted logistic-normal prior covariance Sigma over eta, shape
        (K-1, K-1); the last topic is the softmax reference. The model's own topic
        covariance (cf. topic_correlation, an across-document theta correlation)."""
        ...
    @property
    def prevalence_effects(self) -> numpy.typing.NDArray[numpy.float64]:
        """gamma, shape (num_features, num_topics-1). RuntimeError if no
        prevalence. Prefer topica.stm.estimate_effect for inference."""
        ...
    @property
    def feature_names(self) -> list[str]: ...
    @property
    def topic_word_by_group(self) -> numpy.typing.NDArray[numpy.float64]:
        """Per-group topic-word, shape (num_topics, num_groups, num_words).
        RuntimeError if fit without content covariates."""
        ...
    @property
    def content_kappa(self) -> dict[str, numpy.typing.NDArray[numpy.float64]]:
        """The SAGE content-model kappa decomposition as a dict: 'm' (num_words,),
        'kappa_topic' (num_topics, num_words), 'kappa_cov' (num_groups, num_words),
        'kappa_interaction' (num_topics, num_groups, num_words). Per-group
        log-probabilities are m + kappa_topic + kappa_cov + kappa_interaction
        (softmax over words). These are the additive parts R stm's sageLabels()
        ranks words by; the per-group beta alone does not identify them.
        RuntimeError if fit without content covariates."""
        ...
    @property
    def groups(self) -> list[str]:
        """Content group names (axis-1 of topic_word_by_group). RuntimeError if
        fit without content covariates."""
        ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    @property
    def num_topics(self) -> int: ...
    @property
    def topic_names(self) -> list[str]: ...
    @topic_names.setter
    def topic_names(self, value: list[str]) -> None: ...

    @overload
    def top_words(self, n: int = ..., *, topic: int) -> list[tuple[str, float]]: ...
    @overload
    def top_words(self, n: int = ..., *, topic: None = ...) -> list[list[tuple[str, float]]]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...

    def word_contrast(
        self, topic: int, group_a: str | int, group_b: str | int, n: int = 10
    ) -> list[tuple[str, float]]:
        """Words most distinguishing how `topic` is worded in group_a vs group_b
        (log word-prob ratio; positive favours group_a). Requires content."""
        ...

    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        eta_prior_mean: numpy.typing.NDArray[numpy.float64] | None = None,
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Infer document-topic theta for new documents by the variational
        E-step against the fitted globals (beta and the logistic-normal prior).
        Shape (num_new_docs, num_topics).

        When eta_prior_mean is None (default), the covariate-free baseline mu
        is used for every document. When eta_prior_mean is a
        (num_docs, num_topics-1) array, row d is the prior mean for document d.
        The ergonomic covariate path is topica.stm.transform."""
        ...

    def save(self, path: str) -> None:
        """Persist the fitted model to path. Reload with STM.load."""
        ...

    @staticmethod
    def load(path: str) -> "STM":
        """Load a model previously written by save."""
        ...

    def __repr__(self) -> str: ...


class ECTM:
    """Evolving Content Topic Model: an STM whose content (topic-word) model
    carries a group-by-time interaction. The same stable topic is worded
    differently across a document group, and that difference drifts across
    discrete time periods, with a first-order random-walk prior tying adjacent
    periods. content_word_dist(group, period) returns the topic-word matrix for a
    cell; the topica.ectm helpers read those to report changing group contrasts."""

    def __init__(
        self,
        num_topics: int,
        *,
        sigma_shrink: float = 0.0,
        seed: int = 42,
        variational: str = "laplace",
        init: str = "spectral",
    ) -> None:
        """sigma_shrink in [0,1] shrinks Sigma toward its diagonal each M-step.
        variational is "laplace" (default; full nu = H^-1) or "diagonal"
        (mean-field). init is "spectral" (default; deterministic anchor-word base
        init matching STM/CTM/STS, with the group-by-period content deviations
        starting at zero as R stm does for kappa) or "random" (seeded). The
        spectral base removes the multimodal collapse a random base beta caused
        (issue #220), so the default fit is bit-exact; with init="random" seed
        matters."""
        ...

    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        times: Sequence[float] | Sequence[str],
        content: Sequence[str] | Sequence[int],
        *,
        prevalence: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]] | None = None,
        prevalence_names: list[str] | None = None,
        content_names: list[str] | None = None,
        period_names: list[str] | None = None,
        iters: int = 500,
        convergence_tol: float = 1e-5,
        content_prior_var: float = 1.0,
        period_smooth: float = 5.0,
        interaction_shrink: float = 2.0,
        inference: str = "batch",
        batch_size: int = 256,
        tau: float = 64.0,
        kappa: float = 0.7,
        content_every: int = 0,
        keep_eta_cov: bool = True,
        num_threads: Optional[int] = None,
    ) -> None:
        """Fit. times is one period label per document (numbers or strings; the
        distinct values, sorted, define the period order). content is one group
        label per document. Both are required and form the (group, period) content
        cells. prevalence (optional, (num_docs, F)) adds the STM prevalence
        regression mu_d = X_d gamma (intercept prepended).

        The content deviations are regularized by an L2 prior (variance
        content_prior_var); a first-order random walk across periods with precision
        period_smooth (larger = smoother / more pooling across adjacent periods);
        and an extra L2 factor interaction_shrink on the group-by-time term.

        inference="batch" (default) is full-batch variational EM. inference="svi"
        is minibatch stochastic VI for corpora too large to fit in batch: iters
        becomes the number of epochs, batch_size documents are sampled per step,
        and the globals move with a Robbins-Monro rate (tau + step)^(-kappa). SVI
        is seed-reproducible (it samples minibatches from the model seed), not
        bit-exact like the default spectral batch fit; convergence_tol is unused in
        SVI mode. content_every sets how often (in minibatches) the expensive
        content-κ M-step is re-solved; the cheap μ/Σ/γ updates run every minibatch.
        content_every=0 (default) re-solves κ once per epoch; a small positive value
        re-solves more often (better per-epoch progress, more cost per epoch)."""
        ...

    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]:
        """Cell-averaged topic-word matrix (K, V). Use content_word_dist for a
        specific (group, period) cell."""
        ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def num_topics(self) -> int: ...
    @property
    def num_groups(self) -> int: ...
    @property
    def num_periods(self) -> int: ...
    @property
    def groups(self) -> list[str]:
        """Content group labels, in index/column order."""
        ...
    @property
    def periods(self) -> list[str]:
        """Time period labels, sorted into the random-walk prior order."""
        ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def prevalence_effects(self) -> numpy.typing.NDArray[numpy.float64]:
        """Prevalence coefficients gamma (num_features, K-1). Raises if fit
        without a prevalence design."""
        ...
    @property
    def feature_names(self) -> list[str]: ...
    @property
    def eta_mean(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def eta_cov(self) -> numpy.typing.NDArray[numpy.float32]:
        """Per-document variational posterior covariances nu (num_docs, K-1, K-1)
        as float32. Raises if fit with keep_eta_cov=False."""
        ...
    @property
    def bound(self) -> float: ...
    @property
    def bound_history(self) -> list[float]: ...
    @property
    def converged(self) -> bool: ...
    @property
    def content_shift_history(self) -> list[float]:
        """SVI only: relative L2 change of the content deviations kappa at each
        content M-step solve (empty for a batch fit). A trailing value still large
        means the content model has not settled."""
        ...
    @property
    def content_converged(self) -> bool:
        """Whether the content model settled: always True for a batch fit; for SVI,
        True when the last content shift is below tolerance. False means the
        between-group divergences may be understated (raise iters / lower
        content_every)."""
        ...
    @property
    def fit_history(self) -> list[tuple[int, float]]: ...
    @property
    def variational(self) -> str: ...
    @property
    def topic_names(self) -> list[str]: ...
    @topic_names.setter
    def topic_names(self, names: list[str]) -> None: ...

    def content_word_dist(
        self, group: str | int, period: str | int
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Topic-word matrix (K, V) for one (group, period) cell. group and period
        accept a label (str) or an index (int)."""
        ...

    def top_words(
        self, n: int = 10, *, topic: Optional[int] = None
    ) -> list[list[tuple[str, float]]] | list[tuple[str, float]]:
        """Top n (word, probability) pairs per topic (or one topic) from the
        cell-averaged beta."""
        ...

    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]:
        """UMass topic coherence per topic, shape (num_topics,)."""
        ...

    def save(self, path: str) -> None:
        """Save the fitted model to path. Reload with ECTM.load."""
        ...

    @staticmethod
    def load(path: str) -> "ECTM":
        """Load a model previously written by save."""
        ...

    def __repr__(self) -> str: ...


class STS:
    """Structural Topic and Sentiment-Discourse model (Chen & Mankad 2024): STM
    plus a per-document, per-topic continuous sentiment-discourse latent that
    modulates the topic-word distribution, with both prevalence and sentiment
    driven by document covariates. Fit by Laplace variational EM."""

    def __init__(
        self,
        num_topics: int,
        *,
        seed: int = 42,
        init: str = "spectral",
    ) -> None:
        """init is "spectral" (default; deterministic anchor-word init) or
        "random" (seeded)."""
        ...

    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        sentiment_seed: Sequence[float],
        prevalence: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]] | None = None,
        *,
        prevalence_names: list[str] | None = None,
        iters: int = 30,
        convergence_tol: float = 1e-5,
        kappa_estimation: str = "ridge",
        kappa_ridge: float = 1e-3,
        em_tol: Optional[float] = None,
        covariates: Optional[numpy.typing.NDArray[numpy.float64]] = None,
        keep_eta_cov: bool = True,
    ) -> None:
        """Fit. sentiment_seed (required, one value per document) defines the
        aggregation groups for the topic-word (kappa) Poisson M-step and seeds the
        initial sentiment — e.g. a star rating the sentiment should track.
        prevalence (or covariates, a symmetric alias) is (num_docs, F) covariates
        driving both topic prevalence and sentiment-discourse (alpha_d ~ N(X_d
        Gamma, Sigma); intercept prepended).

        EM stops once the relative change in the variational bound falls below
        convergence_tol or after iters iterations. kappa_estimation chooses the
        topic-word estimator: "ridge" (default, fast; kappa_ridge sets the ridge)
        or "lasso" (an L1 Poisson path with AIC-selected penalty, matching the
        reference R sts exactly at higher cost). Both give the same topics on
        well-conditioned corpora.

        keep_eta_cov=False skips storing the per-document variational covariance
        (nu), saving O(N*(2K-1)^2) memory. The fit is bit-identical. Use
        _recompute_eta_cov() to regenerate on demand.

        em_tol is a deprecated alias for convergence_tol."""
        ...

    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]:
        """Baseline topic-word matrix beta at neutral sentiment, (num_topics, V)."""
        ...
    def topic_word_at(self, level: float) -> numpy.typing.NDArray[numpy.float64]:
        """Topic-word matrix beta at sentiment level `level` (applied to every
        topic), (num_topics, V). Pass percentiles of `sentiment` to inspect the
        wording at positive vs. negative sentiment."""
        ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]:
        """Document-topic prevalence matrix theta, (num_docs, num_topics)."""
        ...
    @property
    def sentiment(self) -> numpy.typing.NDArray[numpy.float64]:
        """Per-document topic sentiment-discourse alpha^(s), (num_docs, num_topics)."""
        ...
    @property
    def prevalence_effects(self) -> numpy.typing.NDArray[numpy.float64]:
        """Prevalence regression coefficients, (num_features, num_topics-1).
        Requires a prevalence design at fit time."""
        ...
    @property
    def sentiment_effects(self) -> numpy.typing.NDArray[numpy.float64]:
        """Sentiment-discourse regression coefficients, (num_features, num_topics).
        Requires a prevalence design at fit time."""
        ...
    @property
    def eta_mean(self) -> numpy.typing.NDArray[numpy.float64]:
        """Per-document variational posterior means of the logistic-normal latent
        eta = [alpha^(p)_{1..K-1}, alpha^(s)_{1..K}], (num_docs, 2*num_topics-1).
        With eta_cov, the joint prevalence/sentiment posterior."""
        ...
    @property
    def eta_cov(self) -> numpy.typing.NDArray[numpy.float32]:
        """Per-document variational posterior covariances of eta,
        (num_docs, 2*num_topics-1, 2*num_topics-1).
        Stored as float32 to halve memory; cast with np.asarray(model.eta_cov, dtype=np.float64) if needed.
        Raises RuntimeError if the model was fit with keep_eta_cov=False."""
        ...
    def _recompute_eta_cov(self) -> numpy.typing.NDArray[numpy.float32]:
        """Recompute the per-document variational covariance nu on demand.
        Use when the model was fit with keep_eta_cov=False. Returns the same
        (num_docs, 2*num_topics-1, 2*num_topics-1) float32 array as eta_cov."""
        ...
    @property
    def doc_names(self) -> list[str]:
        """Document labels (row order of doc_topic); default index strings."""
        ...
    def transform(
        self, data: Corpus | Sequence[Sequence[str]]
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Infer topic prevalence theta for new documents by the Laplace E-step
        against the fitted globals (kappa, m, Sigma) with a zero prior mean.
        Out-of-vocabulary tokens are dropped. Returns (num_docs, num_topics)."""
        ...
    def save(self, path: str) -> None:
        """Save the fitted model to path. Reload with STS.load."""
        ...
    @staticmethod
    def load(path: str) -> STS:
        """Load a model previously written by save."""
        ...
    @property
    def bound(self) -> float: ...
    @property
    def bound_history(self) -> list[float]: ...
    @property
    def converged(self) -> bool: ...
    @property
    def fit_history(self) -> list[tuple[int, float]]: ...
    @property
    def feature_names(self) -> list[str]: ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def num_topics(self) -> int: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[list[tuple[str, float]]] | list[tuple[str, float]]:
        """Top n (word, probability) pairs per topic (or one topic) at neutral
        sentiment."""
        ...
    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    topic_names: list[str]


class HDP:
    """Hierarchical Dirichlet Process topic model (Teh, Jordan, Beal & Blei
    2006): the nonparametric LDA that *infers* the number of topics rather than
    fixing K. Fit by the direct-assignment Gibbs sampler (Chinese Restaurant
    Franchise). The inferred topic count is read from `num_topics` after fit."""

    def __init__(
        self,
        *,
        alpha: float = 0.1,
        gamma: float = 0.1,
        beta: float = 0.01,
        seed: int = 42,
        resample_conc: bool = False,
        eta: Optional[float] = None,
    ) -> None:
        """alpha/gamma are the document- and corpus-level DP concentrations.
        gamma is the dominant lever on the inferred topic count (0.1 is
        conservative; raise it for more topics). resample_conc defaults to False
        (fixed concentrations -> a stable topic count); set it True to adapt the
        concentrations to the data, which is now capped to avoid the runaway
        topic count it used to cause (issue #68). beta is the topic-word Dirichlet
        (base measure). alpha, gamma, beta must be > 0.

        eta is a deprecated alias for beta."""
        ...

    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iters: int = 150,
        progress_interval: int = 0,
        keep_theta_draws: bool = True,
        num_theta_draws: int = 25,
        report_interval: Optional[int] = None,
    ) -> None:
        """Fit by `iters` Gibbs sweeps. The inferred K is then `num_topics`.

        progress_interval controls the discovery/convergence trace
        (topic_count_history / log_likelihood_history / concentration_history):
        0 (default) records ~50 evenly spaced points; a positive value records
        every that-many sweeps.

        report_interval is a deprecated alias for progress_interval."""
        ...

    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]:
        """Topic-word matrix, shape (num_topics, num_words); rows sum to 1."""
        ...
    @property
    def topic_count_history(self) -> list[tuple[int, int]]:
        """Topic-discovery trajectory: (iteration, num_topics) pairs over the
        fit. Watching K stabilize is HDP's headline convergence check."""
        ...
    @property
    def log_likelihood_history(self) -> list[tuple[int, float]]:
        """Convergence trace: (iteration, per-token log-likelihood) pairs."""
        ...
    @property
    def fit_history(self) -> list[tuple[int, float]]:
        """Uniform convergence trace aliasing log_likelihood_history."""
        ...
    @property
    def converged(self) -> bool:
        """Always False; HDP has no early-stop criterion."""
        ...
    @property
    def concentration_history(self) -> list[tuple[int, float, float]]:
        """Learned-concentration trace: (iteration, alpha, gamma) triples
        (informative when resample_conc=True)."""
        ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]:
        """Document-topic matrix, shape (num_docs, num_topics); rows sum to 1."""
        ...
    @property
    def theta_draws(self) -> numpy.typing.NDArray[numpy.float32] | None:
        """Theta draws sampled from Dirichlet(njk[d]+alpha*beta[k]) after the chain
        ends, shape (num_draws, num_docs, num_topics), or None when
        keep_theta_draws=False."""
        ...
    @property
    def doc_lengths(self) -> list[int]:
        """Number of tokens in each training document."""
        ...
    @property
    def num_topics(self) -> int:
        """The inferred number of topics K (RuntimeError before fit)."""
        ...
    @property
    def alpha(self) -> float:
        """The fitted document-level concentration alpha0."""
        ...
    @property
    def gamma(self) -> float:
        """The fitted corpus-level concentration gamma."""
        ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    @property
    def topic_names(self) -> list[str]: ...
    @topic_names.setter
    def topic_names(self, value: list[str]) -> None: ...

    @overload
    def top_words(self, n: int = ..., *, topic: int) -> list[tuple[str, float]]: ...
    @overload
    def top_words(self, n: int = ..., *, topic: None = ...) -> list[list[tuple[str, float]]]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...

    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iters: int = 100,
        burn_in: int = 10,
        num_samples: int = 10,
        sample_interval: int = 5,
        seed: int | None = None,
        iterations: int | None = None,
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Infer theta over the discovered topics for new documents by collapsed
        Gibbs against the fixed topic-word matrix. Shape (num_new_docs,
        num_topics). `iterations` is deprecated; use `iters` instead."""
        ...

    def save(self, path: str) -> None:
        """Persist the fitted model to path. Reload with HDP.load."""
        ...

    @staticmethod
    def load(path: str) -> "HDP":
        """Load a model previously written by save."""
        ...

    def __repr__(self) -> str: ...


class DTM:
    """Dynamic Topic Model (Blei & Lafferty 2006): topics whose word
    distributions evolve across time slices via a Gaussian state-space model.
    Fit variationally with Kalman smoothing (a port of Blei's C dtm /
    gensim's LdaSeqModel). Query a topic's distribution at a slice with
    topic_word(time) and a word's trajectory with word_evolution(topic, word)."""

    def __init__(
        self,
        num_topics: int,
        *,
        alpha: float = 0.01,
        chain_variance: float = 0.005,
        obs_variance: float = 0.5,
        seed: int = 42,
        init: str = "random",
    ) -> None:
        """num_topics >= 2. chain_variance controls how much a topic may drift
        between adjacent slices (larger = freer). alpha, chain_variance,
        obs_variance must be > 0. init is "random" (default; a seeded static-LDA
        seed, matching gensim's LdaSeqModel) or "spectral" (the deterministic
        anchor-word seed shared with STM/CTM/STS/ECTM, reproducible across seeds;
        choose it when you want a single deterministic fit)."""
        ...

    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        times: Sequence[int],
        *,
        iters: int = 20,
    ) -> None:
        """Fit by variational EM. `times` is each document's integer time-slice
        index (0-based, contiguous); the slice count is max(times)+1."""
        ...

    def topic_word(self, time: int) -> numpy.typing.NDArray[numpy.float64]:
        """Topic-word matrix at `time`, shape (num_topics, num_words); rows sum to 1."""
        ...

    def word_evolution(
        self, topic: int, word: str | int
    ) -> numpy.typing.NDArray[numpy.float64]:
        """A word's probability in `topic` across slices, shape (num_times,)."""
        ...

    def top_words(self, topic: int, time: int, n: int = 10) -> list[tuple[str, float]]:
        """Top n words for `topic` at slice `time` as (word, probability) pairs."""
        ...

    def word_drift(
        self, topic: int, *, n: int = 10, from_time: int = 0, to_time: int | None = None
    ) -> dict[str, list[tuple[str, float]]]:
        """Words inside `topic` whose probability changed most between two slices
        (default first and last). Returns {"rising": [(word, delta)], "falling":
        [(word, delta)]} — what makes the topic's vocabulary evolve."""
        ...

    @property
    def num_topics(self) -> int: ...
    @property
    def num_times(self) -> int: ...
    @property
    def bound(self) -> float:
        """The final variational bound (ELBO) reached during fitting."""
        ...
    @property
    def fit_history(self) -> list[tuple[int, float]]:
        """Per-iteration trace: list of (iteration, objective) pairs. Empty for DTM."""
        ...
    @property
    def converged(self) -> bool:
        """Always False; DTM has no early-stop criterion."""
        ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def topic_names(self) -> list[str]: ...
    @topic_names.setter
    def topic_names(self, value: list[str]) -> None: ...

    def save(self, path: str) -> None:
        """Persist the fitted model to path. Reload with DTM.load."""
        ...

    @staticmethod
    def load(path: str) -> "DTM":
        """Load a model previously written by save."""
        ...

    def __repr__(self) -> str: ...


class DETM:
    """Dynamic Embedded Topic Model (Dieng, Ruiz & Blei 2019): ETM extended to
    time-stamped corpora. The topic embeddings (alpha) and the per-time topic prior
    (eta) each follow a Gaussian random walk, so a topic's words drift smoothly
    across time slices. Fit by minibatch Adam on the ELBO with hand-coded gradients
    (structured amortized variational inference; q(eta) is the reference's multi-layer
    LSTM over the per-time bag of words, with hand-coded backprop-through-time). You
    supply the word embeddings rho like ETM. The headline output is the time-varying
    topic-word tensor beta_over_time (num_times, num_topics, vocab); topic_word is its
    mean over time."""

    def __init__(
        self,
        num_topics: int,
        *,
        delta: float = 0.005,
        hidden_size: int = 800,
        eta_hidden_size: int = 200,
        eta_nlayers: int = 3,
        batch_size: int = 1000,
        lr: float = 0.005,
        wdecay: float = 1.2e-6,
        grad_clip: float | None = None,
        convergence_tol: float = 0.0,
        seed: int = 42,
    ) -> None:
        """num_topics >= 2. delta is the random-walk standard-deviation knob on the
        topic-embedding and topic-prior trajectories (smaller = smoother drift).
        hidden_size is the document encoder width; eta_hidden_size/eta_nlayers size the
        LSTM that amortizes the per-time topic prior q(eta) (reference defaults 200/3);
        batch_size/lr/wdecay drive Adam; convergence_tol stops on the relative change
        in the epoch ELBO (0 disables). grad_clip is an optional global gradient-norm
        clip (the reference's --clip), off by default (None); set a positive float to
        rescale each minibatch's gradients to that global L2 norm before the Adam step,
        which stabilizes training on large vocabularies at higher learning rates. The
        variational log-variances are additionally clamped before every exp for
        numerical stability (internal; never reached on a well-behaved fit)."""
        ...

    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        word_embeddings: numpy.typing.ArrayLike,
        vocabulary: Sequence[str],
        *,
        times: Sequence[int] | None = None,
        timestamps: Sequence[int] | None = None,
        iters: int = 100,
        convergence_tol: float | None = None,
    ) -> None:
        """Fit on `data` with `word_embeddings` (len(vocabulary), L) aligned to
        `vocabulary`. `times` is each document's integer time-slice index (0-based,
        contiguous; `timestamps` is the accepted alias). `iters` is the epoch count."""
        ...

    def topic_word_at(self, t: int) -> numpy.typing.NDArray[numpy.float64]:
        """Topic-word matrix at a single time slice, shape (num_topics, vocab)."""
        ...

    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[list[tuple[str, float]]] | list[tuple[str, float]]:
        """Top n words per topic from the time-collapsed topic_word."""
        ...

    def top_words_at(
        self, t: int, n: int = 10, *, topic: int | None = None
    ) -> list[list[tuple[str, float]]] | list[tuple[str, float]]:
        """Top n words for the topics at a single time slice t."""
        ...

    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]:
        """UMass coherence for each topic's top-n words (time-collapsed)."""
        ...

    @property
    def num_topics(self) -> int: ...
    @property
    def num_times(self) -> int: ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]:
        """Time-collapsed topic-word matrix (num_topics, vocab): mean of beta over
        time. Rows sum to 1."""
        ...
    @property
    def beta_over_time(self) -> numpy.typing.NDArray[numpy.float64]:
        """Time-varying topic-word tensor (num_times, num_topics, vocab)."""
        ...
    @property
    def topic_word_over_time(self) -> numpy.typing.NDArray[numpy.float64]:
        """Alias of beta_over_time (num_times, num_topics, vocab)."""
        ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]:
        """Document-topic proportions theta (num_docs, num_topics)."""
        ...
    @property
    def alpha(self) -> numpy.typing.NDArray[numpy.float64]:
        """Topic-embedding trajectories (num_times, num_topics, L)."""
        ...
    @property
    def eta(self) -> numpy.typing.NDArray[numpy.float64]:
        """Time-varying topic prevalence prior (num_times, num_topics)."""
        ...
    @property
    def bound(self) -> float:
        """The final ELBO reached during fitting."""
        ...
    @property
    def converged(self) -> bool: ...
    @property
    def fit_history(self) -> list[tuple[int, float]]:
        """Per-epoch trace: list of (epoch, ELBO) pairs."""
        ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    @property
    def topic_names(self) -> list[str]: ...
    @topic_names.setter
    def topic_names(self, value: list[str]) -> None: ...

    def save(self, path: str) -> None:
        """Persist the fitted model to path. Reload with DETM.load."""
        ...

    @staticmethod
    def load(path: str) -> "DETM":
        """Load a model previously written by save."""
        ...

    def __repr__(self) -> str: ...


class SupervisedLDA:
    """Supervised LDA (Blei & McAuliffe 2007): LDA where each document has a
    real-valued response y_d ~ N(eta^T zbar_d, sigma^2) regressed on its topic
    usage. Topics are shaped to predict the response; `coefficients` (eta) report
    how each topic moves y. Fit by variational EM; `predict` scores new docs."""

    def __init__(self, num_topics: int, *, alpha: float = 0.1, seed: int = 42) -> None:
        """num_topics >= 2. alpha is the Dirichlet concentration on doc-topic
        proportions; both must be > 0."""
        ...

    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        y: Sequence[float],
        *,
        iters: int = 25,
        var_iters: int = 15,
        keep_theta_draws: bool = True,
        num_theta_draws: int = 25,
        convergence_tol: float = 0.0,
        check_every: int = 1,
    ) -> None:
        """Fit by variational EM. `y` is the per-document response (length =
        number of documents)."""
        ...

    def predict(
        self, data: Corpus | Sequence[Sequence[str]], *, var_iters: int = 20
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Predict y-hat for new documents. Out-of-vocabulary words are ignored.
        Returns a 1-D array of length = number of documents."""
        ...

    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def alpha(self) -> numpy.typing.NDArray[numpy.float64]:
        """The symmetric document-topic Dirichlet prior alpha, shape (num_topics,)."""
        ...
    @property
    def theta_draws(self) -> numpy.typing.NDArray[numpy.float32] | None:
        """Theta draws sampled from Dirichlet(gamma_d) at fit end, shape
        (num_draws, num_docs, num_topics), or None when keep_theta_draws=False."""
        ...
    @property
    def doc_lengths(self) -> list[int]:
        """Number of tokens in each training document."""
        ...
    @property
    def coefficients(self) -> numpy.typing.NDArray[numpy.float64]:
        """Regression coefficients eta, shape (num_topics,) — how each topic
        moves the response per unit of topic frequency."""
        ...
    @property
    def sigma2(self) -> float:
        """The fitted response variance sigma^2."""
        ...
    @property
    def num_topics(self) -> int: ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    @property
    def topic_names(self) -> list[str]: ...
    @topic_names.setter
    def topic_names(self, value: list[str]) -> None: ...

    @overload
    def top_words(self, n: int = ..., *, topic: int) -> list[tuple[str, float]]: ...
    @overload
    def top_words(self, n: int = ..., *, topic: None = ...) -> list[list[tuple[str, float]]]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...

    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iters: int = 100,
        burn_in: int = 10,
        num_samples: int = 10,
        sample_interval: int = 5,
        seed: int | None = None,
        iterations: int | None = None,
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Infer document-topic theta for new documents by collapsed Gibbs
        against the fitted topic-word matrix (the response is not used). Shape
        (num_new_docs, num_topics). Predict the response with transform @ eta.
        `iterations` is deprecated; use `iters` instead."""
        ...
    def save(self, path: str) -> None:
        """Persist the fitted model to path. Reload with SupervisedLDA.load."""
        ...

    @staticmethod
    def load(path: str) -> "SupervisedLDA":
        """Load a model previously written by save."""
        ...

    @property
    def fit_history(self) -> list[tuple[int, float]]:
        """Per-iteration ``(iteration, objective)`` trace recorded every
        ``check_every`` sweeps during :meth:`fit` (empty when ``check_every=0``)."""
        ...
    @property
    def converged(self) -> bool:
        """True if fit early-stopped because the relative change in the objective
        fell below ``convergence_tol``; False when the full ``iters`` ran (the
        default, opt-in early stopping)."""
        ...
    def __repr__(self) -> str: ...


class SAGE:
    """Content-covariate topic model (SAGE / the STM content model). Topics are
    shared but each topic's word distribution varies by a document-level group
    covariate, so you can read how a topic is worded differently across groups."""

    def __init__(
        self,
        num_topics: int,
        *,
        alpha: float = 0.1,
        prior_variance: float = 1.0,
        optimize_interval: int = 50,
        burn_in: int = 200,
        seed: int = 42,
        lbfgs_iters: int = 20,
    ) -> None: ...

    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        groups: Sequence[str] | Sequence[int],
        *,
        group_names: list[str] | None = None,
        iters: int = 1000,
        num_samples: int = 5,
        sample_interval: int = 25,
        progress: Optional[object] = None,
        progress_interval: int = 50,
        keep_theta_draws: bool = True,
        num_theta_draws: int = 25,
        convergence_tol: float = 0.0,
        check_every: int = 10,
    ) -> None:
        """Fit. groups is one group label per document (strings or ints);
        group_names fixes group order (default: sorted union)."""
        ...

    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]:
        """Per-group topic-word, shape (num_topics, num_groups, num_words)."""
        ...

    @property
    def topic_word_marginal(self) -> numpy.typing.NDArray[numpy.float64]:
        """Group-averaged topic-word, shape (num_topics, num_words)."""
        ...

    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]:
        """theta, shape (num_docs, num_topics); rows sum to 1."""
        ...

    @property
    def alpha(self) -> numpy.typing.NDArray[numpy.float64]:
        """The symmetric document-topic Dirichlet prior alpha, shape (num_topics,).
        SAGE's sparse additive parameterization is on the word side; the document
        side is an ordinary Dirichlet."""
        ...

    @property
    def theta_draws(self) -> numpy.typing.NDArray[numpy.float32] | None:
        """Thinned MCMC theta draws, shape (num_draws, num_docs, num_topics), float32,
        or None when keep_theta_draws=False."""
        ...

    @property
    def doc_lengths(self) -> list[int]:
        """Number of tokens in each training document."""
        ...

    @property
    def groups(self) -> list[str]:
        """Group names, in the index order of topic_word's second axis."""
        ...

    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    @property
    def num_topics(self) -> int: ...
    @property
    def num_groups(self) -> int: ...
    @property
    def topic_names(self) -> list[str]: ...
    @topic_names.setter
    def topic_names(self, value: list[str]) -> None: ...

    @overload
    def top_words(self, n: int = ..., *, topic: int, group: str | int | None = ...) -> list[tuple[str, float]]: ...
    @overload
    def top_words(self, n: int = ..., *, topic: None = ..., group: str | int | None = ...) -> list[list[tuple[str, float]]]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None, group: str | int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]:
        """Top n (word, prob) pairs. `topic=None` (default) returns a list of
        lists, one per topic. `topic=k` returns that topic's list. `group`
        (name or index) selects a group-specific distribution; None uses the
        group-averaged distribution."""
        ...

    def word_contrast(
        self, topic: int, group_a: str | int, group_b: str | int, n: int = 10
    ) -> list[tuple[str, float]]:
        """Words most distinguishing how `topic` is worded in group_a vs group_b,
        by log word-probability ratio (positive favours group_a)."""
        ...

    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...

    def transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iters: int = 100,
        burn_in: int = 10,
        num_samples: int = 10,
        sample_interval: int = 5,
        seed: int | None = None,
        iterations: int | None = None,
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Infer document-topic theta for new documents by collapsed Gibbs against
        the fitted group-averaged topic-word matrix. The group-specific word
        distributions are held fixed and averaged; no group covariate is needed for
        held-out documents. Shape (num_new_docs, num_topics); rows sum to 1.
        `iterations` is deprecated; use `iters` instead."""
        ...

    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> "SAGE": ...
    @property
    def doc_names(self) -> list[str]: ...
    @property
    def fit_history(self) -> list[tuple[int, float]]:
        """Per-iteration ``(iteration, objective)`` trace recorded every
        ``check_every`` sweeps during :meth:`fit` (empty when ``check_every=0``)."""
        ...
    @property
    def converged(self) -> bool:
        """True if fit early-stopped because the relative change in the objective
        fell below ``convergence_tol``; False when the full ``iters`` ran (the
        default, opt-in early stopping)."""
        ...
    def __repr__(self) -> str: ...


class LabeledLDA:
    """Labeled LDA (Ramage et al. 2009): supervised topics constrained to each
    document's label set. The number of topics equals the number of labels."""

    def __init__(
        self, *, alpha: float = 0.1, beta: float = 0.01, seed: int = 42,
        sampler: str = "sparse",
    ) -> None:
        """Create an unfitted model. alpha is the symmetric per-topic prior.

        sampler selects the backend: "sparse" (default) is the restricted
        collapsed-Gibbs sweep; "cvb0" is deterministic collapsed variational
        Bayes with the label set applied as a mask on the responsibilities (zero
        off the allowed topics). CVB0 enforces the same supervised constraint
        deterministically and tends to higher coherence; it produces no MCMC
        theta_draws. (WarpLDA is not offered here: masked proposals mix poorly,
        whereas masking is free in CVB0.)"""
        ...

    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        labels: Sequence[Sequence[str]],
        *,
        label_names: list[str] | None = None,
        iters: int = 1000,
        num_samples: int = 5,
        sample_interval: int = 25,
        progress: Optional[object] = None,
        progress_interval: int = 50,
        keep_theta_draws: bool = True,
        num_theta_draws: int = 25,
        convergence_tol: float = 0.0,
        check_every: int = 10,
    ) -> None:
        """Fit the model. labels is one label-list per document; the topic set is
        the union of all labels (or label_names, which fixes topic order). An
        empty label list leaves that document unconstrained (all topics)."""
        ...

    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]:
        """phi matrix of shape (num_topics, num_words)."""
        ...

    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]:
        """theta matrix (num_docs, num_topics); only a document's label topics
        are non-zero, rows sum to 1."""
        ...

    @property
    def alpha(self) -> numpy.typing.NDArray[numpy.float64]:
        """The symmetric document-topic Dirichlet prior alpha, shape (num_topics,)."""
        ...

    @property
    def theta_draws(self) -> numpy.typing.NDArray[numpy.float32] | None:
        """Thinned MCMC theta draws, shape (num_draws, num_docs, num_topics), float32,
        or None when keep_theta_draws=False."""
        ...

    @property
    def doc_lengths(self) -> list[int]:
        """Number of tokens in each training document."""
        ...

    @property
    def labels(self) -> list[str]:
        """Label name for each topic, in topic (column) order."""
        ...

    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    @property
    def num_topics(self) -> int: ...
    @property
    def topic_names(self) -> list[str]: ...
    @topic_names.setter
    def topic_names(self, value: list[str]) -> None: ...

    @overload
    def top_words(self, n: int = ..., *, topic: int) -> list[tuple[str, float]]: ...
    @overload
    def top_words(self, n: int = ..., *, topic: None = ...) -> list[list[tuple[str, float]]]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]:
        """Top n (word, probability) pairs for one or all topics."""
        ...

    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]:
        """UMass topic coherence per topic, shape (num_topics,)."""
        ...

    def transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iters: int = 100,
        burn_in: int = 10,
        num_samples: int = 10,
        sample_interval: int = 5,
        seed: int | None = None,
        iterations: int | None = None,
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Infer label (topic) proportions theta for new documents by collapsed
        Gibbs against the fitted topic-word matrix, treating every label as
        available. Shape (num_new_docs, num_topics); columns align with labels.
        `iterations` is deprecated; use `iters` instead."""
        ...

    def save(self, path: str) -> None:
        """Persist the fitted model to path. Reload with LabeledLDA.load."""
        ...

    @staticmethod
    def load(path: str) -> "LabeledLDA":
        """Load a model previously written by save."""
        ...

    @property
    def fit_history(self) -> list[tuple[int, float]]:
        """Per-iteration ``(iteration, objective)`` trace recorded every
        ``check_every`` sweeps during :meth:`fit` (empty when ``check_every=0``)."""
        ...
    @property
    def converged(self) -> bool:
        """True if fit early-stopped because the relative change in the objective
        fell below ``convergence_tol``; False when the full ``iters`` ran (the
        default, opt-in early stopping)."""
        ...
    def __repr__(self) -> str: ...


class LDA:
    """Sparse LDA topic model (MALLET's algorithm) implemented in Rust."""

    def __init__(
        self,
        num_topics: int,
        *,
        alpha_sum: float | None = None,
        beta: float = 0.01,
        optimize_interval: int = 50,
        burn_in: int = 200,
        seed: int = 42,
        num_threads: int = 1,
        sampler: str = "sparse",
        mh_steps: int = 2,
        use_symmetric_alpha: bool = False,
        init: str = "random",
    ) -> None:
        """Create an LDA model. alpha_sum defaults to num_topics if None.

        use_symmetric_alpha mirrors MALLET's --use-symmetric-alpha: when True,
        hyperparameter optimization learns only the alpha concentration and
        keeps every per-topic alpha equal, instead of learning an asymmetric
        per-topic prior (the default, MALLET's Wallach optimization).

        num_threads > 1 enables MALLET-style approximate parallel Gibbs
        sampling in fit() (faster on multicore; results differ from the exact
        single-threaded path but remain deterministic for a fixed
        num_threads + seed). num_threads=1 is the exact, CLI-identical path.

        sampler selects the inference backend: "sparse" (default) is MALLET's
        SparseLDA collapsed Gibbs sampler; "lightlda" is the alias-table
        Metropolis-Hastings sampler of Yuan et al. (2015); "warp" is the
        cache-efficient two-pass MH sampler of Chen et al. (2016, WarpLDA),
        whose per-sweep cost is flat in K; "cvb0" is collapsed variational
        Bayes, zeroth-order (Asuncion et al. 2009) -- deterministic, non-MCMC
        inference that tends to give higher topic coherence at moderate-to-large
        K, at the cost of O(K)-per-token compute (slower, not faster). "sparse"
        is the best speed/quality default up to ~K=200; "warp" is the speed
        choice for large K; "cvb0" is the quality choice when fit time is not the
        constraint. CVB0 produces no MCMC theta draws (theta_draws is None).
        mh_steps is the number of MH proposals per token (lightlda only).

        init selects the initial token-topic assignment: "random" (default,
        MALLET-compatible) draws each token's topic uniformly; "spectral" seeds
        it from a deterministic anchor-word topic-word matrix (the same spectral
        recovery STM/CTM use). Spectral init does not speed convergence, but it
        improves topic coherence at larger K (roughly K >= 50; it is a wash at
        small K) and falls back to the random draw when the corpus is too small
        for anchor recovery. The default leaves the MALLET byte-parity and
        same-seed determinism guarantees unchanged.
        """
        ...

    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iters: int = 1000,
        num_samples: int = 5,
        sample_interval: int = 25,
        progress: Optional[object] = None,
        progress_interval: int = 50,
        keep_theta_draws: bool = True,
        num_theta_draws: int = 25,
        convergence_tol: float = 0.0,
        check_every: int = 10,
        num_threads: Optional[int] = None,
        turbo_merge_every: int = 1,
    ) -> None:
        """Run Gibbs sampling to fit the model on data.

        With ``keep_theta_draws`` (default on), the last ``num_theta_draws``
        thinned MCMC theta snapshots are retained as :attr:`theta_draws` for
        ``composition_theta`` standard errors. Set ``keep_theta_draws=False`` to
        save memory (``num_theta_draws x num_docs x num_topics`` f32).

        ``check_every`` controls how often (in iterations) the log-likelihood is
        recorded in :attr:`fit_history`. Set ``check_every=0`` to disable tracing.
        ``convergence_tol > 0`` enables early stopping: training halts when the
        relative change in log-likelihood across two consecutive check points falls
        below ``convergence_tol``. The default (0.0) disables early stopping and
        reproduces the historical fit bit-for-bit.

        ``num_threads`` overrides the constructor's num_threads for this fit call
        only (None = use constructor value).

        ``turbo_merge_every`` (default 1, exact) is an opt-in approximate-speed
        knob for multi-threaded runs. With ``m > 1`` each worker runs ``m`` sweeps
        against its own counts before the shared topic-word table is reconciled,
        so the per-sweep merge -- the thread-scaling ceiling -- happens once per
        ``m`` sweeps instead of every sweep. Results differ from the exact path
        and are not bit-reproducible against it; ``m = 1`` (or single-threaded, or
        the lightlda/warp/cvb0 samplers) runs the exact per-sweep path unchanged.
        On a large wide-vocabulary corpus (30k docs, 30k vocabulary, K=400, 8
        threads), ``m = 3`` ran 1.55x faster for a 0.010 drop in c_npmi
        coherence. The win appears only when the merge dominates (large corpus,
        wide vocabulary, high K, many threads); on smaller corpora it does not
        help and can run slower. Recommended range when it helps: 3 to 4.
        """
        ...

    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]:
        """phi matrix of shape (num_topics, num_words)."""
        ...

    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]:
        """theta matrix of shape (num_docs, num_topics); rows sum to 1."""
        ...

    @property
    def theta_draws(self) -> Optional[numpy.typing.NDArray[numpy.float32]]:
        """Thinned MCMC theta draws, shape (num_draws, num_docs, num_topics), or
        None when fit with keep_theta_draws=False. Real cross-sweep posterior
        samples that composition_theta prefers over the Dirichlet approximation."""
        ...

    @property
    def doc_lengths(self) -> list[int]:
        """Per-document token counts (length num_docs), in doc_topic row order.
        Lets composition_theta recover N_d without re-threading the Corpus."""
        ...

    @property
    def vocabulary(self) -> list[str]:
        """Vocabulary list; column order matches topic_word."""
        ...

    @property
    def doc_names(self) -> list[str]:
        """Document names; row order matches doc_topic."""
        ...

    @property
    def alpha(self) -> numpy.typing.NDArray[numpy.float64]:
        """Per-topic alpha (Dirichlet prior), shape (num_topics,)."""
        ...

    @property
    def beta(self) -> float:
        """Scalar beta hyperparameter."""
        ...

    @property
    def num_topics(self) -> int:
        """Number of topics (available before fit)."""
        ...
    @property
    def topic_names(self) -> list[str]: ...
    @topic_names.setter
    def topic_names(self, value: list[str]) -> None: ...

    @overload
    def top_words(
        self, n: int = ..., *, topic: int
    ) -> list[tuple[str, float]]: ...

    @overload
    def top_words(
        self, n: int = ..., *, topic: None = ...
    ) -> list[list[tuple[str, float]]]: ...

    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]:
        """Return top n (word, probability) pairs for one or all topics."""
        ...

    def log_likelihood(self) -> float:
        """MALLET-formula model log-likelihood of the final sampler state (in-sample)."""
        ...

    @property
    def log_likelihood_history(self) -> list[tuple[int, float]]:
        """Per-iteration log-likelihood trace: list of (iteration, log_likelihood) pairs,
        recorded every ``check_every`` iterations. Empty when ``check_every=0``.
        Alias of fit_history for LDA."""
        ...

    @property
    def fit_history(self) -> list[tuple[int, float]]:
        """Per-iteration log-likelihood trace: list of (iteration, log_likelihood) pairs,
        recorded every ``check_every`` iterations. Empty when ``check_every=0``."""
        ...

    @property
    def converged(self) -> bool:
        """True if early stopping fired (``convergence_tol > 0`` and the relative
        change in log-likelihood fell below the tolerance). False by default."""
        ...

    def evaluate(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        num_particles: int = 10,
        seed: int | None = None,
    ) -> dict[str, Union[float, int]]:
        """Held-out evaluation via the Wallach (2009) left-to-right estimator.

        Returns a dict with `log_likelihood`, `perplexity`, `num_tokens`, `num_oov`.
        Out-of-vocabulary tokens (not seen in training) are dropped and counted.
        """
        ...

    def perplexity(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        num_particles: int = 10,
        seed: int | None = None,
    ) -> float:
        """Held-out perplexity (lower is better); convenience wrapper over evaluate()."""
        ...

    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]:
        """UMass topic coherence per topic, shape (num_topics,). Higher (nearer 0) is better."""
        ...

    def diagnostics(self, n: int = 10) -> list[dict[str, Any]]:
        """Per-topic diagnostics (MALLET-style), one dict per topic.

        Keys: topic, tokens, coherence, exclusivity, effective_words,
        document_entropy, uniform_dist, corpus_dist, rank1_docs, alpha,
        top_words. Suitable for pandas.DataFrame(...).
        """
        ...

    def transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iters: int = 100,
        burn_in: int = 10,
        num_samples: int = 10,
        sample_interval: int = 5,
        seed: int | None = None,
        iterations: int | None = None,
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Infer document-topic distributions for new, unseen documents under
        the fitted model. Returns shape (num_new_docs, num_topics); rows sum to 1.
        `iterations` is deprecated; use `iters` instead."""
        ...

    def top_documents(self, topic: int, n: int = 10) -> list[tuple[str, float]]:
        """The n training documents most associated with `topic`, as
        (doc_name, weight) pairs sorted by descending theta."""
        ...

    @property
    def topic_divergence(self) -> numpy.typing.NDArray[numpy.float64]:
        """Pairwise Jensen-Shannon divergence between topic-word distributions,
        shape (num_topics, num_topics), base 2 in [0, 1]; 0 on the diagonal."""
        ...

    def similar_documents(self, doc: int, n: int = 10) -> list[tuple[str, float]]:
        """The n training documents most similar to document `doc` (by index),
        as (doc_name, divergence) pairs sorted by ascending JS divergence."""
        ...

    def save_topic_word(self, path: str) -> None:
        """Write topic-word matrix to a TSV file (topic, word, probability)."""
        ...

    def save_doc_topic(self, path: str) -> None:
        """Write doc-topic matrix to a TSV file (doc[, label], topic_0, ...)."""
        ...

    def save_state(self, path: str) -> None:
        """Write the token-level Gibbs state to a gzipped file in MALLET's
        --output-state format: a header, #alpha/#beta lines, then one row per
        token (doc source pos typeindex type topic) giving the final topic
        assignment of every token in the training corpus. Use it to feed custom
        visualizations (e.g. pyLDAvis) or corpus metrics."""
        ...

    @staticmethod
    def load_state(path: str) -> "LDA":
        """Reconstruct a fitted LDA from a MALLET-format Gibbs state file (the
        inverse of save_state; MALLET --input-state). The file may be gzipped or
        plain text. Vocabulary, documents, per-token topic assignments, and the
        #alpha/#beta hyperparameters are restored, so the model supports the
        read-only surface (topic_word, doc_topic, top_words, ...) and transform
        on new documents."""
        ...

    def save(self, path: str) -> None:
        """Persist the fitted model (topic-word state, hyperparameters, and the
        training corpus) to `path`. Reload with `LDA.load` to run `transform`
        inference later without retraining (MALLET --output-model)."""
        ...

    @staticmethod
    def load(path: str) -> "LDA":
        """Load a model written by `save`, ready for `transform` inference on new
        documents (MALLET --input-model / --inferencer-filename)."""
        ...

    def __repr__(self) -> str: ...


class PT:
    """Pseudo-document Topic model (Zuo et al. 2016) for short texts: aggregates
    documents into `num_pseudo` pseudo-documents so LDA-style mixed membership is
    estimable on short, sparse texts. Fit by collapsed Gibbs."""

    def __init__(
        self,
        num_topics: int,
        *,
        num_pseudo: int = 100,
        alpha: float = 0.1,
        beta: float = 0.01,
        seed: int = 42,
    ) -> None: ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iters: int = 1000,
        keep_theta_draws: bool = True,
        num_theta_draws: int = 25,
        convergence_tol: float = 0.0,
        check_every: int = 10,
    ) -> None: ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def alpha(self) -> numpy.typing.NDArray[numpy.float64]:
        """The symmetric document-topic Dirichlet prior alpha, shape (num_topics,)."""
        ...
    @property
    def theta_draws(self) -> numpy.typing.NDArray[numpy.float32] | None:
        """Thinned MCMC theta draws, shape (num_draws, num_docs, num_topics), float32,
        or None when keep_theta_draws=False."""
        ...
    @property
    def doc_lengths(self) -> list[int]:
        """Number of tokens in each training document."""
        ...
    @property
    def num_topics(self) -> int: ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    @property
    def topic_names(self) -> list[str]: ...
    @topic_names.setter
    def topic_names(self, value: list[str]) -> None: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...
    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iters: int = 100,
        burn_in: int = 10,
        num_samples: int = 10,
        sample_interval: int = 5,
        seed: int | None = None,
        iterations: int | None = None,
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Infer document-topic theta for new documents by collapsed Gibbs against the
        fitted topic-word matrix. The pseudo-document layer is a training-time device;
        held-out documents infer theta over the K topics directly under the fitted phi.
        Shape (num_new_docs, num_topics); rows sum to 1.
        `iterations` is deprecated; use `iters` instead."""
        ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> "PT": ...
    @property
    def fit_history(self) -> list[tuple[int, float]]:
        """Per-iteration ``(iteration, objective)`` trace recorded every
        ``check_every`` sweeps during :meth:`fit` (empty when ``check_every=0``)."""
        ...
    @property
    def converged(self) -> bool:
        """True if fit early-stopped because the relative change in the objective
        fell below ``convergence_tol``; False when the full ``iters`` ran (the
        default, opt-in early stopping)."""
        ...
    def __repr__(self) -> str: ...


class GSDMM:
    """Gibbs Sampling Dirichlet Multinomial Mixture, the "Movie Group Process"
    (Yin & Wang 2014): a one-topic-per-document mixture for short texts. You set
    an upper bound K (`num_topics`); empty clusters die out, so the effective
    number of topics is read from `num_topics` after fit."""

    def __init__(
        self,
        num_topics: int,
        *,
        alpha: float = 0.1,
        beta: float = 0.1,
        seed: int = 42,
    ) -> None: ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iters: int = 30,
        progress_interval: int = 0,
        report_interval: Optional[int] = None,
    ) -> None:
        """Fit by the Movie Group Process. progress_interval controls the
        cluster-discovery trace (0 = auto ~50 points).

        report_interval is a deprecated alias for progress_interval."""
        ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def cluster_count_history(self) -> list[tuple[int, int]]:
        """Cluster-discovery trajectory: (iteration, num_clusters) pairs.
        Watching the count collapse to a stable value is GSDMM's headline
        convergence check."""
        ...
    @property
    def log_likelihood_history(self) -> list[tuple[int, float]]:
        """Convergence trace: (iteration, per-token log-likelihood) pairs."""
        ...
    @property
    def fit_history(self) -> list[tuple[int, float]]:
        """Uniform convergence trace aliasing log_likelihood_history."""
        ...
    @property
    def converged(self) -> bool:
        """Always False; GSDMM has no early-stop criterion."""
        ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def doc_cluster(self) -> numpy.typing.NDArray[numpy.int64]:
        """Hard cluster assignment per document, shape (num_docs,)."""
        ...
    @property
    def num_topics(self) -> int:
        """The number of non-empty clusters after fitting (the effective K)."""
        ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    @property
    def topic_names(self) -> list[str]: ...
    @topic_names.setter
    def topic_names(self, value: list[str]) -> None: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...
    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> "GSDMM": ...
    def __repr__(self) -> str: ...


class PA:
    """Pachinko Allocation Model (Li & McCallum 2006): a DAG of `num_super`
    super-topics over `num_sub` shared sub-topics over words, capturing topic
    correlations. `super_sub` reports which sub-topics each super-topic groups."""

    def __init__(
        self,
        num_super: int,
        num_sub: int,
        *,
        alpha: float = 0.1,
        beta: float = 0.01,
        seed: int = 42,
    ) -> None: ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iters: int = 1000,
        keep_theta_draws: bool = True,
        num_theta_draws: int = 25,
        convergence_tol: float = 0.0,
        check_every: int = 10,
    ) -> None: ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]:
        """Sub-topic word matrix, shape (num_sub, num_words); rows sum to 1."""
        ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]:
        """Document sub-topic matrix, shape (num_docs, num_sub)."""
        ...
    @property
    def alpha(self) -> numpy.typing.NDArray[numpy.float64]:
        """The symmetric sub-topic Dirichlet prior alpha, shape (num_sub,)."""
        ...
    @property
    def theta_draws(self) -> numpy.typing.NDArray[numpy.float32] | None:
        """Thinned MCMC theta draws (sub-topic proportions), shape
        (num_draws, num_docs, num_sub), or None when keep_theta_draws=False."""
        ...
    @property
    def doc_lengths(self) -> list[int]:
        """Number of tokens in each training document."""
        ...
    @property
    def super_sub(self) -> numpy.typing.NDArray[numpy.float64]:
        """Super-topic to sub-topic association, shape (num_super, num_sub)."""
        ...
    @property
    def num_super(self) -> int: ...
    @property
    def num_sub(self) -> int: ...
    @property
    def num_topics(self) -> int: ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    @property
    def topic_names(self) -> list[str]: ...
    @topic_names.setter
    def topic_names(self, value: list[str]) -> None: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...
    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iters: int = 100,
        burn_in: int = 10,
        num_samples: int = 10,
        sample_interval: int = 5,
        seed: int | None = None,
        iterations: int | None = None,
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Infer sub-topic proportions for new documents by collapsed Gibbs against the
        fitted sub-topic-word matrix. Projects onto the num_sub sub-topics, marginalizing
        the super-topic layer. Shape (num_new_docs, num_sub); rows sum to 1.
        `iterations` is deprecated; use `iters` instead."""
        ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> "PA": ...
    @property
    def fit_history(self) -> list[tuple[int, float]]:
        """Per-iteration ``(iteration, objective)`` trace recorded every
        ``check_every`` sweeps during :meth:`fit` (empty when ``check_every=0``)."""
        ...
    @property
    def converged(self) -> bool:
        """True if fit early-stopped because the relative change in the objective
        fell below ``convergence_tol``; False when the full ``iters`` ran (the
        default, opt-in early stopping)."""
        ...
    def __repr__(self) -> str: ...


class HLDA:
    """Hierarchical LDA (Blei et al.): topics arranged in a `depth`-level tree via
    the nested Chinese Restaurant Process. Each document follows a root-to-leaf
    path; general words sit near the root, specific words near the leaves."""

    def __init__(
        self,
        *,
        depth: int = 3,
        gamma: float = 1.0,
        beta: float = 0.01,
        alpha: float = 0.1,
        seed: int = 42,
        eta: Optional[float] = None,
    ) -> None:
        """depth is the number of levels in the topic tree. gamma controls
        branching (higher = more nodes). beta is the topic-word Dirichlet base
        measure. alpha is the document-path concentration.

        eta is a deprecated alias for beta."""
        ...
    def fit(self, data: Corpus | Sequence[Sequence[str]], *, iters: int = 500) -> None: ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]:
        """Node-word matrix, shape (num_nodes, num_words); rows sum to 1."""
        ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def node_levels(self) -> list[int]:
        """Tree level of each node (0 = root)."""
        ...
    @property
    def node_parents(self) -> list[int]:
        """Parent node index of each node (-1 for the root)."""
        ...
    @property
    def doc_paths(self) -> list[list[int]]:
        """Each document's root-to-leaf path as a list of node indices."""
        ...
    @property
    def leaves(self) -> list[int]: ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def topic_names(self) -> list[str]:
        """Node labels (length = num_nodes). Settable; length must equal num_nodes."""
        ...
    @topic_names.setter
    def topic_names(self, value: list[str]) -> None: ...
    def top_words(self, node: int, n: int = 10) -> list[tuple[str, float]]: ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> "HLDA": ...
    @property
    def fit_history(self) -> list[tuple[int, float]]:
        """Per-iteration trace. Empty; HLDA has no flat K-topic objective."""
        ...
    @property
    def converged(self) -> bool:
        """Always False; no early-stop criterion."""
        ...
    def __repr__(self) -> str: ...


class SeededLDA:
    """Seeded (guided) LDA: supply a few seed words per topic and the model is
    steered so those topics form around them, while the rest of each topic's
    vocabulary and any `residual` unseeded topics are still learned. Seeding
    follows the seededlda package (seed words get a `weight * 100` prior
    pseudocount in their topic, plus seeded initialization)."""

    def __init__(
        self,
        seed_words: dict[str, Sequence[str]],
        *,
        residual: int = 0,
        alpha: float = 0.1,
        beta: float = 0.01,
        weight: float = 0.01,
        seed: int = 42,
        sampler: str = "sparse",
    ) -> None:
        """sampler selects the backend: "sparse" (default) is the seeded
        collapsed-Gibbs sweep; "warp" is the WarpLDA cache-efficient sampler
        (seeded word phase), whose per-sweep cost is flat in K; "cvb0" is
        deterministic collapsed variational Bayes (seeded β), the quality choice.
        SeededLDA's sparse sweep scores all K topics per token, so "warp" is
        dramatically faster at large K (e.g. ~40x at K=500 on a 2,000-document
        corpus) at comparable coherence. The "warp" and "cvb0" paths do not yet
        support doc_topic_prior."""
        ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iters: int = 2000,
        doc_topic_prior: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]] | None = None,
        keep_theta_draws: bool = True,
        num_theta_draws: int = 25,
        convergence_tol: float = 0.0,
        check_every: int = 10,
    ) -> None:
        """Fit the seeded model. `convergence_tol` (default 0.0, disabled) enables
        opt-in log-likelihood early stopping on the default ("sparse") sampler,
        recording the `fit_history` trace every `check_every` sweeps; the "cvb0"
        and "warp" backends ignore both (no per-iteration trace, `converged`
        stays False)."""
        ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def theta_draws(self) -> Optional[numpy.typing.NDArray[numpy.float32]]:
        """Thinned MCMC theta draws, shape (num_draws, num_docs, num_topics), or
        None when fit with keep_theta_draws=False."""
        ...
    @property
    def doc_lengths(self) -> list[int]:
        """Per-document token counts (length num_docs), in doc_topic row order."""
        ...
    @property
    def num_topics(self) -> int: ...
    @property
    def alpha(self) -> numpy.typing.NDArray[numpy.float64]:
        """The symmetric document-topic Dirichlet prior alpha, shape (num_topics,)."""
        ...
    @property
    def topic_names(self) -> list[str]:
        """The seed names you gave, then 'residual_1' ... for unseeded topics.
        Settable after fit; length must equal num_topics."""
        ...
    @topic_names.setter
    def topic_names(self, value: list[str]) -> None: ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...
    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iters: int = 100,
        burn_in: int = 10,
        num_samples: int = 10,
        sample_interval: int = 5,
        seed: int | None = None,
        iterations: int | None = None,
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Infer document-topic theta for new documents by collapsed Gibbs against the
        fitted topic-word matrix. The seed-word boost is baked into the fitted phi;
        new documents infer theta under those distributions without re-estimating the
        seed prior. Shape (num_new_docs, num_topics); rows sum to 1.
        `iterations` is deprecated; use `iters` instead."""
        ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> "SeededLDA": ...
    @property
    def fit_history(self) -> list[tuple[int, float]]:
        """Per-iteration ``(iteration, objective)`` trace recorded every
        ``check_every`` sweeps during :meth:`fit` (empty when ``check_every=0``)."""
        ...
    @property
    def converged(self) -> bool:
        """True if fit early-stopped because the relative change in the objective
        fell below ``convergence_tol``; False when the full ``iters`` ran (the
        default, opt-in early stopping)."""
        ...
    def __repr__(self) -> str: ...


class Top2Vec:
    """Top2Vec (Angelov 2020): topics by clustering document embeddings. The
    embeddings are reduced (randomized PCA), density-clustered (HDBSCAN), and
    each topic is read off its cluster: the topic vector is the mean of its
    documents' embeddings and its words are the nearest vocabulary terms. You
    bring the embeddings; the topic count is discovered, not set. No embedder of
    your own? ``topica.llm_embed(texts, model=...)`` builds the matrix."""

    def __init__(
        self,
        *,
        n_components: int = 5,
        min_cluster_size: int = 15,
        min_samples: int | None = None,
        reducer: str = "pca",
        n_neighbors: int = 15,
        clusterer: str = "hdbscan",
        num_clusters: int | None = None,
        seed: int = 42,
    ) -> None: ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        doc_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]],
        *,
        word_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]] | None = None,
        vocabulary: Sequence[str] | None = None,
    ) -> None:
        """Fit on token documents plus one `doc_embeddings` row per document.
        Pass `word_embeddings` with the aligned `vocabulary` (same space) to
        enable `topic_neighbors`; they are realigned to topica's vocabulary."""
        ...
    @property
    def num_topics(self) -> int: ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def topic_vectors(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def labels(self) -> list[int]: ...
    @property
    def topic_names(self) -> list[str]: ...
    @topic_names.setter
    def topic_names(self, value: list[str]) -> None: ...
    @property
    def vocabulary(self) -> list[str]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None, representation: str | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...
    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def topic_neighbors(self, topic: int, *, n: int = 10) -> list[tuple[str, float]]: ...
    def transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        doc_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]] | None = None,
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Assign new documents to topics from their embeddings. `doc_embeddings`
        is required (raises ValueError if omitted); `data` is accepted for API
        consistency but not used in assignment. Shape (num_docs, num_topics)."""
        ...
    def fit_transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        doc_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]],
        *,
        word_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]] | None = None,
        vocabulary: Sequence[str] | None = None,
    ) -> numpy.typing.NDArray[numpy.float64]: ...
    def merge_topics(self, groups: Sequence[Sequence[int]]) -> None: ...
    def reduce_outliers(self) -> int: ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> "Top2Vec": ...
    @property
    def fit_history(self) -> list[tuple[int, float]]:
        """Always []; Top2Vec is not an iterative sampler."""
        ...
    @property
    def converged(self) -> None:  # type: ignore[override]
        """Always None; Top2Vec is a cluster model with no iterative objective."""
        ...
    def __repr__(self) -> str: ...


class BERTopic:
    """BERTopic (Grootendorst 2022): the same reduce/cluster pipeline as Top2Vec,
    but topics are defined by class-based TF-IDF over their documents' words, so
    no word embeddings are needed. `nr_topics` merges the most similar topics down
    to a target; `doc_topic` is the approximate distribution. You bring the
    document embeddings; the topic count is discovered (before any reduction).
    No embedder of your own? ``topica.llm_embed(texts, model=...)`` builds it."""

    def __init__(
        self,
        *,
        n_components: int = 5,
        min_cluster_size: int = 15,
        min_samples: int | None = None,
        nr_topics: int | None = None,
        window: int = 4,
        stride: int = 1,
        reducer: str = "pca",
        n_neighbors: int = 15,
        bm25: bool = False,
        reduce_frequent: bool = False,
        clusterer: str = "hdbscan",
        num_clusters: int | None = None,
        seed: int = 42,
    ) -> None: ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        doc_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]],
    ) -> None:
        """Fit on token documents plus one `doc_embeddings` row per document."""
        ...
    @property
    def num_topics(self) -> int: ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def labels(self) -> list[int]: ...
    @property
    def topic_names(self) -> list[str]: ...
    @topic_names.setter
    def topic_names(self, value: list[str]) -> None: ...
    @property
    def vocabulary(self) -> list[str]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...
    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def approximate_distribution(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        window: int | None = None,
        stride: int | None = None,
    ) -> numpy.typing.NDArray[numpy.float64]: ...
    def transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        doc_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]] | None = None,
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Soft topic distribution for new documents. BERTopic reads topics from
        text; `doc_embeddings` is accepted for API consistency but not used.
        Shape (num_docs, num_topics)."""
        ...
    def fit_transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        doc_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]],
    ) -> numpy.typing.NDArray[numpy.float64]: ...
    def merge_topics(self, groups: Sequence[Sequence[int]]) -> None: ...
    def reduce_outliers(self) -> int: ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> "BERTopic": ...
    @property
    def fit_history(self) -> list[tuple[int, float]]:
        """Always []; BERTopic is not an iterative sampler."""
        ...
    @property
    def converged(self) -> None:  # type: ignore[override]
        """Always None; BERTopic is a cluster model with no iterative objective."""
        ...
    def __repr__(self) -> str: ...


class ETM:
    """Embedded Topic Model (Dieng, Ruiz & Blei 2020): LDA with the topic-word
    matrix factored through embeddings, beta_{k,v} = softmax_v(rho_v . alpha_k),
    and a logistic-normal document prior. You bring the word embeddings rho;
    topica fits the topic embeddings alpha. `inference="em"` (default) uses
    per-document variational EM; `inference="vae"` uses the reference's amortized
    autoencoder, which scales to large corpora and maps new documents with a single
    encoder pass. Neither uses PyTorch. No embedder of your own?
    ``topica.llm_embed(vocabulary, model=...)`` builds the word embeddings rho."""

    def __init__(
        self,
        num_topics: int,
        *,
        inference: str = "em",
        convergence_tol: float = 1e-4,
        sigma_shrink: float = 0.0,
        prior_variance: float = 1e6,
        max_inner: int = 25,
        hidden_size: int = 800,
        batch_size: int = 1000,
        lr: float = 0.005,
        wdecay: float = 1.2e-6,
        seed: int = 42,
        prior: str = "laplace",
        contrastive: bool = False,
        contrastive_weight: float = 0.5,
        contrastive_temp: float = 0.5,
        em_tol: Optional[float] = None,
    ) -> None:
        """em_tol is a deprecated alias for convergence_tol. On the VAE path,
        ``prior`` selects ``"laplace"`` (default), ``"dirichlet"`` (Weibull
        reparameterization), or ``"stick_breaking"`` (Gaussian stick-breaking,
        Miao et al. 2017), and ``contrastive`` adds an InfoNCE term on the topic
        vectors scaled by ``contrastive_weight`` at temperature ``contrastive_temp``.
        Both are ignored on the EM path."""
        ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        word_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]],
        vocabulary: Sequence[str],
        *,
        iters: int | None = None,
        convergence_tol: Optional[float] = None,
    ) -> None:
        """Fit on token documents plus word embeddings (len(vocabulary) x E) and
        the aligned vocabulary, which defines the word ids. `iters` sets the number
        of training iterations (EM iterations or VAE epochs).

        convergence_tol overrides the constructor's convergence_tol for this fit
        call only (None = use constructor value)."""
        ...
    @property
    def num_topics(self) -> int: ...
    @property
    def inference(self) -> str: ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def topic_embeddings(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def bound(self) -> float: ...
    @property
    def converged(self) -> bool: ...
    @property
    def fit_history(self) -> list[tuple[int, float]]:
        """Per-iteration variational bound trace: list of (iteration, bound) pairs."""
        ...
    @property
    def topic_names(self) -> list[str]: ...
    @topic_names.setter
    def topic_names(self, value: list[str]) -> None: ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...
    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        doc_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]] | None = None,
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Held-out topic proportions for new documents. ETM reads topics from
        text; `doc_embeddings` is accepted for API consistency but not used.
        Shape (num_docs, num_topics)."""
        ...
    def fit_transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        word_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]],
        vocabulary: Sequence[str],
        *,
        iters: int | None = None,
        convergence_tol: Optional[float] = None,
    ) -> numpy.typing.NDArray[numpy.float64]: ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> ETM: ...
    def __repr__(self) -> str: ...


class InfoCTM:
    """InfoCTM (Wu et al. 2023), a cross-lingual neural topic model. Two ProdLDA
    models -- one per language over independent vocabularies, sharing the topic
    index -- are fit jointly and aligned by a Topic-Alignment Mutual-Information
    term (a masked cross-lingual InfoNCE over topic-word columns) seeded by a
    bilingual ``dictionary`` (optionally densified by per-language ``embeddings``).
    After fitting, topic ``k`` denotes the same theme in both languages."""

    def __init__(
        self,
        num_topics: int,
        *,
        mi_weight: float = 30.0,
        mi_temperature: float = 0.2,
        pos_threshold: float = 0.4,
        hidden_size: int = 100,
        dropout: float = 0.0,
        lr: float = 0.002,
        convergence_tol: float = 0.0,
        seed: int = 42,
        languages: tuple[str, str] | None = None,
    ) -> None:
        """``mi_weight`` scales the alignment term (reference 30-50);
        ``mi_temperature`` is the InfoNCE temperature; ``pos_threshold`` is the
        cosine cutoff for the embedding-densified positive mask (used only with
        embeddings). ``languages`` names the two corpora for the ``lang=`` selector
        (default ``("a", "b")``)."""
        ...
    def fit(
        self,
        data_a: Corpus | Sequence[Sequence[str]],
        data_b: Corpus | Sequence[Sequence[str]],
        *,
        dictionary: Sequence[tuple[str, str]],
        embeddings_a: dict[str, Sequence[float]] | None = None,
        embeddings_b: dict[str, Sequence[float]] | None = None,
        iters: int | None = None,
        batch_size: int = 128,
    ) -> None:
        """Fit both languages jointly. ``dictionary`` is an iterable of
        ``(word_a, word_b)`` pairs; ``embeddings_*`` are optional ``{word: vector}``
        maps that densify the alignment mask. ``iters`` is the epoch count (500)."""
        ...
    @property
    def num_topics(self) -> int: ...
    def topic_word(self, lang: str = "a") -> numpy.typing.NDArray[numpy.float64]:
        """Topic-word matrix (num_topics, vocab) for ``lang``; rows are softmax(beta_k)."""
        ...
    def doc_topic(self, lang: str = "a") -> numpy.typing.NDArray[numpy.float64]:
        """Document-topic proportions (num_docs, num_topics) for ``lang``."""
        ...
    def vocabulary(self, lang: str = "a") -> list[str]: ...
    def top_words(self, n: int = 10, *, lang: str = "a") -> list[list[tuple[str, float]]]: ...
    def transform(
        self, data: Corpus | Sequence[Sequence[str]], *, lang: str = "a"
    ) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def fit_history(self) -> list[float]: ...
    @property
    def converged(self) -> bool | None: ...


class ProdLDA:
    """ProdLDA (Srivastava & Sutton 2017), the AVITM autoencoding-variational topic
    model. LDA with the word-level mixture replaced by a product of experts:
    the word distribution is softmax(beta . theta) with an unnormalized beta,
    yielding more coherent topics. Inference is an amortized VAE trained by
    minibatch Adam on the ELBO; batch normalization and high-momentum Adam guard
    against component collapse. Unlike ETM you bring no embeddings: beta is learned
    directly. New documents transform with a single encoder forward pass."""

    def __init__(
        self,
        num_topics: int,
        *,
        alpha: float = 1.0,
        hidden_size: int = 100,
        dropout: float = 0.2,
        batch_size: int = 200,
        lr: float = 0.002,
        convergence_tol: float = 0.0,
        seed: int = 42,
        prior: str = "laplace",
        contrastive: bool = False,
        contrastive_weight: float = 0.5,
        contrastive_temp: float = 0.5,
        em_tol: Optional[float] = None,
    ) -> None:
        """em_tol is a deprecated alias for convergence_tol. ``prior`` selects
        ``"laplace"`` (default), ``"dirichlet"`` (Weibull reparameterization), or
        ``"stick_breaking"`` (Gaussian stick-breaking, Miao et al. 2017);
        ``contrastive`` adds an InfoNCE term on the topic vectors scaled by
        ``contrastive_weight`` at temperature ``contrastive_temp``."""
        ...
    @property
    def prior(self) -> str: ...
    @property
    def contrastive(self) -> bool: ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iters: int | None = None,
        convergence_tol: Optional[float] = None,
    ) -> None:
        """Fit on a Corpus or a list of token lists. `iters` sets the number of epochs.

        convergence_tol overrides the constructor's convergence_tol for this fit
        call only (None = use constructor value)."""
        ...
    @property
    def num_topics(self) -> int: ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def bound(self) -> float: ...
    @property
    def bound_history(self) -> list[float]: ...
    @property
    def converged(self) -> bool: ...
    @property
    def fit_history(self) -> list[tuple[int, float]]:
        """Per-iteration variational ELBO trace: list of (iteration, bound) pairs."""
        ...
    @property
    def epochs_run(self) -> int: ...
    @property
    def topic_names(self) -> list[str]: ...
    @topic_names.setter
    def topic_names(self, value: list[str]) -> None: ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...
    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def transform(
        self, data: Corpus | Sequence[Sequence[str]]
    ) -> numpy.typing.NDArray[numpy.float64]: ...
    def fit_transform(
        self, data: Corpus | Sequence[Sequence[str]]
    ) -> numpy.typing.NDArray[numpy.float64]: ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> ProdLDA: ...
    def __repr__(self) -> str: ...


class CombinedTM:
    """CombinedTM (Bianchi, Terragni & Hovy 2021), a contextualized topic model.
    ProdLDA whose encoder reads the normalized bag of words concatenated with a
    caller-supplied document embedding; the product-of-experts decoder still
    reconstructs the bag of words. Bring the embeddings at fit() as a
    (num_docs, E) array, aligned to the documents. Reference implementation:
    contextualized-topic-models (Bianchi et al., MIT)."""

    def __init__(
        self,
        num_topics: int,
        *,
        alpha: float = 1.0,
        hidden_size: int = 100,
        dropout: float = 0.2,
        batch_size: int = 200,
        lr: float = 0.002,
        convergence_tol: float = 0.0,
        seed: int = 42,
        prior: str = "laplace",
        contrastive: bool = False,
        contrastive_weight: float = 0.5,
        contrastive_temp: float = 0.5,
    ) -> None:
        """``prior`` selects ``"laplace"`` (default), ``"dirichlet"`` (Weibull
        reparameterization), or ``"stick_breaking"`` (Gaussian stick-breaking, Miao
        et al. 2017); ``contrastive`` adds an InfoNCE term on the topic vectors
        scaled by ``contrastive_weight`` at temperature ``contrastive_temp``."""
        ...
    @property
    def prior(self) -> str: ...
    @property
    def contrastive(self) -> bool: ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        doc_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]],
        *,
        iters: int | None = None,
        convergence_tol: Optional[float] = None,
    ) -> None:
        """Fit on token documents plus per-document embeddings (num_docs x E,
        in corpus order). `iters` sets the number of training epochs.

        convergence_tol overrides the constructor's convergence_tol for this fit
        call only (None = use constructor value)."""
        ...
    @property
    def num_topics(self) -> int: ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def bound(self) -> float: ...
    @property
    def bound_history(self) -> list[float]: ...
    @property
    def converged(self) -> bool: ...
    @property
    def fit_history(self) -> list[tuple[int, float]]: ...
    @property
    def epochs_run(self) -> int: ...
    @property
    def topic_names(self) -> list[str]: ...
    @topic_names.setter
    def topic_names(self, value: list[str]) -> None: ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...
    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        doc_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]],
    ) -> numpy.typing.NDArray[numpy.float64]: ...
    def fit_transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        doc_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]],
        *,
        iters: int | None = None,
    ) -> numpy.typing.NDArray[numpy.float64]: ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> CombinedTM: ...
    def __repr__(self) -> str: ...


class ZeroShotTM:
    """ZeroShotTM (Bianchi, Nozza & Hovy 2021), a contextualized topic model.
    ProdLDA whose encoder reads only a caller-supplied document embedding (no bag
    of words); the product-of-experts decoder still reconstructs the bag of words.
    Because topics are inferred from the embedding alone, a document embedded with
    a multilingual encoder maps to the trained topics without any bag of words,
    enabling cross-lingual transfer: fit on one language, transform documents in
    another. Bring the embeddings at fit() as a (num_docs, E) array, aligned to the
    documents. Reference implementation: contextualized-topic-models (Bianchi et
    al., MIT)."""

    def __init__(
        self,
        num_topics: int,
        *,
        alpha: float = 1.0,
        hidden_size: int = 100,
        dropout: float = 0.2,
        batch_size: int = 200,
        lr: float = 0.002,
        convergence_tol: float = 0.0,
        seed: int = 42,
        prior: str = "laplace",
        contrastive: bool = False,
        contrastive_weight: float = 0.5,
        contrastive_temp: float = 0.5,
    ) -> None:
        """``prior`` selects ``"laplace"`` (default), ``"dirichlet"`` (Weibull
        reparameterization), or ``"stick_breaking"`` (Gaussian stick-breaking, Miao
        et al. 2017); ``contrastive`` adds an InfoNCE term on the topic vectors
        scaled by ``contrastive_weight`` at temperature ``contrastive_temp``."""
        ...
    @property
    def prior(self) -> str: ...
    @property
    def contrastive(self) -> bool: ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        doc_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]],
        *,
        iters: int | None = None,
        convergence_tol: Optional[float] = None,
    ) -> None:
        """Fit on token documents plus per-document embeddings (num_docs x E,
        in corpus order). The encoder uses the embeddings alone. `iters` sets the
        number of training epochs.

        convergence_tol overrides the constructor's convergence_tol for this fit
        call only (None = use constructor value)."""
        ...
    @property
    def num_topics(self) -> int: ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def bound(self) -> float: ...
    @property
    def bound_history(self) -> list[float]: ...
    @property
    def converged(self) -> bool: ...
    @property
    def fit_history(self) -> list[tuple[int, float]]: ...
    @property
    def epochs_run(self) -> int: ...
    @property
    def topic_names(self) -> list[str]: ...
    @topic_names.setter
    def topic_names(self, value: list[str]) -> None: ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...
    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        doc_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]],
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Held-out topic proportions. For cross-lingual transfer, embed the new
        documents with the same multilingual encoder used at fit()."""
        ...
    def fit_transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        doc_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]],
        *,
        iters: int | None = None,
    ) -> numpy.typing.NDArray[numpy.float64]: ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> ZeroShotTM: ...
    def __repr__(self) -> str: ...


class NMF:
    """NMF, non-negative matrix factorization for topic modeling (Lee & Seung
    2001; Boutsidis & Gallopoulos 2008). We factor the non-negative document-term
    matrix X (D x V) as X ~ W H with W, H >= 0 by multiplicative updates. Two
    divergences are available through beta_loss: the squared Frobenius loss
    (default) and the generalized Kullback-Leibler divergence. The reference is
    scikit-learn's sklearn.decomposition.NMF (BSD-3-Clause). The topic-word matrix
    is each row of H normalized to sum 1; the document-topic matrix is each row of
    W normalized to sum 1."""

    def __init__(
        self,
        num_topics: int,
        *,
        beta_loss: str = "frobenius",
        init: str = "nndsvd",
        weighting: str = "count",
        convergence_tol: float = 1e-4,
        seed: int = 42,
    ) -> None:
        """beta_loss is 'frobenius' or 'kullback-leibler' (alias 'kl'); init is
        'nndsvd' or 'random'; weighting is 'count' or 'tfidf'. seed affects only
        init='random'."""
        ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iters: int | None = None,
        convergence_tol: Optional[float] = None,
    ) -> None:
        """Fit on a Corpus or a list of token lists. `iters` is the maximum number
        of multiplicative-update iterations (default 200). convergence_tol
        overrides the constructor value for this fit call only."""
        ...
    @property
    def num_topics(self) -> int: ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def reconstruction_error(self) -> float: ...
    @property
    def error_history(self) -> list[float]: ...
    @property
    def converged(self) -> bool: ...
    @property
    def fit_history(self) -> list[tuple[int, float]]:
        """Per-iteration reconstruction-error trace: list of (iter, error) pairs."""
        ...
    @property
    def iters_run(self) -> int: ...
    @property
    def topic_names(self) -> list[str]: ...
    @topic_names.setter
    def topic_names(self, value: list[str]) -> None: ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...
    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> NMF: ...
    def __repr__(self) -> str: ...

class LSA:
    """LSA / LSI, latent semantic analysis (Deerwester et al. 1990; randomized
    truncated SVD per Halko et al. 2011). A truncated SVD of the weighted
    document-term matrix X (D x V) ~ U_k Sigma_k V_k^T. The reference is
    scikit-learn's sklearn.decomposition.TruncatedSVD (BSD-3-Clause).

    Outputs are SIGNED latent coordinates, not probabilities. topic_word (K x V)
    is the right singular vectors V_k (signed term loadings; top_words ranks by
    absolute value). doc_topic (D x K) is U_k Sigma_k (signed document
    coordinates; rows do not sum to 1). singular_values (K) is Sigma_k. A
    deterministic svd_flip sign convention matches scikit-learn's output."""

    def __init__(
        self,
        num_topics: int,
        *,
        weighting: str = "tfidf",
        seed: int = 42,
    ) -> None:
        """weighting is 'tfidf' (default, classic LSI) or 'count'. seed seeds the
        randomized-SVD sketch."""
        ...
    def fit(self, data: Corpus | Sequence[Sequence[str]]) -> None:
        """Fit on a Corpus or a list of token lists. The SVD is a direct solve, so
        there is no iters argument."""
        ...
    @property
    def num_topics(self) -> int: ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]:
        """(num_topics, vocab) signed right singular vectors V_k. Term loadings,
        not probabilities."""
        ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]:
        """(num_docs, num_topics) signed document coordinates U_k Sigma_k. Rows do
        not sum to 1."""
        ...
    @property
    def singular_values(self) -> numpy.typing.NDArray[numpy.float64]:
        """(num_topics,) truncated singular values Sigma_k."""
        ...
    @property
    def fit_history(self) -> list[tuple[int, float]]:
        """Empty: the SVD is a direct solve with no iterative trace."""
        ...
    @property
    def converged(self) -> bool | None:
        """None: convergence is not meaningful for a one-shot SVD."""
        ...
    @property
    def topic_names(self) -> list[str]: ...
    @topic_names.setter
    def topic_names(self, value: list[str]) -> None: ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]:
        """Top-n words per component, ranked by absolute loading; each entry is
        (word, signed_loading)."""
        ...
    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> LSA: ...
    def __repr__(self) -> str: ...


class FASTopic:
    """FASTopic (Wu et al. 2024): a topic model with no encoder or neural network.
    The topic proportions theta and topic-word matrix beta are read off two
    entropic optimal-transport plans between embedding sets. You bring the document
    embeddings; topica learns the topic embeddings, word embeddings (same space),
    and transport marginals, minimizing a bag-of-words reconstruction plus the two
    transport costs. Held-out documents are mapped by a distance-softmax over the
    fitted topic embeddings, so `transform` needs only their embeddings. No
    embedder of your own? ``topica.llm_embed(texts, model=...)`` builds it."""

    def __init__(
        self,
        num_topics: int,
        *,
        lr: float = 0.002,
        dt_alpha: float = 3.0,
        tw_alpha: float = 2.0,
        theta_temp: float = 1.0,
        convergence_tol: float = 1e-6,
        sinkhorn_iters: int = 50,
        sinkhorn_tol: float = 1e-4,
        seed: int = 42,
        em_tol: Optional[float] = None,
    ) -> None:
        """em_tol is a deprecated alias for convergence_tol."""
        ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        doc_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]],
        *,
        iters: int | None = None,
        convergence_tol: Optional[float] = None,
    ) -> None:
        """Fit on token documents plus frozen document embeddings (num_docs x E).
        The vocabulary is taken from the corpus; the word embeddings are learned.
        `iters` sets the number of training epochs.

        convergence_tol overrides the constructor's convergence_tol for this fit
        call only (None = use constructor value)."""
        ...
    @property
    def num_topics(self) -> int: ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def topic_embeddings(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def word_embeddings(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def loss_history(self) -> list[float]: ...
    @property
    def converged(self) -> bool: ...
    @property
    def fit_history(self) -> list[tuple[int, float]]:
        """Per-iteration Sinkhorn loss trace (negated for higher-is-better): list of (iteration, value)."""
        ...
    @property
    def topic_names(self) -> list[str]: ...
    @topic_names.setter
    def topic_names(self, value: list[str]) -> None: ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...
    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def transform(
        self,
        data: Corpus | Sequence[Sequence[str]] | None = None,
        doc_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]] | None = None,
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Held-out topic proportions from document embeddings. `doc_embeddings`
        is required (raises ValueError if omitted); `data` is accepted for API
        consistency but not used. Shape (num_docs, num_topics)."""
        ...
    def fit_transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        doc_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]],
    ) -> numpy.typing.NDArray[numpy.float64]: ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> FASTopic: ...
    def __repr__(self) -> str: ...


class KeyATM:
    """Keyword-Assisted Topic Model (keyATM Base; Eshima, Imai & Sasaki 2024).
    Some topics carry a keyword list; a token in a keyword topic comes either from
    a distribution over only that topic's keywords or from its full distribution,
    anchoring keyword topics to their keywords. `num_topics` may exceed the number
    of keyword topics to add regular, no-keyword topics."""

    def __init__(
        self,
        keywords: dict[str, Sequence[str]],
        *,
        num_topics: int | None = None,
        alpha: float | None = None,
        beta: float = 0.01,
        beta_keyword: float = 0.1,
        gamma1: float = 1.0,
        gamma2: float = 1.0,
        seed: int = 42,
        estimate_alpha: bool = True,
        sampler: str = "sparse",
        num_threads: int = 1,
    ) -> None:
        """estimate_alpha (default True, matching R keyATM) slice-samples an
        asymmetric document-topic prior alpha each sweep. Set it False for a
        fixed symmetric alpha: a faster fit (it skips the dominant non-sweep
        cost) at the price of the R-matching asymmetric prior. The base model
        only; the covariate and dynamic models always learn their priors.

        sampler selects inference: "sparse" (default) is the collapsed-Gibbs
        sampler validated against R keyATM; "cvb0" is deterministic collapsed
        variational Bayes over the (topic, keyword-switch) states. CVB0 is an
        opt-in alternative for the base model only (it errors with covariates,
        timestamps, or a prior_offset) and does NOT preserve R-parity -- it is a
        different, deterministic estimator with no MCMC theta_draws, useful when
        you want reproducibility/quality over R-faithfulness.

        num_threads > 1 enables approximate parallel Gibbs (AD-LDA-style);
        can be overridden per-call in fit()."""
        ...
    @staticmethod
    def weighted_lda(
        num_topics: int,
        *,
        alpha: float = 0.1,
        beta: float = 0.01,
        seed: int = 42,
    ) -> "KeyATM":
        """keyATM's weightedLDA: a keyword-free model (no keyword topics) — plain
        LDA fit with keyATM's token weighting and estimated asymmetric alpha. Fit
        it like a KeyATM; keyword outputs (keyword_rate, pi_history) are empty."""
        ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iters: int = 1500,
        covariates: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]] | None = None,
        feature_names: list[str] | None = None,
        times: Sequence[float] | Sequence[str] | None = None,
        timestamps: Sequence[float] | Sequence[str] | None = None,
        num_states: int = 5,
        weights: str = "information-theory",
        num_threads: Optional[int] = None,
        optimize_interval: int = 50,
        burn_in: int = 200,
        prior_variance: float = 1.0,
        lbfgs_iters: int = 20,
        progress_interval: int = 0,
        prior_offset: Optional[float] = None,
        keep_theta_draws: bool = True,
        num_theta_draws: int = 25,
        convergence_tol: float = 0.0,
        report_interval: Optional[int] = None,
        turbo_alpha_stride: int = 1,
    ) -> None:
        """Fit by collapsed Gibbs. Pass `covariates` (num_docs x F) for the
        covariate keyATM: the document-topic prior becomes a DMR,
        alpha_{d,k} = exp(x_d . lambda_k) (an intercept is prepended), and the
        learned lambda is exposed as `feature_effects`.

        Pass `timestamps` (one per document) for the dynamic keyATM: a Chib (1998)
        change-point HMM lets topic prevalence shift over `num_states` regimes.
        The smoothed path is exposed as `time_prevalence` (aligned with
        `time_labels`) and the per-segment regime as `time_state`. `timestamps`
        and `covariates` are mutually exclusive.

        `weights` is keyATM's token weighting: 'information-theory' (default,
        each token counts by its word's surprisal in bits), 'inv-freq', or
        'none' (unweighted). Weighting downweights frequent words and applies to
        every variant (base, covariate, dynamic).

        `num_threads` overrides the constructor's num_threads for this fit call
        only (None = use constructor value).

        `progress_interval` sets how often model_fit is recorded for
        `log_likelihood_history` (keyATM's model_fit / plot_modelfit): 0
        (default) records ~50 evenly spaced points across the run; a positive
        value records every that-many sweeps.

        `convergence_tol` (default 0.0, disabled) enables opt-in early stopping
        on the recorded model_fit trace: the Gibbs sweep stops once the relative
        change in the log-likelihood between two recorded points falls below the
        tolerance, setting `converged` to True (the relative-bound criterion R
        stm uses; 0.0 runs the full `iters`, the field convention for collapsed
        samplers). Applies to the base/covariate/dynamic Gibbs backends; ignored
        by the CVB0 backend, which keeps no trace.

        report_interval is a deprecated alias for progress_interval.

        turbo_alpha_stride (default 1, exact) is an opt-in approximate speedup
        for the base model's asymmetric-alpha slice sampler, the dominant
        non-sweep cost on large corpora. With a stride s > 1 the slice sampler
        evaluates its Dirichlet-multinomial data term over every s-th document and
        scales that sum up by s, an unbiased estimate of the full term that touches
        ~1/s of the documents. It changes the estimated alpha (and therefore the
        fit), so it is off by default; it applies to the base model only (it errors
        with covariates or timestamps) and only when estimate_alpha is True."""
        ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def log_likelihood_history(self) -> list[tuple[int, float, float]]:
        """Convergence trace: (iteration, log_likelihood, perplexity) triples —
        the three columns of keyATM's model_fit / plot_modelfit. The
        log-likelihood is the collapsed marginal and perplexity is
        exp(-loglik / total_weighted_tokens), both on R keyATM's scale. Empty if
        tracing was disabled."""
        ...
    @property
    def alpha_history(self) -> list[tuple[int, list[float]]]:
        """Trace of the estimated document-topic prior alpha: (iteration, alpha)
        pairs (alpha length K) — keyATM's plot_alpha / values_iter$alpha_iter.
        Base model only; empty for covariate (traces lambda) and dynamic."""
        ...
    @property
    def pi_history(self) -> list[tuple[int, list[float]]]:
        """Trace of the per-topic keyword switch rate pi: (iteration, pi) pairs
        (pi length K, 0 for regular topics) — keyATM's plot_pi /
        values_iter$pi_iter. Empty for a keyword-free model."""
        ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def theta_draws(self) -> Optional[numpy.typing.NDArray[numpy.float32]]:
        """Thinned MCMC theta draws, shape (num_draws, num_docs, num_topics), or
        None when fit with keep_theta_draws=False. Real cross-sweep posterior
        samples that composition_theta prefers over the Dirichlet approximation.
        Collected for the base, covariate, and dynamic variants."""
        ...
    @property
    def doc_lengths(self) -> list[int]:
        """Per-document token counts (length num_docs), in doc_topic row order."""
        ...
    @property
    def feature_effects(self) -> numpy.typing.NDArray[numpy.float64]:
        """Covariate model: learned lambda, shape (num_topics, F+1); column 0 is
        the intercept. Raises if fit without covariates."""
        ...
    @property
    def feature_names(self) -> list[str]:
        """Covariate model: names for feature_effects columns ('intercept' first)."""
        ...
    @property
    def keyword_rate(self) -> numpy.typing.NDArray[numpy.float64]:
        """Per-topic keyword switch rate (0 for regular topics)."""
        ...
    @property
    def alpha(self) -> numpy.typing.NDArray[numpy.float64]:
        """The document-topic Dirichlet prior alpha, shape (num_topics,). Base model:
        the estimated asymmetric prior; covariate/dynamic models fall back to the
        symmetric base value."""
        ...
    @property
    def time_prevalence(self) -> numpy.typing.NDArray[numpy.float64]:
        """Dynamic model: smoothed topic prevalence per time segment, shape
        (T, num_topics), aligned with `time_labels`. Raises if fit without
        `timestamps`."""
        ...
    @property
    def time_state(self) -> list[int]:
        """Dynamic model: latent HMM regime of each time segment (length T).
        Empty for non-dynamic models."""
        ...
    @property
    def time_labels(self) -> list[str]:
        """Dynamic model: sorted distinct timestamp labels, one per time segment.
        Empty for non-dynamic models."""
        ...
    @property
    def transition_matrix(self) -> numpy.typing.NDArray[numpy.float64]:
        """Dynamic model: left-to-right state transition matrix, shape
        (num_states, num_states). Raises if fit without `timestamps`."""
        ...
    @property
    def num_topics(self) -> int: ...
    @property
    def topic_names(self) -> list[str]: ...
    @topic_names.setter
    def topic_names(self, value: list[str]) -> None: ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...
    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iters: int = 100,
        burn_in: int = 10,
        num_samples: int = 10,
        sample_interval: int = 5,
        seed: int | None = None,
        iterations: int | None = None,
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Infer document-topic theta for new documents by collapsed Gibbs against the
        fitted effective topic-word matrix. The effective P(w|topic) already marginalizes
        over the keyword switch; held-out inference does not re-estimate the switch for
        new tokens. Uses the estimated asymmetric alpha when available. Shape
        (num_new_docs, num_topics); rows sum to 1.
        `iterations` is deprecated; use `iters` instead."""
        ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> "KeyATM": ...
    @property
    def fit_history(self) -> list[tuple[int, float]]:
        """Per-iteration log-likelihood trace: list of (iteration, log_likelihood) pairs."""
        ...
    @property
    def converged(self) -> bool:
        """True if the Gibbs run early-stopped because the relative change in the
        recorded model_fit log-likelihood fell below `convergence_tol`; False when
        the full `iters` ran (the default; always False for the CVB0 backend)."""
        ...
    def __repr__(self) -> str: ...
