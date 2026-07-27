from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence, Union, overload
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
    min_dist: float = 0.0,
    spread: float = 1.0,
    n_epochs: int = 0,
    negative_sample_rate: int = 5,
    repulsion_strength: float = 1.0,
    metric: str = "cosine",
    seed: int = 0,
) -> numpy.typing.NDArray[numpy.float64]:
    """Project a high-dimensional array to `n_components` for plotting or clustering.

    `method` is "pca" (default, deterministic, distance-faithful), "umap", or
    "tsne". UMAP and t-SNE preserve local neighborhoods but distort global geometry
    (between-cluster distances and cluster sizes are not meaningful) and are not
    reproducible across runs (a warning is issued); PCA is the reproducible default.
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

def inspect_residual_dispersion(
    beta: list[list[float]],
    theta: list[list[float]],
    docs: list[list[int]],
    tol: float = 0.01,
) -> tuple[float, float, float, float, float]:
    """stm-faithful multinomial residual dispersion (checkResiduals / Taddy 2012).

    Returns ``(dispersion, df, num_params, statistic, nhat)`` (internal).
    """
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

    @property
    def preprocessing(self) -> dict | None:
        """The vocabulary-filtering parameters Topica applied when this corpus was
        built (min_doc_freq/max_doc_fraction/min_cf/rm_top/max_features, plus a
        ``vocabulary`` flag for a fixed-vocabulary build), or None after load."""
        ...

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
        max_features: int | None = None,
        vocabulary: list[str] | None = None,
    ) -> Corpus:
        """Build a Corpus from a list of token lists.

        A document left with no tokens by pruning is dropped, so ``num_docs`` can
        be smaller than ``len(documents)``; the surviving original indices are in
        ``kept_indices`` (realign external covariates with ``X[corpus.kept_indices]``).
        ``min_cf`` drops words whose corpus frequency is below the threshold.
        ``rm_top`` removes the top-N most frequent words before any other pruning.
        ``max_features`` then caps the vocabulary to the N most frequent surviving
        terms (scikit-learn's ``CountVectorizer(max_features=)``); None is unbounded.
        ``vocabulary`` pins the vocabulary to a fixed, ordered term list
        (scikit-learn's ``vocabulary=``): out-of-vocabulary tokens are dropped and
        the frequency filters are not applied, so it cannot be combined with them.
        To vectorize held-out documents against an existing corpus, use
        :meth:`transform`.
        """
        ...

    def transform(
        self,
        documents: list[list[str]],
        *,
        doc_names: list[str] | None = None,
        doc_labels: list[str] | None = None,
    ) -> Corpus:
        """Vectorize new documents against this corpus's vocabulary.

        Returns a Corpus sharing this one's vocabulary exactly (same terms, order,
        and ids, at full width), so a model fitted here keeps its ``topic_word``
        columns aligned. Out-of-vocabulary tokens are dropped; documents left with
        no in-vocabulary token are dropped, with the survivors' indices in the
        result's ``kept_indices``. scikit-learn's ``vectorizer.transform`` /
        gensim's ``doc2bow`` on held-out text. Raises if no document survives.
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
    features: alpha_{d,t} = exp(lambda_t . x_d) + alpha_epsilon. The lambda prior
    is centered at log(alpha) on the intercept, so with a null covariate the model
    reduces to LDA with symmetric prior `alpha` (matching tomotopy's DMR default
    0.1). After fitting, the learned weights are in `feature_effects`.
    """
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

    def __init__(
        self,
        num_topics: int,
        *,
        beta: float = 0.01,
        optimize_interval: int = 50,
        burn_in: int = 200,
        seed: int = 42,
        alpha: float = 0.1,
        prior_variance: float = 1.0,
        alpha_epsilon: float = 1e-10,
        lbfgs_iters: int = 20,
        sampler: str = "sparse",
        num_threads: int = 1,
    ) -> None:
        """Create an unfitted DMR model. prior_variance is the Gaussian prior
        variance on the feature weights; lbfgs_iters caps L-BFGS steps per round.

        sampler selects the inference backend: "sparse" (default) is the
        SparseLDA collapsed-Gibbs sweep with the per-document DMR prior; "warp"
        is the WarpLDA cache-efficient sampler (per-document-α doc phase), whose
        per-sweep cost is flat in K; "cvb0" is deterministic collapsed
        variational Bayes (per-document α; the soft expected counts feed the λ
        optimizer directly), the quality choice at the cost of O(K)-per-token
        compute. As with plain LDA, warp is flat in K and overtakes sparse on
        speed around K~=50 (dominating at large K), while sparse keeps a small-K
        coherence edge and a convergence trace; the "warp"/"cvb0" paths record no
        convergence trace, so convergence_tol has no effect there.

        num_threads > 1 runs MALLET-style approximate-parallel Gibbs on the
        default sparse backend (partition documents, sample against per-worker
        count copies, merge; deterministic for a fixed num_threads+seed); 1 is the
        exact serial path. It is ignored by the warp/cvb0 backends and can be
        overridden per call via fit(num_threads=)."""
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
        offset: Optional[numpy.typing.NDArray[numpy.float64]] = None,
        num_threads: int | None = None,
    ) -> "DMR":
        """Fit by collapsed Gibbs with the per-document Dirichlet prior
        alpha_{d,t} = exp(lambda_t . x_d + offset[d, t]). `features` (or
        `covariates`, a symmetric alias) is required: an (num_docs, F) covariate
        matrix (no intercept column — one is prepended), with feature_names naming
        the F columns. `offset` is an optional fixed (num_docs, num_topics) term
        added inside the exponent (e.g. a constant log(alpha) to set the baseline
        concentration); it shifts the predictor but is not itself estimated. The
        L-BFGS optimization of lambda runs every optimize_interval sweeps after
        burn_in; topic-word phi is averaged over num_samples samples taken every
        sample_interval sweeps."""
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
    def feature_effect_se(self) -> numpy.typing.NDArray[numpy.float64] | None:
        """Standard error of each feature weight lambda, shape (num_topics,
        num_features), from the observed information of the penalized
        Dirichlet-multinomial likelihood evaluated at the counts lambda was last
        optimized against. Aligned to feature_effects; an effect more than ~2 SEs
        from zero is the usual significance cue. None when lambda was never
        optimized to a stationary point (e.g. optimize_interval=0, burn_in>=iters,
        lbfgs_iters=0, or L-BFGS did not converge), since the observed information
        is only a valid covariance at an optimum; also None for models saved before
        this was added."""
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
    @property
    def initialization(self) -> str | None:
        """The initialization route the fit took (#410): 'spectral',
        'random-fallback', or 'random'; None before fit."""
        ...

    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

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
        spectral_projection_threshold: int = 10000,
    ) -> "CTM":
        """EM stops once the relative change in the variational bound falls below
        convergence_tol or after iters iterations, whichever comes first. Pass
        convergence_tol=0 to always run iters steps. Check converged and bound
        afterward.

        inference selects the backend: "batch" (default, full variational EM) or
        "svi" (stochastic/online VB, Hoffman et al. 2013) for very large corpora.
        Under "svi", iters is the number of epochs (passes over the data), and the
        global parameters update from minibatches of batch_size docs with a Robbins-
        Monro step rho_t = (tau + t)^(-kappa). tau (>= 0) and kappa in (0.5, 1] set
        the learning-rate schedule. convergence_tol early-stops on the relative
        epoch-to-epoch change in the running training ELBO (each epoch's per-minibatch
        bounds summed; a streaming monitoring signal, not a fixed-parameter corpus
        bound), and fit_history reports that per-epoch trace; pass convergence_tol=0 to
        run the full epoch budget.

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
    @property
    def initialization(self) -> str | None:
        """The initialization route the fit took (#410): 'spectral',
        'random-fallback', or 'random'; None before fit."""
        ...

    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

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
        content_time: Sequence[str] | Sequence[int] | None = None,
        content_smooth: float = 1.0,
        content_prior_var: float = 0.5,
        content_prior: str = "l2",
        iters: int = 500,
        convergence_tol: float = 1e-5,
        gamma_prior: str = "pooled",
        gamma_enet: float = 1.0,
        beta_init: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]] | None = None,
        em_tol: Optional[float] = None,
        covariates: Optional[numpy.typing.NDArray[numpy.float64]] = None,
        keep_eta_cov: bool = True,
        num_threads: Optional[int] = None,
        spectral_projection_threshold: int = 10000,
    ) -> "STM":
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
    def num_base_groups(self) -> int: ...
    @property
    def num_time_periods(self) -> int: ...
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


class STS:
    """Structural Topic and Sentiment-Discourse model (Chen & Mankad 2024): STM
    plus a per-document, per-topic continuous sentiment-discourse latent that
    modulates the topic-word distribution, with both prevalence and sentiment
    driven by document covariates. Fit by Laplace variational EM."""
    @property
    def initialization(self) -> str | None:
        """The initialization route the fit took (#410): 'spectral',
        'random-fallback', or 'random'; None before fit."""
        ...

    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

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
        kappa_estimation: str | None = None,
        kappa_ridge: float = 1e-3,
        em_tol: Optional[float] = None,
        covariates: Optional[numpy.typing.NDArray[numpy.float64]] = None,
        keep_eta_cov: bool = True,
        reference: str = "none",
    ) -> "STS":
        """Fit. sentiment_seed (required, one value per document) defines the
        aggregation groups for the topic-word (kappa) Poisson M-step and seeds the
        initial sentiment — e.g. a star rating the sentiment should track.
        prevalence (or covariates, a symmetric alias) is (num_docs, F) covariates
        driving both topic prevalence and sentiment-discourse (alpha_d ~ N(X_d
        Gamma, Sigma); intercept prepended).

        EM stops once the relative change in the variational bound falls below
        convergence_tol or after iters iterations. kappa_estimation chooses the
        topic-word estimator: None (default) uses "lasso" -- R STS.R's default --
        unless a reference profile overrides it; "lasso" (an L1 Poisson path with
        AIC-selected penalty, the reference opt.kappa.R glmnet default), "ridge"
        (topica-native ridge-penalized Poisson; kappa_ridge sets the ridge, an opt-in
        for large-K regimes where it is faster), or "adjusted" (the CRAN sts public
        default: the same L1/AIC solve with a phi-mass-weighted sentiment aggregation).

        reference selects a reference-fidelity profile: "none" (default: the
        reference STS.R init -- prevalence latents at 0 and prior covariance diag(20)
        -- honoring kappa_estimation, so the default lasso fit reproduces the
        reference), "paper" (Chen & Mankad 2024: their stm(max.em.its=0)
        seed -- prevalence latents at 0 and prior covariance diag(20) -- the "lasso"
        estimator, no kappa damping), or "cran" (CRAN sts: same init, the "adjusted"
        estimator, reference half-step kappa damping). A reference profile forces its own estimator, so
        pairing it with any explicit kappa_estimation that differs (including an
        explicit "ridge") raises.

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
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

    def __init__(
        self,
        *,
        alpha: float = 0.1,
        gamma: float = 0.1,
        beta: float = 0.01,
        seed: int = 42,
        resample_conc: bool = False,
        concentration_max: float = 2.0,
        eta: Optional[float] = None,
    ) -> None:
        """alpha/gamma are the document- and corpus-level DP concentrations.
        gamma is the dominant lever on the inferred topic count (0.1 is
        conservative; raise it for more topics). resample_conc defaults to False
        (fixed concentrations -> a stable topic count); set it True to adapt the
        concentrations to the data, which is capped at concentration_max to avoid
        the runaway topic count it used to cause (issue #68). beta is the
        topic-word Dirichlet (base measure). alpha, gamma, beta must be > 0.

        concentration_max (default 2.0) bounds the resampled alpha/gamma when
        resample_conc=True. It is a divergence backstop, not a statistical prior:
        a posterior with mass above it is pinned at the cap, biasing the
        concentrations (and the topic count) downward, so raise it for corpora
        that legitimately support larger concentrations. No effect when
        resample_conc=False. Must be finite and > 1e-3.

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
    ) -> "HDP":
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
    topic_word(time) and a word's trajectory with word_evolution(topic, word).

    Topics are shared across slices, so it also exposes per-document topic
    proportions via `doc_topic` (the final-iteration variational gammas,
    row-normalized); the topic-word distributions are what evolve (#494)."""
    @property
    def initialization(self) -> str | None:
        """The initialization route the fit took (#410): 'spectral',
        'random-fallback', or 'random'; None before fit."""
        ...

    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

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
        anchor-word seed shared with STM/CTM/STS, reproducible across seeds;
        choose it when you want a single deterministic fit)."""
        ...

    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        times: Sequence[int],
        *,
        iters: int = 20,
    ) -> "DTM":
        """Fit by variational EM. `times` is each document's integer time-slice
        index (0-based, contiguous); the slice count is max(times)+1."""
        ...

    def topic_word(self, time: int) -> numpy.typing.NDArray[numpy.float64]:
        """Topic-word matrix at `time`, shape (num_topics, num_words); rows sum to 1."""
        ...

    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]:
        """Per-document topic proportions, shape (num_docs, num_topics); rows
        sum to 1 (#494). The final-iteration variational gammas (gensim's
        self.gammas), row-normalized. Topics are shared across slices."""
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
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

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
        """num_topics >= 2. delta is the random-walk variance knob on the
        topic-embedding and topic-prior trajectories (the prior variance for a step
        is delta; the reference sets the prior log-variance to log(delta); smaller =
        smoother drift).
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
    ) -> "DETM":
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
    how each topic moves y. `predict` scores new docs.

    inference="variational" (default) is the original Blei & McAuliffe (2007)
    variational EM; inference="gibbs" is the collapsed Gibbs sampler used by
    tomotopy."""
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

    @property
    def inference(self) -> str:
        """The inference backend ("variational" or "gibbs")."""
        ...

    def __init__(
        self,
        num_topics: int,
        *,
        alpha: float = 0.1,
        seed: int = 42,
        inference: str = "variational",
    ) -> None:
        """num_topics >= 2. alpha is the Dirichlet concentration on doc-topic
        proportions; both must be > 0. inference is "variational" (Blei &
        McAuliffe variational EM) or "gibbs" (tomotopy-style collapsed Gibbs)."""
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
    ) -> "SupervisedLDA":
        """Fit the model. `y` is the per-document response (length = number of
        documents). With inference="variational", `iters` is EM iterations and
        `var_iters` the per-document E-step iterations. With inference="gibbs",
        `iters` is the number of collapsed-Gibbs sweeps (use more, e.g. 1000) and
        `var_iters`/`convergence_tol` do not apply."""
        ...

    def predict(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        var_iters: int = 20,
        return_std: bool = False,
    ) -> (
        numpy.typing.NDArray[numpy.float64]
        | tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]
    ):
        """Predict y-hat for new documents. Out-of-vocabulary words are ignored.

        With return_std=False (default) returns a 1-D array of predictions. With
        return_std=True returns (mean, std), where std propagates the new document's
        variational topic uncertainty through the regression plus the residual
        variance sigma^2. This is a conditional predictive spread (it holds the
        fitted beta, eta, sigma^2 fixed and uses the mean-field Cov(zbar)), not a
        full Bayesian posterior-predictive interval; mean +/- 1.96 * std is a
        Gaussian approximation under those conditions."""
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
    def coefficient_se(self) -> numpy.typing.NDArray[numpy.float64] | None:
        """Standard error of each regression coefficient eta, shape (num_topics,),
        from the OLS covariance sigma^2 * M^-1 where M = sum_d E[zbar zbar^T] is the
        normal-equations matrix the fit solves for eta. This is a conditional
        approximation — it treats the fitted topics, beta, and the variational
        moments as fixed, so it does not propagate topic/beta uncertainty. Read
        |eta| > ~2*SE as an informal importance cue under those assumptions, not a
        calibrated significance test. Aligned to coefficients. None for models saved
        before this was added."""
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
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

    def __init__(
        self,
        num_topics: int,
        *,
        alpha: float = 0.1,
        prior: str = "laplace",
        prior_variance: float = 1.0,
        optimize_interval: int = 50,
        burn_in: int = 200,
        seed: int = 42,
        lbfgs_iters: int = 20,
    ) -> None: ...
    @property
    def prior(self) -> str:
        """The prior on the κ content deviations (``"laplace"``, ``"gaussian"``,
        or ``"jeffreys"``)."""
        ...
    @property
    def content_kappa(self) -> dict:
        """The fitted content deviations κ as a dict of numpy arrays: ``"topic"``
        (K×V), ``"group"`` (G×V), and ``"interaction"`` (K·G×V). Under a sparse
        ``prior`` most entries are ~0."""
        ...

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
    ) -> "SAGE":
        """Fit. groups is one group label per document (strings or ints);
        group_names fixes group order (default: sorted union)."""
        ...

    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]:
        """Per-group topic-word, shape (num_topics, num_groups, num_words)."""
        ...

    @property
    def topic_word_marginal(self) -> numpy.typing.NDArray[numpy.float64]:
        """Group-neutral topic-word, shape (num_topics, num_words): the per-group
        beta averaged with equal weight over groups (a uniform group prior), a
        deliberate group-neutral summary of each topic. Not the empirical marginal
        sum_g P(g|z=k) beta_{k,g}, which would tilt toward more prevalent groups;
        use topic_word for the full per-group distributions."""
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
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

    def __init__(
        self, *, alpha: float = 0.1, beta: float = 0.01, seed: int = 42,
        sampler: str = "sparse", num_threads: int = 1,
    ) -> None:
        """Create an unfitted model. alpha is the symmetric per-topic prior.

        sampler selects the backend: "sparse" (default) is the restricted
        collapsed-Gibbs sweep; "cvb0" is deterministic collapsed variational
        Bayes with the label set applied as a mask on the responsibilities (zero
        off the allowed topics). CVB0 enforces the same supervised constraint
        deterministically and tends to higher coherence; it produces no MCMC
        theta_draws. (WarpLDA is not offered here: masked proposals mix poorly,
        whereas masking is free in CVB0.)

        num_threads > 1 runs the sparse backend as MALLET-style approximate-parallel
        restricted Gibbs (partition documents, sample against per-worker count
        copies, merge; deterministic for a fixed num_threads+seed); 1 is the exact
        serial path. It is ignored by the cvb0 backend and can be overridden per
        call via fit(num_threads=)."""
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
        num_threads: int | None = None,
    ) -> "LabeledLDA":
        """Fit the model. labels is one label-list per document; the topic set is
        the union of all labels (or label_names, which fixes topic order and must
        contain every non-empty observed label exactly once). An empty label list
        leaves that document unconstrained (all topics).

        num_threads overrides the constructor's worker count for this fit only
        (None = constructor value); >1 runs the sparse sweep as approximate-parallel
        AD-LDA (deterministic for a fixed num_threads+seed), 1 is the exact serial
        path, and it is ignored by the cvb0 backend."""
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


class OnlineLDA:
    """Online (streaming) variational-Bayes LDA (Hoffman, Blei & Bach 2010).

    Minibatch stochastic VB on the Dirichlet LDA model -- the analogue of
    gensim's ``LdaModel``. Fits in minibatches with a decaying learning rate
    ``rho_t = (tau + t)**(-kappa)`` without holding the whole corpus in memory,
    and supports a streaming :meth:`partial_fit` that folds new documents into an
    already-fitted model. Prefer it over the batch-Gibbs :class:`LDA` for very
    large or streaming corpora.
    """

    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...

    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

    def __init__(
        self,
        num_topics: int,
        *,
        alpha_sum: float | None = None,
        beta: float = 0.01,
        tau: float = 1.0,
        kappa: float = 0.7,
        batch_size: int = 256,
        inner_iters: int = 100,
        mean_change_tol: float = 1e-3,
        total_docs: float | None = None,
        seed: int = 42,
    ) -> None:
        """Create an unfitted OnlineLDA model.

        ``alpha_sum`` is the total document-topic Dirichlet mass (default
        ``num_topics``, i.e. 1.0 per topic); the symmetric per-topic alpha is
        ``alpha_sum / num_topics``. ``beta`` is the symmetric topic-word Dirichlet
        prior (gensim's ``eta``). ``tau`` (offset >= 0, gensim's ``offset``)
        down-weights early minibatches and ``kappa`` (decay in (0.5, 1], gensim's
        ``decay``) sets the forgetting rate of ``rho_t = (tau + t)**(-kappa)``.
        ``batch_size`` is the minibatch size (gensim's ``chunksize``).
        ``inner_iters`` caps the per-document E-step iterations (gensim's
        ``iterations``), stopping once the mean change in gamma is below
        ``mean_change_tol``. ``total_docs`` is the assumed corpus size D used for
        the ``D / batch_size`` gradient scaling -- set it when streaming a corpus
        larger than the first :meth:`fit` batch (default: the fit corpus size).
        ``seed`` seeds the initial lambda and the per-pass shuffle.
        """
        ...

    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iters: int = 100,
        convergence_tol: float = 0.0,
    ) -> "OnlineLDA":
        """Fit by online VB: sweep the corpus for ``iters`` passes (gensim's
        ``passes``), one stochastic lambda update per minibatch.
        ``convergence_tol > 0`` early-stops on the relative change in the per-pass
        evidence lower bound. Returns ``self``."""
        ...

    def partial_fit(
        self, data: Corpus | Sequence[Sequence[str]]
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Fold one fresh minibatch into the fitted model: a single stochastic
        lambda update that advances the Robbins-Monro schedule. Out-of-vocabulary
        tokens are dropped (the vocabulary is fixed at the first :meth:`fit`).
        Updates :attr:`topic_word` and returns the minibatch's document-topic
        matrix, shape (num_docs, num_topics). Requires a prior :meth:`fit`."""
        ...

    def transform(
        self, data: Corpus | Sequence[Sequence[str]]
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Infer the document-topic matrix for held-out ``data`` at the current
        lambda, without updating the model (a pure E-step). Out-of-vocabulary
        tokens are dropped. Returns a (num_docs, num_topics) row-stochastic array."""
        ...

    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]:
        """phi matrix of shape (num_topics, num_words); each row is lambda
        normalized to sum 1."""
        ...

    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]:
        """theta matrix of shape (num_docs, num_topics); rows sum to 1. After
        :meth:`fit` this covers the training corpus; after :meth:`partial_fit` it
        is the most recently processed minibatch."""
        ...

    @property
    def updates(self) -> int:
        """Number of stochastic lambda updates applied so far (the Robbins-Monro
        step index); advanced by every minibatch of fit and each partial_fit."""
        ...

    @property
    def alpha(self) -> numpy.typing.NDArray[numpy.float64]:
        """Per-topic document-topic Dirichlet prior, shape (num_topics,)."""
        ...

    @property
    def beta(self) -> float:
        """Scalar topic-word Dirichlet prior."""
        ...

    @property
    def doc_lengths(self) -> list[int]:
        """Per-document token counts, in doc_topic row order."""
        ...

    @property
    def theta_draws(self) -> None:
        """Always None: online VB keeps one variational posterior, not MCMC
        draws. Present for the Dirichlet-family contract."""
        ...

    @property
    def converged(self) -> bool:
        """Whether fit stopped early on the per-pass bound tolerance."""
        ...

    @property
    def fit_history(self) -> list[tuple[int, float]]:
        """Per-pass evidence-lower-bound trace as (pass, elbo) pairs."""
        ...

    @property
    def vocabulary(self) -> list[str]:
        """Vocabulary list; column order matches topic_word."""
        ...

    @property
    def doc_names(self) -> list[str]:
        """Document names; row order matches the training doc_topic."""
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
    def top_words(self, n: int = ..., *, topic: int) -> list[tuple[str, float]]: ...
    @overload
    def top_words(
        self, n: int = ..., *, topic: None = ...
    ) -> list[list[tuple[str, float]]]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]:
        """Return top n (word, probability) pairs for one or all topics."""
        ...

    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]:
        """Per-topic u_mass coherence over the top ``n`` words."""
        ...

    def save(self, path: str) -> None:
        """Save the fitted model to ``path`` (topica's binary format)."""
        ...

    @staticmethod
    def load(path: str) -> "OnlineLDA":
        """Load a model previously written by :meth:`save`."""
        ...


class LDA:
    """Sparse LDA topic model (MALLET's algorithm) implemented in Rust."""
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

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
        K, at the cost of O(K)-per-token compute (slower, not faster). On speed,
        "warp" is flat in K while "sparse" grows with it, so warp overtakes sparse
        around K~=50 and wins by several-fold at large K (measured on poliblog:
        warp ~13 ms/sweep at every K; sparse 16 ms at K=10 rising to 28 ms at
        K=200). "sparse" remains the default because at small K its collapsed-Gibbs
        posterior gives slightly better coherence than warp's MAP/MCEM target, and
        it records a convergence trace warp does not. Rules of thumb: "sparse" for
        small K or when you need the trace; "warp" when per-sweep speed dominates
        (large K, or K>=~50 with the trace not needed); "cvb0" for the best
        coherence when fit time is not the constraint. CVB0 produces no MCMC theta
        draws (theta_draws is None).
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
    ) -> "LDA":
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
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

    def __init__(
        self,
        num_topics: int,
        *,
        num_pseudo: int = 100,
        alpha: float = 0.1,
        beta: float = 0.01,
        pseudo_doc_prior: float = 0.1,
        seed: int = 42,
        num_threads: int = 1,
    ) -> None:
        """pseudo_doc_prior (lambda) is the symmetric Dirichlet prior on the
        pseudo-document mixture; it drives PTM's (m_p + lambda) rich-get-richer
        aggregation (smaller = stronger popularity bias, larger flattens it).
        PTM's regime is P << D: keep num_pseudo well below the corpus size.
        Fitting with num_pseudo >= num_docs warns and collapses toward
        per-document LDA. num_threads > 1 runs the two-phase collapsed-Gibbs sweep
        as MALLET-style approximate-parallel AD-LDA (documents partitioned across
        workers sampling private count copies, then merged; deterministic for a
        fixed num_threads+seed); 1 is the exact serial path. Overridden per call
        via fit(num_threads=)."""
        ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iters: int = 1000,
        keep_theta_draws: bool = True,
        num_theta_draws: int = 25,
        convergence_tol: float = 0.0,
        check_every: int = 10,
        num_threads: int | None = None,
    ) -> "PT": ...
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
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

    def __init__(
        self,
        num_topics: int,
        *,
        alpha: float = 0.1,
        beta: float = 0.1,
        seed: int = 42,
        num_threads: int = 1,
    ) -> None:
        """num_threads must be 1. Unlike the other collapsed-Gibbs count models
        (DMR, LabeledLDA, SeededLDA, BTM), GSDMM is not parallelized: its Movie
        Group Process discovers the cluster count K via within-sweep reinforcement,
        which approximate-parallel (AD-LDA) sampling would break, making K depend on
        the thread count. Passing num_threads > 1 raises ValueError."""
        ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iters: int = 30,
        progress_interval: int = 0,
        report_interval: Optional[int] = None,
        num_threads: int = 1,
    ) -> "GSDMM":
        """Fit by the Movie Group Process. progress_interval controls the
        cluster-discovery trace (0 = auto ~50 points).

        report_interval is a deprecated alias for progress_interval.

        num_threads must be 1; GSDMM is not parallelized (cluster-count discovery
        is inherently sequential). num_threads > 1 raises ValueError."""
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
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]:
        """Document-topic matrix, shape (num_docs, num_topics); rows sum to 1.

        The in-sample Movie-Group-Process soft conditional (Yin & Wang Eq. 4) over
        the discovered clusters, with the document's own words still counted (not
        held out), matching the reference gsdmm `score()`. It is a plug-in estimate,
        not a Gibbs-averaged posterior, and its argmax is usually but not guaranteed
        to equal the hard `doc_cluster`. Use `doc_cluster` for the hard label,
        `transform` for the same conditional on held-out docs.
        """
        ...
    def transform(self, data: Corpus | list[list[str]]) -> numpy.typing.NDArray[numpy.float64]:
        """Soft cluster assignment of held-out documents, shape (num_docs, num_topics).

        Scores each document with the fitted Movie-Group-Process conditional
        (Yin & Wang Eq. 4), restricted to the discovered clusters and renormalized;
        out-of-vocabulary words are dropped. The fitted counts are not modified.
        """
        ...
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


class BTM:
    """Biterm Topic Model (Yan, Guo, Lan & Cheng 2013): a word co-occurrence topic
    model for short text. Rather than a per-document topic mixture (which short
    texts are too sparse to estimate), BTM learns one global topic distribution and
    per-topic word distributions from the corpus's biterms -- unordered word pairs
    co-occurring within a window. `alpha` defaults to 50/num_topics."""
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

    def __init__(
        self,
        num_topics: int,
        *,
        alpha: float | None = None,
        beta: float = 0.01,
        iters: int = 1000,
        window: int = 15,
        background: bool = False,
        seed: int = 42,
        num_threads: int = 1,
    ) -> None:
        """num_threads > 1 runs the biterm Gibbs sweep as MALLET-style
        approximate-parallel (AD-LDA) sampling (partition the biterms, sample
        against per-worker count copies, merge; deterministic for a fixed
        num_threads+seed); 1 is the exact serial path. Override per call via
        fit(num_threads=)."""
        ...
    def fit(
        self, data: Corpus | Sequence[Sequence[str]], *, iters: int | None = None,
        num_threads: int | None = None,
    ) -> "BTM":
        """num_threads overrides the constructor's worker count for this fit only
        (None = constructor value); >1 runs the biterm sweep as approximate-parallel
        AD-LDA (deterministic for a fixed num_threads+seed), 1 is the exact serial
        path."""
        ...
    def transform(
        self, data: Corpus | Sequence[Sequence[str]]
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Infer document-topic distributions for new documents (the sum_b scheme)."""
        ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def theta(self) -> numpy.typing.NDArray[numpy.float64]:
        """Global topic distribution p(z) (num_topics,); sums to 1. BTM's
        corpus-level topic prevalence, the counterpart to a per-document mixture."""
        ...
    @property
    def num_biterms(self) -> int:
        """Number of biterms extracted from the training corpus."""
        ...
    @property
    def num_topics(self) -> int: ...
    @property
    def model_family(self) -> str: ...
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
    def load(path: str) -> "BTM": ...
    def __repr__(self) -> str: ...


class PolylingualLDA:
    """Polylingual Topic Model (Mimno, Wallach, Naradowsky, Smith & McCallum 2009):
    LDA for aligned document tuples across L languages. Every document in a tuple
    shares one topic distribution theta; each topic carries a per-language word
    distribution phi^l, so topic k denotes the same theme in every language (aligned
    by construction). `alpha` is the per-topic document-topic prior (default 0.01),
    `beta` the topic-word prior applied to every language (default 0.01); with
    `optimize_alpha` the asymmetric alpha.m prior is re-estimated every
    `optimize_interval` Gibbs iterations after an `optimize_burn_in` warm-up."""
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

    def __init__(
        self,
        num_topics: int,
        *,
        alpha: float | None = None,
        beta: float = 0.01,
        iters: int = 1000,
        optimize_alpha: bool = True,
        optimize_interval: int = 10,
        optimize_burn_in: int = 200,
        seed: int = 42,
    ) -> None: ...
    def fit(
        self,
        data: Mapping[str, Corpus | Sequence[Sequence[str]]]
        | Sequence[Corpus | Sequence[Sequence[str]]],
        *,
        iters: int | None = None,
    ) -> "PolylingualLDA":
        """Fit on aligned document tuples: a dict {language: docs} (preferred) or a
        list of per-language corpora. Every language must have the same number of
        tuples, aligned by index."""
        ...
    def transform(
        self,
        data: Mapping[str, Corpus | Sequence[Sequence[str]]]
        | Sequence[Corpus | Sequence[Sequence[str]]],
        *,
        sweeps: int = 100,
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Infer tuple-topic distributions theta for new aligned tuples, holding the
        fitted per-language phi fixed."""
        ...
    def topic_word(
        self, lang: str | None = None
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Per-language topic-word matrix phi^l (num_topics, vocab_l); rows sum to 1.
        `lang` selects the language by name or index (default: the first)."""
        ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]:
        """Tuple-topic matrix theta (num_tuples, num_topics); shared across languages."""
        ...
    @property
    def alpha(self) -> numpy.typing.NDArray[numpy.float64]:
        """The learned asymmetric document-topic prior alpha.m (num_topics)."""
        ...
    @property
    def languages(self) -> list[str]:
        """The languages, in the order supplied to fit."""
        ...
    @property
    def num_topics(self) -> int: ...
    @property
    def model_family(self) -> str: ...
    def vocabulary(self, lang: str | None = None) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    @property
    def topic_names(self) -> list[str]: ...
    @topic_names.setter
    def topic_names(self, value: list[str]) -> None: ...
    def top_words(
        self, n: int = 10, *, lang: str | None = None, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...
    def coherence(
        self, n: int = 10, *, lang: str | None = None
    ) -> numpy.typing.NDArray[numpy.float64]: ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> "PolylingualLDA": ...
    def __repr__(self) -> str: ...


class DiscLDA:
    """DiscLDA (Lacoste-Julien, Sha & Jordan 2008): a discriminative topic model.
    The actual topics partition into `k_class` topics specific to each class (one
    block per class) plus `k_shared` shared topics; a document of a given class uses
    only its class block and the shared block. Reads how classes talk differently
    (`class_topics`) vs their common ground (`shared_topics`), and gives a
    class-carrying document representation (`transform`/`predict`). Fixed
    block-transform variant (paper section 4.1)."""
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

    def __init__(
        self,
        k_class: int,
        k_shared: int,
        *,
        alpha: float | None = None,
        beta: float = 0.01,
        iters: int = 1000,
        infer_sweeps: int = 100,
        class_prior: str | Sequence[float] | None = None,
        seed: int = 42,
    ) -> None:
        """class_prior sets the prior the direct classifier combines with each
        document's plug-in likelihood: "empirical" (default) uses the observed
        class frequencies from fit, so predict_proba is calibrated to class
        prevalence; "uniform" gives every class an equal prior; or pass a sequence
        of positive per-class weights (in the sorted-class order of `classes`),
        which is normalised."""
        ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        y: Sequence[str] | Sequence[int],
        *,
        iters: int | None = None,
    ) -> "DiscLDA":
        """Fit on documents with one class label `y` per document (str or int)."""
        ...
    def transform(
        self, data: Corpus | Sequence[Sequence[str]]
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Class-marginalized discriminative representation (num_docs, num_topics)."""
        ...
    def predict(self, data: Corpus | Sequence[Sequence[str]]) -> list[str]:
        """Predicted class label per document (argmax of predict_proba). An empty /
        all-OOV document resolves to the most probable class under class_prior (the
        majority class for the default empirical prior)."""
        ...
    def predict_proba(
        self, data: Corpus | Sequence[Sequence[str]]
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Approximate class posteriors (num_docs, num_classes), columns in
        `classes` order. A topica-native plug-in classifier: each class score is
        the posterior-mean-theta likelihood combined with class_prior (empirical by
        default) and softmaxed — a prior-calibrated score, not exact evidence."""
        ...
    @property
    def classes(self) -> list[str]: ...
    @property
    def class_prior(self) -> list[float]:
        """The resolved class prior p(c) used by predict/predict_proba, in `classes`
        order (sums to 1)."""
        ...
    @property
    def class_counts(self) -> list[int]:
        """Observed per-class document counts from fit, in `classes` order."""
        ...
    @property
    def num_topics(self) -> int: ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    def class_topic_ids(self, label: str) -> list[int]: ...
    def shared_topic_ids(self) -> list[int]: ...
    def class_topics(
        self, label: str, n: int = 10
    ) -> list[list[tuple[str, float]]]: ...
    def shared_topics(self, n: int = 10) -> list[list[tuple[str, float]]]: ...
    @property
    def model_family(self) -> str: ...
    @property
    def fit_history(self) -> list[tuple[int, float]]: ...
    @property
    def converged(self) -> bool | None: ...
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
    def load(path: str) -> "DiscLDA": ...
    def __repr__(self) -> str: ...


class RTM:
    """RTM: the Relational Topic Model (Chang & Blei, "Hierarchical Relational
    Models for Document Networks", AOAS 2010). LDA plus a link model: for each
    observed pair of documents a binary link is drawn from a function of the two
    documents' mean topic assignments, so the same topics explain both words and
    links. Fit with ``fit(docs, links=edges)`` on a document graph (citations,
    hyperlinks, co-sponsorship, adjacency); predict links from words for unseen
    documents with ``suggest_links``. Undirected links; ``link="logistic"``
    (default) or ``"exponential"``."""
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400). ``alpha``/``rho`` are
        ``None`` when left to resolve at fit."""
        ...

    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

    def __init__(
        self,
        num_topics: int,
        *,
        link: str | None = None,
        inference: str = "variational",
        alpha: float | None = None,
        beta: float = 0.1,
        rho: float | None = None,
        negative_ratio: float = 1.0,
        ridge: float = 1.0,
        seed: int = 42,
    ) -> None:
        """`inference` selects the backend: "variational" (default, the shipped
        variational EM) or "gibbs" (collapsed Gibbs matching R lda's `rtm.em` /
        `rtm.collapsed.gibbs.sampler`, for same-algorithm parity). The Gibbs
        backend uses the exponential link only (R lda's sole option), so `link`
        must be "exponential" (its default) under it. `beta` is the topic-word
        Dirichlet smoothing used by the Gibbs backend (R lda's `eta`). The Gibbs
        backend reproduces R lda's link coefficient, which is negative even on
        strongly-linked data — use "variational" for link scores.

        `rho` (or `negative_ratio`, which scales it by the link count) is the
        variational pseudo-negative regularization that prevents the degenerate
        positive-links-only fit — the paper's rho, R lda's `lambda` — and must be
        strictly positive; `rho=None` resolves to `negative_ratio * num_links`.
        `ridge` is the separate L2 Gaussian prior on the link coefficients and may
        be 0 (plain MLE)."""
        ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        links: Sequence[tuple[int, int]],
        *,
        iters: int = 50,
        e_sweeps: int = 3,
        e_inner: int = 5,
    ) -> "RTM":
        """Fit RTM on a document graph. ``links`` is a sequence of undirected
        ``(i, j)`` document-index pairs (only observed links are modelled)."""
        ...
    def predict_link(self, i: int, j: int) -> float:
        """Plug-in link probability between two training documents."""
        ...
    def suggest_links(
        self,
        doc: Sequence[str],
        *,
        top_n: int = 20,
        exclude: Sequence[int] | None = None,
        infer_iters: int = 50,
    ) -> list[tuple[int, float]]:
        """Rank training documents as link candidates for a new document, from its
        words alone. Returns ``(doc_index, probability)`` pairs, highest first."""
        ...
    @property
    def num_topics(self) -> int: ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def phi_bar(self) -> numpy.typing.NDArray[numpy.float64]:
        """Mean topic-assignment vectors (D, K) — the quantity the link function
        reads. Distinct from ``doc_topic`` (the normalized Dirichlet mean)."""
        ...
    @property
    def eta(self) -> numpy.typing.NDArray[numpy.float64]:
        """Link-function coefficients (length K)."""
        ...
    @property
    def nu(self) -> float:
        """Link-function intercept."""
        ...
    @property
    def link(self) -> str: ...
    @property
    def fit_history(self) -> list[tuple[int, float]]: ...
    @property
    def converged(self) -> bool: ...
    @property
    def vocabulary(self) -> list[str]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...
    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "RTM": ...
    def __repr__(self) -> str: ...


class PA:
    """Pachinko Allocation Model (Li & McCallum 2006): a DAG of `num_super`
    super-topics over `num_sub` shared sub-topics over words, capturing topic
    correlations. `super_sub` reports which sub-topics each super-topic groups.

    Behavioral differences from MALLET's PAM: a small default `alpha=0.1` with a
    hard single-super commitment at init, and α_s adaptation only in the final
    quarter of sweeps. See ``help(PA)`` (#497)."""
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

    def __init__(
        self,
        num_super: int,
        num_sub: int,
        *,
        alpha: float = 0.1,
        beta: float = 0.01,
        seed: int = 42,
        num_threads: int = 1,
    ) -> None:
        """num_threads > 1 runs the collapsed-Gibbs sweep as MALLET-style
        approximate-parallel AD-LDA (documents partitioned across workers sampling
        private sub-topic-word count copies, then merged; deterministic for a fixed
        num_threads+seed); 1 is the exact serial path. Overridden per call via
        fit(num_threads=)."""
        ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iters: int = 1000,
        keep_theta_draws: bool = True,
        num_theta_draws: int = 25,
        convergence_tol: float = 0.0,
        check_every: int = 10,
        num_threads: int | None = None,
    ) -> "PA": ...
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
    def doc_super(self) -> numpy.typing.NDArray[numpy.float64]:
        """Document by super-topic proportions, shape (num_docs, num_super);
        row d is document d's posterior-mean mixture over super-topics
        (n_ds + alpha, normalized), the per-document companion to super_sub."""
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
    path; general words sit near the root, specific words near the leaves.

    Simplifies hlda-c: a symmetric level Dirichlet `alpha` (not the GEM stick),
    a scalar `beta` (not per-level), and fixed hyperparameters; the default
    `beta=0.01` is a sharp topica calibration. See ``help(HLDA)`` (#496)."""
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

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
        """depth is the number of levels in the topic tree. gamma is the nCRP
        branching concentration (higher = more nodes). beta is the topic-word
        Dirichlet base measure. alpha is the symmetric Dirichlet smoothing over the
        L path levels (the per-document level distribution).

        eta is a deprecated alias for beta."""
        ...
    def fit(self, data: Corpus | Sequence[Sequence[str]], *, iters: int = 500) -> "HLDA": ...
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
    follows the seededlda package: by default each seed word's prior pseudocount
    scales with its corpus frequency (`count * weight * 100`) and tokens are
    initialized at random (`seed_prior="frequency"`; alpha/beta default to the
    package's 0.5/0.1). `seed_prior="uniform"` is topica's original scheme (a flat
    `weight * 100` per seed word with seeded initialization)."""
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

    def __init__(
        self,
        seed_words: dict[str, Sequence[str]],
        *,
        residual: int = 0,
        alpha: float = 0.5,
        beta: float = 0.1,
        weight: float = 0.01,
        seed: int = 42,
        seed_prior: str = "frequency",
        sampler: str = "sparse",
        seed_match: str = "fixed",
        case_insensitive: bool = False,
        num_threads: int = 1,
    ) -> None:
        """seed_match selects how each seed pattern is matched to the vocabulary:
        "fixed" (default) exact literal equality; "glob" reads `*`/`?` wildcards
        anchored to the whole token (e.g. "tax*" seeds tax, taxes, taxation);
        "regex" matches a regular expression anywhere in the token. These mirror
        quanteda's dictionary valuetype (the seededlda package's matcher).
        case_insensitive (default False) folds case; set it True with
        seed_match="glob" to reproduce quanteda's dictionary defaults. The "regex"
        dialect is Rust's linear-time regex crate, not R's ICU/stringi: common
        syntax matches identically, but backreferences and lookaround are
        unsupported.

        seed_prior selects how each seed word's prior pseudocount is built:
        "frequency" (default) reproduces the seededlda package (pseudocount =
        corpus-frequency * weight * 100, random initialization); "uniform" gives
        every seed word the same weight * 100 pseudocount with seed-word tokens
        anchored to their topic at initialization (topica's original scheme).

        sampler selects the backend: "sparse" (default) is the seeded
        collapsed-Gibbs sweep; "warp" is the WarpLDA cache-efficient sampler
        (seeded word phase), whose per-sweep cost is flat in K; "cvb0" is
        deterministic collapsed variational Bayes (seeded β), the quality choice.
        SeededLDA's sparse sweep scores all K topics per token, so "warp" is
        dramatically faster at large K (e.g. ~40x at K=500 on a 2,000-document
        corpus) at comparable coherence. The "warp" and "cvb0" paths do not yet
        support doc_topic_prior.

        num_threads > 1 runs the default sparse backend as MALLET-style
        approximate-parallel (AD-LDA) seeded Gibbs (partition documents, sample
        against per-worker count copies, merge; deterministic for a fixed
        num_threads+seed); 1 is the exact serial path. It is ignored by the
        warp/cvb0 backends and can be overridden per call via fit(num_threads=)."""
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
        num_threads: int | None = None,
    ) -> "SeededLDA":
        """Fit the seeded model. `convergence_tol` (default 0.0, disabled) enables
        opt-in log-likelihood early stopping on the default ("sparse") sampler,
        recording the `fit_history` trace every `check_every` sweeps; the "cvb0"
        and "warp" backends ignore both (no per-iteration trace, `converged`
        stays False).

        num_threads overrides the constructor's worker count for this fit only
        (None = constructor value); >1 runs the sparse sweep as approximate-parallel
        AD-LDA (deterministic for a fixed num_threads+seed), 1 is the exact serial
        path, and it is ignored by the warp/cvb0 backends."""
        ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def seed_prior_matrix(self) -> numpy.typing.NDArray[numpy.float64]:
        """The seed pseudocount matrix m[k, w] applied at fit, shape
        (num_topics, vocab_size), aligned to vocabulary and topic_names. Under
        seed_prior="frequency" this reproduces the seededlda package's tfm matrix
        (corpus-frequency(word) * weight * 100); under "uniform" each seed word's
        entry is weight * 100. Residual topics are all-zero rows."""
        ...
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
    embeddings are reduced with UMAP (matching the original, which always uses
    UMAP; pass ``reducer="pca"`` for a linear projection), density-clustered
    (HDBSCAN), and
    each topic is read off its cluster: the topic vector is the mean of its
    documents' embeddings and its words are the nearest vocabulary terms. You
    bring the embeddings; the topic count is discovered, not set. No embedder of
    your own? ``topica.llm_embed(texts, model=...)`` builds the matrix."""
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

    def __init__(
        self,
        *,
        n_components: int = 5,
        min_cluster_size: int = 15,
        min_samples: int | None = None,
        reducer: str = "umap",
        n_neighbors: int = 15,
        clusterer: str = "hdbscan",
        num_clusters: int | None = None,
        resolution: float = 1.0,
        knn_neighbors: int = 15,
        diagnostics: bool = True,
        min_dist: float = 0.1,
        spread: float = 1.0,
        n_epochs: int = 0,
        negative_sample_rate: int = 5,
        repulsion_strength: float = 1.0,
        metric: str = "cosine",
        seed: int = 42,
    ) -> None: ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        doc_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]],
        *,
        word_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]] | None = None,
        vocabulary: Sequence[str] | None = None,
    ) -> "Top2Vec":
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
    @property
    def topic_sizes(self) -> list[int]: ...
    def search_words_by_vector(
        self, vector: Sequence[float], *, n: int = 10
    ) -> list[tuple[str, float]]: ...
    def similar_words(
        self, keywords: Sequence[str], *, n: int = 10
    ) -> list[tuple[str, float]]: ...
    def search_topics(
        self, keywords: Sequence[str], *, n: int | None = None
    ) -> list[tuple[int, float]]: ...
    def search_documents_by_topic(
        self, topic: int, *, num_docs: int = 10
    ) -> list[tuple[int, float]]: ...
    def search_documents_by_keywords(
        self, keywords: Sequence[str], *, num_docs: int = 10
    ) -> list[tuple[int, float]]: ...
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
    def hierarchical_topic_reduction(self, num_topics: int) -> None: ...
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
    no word embeddings are needed. `nr_topics` reduces the discovered real topics
    down to a target (topica's greedy c-TF-IDF merge, not the upstream package's
    ward agglomeration over topic embeddings, and it counts real topics only, not
    the -1 noise topic); `doc_topic` is the approximate distribution. You bring the
    document embeddings; the topic count is discovered (before any reduction).
    topica defaults to ``reducer="umap"`` to match the upstream package (its
    UMAP is the in-house, seed-reproducible reducer); pass ``reducer="pca"`` for a
    linear, lighter projection (L2-normalized onto the unit sphere before
    clustering). topica keeps ``min_cluster_size=15`` where upstream uses
    ``min_topic_size=10`` (issue #488).
    No embedder of your own? ``topica.llm_embed(texts, model=...)`` builds it."""
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

    def __init__(
        self,
        *,
        n_components: int = 5,
        min_cluster_size: int = 15,
        min_samples: int | None = None,
        nr_topics: int | None = None,
        window: int = 4,
        stride: int = 1,
        reducer: str = "umap",
        n_neighbors: int = 15,
        bm25: bool = False,
        reduce_frequent: bool = False,
        min_similarity: float = 0.0,
        clusterer: str = "hdbscan",
        num_clusters: int | None = None,
        resolution: float = 1.0,
        knn_neighbors: int = 15,
        diagnostics: bool = True,
        min_dist: float = 0.0,
        spread: float = 1.0,
        n_epochs: int = 0,
        negative_sample_rate: int = 5,
        repulsion_strength: float = 1.0,
        metric: str = "cosine",
        seed: int = 42,
    ) -> None: ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        doc_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]],
    ) -> "BERTopic":
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
        min_similarity: float | None = None,
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
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

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
    ) -> "ETM":
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
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

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
    ) -> "InfoCTM":
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
    def save(self, path: str) -> None:
        """Persist the fitted model (both languages) to path. Reload with InfoCTM.load."""
        ...
    @staticmethod
    def load(path: str) -> "InfoCTM":
        """Load a model previously written by save."""
        ...


class ProdLDA:
    """ProdLDA (Srivastava & Sutton 2017), the AVITM autoencoding-variational topic
    model. LDA with the word-level mixture replaced by a product of experts:
    the word distribution is softmax(beta . theta) with an unnormalized beta,
    yielding more coherent topics. Inference is an amortized VAE trained by
    minibatch Adam on the ELBO; batch normalization and high-momentum Adam guard
    against component collapse. Unlike ETM you bring no embeddings: beta is learned
    directly. New documents transform with a single encoder forward pass."""
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

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
    ) -> "ProdLDA":
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


class Scholar:
    """SCHOLAR (Card, Tan & Smith 2018) with prior (prevalence) covariates. A
    ProdLDA/AVITM VAE whose document-topic prior mean is shifted by document
    metadata, ``mu_0 = W . covariates``: a covariate that co-occurs with a topic
    raises that topic's prevalence, the neural analog of STM/DMR prevalence
    covariates, learned jointly with the topics (not post-hoc). ``covariate_effects``
    is the fitted covariate-by-topic prevalence matrix. Covariates also enter the
    encoder. Built on topica's ProdLDA backbone. Reference implementation:
    dallascard/scholar (Apache-2.0)."""
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

    def __init__(
        self,
        num_topics: int,
        *,
        covariates: object | None = None,
        covariate_names: list[str] | None = None,
        content: object | None = None,
        content_names: list[str] | None = None,
        interactions: bool = False,
        alpha: float = 1.0,
        hidden_size: int = 100,
        dropout: float = 0.2,
        batch_size: int = 200,
        lr: float = 0.002,
        l2_prior_reg: float = 0.0,
        l1_content_reg: float = 0.0,
        convergence_tol: float = 0.0,
        seed: int = 42,
    ) -> None:
        """``covariates`` (prevalence) and ``content`` (topic-covariate) numeric
        matrices may be given here or at fit(). ``interactions`` adds topic-covariate
        interaction deviations; ``l2_prior_reg``/``l1_content_reg`` regularize the
        covariate/content weights."""
        ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        covariates: object | None = None,
        labels: object | None = None,
        content: object | None = None,
        iters: int | None = None,
        convergence_tol: Optional[float] = None,
    ) -> "Scholar":
        """Fit on a Corpus or list of token lists with prior ``covariates``, supervised
        ``labels`` (str/int, one per document), and/or topic-covariate ``content``. At
        least one of covariates, labels, or content must be given."""
        ...
    @property
    def num_topics(self) -> int: ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def covariate_effects(self) -> numpy.typing.NDArray[numpy.float64]:
        """Covariate-by-topic prevalence effects, shape (n_covars, num_topics)."""
        ...
    @property
    def covariate_names(self) -> list[str]: ...
    @property
    def content_effects(self) -> numpy.typing.NDArray[numpy.float64]:
        """Content (topic-covariate) word deviations, shape (n_content, vocab)."""
        ...
    @property
    def content_names(self) -> list[str]: ...
    @property
    def classes(self) -> list[str]:
        """Sorted class labels (predict_proba column order); empty if fit without labels."""
        ...
    @property
    def model_family(self) -> str: ...
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
        covariates: object | None = None,
        content: object | None = None,
    ) -> numpy.typing.NDArray[numpy.float64]: ...
    def predict_proba(
        self,
        data: Corpus | Sequence[Sequence[str]],
        covariates: object | None = None,
        content: object | None = None,
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Class probabilities (num_docs, n_classes), columns in `classes` order.
        Requires a label-trained model."""
        ...
    def predict(
        self,
        data: Corpus | Sequence[Sequence[str]],
        covariates: object | None = None,
        content: object | None = None,
    ) -> list[str]:
        """Predicted class label per document. Requires a label-trained model."""
        ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> Scholar: ...
    def __repr__(self) -> str: ...


class CombinedTM:
    """CombinedTM (Bianchi, Terragni & Hovy 2021), a contextualized topic model.
    ProdLDA whose encoder reads the bag of words concatenated with a
    caller-supplied document embedding; following the reference
    CombinedInferenceNetwork, the embedding is first passed through a learned
    adapt_bert linear projection into vocabulary space before being concatenated
    with the raw bag-of-words counts (first layer Linear(2V, hidden)). The
    product-of-experts decoder still reconstructs the bag of words. Bring the
    embeddings at fit() as a (num_docs, E) array, aligned to the documents.
    Reference implementation: contextualized-topic-models (Bianchi et al., MIT)."""
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

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
    ) -> "CombinedTM":
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
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

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
    ) -> "ZeroShotTM":
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
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

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
        init='random'. The 'nndsvd' init is scikit-learn's NNDSVDa variant (exact
        zeros filled with the data mean) and requires num_topics <=
        min(num_documents, num_words); use 'random' above that rank."""
        ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iters: int | None = None,
        convergence_tol: Optional[float] = None,
    ) -> "NMF":
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
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

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
    def fit(self, data: Corpus | Sequence[Sequence[str]]) -> "LSA":
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


class TensorLDA:
    """Online Tensor LDA (TensorLDA) topic model (Kangaslahti et al. 2026).
    Method-of-moments topic modeling using second and third-order cumulants.
    Gated behind `topica.enable_experimental()`."""
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

    def __init__(
        self,
        num_topics: int,
        *,
        alpha_0: float = 1.0,
        n_iter_train: int = 100,
        n_iter_test: int = 30,
        learning_rate: float = 0.01,
        batch_size: int = 10,
        smoothing: float = 0.01,
        theta: float = 1.0,
        n_eigenvec: int | None = None,
        pca_batch_size: int = 128,
        seed: int = 42,
    ) -> None: ...
    def fit(self, data: Corpus | Sequence[Sequence[str]], *, iters: int | None = None) -> "TensorLDA": ...
    def partial_fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        batch_index: int,
        *,
        vocabulary: list[str] | None = None,
    ) -> None: ...
    def finalize(self) -> None: ...
    def transform(self, data: Corpus | Sequence[Sequence[str]], *, seed: int | None = None) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def num_topics(self) -> int: ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def weights(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def unwhitened_raw(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def fit_history(self) -> list[tuple[int, float]]: ...
    @property
    def converged(self) -> bool: ...
    @property
    def topic_names(self) -> list[str]: ...
    @topic_names.setter
    def topic_names(self, value: list[str]) -> None: ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    @property
    def model_family(self) -> str: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...
    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> TensorLDA: ...
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
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

    def __init__(
        self,
        num_topics: int,
        *,
        lr: float = 0.002,
        dt_alpha: float = 3.0,
        tw_alpha: float = 2.0,
        theta_temp: float = 1.0,
        convergence_tol: float = 1e-6,
        sinkhorn_iters: int = 5000,
        sinkhorn_tol: float = 5e-3,
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
    ) -> "FASTopic":
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
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

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
    ) -> "KeyATM":
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

        The covariate model's lambda is re-estimated by L-BFGS:
        `optimize_interval` sweeps between updates, starting after `burn_in`
        sweeps, `lbfgs_iters` L-BFGS steps per update, under a Gaussian prior of
        variance `prior_variance` on lambda. `prior_offset` (an optional
        (num_docs, num_topics) array) adds a fixed per-document log-prior offset.
        These apply to the covariate variant only and are ignored otherwise.

        `keep_theta_draws` (default True) retains `num_theta_draws` thinned MCMC
        theta snapshots in `theta_draws` (the cross-sweep posterior samples
        `composition_theta` prefers over the Dirichlet approximation); set it
        False to save memory.

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
        evaluates its Dirichlet-multinomial data term over every s-th document
        (fixed stride in corpus order) and scales that sum up by s. This is not
        unbiased: the slice sampler then targets the subsampled posterior rather
        than the full-data one, and because the subset is deterministic the bias
        also depends on document order. It changes the estimated alpha (and
        therefore the fit), so it is off by default; use stride=1 for the exact
        alpha. It applies to the base model only (it errors with covariates or
        timestamps) and only when estimate_alpha is True."""
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
    def feature_effect_se(self) -> numpy.typing.NDArray[numpy.float64] | None:
        """Covariate model: standard errors of feature_effects (lambda), same
        shape and column order, on the original covariate scale (observed
        information in the standardized fit space mapped back by the
        standardization Jacobian, issue #316). NaN where the standardized lambda
        hit the +/-5 bound. None when lambda was never optimized to a stationary
        point (optimize_interval past iters, burn_in>=iters, lbfgs_iters=0, or
        non-convergence), since the observed information is only a valid covariance
        at an optimum (#418). Raises if fit without covariates."""
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

class IdealPointTM:
    """IdealPointTM, a topic model with a latent ideal-point head (EXPERIMENTAL,
    original construction). Each topic carries a per-author low-dimensional position
    and a per-topic loading, so an author's position shifts word choice *within* a
    topic (content), with a per-topic discrimination (||W_k||) flagging which topics
    carry the cleavage. The position is latent and estimated, making this the
    unsupervised, latent-trait twin of the STM content covariate.

    IdealPointTM consumes word tokens in one of two representations, selected at fit
    time by whether you pass `word_embeddings`: omit them and the topic-word matrix
    is parameterized directly over the vocabulary (counts; an author-displaced
    multinomial whose within-topic word choice coincides with Wordfish's through the
    Poisson-multinomial equivalence); pass them and it is factored through word
    embeddings, as in ETM. Both are the same model. Gated behind
    topica.enable_experimental()."""
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

    def __init__(
        self,
        num_topics: int,
        *,
        num_dims: int = 1,
        convergence_tol: float = 1e-4,
        sigma_shrink: float = 0.0,
        prior_variance: float = 1e6,
        w_prior_variance: float = 10.0,
        x_prior_variance: float = 1.0,
        max_inner: int = 15,
        min_count: int = 1,
        seed: int = 42,
    ) -> None:
        """num_dims is the dimensionality d of the latent ideal point. For
        num_dims > 1 the positions are identified only up to an orthogonal rotation
        (and a per-dimension sign), so read them through the loadings, not
        coordinate-by-coordinate; see author_positions.
        prior_variance is the Gaussian prior on the topic profiles (weak, as ETM);
        w_prior_variance regularizes the position loadings (smaller = more shrinkage
        toward neutral topics); x_prior_variance is the prior on the positions (1.0
        matches the unit-variance standardization); min_count drops words below that
        corpus frequency (count representation only; ignored when word_embeddings is
        passed, where the vocabulary is supplied)."""
        ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        word_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]] | None = None,
        vocabulary: Sequence[str] | None = None,
        group: Sequence[str] | None = None,
        anchors: dict[str, float] | None = None,
        iters: int | None = None,
        convergence_tol: Optional[float] = None,
    ) -> "IdealPointTM":
        """Fit on token documents. Omit `word_embeddings` for the count
        representation (the vocabulary is built from the corpus by min_count; do not
        pass `vocabulary`). Pass `word_embeddings` (len(vocabulary) x E) with the
        aligned `vocabulary` for the word-embedding representation, as in ETM.
        `group` is an optional list of author labels (length num_docs): documents
        sharing a label share one latent position; if omitted, each document is its
        own author. `anchors` is an optional {author_label: value} mapping that
        orients the sign of the first latent dimension. `iters` is the EM iteration
        count (default 50)."""
        ...
    @property
    def num_topics(self) -> int: ...
    @property
    def num_dims(self) -> int: ...
    @property
    def representation(self) -> Optional[str]:
        """"word2vec" if fitted with word_embeddings, "counts" if fitted without,
        None if unfitted."""
        ...
    @property
    def num_authors(self) -> int: ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]:
        """Topic-word matrix at the neutral position x=0 (num_topics, vocab)."""
        ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def author_positions(self) -> numpy.typing.NDArray[numpy.float64]:
        """The latent ideal points (num_authors, num_dims), standardized to mean 0 /
        unit variance per dimension.

        Identifiability: the scale is fixed but the axis is identified only up to
        sign per dimension, and for num_dims > 1 up to an arbitrary rotation of the
        axes (the likelihood is invariant under x -> x @ R, W -> R^-1 @ W). Pass
        `anchors` to fit() to fix the sign of dimension 0; without them the
        orientation is deterministic for a given seed but otherwise arbitrary (it can
        flip across seeds/corpora), and multi-dimensional positions are best read
        through the loadings, not coordinate-by-coordinate."""
        ...
    @property
    def position_se(self) -> numpy.typing.NDArray[numpy.float64]:
        """Asymptotic standard error of each author position (num_authors,
        num_dims), from the observed information of the penalized position objective
        at the fit. The multinomial-content analog of Wordfish's Hessian-based
        `se.theta`: it conditions on the fitted topic content and shrinks with the
        number of tokens an author contributes. Aligned to author_positions."""
        ...
    @property
    def author_names(self) -> list[str]: ...
    @property
    def topic_discrimination(self) -> numpy.typing.NDArray[numpy.float64]:
        """Per-topic ||W_k|| (num_topics); large where the topic separates
        positions, ~0 where it is neutral."""
        ...
    def position_shift(
        self, topic: int, *, n: int = 10, magnitude: float = 1.0, dim: int = 0,
        weighting: str = "prob",
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """The words whose within-topic use rises at the positive vs negative end
        of latent dimension `dim`. Returns (positive, negative), each a list of
        (word, score) for the top-n words. `weighting="prob"` (default) scores by
        the probability difference beta(+)-beta(-), keeping the contrast inside the
        topic's own vocabulary; "logratio" uses the log-ratio, which is more
        sensitive but dominated by rare, often off-topic words."""
        ...
    @property
    def loadings(self) -> numpy.typing.NDArray[numpy.float64]:
        """Position loadings W, shape (num_topics, num_dims*feature_dim), row-major
        over (dim, feature), where the feature dimension is the embedding dimension
        (word-embedding representation) or the vocabulary (count representation).
        Per-topic discrimination directions; their pairwise cosine shows whether
        topics share one axis or split on several."""
        ...
    @property
    def fit_history(self) -> list[tuple[int, float]]: ...
    @property
    def converged(self) -> Optional[bool]: ...
    @property
    def bound(self) -> float: ...
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
    def load(path: str) -> IdealPointTM: ...
    def __repr__(self) -> str: ...


class Wordfish:
    """Wordfish (Slapin & Proksch 2008): a word-frequency ideal-point scaler with
    no topics and no embeddings. Counts are modeled as
    y_ij ~ Poisson(exp(alpha_i + psi_j + beta_j * theta_i)); theta_i is the author's
    latent position, beta_j the word's discrimination. The word-frequency baseline
    companion to IdealPointTM. The fit is deterministic. Validated against R
    quanteda's textmodel_wordfish (parity/wordfish_r_compare.py)."""
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

    def __init__(
        self,
        *,
        beta_prior_sd: float = 3.0,
        theta_prior_sd: float = 1.0,
        min_count: int = 1,
        convergence_tol: float = 1e-6,
        seed: int = 42,
    ) -> None:
        """beta_prior_sd / theta_prior_sd are the standard deviations of the weak
        Gaussian priors regularizing word discriminations and positions (pass
        math.inf for none); min_count drops words with corpus frequency below it.
        seed is accepted for API uniformity; the fit is deterministic."""
        ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        group: Sequence[str] | None = None,
        control: Sequence[str] | None = None,
        anchors: dict[str, float] | None = None,
        iters: int | None = None,
        convergence_tol: float | None = None,
    ) -> "Wordfish":
        """group pools documents sharing a label into one unit with one position;
        control is an optional categorical confound (constant within each author)
        whose level-specific word usage is absorbed into per-level offsets so it does
        not contaminate the position; anchors orients the sign of the axis."""
        ...
    @property
    def num_authors(self) -> int: ...
    @property
    def author_positions(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def position_se(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def author_names(self) -> list[str]: ...
    @property
    def control_names(self) -> list[str]: ...
    @property
    def control_word_offsets(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def word_discrimination(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def word_intercept(self) -> numpy.typing.NDArray[numpy.float64]: ...
    def discriminating_words(
        self, n: int = 10
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]: ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def log_likelihood(self) -> float: ...
    @property
    def fit_history(self) -> list[tuple[int, float]]: ...
    @property
    def converged(self) -> bool | None: ...
    @property
    def iters_run(self) -> int: ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> Wordfish: ...
    def __repr__(self) -> str: ...


class IdealPointSentenceTM:
    """IdealPointSentenceTM, a continuous ideal-point topic model over sentence or
    document embeddings (EXPERIMENTAL). Topics are Gaussian clusters in embedding
    space whose centroids are displaced by a latent author position
    (e ~ N(mu_k + x_a . V_k, sigma^2)); ||V_k|| is the topic's discrimination. The
    sentence-embedding sibling of IdealPointTM, fit by EM. Gated behind
    topica.enable_experimental()."""
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

    def __init__(
        self,
        num_topics: int,
        *,
        num_dims: int = 1,
        convergence_tol: float = 1e-4,
        x_prior_variance: float = 1.0,
        seed: int = 42,
    ) -> None: ...
    def fit(
        self,
        embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]],
        *,
        group: Sequence[str] | None = None,
        anchors: dict[str, float] | None = None,
        iters: int | None = None,
        convergence_tol: float | None = None,
    ) -> "IdealPointSentenceTM":
        """embeddings is an (N, D) array of per-observation sentence/document
        embeddings; group (length N) gives the author of each observation."""
        ...
    @property
    def num_topics(self) -> int: ...
    @property
    def num_dims(self) -> int: ...
    @property
    def num_authors(self) -> int: ...
    @property
    def author_positions(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def position_se(self) -> numpy.typing.NDArray[numpy.float64]:
        """Standard error of each author position (num_authors, num_dims). The
        position is a linear-Gaussian least squares given the topic responsibilities,
        so this is the exact Laplace posterior SE; it shrinks with the number of the
        author's observations. Aligned to author_positions."""
        ...
    @property
    def author_names(self) -> list[str]: ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def topic_centroids(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def topic_discrimination(self) -> numpy.typing.NDArray[numpy.float64]: ...
    def position_centroid(
        self, topic: int, x: Sequence[float]
    ) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def log_likelihood(self) -> float: ...
    @property
    def fit_history(self) -> list[tuple[int, float]]: ...
    @property
    def converged(self) -> bool | None: ...
    @property
    def iters_run(self) -> int: ...
    @property
    def topic_names(self) -> list[str]: ...
    @topic_names.setter
    def topic_names(self, names: Sequence[str]) -> None: ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> IdealPointSentenceTM: ...
    def __repr__(self) -> str: ...


class TBIP:
    """TBIP, Text-Based Ideal Points (Vafa, Naidu & Blei 2020). A
    Poisson factorization in which an author's latent ideal point x_s rescales a
    neutral topic-word intensity beta_kv by a per-word ideological factor
    exp(x_s * eta_kv); documents mix topics with positive per-doc intensities
    theta_dk. Fit by the paper's mean-field variational inference (reparameterized
    single-sample SVI, Adam, document minibatching). Recovers ideological scales
    from unlabeled text. Validated by synthetic planted-position recovery
    (parity/tbip_parity.py); the reference TF1.14/TFP0.7 code is not run as a
    cross-implementation check."""
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

    def __init__(
        self,
        num_topics: int,
        *,
        a_gamma: float = 0.3,
        b_gamma: float = 0.3,
        iters: int = 7000,
        batch_size: int = 512,
        learning_rate: float = 0.05,
        min_count: int = 1,
        seed: int = 42,
    ) -> None: ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        group: Sequence[str] | None = None,
        iters: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
    ) -> "TBIP":
        """data is a Corpus or list of token lists; group (length num_docs) gives the
        author of each document (documents sharing a label share one ideal point)."""
        ...
    @property
    def num_topics(self) -> int: ...
    @property
    def num_authors(self) -> int: ...
    @property
    def ideal_points(self) -> numpy.typing.NDArray[numpy.float64]:
        """Author ideal points (num_authors,), the posterior-mean positions mu_x.

        Identifiability: identified only up to sign -- the model is invariant under
        x -> -x, eta -> -eta, so the direction is arbitrary and determined by the
        seed (TBIP has no anchoring). Compare runs by absolute correlation, or flip
        to a chosen reference author. The scale is only softly pinned by the N(0, 1)
        prior."""
        ...
    @property
    def position_se(self) -> numpy.typing.NDArray[numpy.float64]:
        """Standard error of each author ideal point (num_authors,): the standard
        deviation of the Gaussian variational posterior q(x_s), estimated jointly
        with the mean. As with any mean-field VI this can understate the true
        posterior spread. Aligned to ideal_points / author_names."""
        ...
    @property
    def author_names(self) -> list[str]: ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def ideological_topics(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def fit_history(self) -> list[tuple[int, float]]: ...
    @property
    def iters_run(self) -> int: ...
    @property
    def topic_names(self) -> list[str]: ...
    @topic_names.setter
    def topic_names(self, names: Sequence[str]) -> None: ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    def top_words(self, n: int = 10, *, topic: int | None = None) -> Any: ...
    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> TBIP: ...
    def __repr__(self) -> str: ...


class PartyEmbeddings:
    """PartyEmbeddings (Rheault & Cochrane 2020): a PV-DM (distributed-memory
    paragraph-vector) model trained by negative sampling, where each document is
    tagged with party-period metadata. The learned party (tag) vectors share a
    space with the word vectors; the leading principal components of the party
    vectors give the ideological placement (column 0 is the left-right scale). A
    scaling model with no topics, in the ideal-point family alongside Wordfish.
    Implemented from Mikolov et al. (2013) and Le & Mikolov (2014); validated by
    planted-position recovery and correlation against the gensim reference
    (parity/party_embeddings_compare.py)."""
    @property
    def settings(self) -> dict:
        """The constructor configuration as a JSON-serialisable dict,
        keyword-named to match ``__init__`` (issue #400)."""
        ...


    @property
    def seed(self) -> int:
        """The random seed the model was constructed with."""
        ...

    def __init__(
        self,
        num_dims: int = 2,
        *,
        vector_size: int = 200,
        window: int = 20,
        min_count: int = 5,
        negative: int = 5,
        sample: float = 1e-4,
        learning_rate: float = 0.025,
        seed: int = 42,
    ) -> None:
        """num_dims is the number of placement dimensions returned in
        author_positions (leading principal components of the party vectors).
        vector_size is the embedding dimension M; window the context width;
        min_count drops words below that corpus frequency; negative the number of
        negative samples; sample the frequent-word subsampling threshold;
        learning_rate the initial SGD step. The fit is single-threaded and
        reproducible from seed."""
        ...
    def fit(
        self,
        data: Any,
        *,
        group: Sequence[str],
        control: Sequence[str] | None = None,
        anchors: dict[str, float] | None = None,
        iters: int = 5,
    ) -> "PartyEmbeddings":
        """data is a Corpus or list of token lists; group (length num_docs) is the
        party-period label of each document. control is an optional second
        per-document metadata tag (estimated but not placed); anchors
        ({group_label: value}) orients the sign of each placement dimension; iters
        is the number of training epochs."""
        ...
    @property
    def num_authors(self) -> int: ...
    @property
    def author_positions(self) -> numpy.typing.NDArray[numpy.float64]:
        """Party placements as (num_authors, num_dims): the leading principal
        components of the party vectors, standardized to mean 0 / unit variance
        (column 0 is the left-right scale). Identifiability: without anchors the
        sign of each dimension is arbitrary (PCA fixes the axes by variance, so
        each column is identified only up to sign); pass anchors to fit() to fix
        the signs. Columns are standardized independently, so the relative scale
        of the components is not preserved; read author_vectors if you need it."""
        ...
    @property
    def author_names(self) -> list[str]: ...
    @property
    def author_vectors(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def word_vectors(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def fit_history(self) -> list[tuple[int, float]]: ...
    @property
    def converged(self) -> bool | None: ...
    def nearest_words(self, group: str, n: int = 10) -> list[tuple[str, float]]:
        """The top-n words by cosine to a party's vector (linguistic specificity).
        A raw ranking: high-frequency words can crowd the top, so read it relative
        to another party or the average party rather than in isolation."""
        ...
    def guided_positions(
        self, left: Sequence[str], right: Sequence[str]
    ) -> numpy.typing.NDArray[numpy.float64]: ...
    def distance(self, group_a: str, group_b: str) -> float: ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> PartyEmbeddings: ...
    def __repr__(self) -> str: ...
