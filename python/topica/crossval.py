"""Cross-validation evaluation framework (#701).

Reproducible, shared cross-validation as evidence about *predictive behavior* for
both topic models and supervised/measurement models. Document-level folds (ordinary
k-fold, grouped, temporal) with fixed seeds recorded in an :class:`AnalysisManifest`;
refit a fresh model per fold via a factory; evaluate held-out.

Non-goal: this never auto-chooses ``K`` or declares a topic substantively valid. It
reports evidence; the researcher adjudicates.

Public surface
--------------
- :func:`make_folds` — the fold engine (the single leakage guard).
- :func:`cross_validate` — the orchestrator (PR2 adds the supervised path).
- :class:`Folds`, :class:`CrossValResult` — the returned dataclasses.

Design notes (Gate A, #701)
---------------------------
- Fold seeds come from :class:`numpy.random.SeedSequence`, never ``hash()`` (which is
  non-deterministic across interpreter runs).
- Temporal ties are atomic: documents sharing a timestamp never straddle the
  train/test boundary, and the guard is a *strict* ``max(train) < min(test)``.
- Vocabulary is rebuilt per fold by default (leakage-free), so held-out perplexity is
  comparable only *within* a fold and is never pooled across folds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["make_folds", "Folds", "cross_validate", "CrossValResult"]


# ---------------------------------------------------------------------------
# Folds — the fold engine and its dataclass
# ---------------------------------------------------------------------------


@dataclass
class Folds:
    """A reproducible set of ``(train_idx, test_idx)`` document splits.

    The only place leakage is guarded. Every index array is ``int64``, in-bounds,
    and unique within itself; train and test are disjoint and nonempty.

    Attributes
    ----------
    splits : list of ``(train_idx, test_idx)`` int64 arrays.
    strategy : "kfold" | "grouped" | "temporal".
    seed : the base seed the split was derived from.
    fold_seeds : per-fold derived seeds (from ``SeedSequence(seed).spawn``), the
        integer each fold's ``factory(seed_fold)`` receives.
    oof_mask : boolean array over ``n_docs``; True where a document is tested in
        some fold. Always all-True for kfold/grouped (exactly-once test coverage);
        for temporal the initial training window is never tested, so those entries
        are False and are excluded from pooled out-of-fold metrics.
    n_docs : the document count these folds index.
    window : temporal rolling-window size, or None for an expanding window.
    groups_hash / times_hash : provenance hashes recorded in the manifest.
    """

    splits: list[tuple[np.ndarray, np.ndarray]]
    strategy: str
    seed: int
    fold_seeds: list[int]
    oof_mask: np.ndarray
    n_docs: int
    window: int | None = None
    groups_hash: str | None = None
    times_hash: str | None = None

    def __len__(self) -> int:
        return len(self.splits)

    def __iter__(self):
        return iter(self.splits)

    def __getitem__(self, i):
        return self.splits[i]

    def __repr__(self) -> str:
        return (
            f"Folds(strategy={self.strategy!r}, folds={len(self.splits)}, "
            f"n_docs={self.n_docs}, tested={int(self.oof_mask.sum())}/{self.n_docs})"
        )


def _hash_array(a) -> str | None:
    """A short stable content hash for manifest provenance (order-sensitive)."""
    if a is None:
        return None
    import hashlib

    arr = np.asarray(a)
    return hashlib.sha256(arr.tobytes() + str(arr.dtype).encode() + str(arr.shape).encode()).hexdigest()[:16]


def _derive_fold_seeds(seed: int, k: int) -> tuple[np.random.SeedSequence, list[int]]:
    """Base SeedSequence + K reproducible child seeds. No ``hash()`` anywhere."""
    ss = np.random.SeedSequence(int(seed))
    children = ss.spawn(k)
    # A stable, human-readable integer seed per fold (uint32, positive).
    fold_seeds = [int(c.generate_state(1, dtype=np.uint32)[0]) for c in children]
    return ss, fold_seeds


def _validate_split(train, test, n_docs):
    """Enforce the per-fold invariants (Gate A-A3)."""
    for name, idx in (("train", train), ("test", test)):
        if idx.dtype != np.int64:
            raise ValueError(f"{name} indices must be int64, got {idx.dtype}")
        if idx.size == 0:
            raise ValueError(f"a fold has an empty {name} set")
        if idx.min() < 0 or idx.max() >= n_docs:
            raise ValueError(f"{name} index out of bounds [0, {n_docs})")
        if np.unique(idx).size != idx.size:
            raise ValueError(f"duplicate indices within a fold {name} set")
    if np.intersect1d(train, test).size:
        raise ValueError("a document appears in both train and test of a fold")


def _validate_folds_obj(fold_obj, n_docs):
    """Re-run the per-fold invariants + seed-count check on a user-supplied Folds,
    so an externally constructed (or hand-edited) Folds can never smuggle a leak in."""
    if len(fold_obj.fold_seeds) != len(fold_obj.splits):
        raise ValueError(
            f"Folds has {len(fold_obj.splits)} splits but "
            f"{len(fold_obj.fold_seeds)} fold seeds; they must match"
        )
    for train, test in fold_obj.splits:
        _validate_split(np.asarray(train), np.asarray(test), n_docs)
    tested = np.concatenate([np.asarray(test) for _, test in fold_obj.splits])
    # Every strategy needs test indices unique ACROSS folds: a document tested in two
    # folds would make the out-of-fold scatter overwrite one prediction (temporal is
    # the sneaky case — its blocks are disjoint when generated, but a supplied Folds
    # could overlap them).
    if np.unique(tested).size != tested.size:
        raise ValueError(
            "a document is a test document in more than one fold; test blocks must be "
            "disjoint across folds (out-of-fold assembly needs one prediction per doc)"
        )
    if fold_obj.strategy in ("kfold", "grouped") and np.unique(tested).size != n_docs:
        raise ValueError(
            f"{fold_obj.strategy} Folds must test every document exactly once"
        )
    # oof_mask must equal the set of tested documents.
    expected = np.zeros(n_docs, dtype=bool)
    expected[np.unique(tested)] = True
    if not np.array_equal(np.asarray(fold_obj.oof_mask, dtype=bool), expected):
        raise ValueError("Folds.oof_mask does not match the union of test indices")


def _make_kfold(n_docs, folds, rng):
    perm = rng.permutation(n_docs)
    test_blocks = np.array_split(perm, folds)
    splits = []
    for block in test_blocks:
        test = np.sort(block).astype(np.int64)
        train = np.setdiff1d(np.arange(n_docs, dtype=np.int64), test, assume_unique=False)
        splits.append((train, test))
    return splits


def _make_grouped(n_docs, folds, groups, rng):
    groups = np.asarray(groups)
    if groups.shape[0] != n_docs:
        raise ValueError(f"groups has length {groups.shape[0]}, expected {n_docs}")
    # Missing group ids are not allowed (they cannot be held out coherently).
    missing = np.array([g is None or (isinstance(g, float) and np.isnan(g)) for g in groups])
    if missing.any():
        raise ValueError(f"{int(missing.sum())} documents have a missing group id")

    uniq, counts = np.unique(groups, return_counts=True)
    if uniq.size < folds:
        raise ValueError(
            f"grouped CV needs at least folds={folds} distinct groups, got {uniq.size}"
        )
    # Dominant-group guard (Gate A-B6): greedily bin groups into K folds by
    # descending size (multiway partition); if the largest group alone forces a
    # fold to swallow more than a balanced share we still proceed, but if a single
    # group exceeds what leaves K nonempty folds, fail loudly.
    order = np.argsort(-counts)
    bins = [[] for _ in range(folds)]
    bin_sizes = np.zeros(folds, dtype=np.int64)
    for gi in order:
        target = int(np.argmin(bin_sizes))
        bins[target].append(uniq[gi])
        bin_sizes[target] += counts[gi]
    if (bin_sizes == 0).any():
        biggest = uniq[order[0]]
        raise ValueError(
            f"cannot build {folds} group-disjoint folds: group {biggest!r} holds "
            f"{counts[order[0]]}/{n_docs} documents, leaving an empty fold. Reduce "
            "folds or merge groups."
        )
    # A dominant group produces valid but severely imbalanced folds; the split is
    # honest (no leakage, no dropped fold), but the researcher should know their
    # held-out sizes are lopsided (Gate A-B6).
    balanced = n_docs / folds
    if bin_sizes.max() > 2.0 * balanced:
        import warnings

        biggest = uniq[order[0]]
        warnings.warn(
            f"grouped folds are imbalanced: the largest test fold holds "
            f"{int(bin_sizes.max())} docs vs a balanced ~{balanced:.0f} (group "
            f"{biggest!r} alone is {counts[order[0]]}/{n_docs}). Held-out metrics "
            "will weight that fold heavily.",
            stacklevel=2,
        )
    all_idx = np.arange(n_docs, dtype=np.int64)
    splits = []
    for b in bins:
        test_mask = np.isin(groups, b)
        test = all_idx[test_mask]
        train = all_idx[~test_mask]
        splits.append((train, test))
    return splits


def _make_temporal(n_docs, folds, times, window):
    times = np.asarray(times)
    if times.shape[0] != n_docs:
        raise ValueError(f"times has length {times.shape[0]}, expected {n_docs}")
    # NaN sorts unpredictably and defeats the strict-ordering guard (NaN comparisons
    # are always False), so reject missing timestamps outright.
    if times.dtype.kind == "f" and np.isnan(times).any():
        raise ValueError(
            f"{int(np.isnan(times).sum())} documents have a missing (NaN) timestamp"
        )
    order = np.argsort(times, kind="stable")
    sorted_times = times[order]
    # Tie groups: contiguous runs of equal timestamp in sorted order. A tie group
    # is atomic — never split across the train/test boundary (Gate A-A2/B4).
    boundaries = np.flatnonzero(sorted_times[1:] != sorted_times[:-1]) + 1
    tie_starts = np.concatenate(([0], boundaries))
    tie_ends = np.concatenate((boundaries, [n_docs]))
    n_ties = tie_starts.size
    if n_ties < folds + 1:
        raise ValueError(
            f"temporal CV with folds={folds} needs at least {folds + 1} distinct "
            f"timestamps, got {n_ties}"
        )
    # Split the tie groups into folds+1 contiguous blocks: block 0 is the initial
    # training window; folds f=1..K test on block f, training on all earlier blocks.
    block_of_tie = np.array_split(np.arange(n_ties), folds + 1)
    # Map tie-block -> the sorted positions it covers.
    def tie_block_positions(tie_ids):
        lo = tie_starts[tie_ids[0]]
        hi = tie_ends[tie_ids[-1]]
        return lo, hi

    splits = []
    for f in range(1, folds + 1):
        test_lo, test_hi = tie_block_positions(block_of_tie[f])
        if window is None:
            train_lo = 0
        else:
            # rolling window: the `window` tie-blocks immediately before this test.
            first_train_block = max(0, f - window)
            train_lo, _ = tie_block_positions(block_of_tie[first_train_block])
        train_hi = test_lo
        train = np.sort(order[train_lo:train_hi]).astype(np.int64)
        test = np.sort(order[test_lo:test_hi]).astype(np.int64)
        # Strict-ordering guard: no timestamp shared across the boundary.
        if train.size and times[train].max() >= times[test].min():
            raise ValueError(
                "temporal split would place a shared timestamp in both train and "
                "test; a single tie group is larger than a fold block"
            )
        splits.append((train, test))

    # Ties are atomic (split over tie-group ids, never positions), so a shared
    # timestamp can never straddle the boundary — the guard above is defensive. But
    # a tie group larger than a balanced block yields a valid, lopsided test fold;
    # surface that rather than let it pass silently.
    test_sizes = np.array([e.size for _, e in splits])
    balanced = n_docs / (folds + 1)
    if test_sizes.max() > 3.0 * balanced:
        import warnings

        warnings.warn(
            f"temporal folds are imbalanced: a tied-timestamp block gives a test "
            f"fold of {int(test_sizes.max())} docs vs a balanced ~{balanced:.0f}. "
            "Held-out metrics will weight that fold heavily.",
            stacklevel=2,
        )
    return splits


def make_folds(
    n_docs: int,
    *,
    strategy: str = "kfold",
    folds: int = 5,
    groups=None,
    times=None,
    window: int | None = None,
    seed: int = 13,
) -> Folds:
    """Build a reproducible :class:`Folds` object — the CV leakage guard.

    Parameters
    ----------
    n_docs : number of documents to split.
    strategy : "kfold" (shuffle + K contiguous test blocks), "grouped" (whole
        groups held out together; requires ``groups``), or "temporal" (ordered by
        ``times``, test always strictly after train; requires ``times``).
    folds : number of folds K.
    groups : per-document group id (length ``n_docs``), required for "grouped".
    times : per-document order key (length ``n_docs``), required for "temporal".
    window : temporal only — None for an expanding training window (default), or an
        int rolling-window size in tie-blocks.
    seed : base seed; per-fold seeds are derived via ``SeedSequence(seed).spawn``.

    Returns
    -------
    Folds
    """
    if not isinstance(n_docs, (int, np.integer)) or n_docs < 2:
        raise ValueError(f"n_docs must be an int >= 2, got {n_docs!r}")
    if not isinstance(folds, (int, np.integer)) or folds < 2:
        raise ValueError(f"folds must be an int >= 2, got {folds!r}")
    if folds > n_docs:
        raise ValueError(f"folds={folds} exceeds n_docs={n_docs}")

    ss, fold_seeds = _derive_fold_seeds(seed, folds)
    rng = np.random.default_rng(ss)

    if strategy == "kfold":
        splits = _make_kfold(n_docs, folds, rng)
    elif strategy == "grouped":
        if groups is None:
            raise ValueError("strategy='grouped' requires groups=")
        splits = _make_grouped(n_docs, folds, groups, rng)
    elif strategy == "temporal":
        if times is None:
            raise ValueError("strategy='temporal' requires times=")
        splits = _make_temporal(n_docs, folds, times, window)
    else:
        raise ValueError(
            f"unknown strategy {strategy!r}; use 'kfold', 'grouped', or 'temporal'"
        )

    for train, test in splits:
        _validate_split(train, test, n_docs)

    # Exactly-once test coverage for kfold/grouped (Gate A-A3).
    tested = np.concatenate([test for _, test in splits])
    if strategy in ("kfold", "grouped"):
        if np.unique(tested).size != n_docs or tested.size != n_docs:
            raise ValueError(
                f"{strategy} folds must test every document exactly once "
                f"(covered {np.unique(tested).size}/{n_docs})"
            )
    oof_mask = np.zeros(n_docs, dtype=bool)
    oof_mask[np.unique(tested)] = True

    return Folds(
        splits=splits,
        strategy=strategy,
        seed=int(seed),
        fold_seeds=fold_seeds,
        oof_mask=oof_mask,
        n_docs=int(n_docs),
        window=window,
        groups_hash=_hash_array(groups),
        times_hash=_hash_array(times),
    )


# ---------------------------------------------------------------------------
# cross_validate — the topic-model orchestrator (PR1)
# ---------------------------------------------------------------------------


@dataclass
class CrossValResult:
    """Result of :func:`cross_validate`.

    Attributes
    ----------
    per_fold : one dict per fold (fold index, train/test sizes, seed_fold,
        vocab_size, oov_dropped, and each metric).
    aggregate : macro mean/std per metric across folds. Perplexity is macro-only
        with a different-vocabulary caveat (per-fold vocabularies differ under the
        default ``vocab="per_fold"``, so pooling perplexity is not meaningful).
    folds : the :class:`Folds` object (assignments + per-fold seeds).
    manifest : the :class:`~topica.manifest.AnalysisManifest`, or None.
    vocab : the vocabulary mode used ("per_fold" or "fixed").
    """

    per_fold: list[dict]
    aggregate: dict
    folds: Folds
    manifest: Any = None
    vocab: str = "per_fold"
    stability: dict | None = None
    """Cross-fold topic stability (mean +/- std matched-topic cosine over all fold
    pairs), computed vocabulary-aware via align_topics since fold vocabularies
    differ. None when there are too few comparable fits."""
    covariate_stability: dict | None = None
    """Covariate-effect fold-stability for models that learn covariate coefficients
    (keyATM covariate, DMR, GDMR): after aligning topics across every fold pair,
    the sign-agreement rate and magnitude (Pearson) correlation of the fold-aligned
    covariate effects (lambda), per feature and pooled.

    This is deliberately NOT a predictive-coverage statistic and is never stored in
    a ``coverage_*`` field: lambda has no per-document ground truth, and the
    conditional ``feature_effect_se`` is under-dispersed, so a +/-2*SE-overlap
    "stability" would read as spuriously stable. Sign-agreement + magnitude
    correlation ask the honest question instead — do the folds recover effects of
    the same direction and relative size? None when fewer than two fold models
    expose ``feature_effects``."""
    kind: str = "topic"
    """"topic" (held-out completion) or "supervised" (out-of-fold prediction)."""
    oof_predictions: Any = None
    """Supervised path: assembled out-of-fold y-hat, length n_docs, NaN off the
    scored population."""
    oof_std: Any = None
    """Supervised path: per-doc predictive std from predict(return_std=True)."""
    scored_mask: Any = None
    """Supervised path: bool array (n_docs) — the authoritative evaluated population
    (tested AND survived per-fold pruning). Drives every pooled metric, calibration,
    and coverage identically."""
    calibration_table: Any = None
    """Supervised path: reliability table (DataFrame) of binned predicted vs observed."""

    def to_frame(self):
        """Per-fold metrics as a pandas DataFrame (a results-appendix table)."""
        import pandas as pd

        return pd.DataFrame(self.per_fold)

    def covariate_stability_frame(self):
        """Per-feature covariate-effect stability as a pandas DataFrame (one row per
        covariate: feature, sign_agreement, magnitude_correlation, n). None when this
        run has no covariate-effect stability (see :attr:`covariate_stability`)."""
        if self.covariate_stability is None:
            return None
        import pandas as pd

        rows = [
            {"feature": name, **stat}
            for name, stat in self.covariate_stability["per_feature"].items()
        ]
        return pd.DataFrame(rows)

    def summary(self) -> str:
        if self.kind == "supervised":
            return self._supervised_summary()
        lines = [
            f"cross_validate: {len(self.per_fold)} folds, "
            f"strategy={self.folds.strategy}, vocab={self.vocab}",
        ]
        for m, agg in self.aggregate.items():
            lines.append(f"  {m}: {agg['mean']:.4g} +/- {agg['std']:.4g}")
        # Surface covariate conditioning in the headline: a researcher who passed
        # covariates must see whether held-out scoring actually conditioned on them,
        # not have to dig into to_frame() to discover a marginal fallback.
        cond = [r["covariate_conditioned"] for r in self.per_fold
                if "covariate_conditioned" in r]
        if cond:
            if all(cond):
                lines.append("  covariates: conditioned held-out inference")
            elif not any(cond):
                lines.append(
                    "  covariates: MARGINAL fallback — held-out scoring did NOT "
                    "condition on them (this model's transform ignores covariates)"
                )
            else:
                lines.append(
                    f"  covariates: conditioned on {sum(cond)}/{len(cond)} folds "
                    "(marginal on the rest)"
                )
        if self.stability is not None:
            lines.append(
                f"  stability (cross-fold cosine): {self.stability['mean']:.4g} "
                f"+/- {self.stability['std']:.4g}"
            )
        if self.covariate_stability is not None:
            cs = self.covariate_stability
            if cs.get("effects_near_zero"):
                lines.append(
                    "  covariate-effect stability: UNDEFINED — the learned effects "
                    "are ~0 (the model likely needs more fit iterations before its "
                    "covariate effects mean anything)"
                )
            else:
                mc = cs["magnitude_correlation"]
                mc_str = ("n/a (no variance)" if not np.isfinite(mc) else f"{mc:.3f}")
                lines.append(
                    f"  covariate-effect stability: sign-agreement "
                    f"{cs['sign_agreement']:.3f}, magnitude corr {mc_str} "
                    f"(feature-macro over {cs['n_features']} covariate(s), "
                    f"{cs['n_pairs']} fold pairs; NOT predictive coverage; effects are "
                    f"learned per fold — unaffected by any marginal held-out fallback)"
                )
            if cs.get("partial_alignment"):
                tc = cs["topics_compared"]
                lines.append(
                    f"    (partial alignment: some fold pairs compared as few as "
                    f"{tc['min']}/{cs['n_topics']} topics — unaligned topics are "
                    f"excluded, which can bias stability upward)"
                )
            if cs.get("caveat"):
                lines.append(f"    (caveat: {cs['caveat']})")
        if self.vocab == "per_fold":
            lines.append(
                "  (perplexity is per-fold; vocabularies differ so it is not pooled)"
            )
        return "\n".join(lines)

    def _supervised_summary(self) -> str:
        a = self.aggregate
        n = int(self.scored_mask.sum()) if self.scored_mask is not None else 0
        lines = [
            f"cross_validate (supervised): {len(self.per_fold)} folds, "
            f"strategy={self.folds.strategy}, {n} out-of-fold predictions",
            f"  RMSE (pooled): {a.get('rmse_pooled', float('nan')):.4g}   "
            f"MAE: {a.get('mae_pooled', float('nan')):.4g}   "
            f"R2: {a.get('r2_pooled', float('nan')):.4g}",
        ]
        if "rmse_macro" in a:
            m = a["rmse_macro"]
            lines.append(
                f"  RMSE (per-fold): {m['mean']:.4g} +/- {m['std']:.4g} "
                f"over {m['n_valid_folds']} folds"
            )
        if "coverage_90" in a:
            lines.append(
                f"  interval coverage: 90% -> {a['coverage_90']:.3f}, "
                f"95% -> {a.get('coverage_95', float('nan')):.3f}"
            )
        if "calibration_slope" in a:
            lines.append(
                f"  calibration: intercept {a['calibration_intercept']:.3g}, "
                f"slope {a['calibration_slope']:.3g} (ideal 0, 1)"
            )
        lines.append(
            "  (coverage uses a conditional Gaussian interval with in-sample sigma^2; "
            "out-of-fold coverage below nominal is partly expected, not proof of "
            "miscalibration)"
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"CrossValResult(folds={len(self.per_fold)}, metrics={list(self.aggregate)})"


# Named covariate routing: how each family takes covariates at fit time and how it
# conditions held-out inference on test covariates (Gate A-A5). Families absent here
# are scored marginally (documents only), which is labeled honestly.
def _route_transform(model, est_docs, cov_slice, seed):
    """Covariate-aware theta for held-out inference. Returns (theta, conditioned).

    ``conditioned`` is False when the model's transform cannot take covariates, so
    the held-out score is marginal and must be labeled as such.
    """
    name = type(model).__name__
    if name == "STM":
        from . import stm as _stm

        prev = cov_slice.get("prevalence")
        theta = _stm.transform(model, est_docs, prevalence=prev)
        return np.asarray(theta, dtype=np.float64), prev is not None
    if name in ("DMR", "GDMR"):
        feats = cov_slice.get("features")
        if name == "GDMR":
            feats = cov_slice.get("metadata", feats)
        theta = model.transform(est_docs, feats) if feats is not None else model.transform(est_docs)
        return np.asarray(theta, dtype=np.float64), feats is not None
    # Marginal path (LDA, CTM, HDP, keyATM, ...): documents only.
    fn = getattr(model, "transform", None)
    if not callable(fn):
        raise ValueError(
            f"{name} has no transform(); held-out completion needs a generative "
            "model that infers topics for new documents"
        )
    from .validation import _accepts_kwarg

    theta = fn(est_docs, seed=seed) if _accepts_kwarg(fn, "seed") else fn(est_docs)
    return np.asarray(theta, dtype=np.float64), False


# Which fit kwargs each family accepts as per-document covariates, with aliases.
# Keys are the exact public class names (type(model).__name__).
_FIT_COV = {
    "STM": (("prevalence", "content", "content_time"), {"covariates": "prevalence"}),
    "DMR": (("features", "offset"), {"covariates": "features"}),
    "GDMR": (("features", "metadata"), {"covariates": "features"}),
    "KeyATM": (("covariates", "times", "prior_offset"), {"timestamps": "times"}),
}


def _canonical_family_cov(cov, model_name):
    """Resolve user covariate keys to a family's canonical fit kwargs, hard-failing
    any key the model does not accept. The same canonical dict drives both the fit
    and the held-out scorer, so an aliased covariate conditions test inference too."""
    if not cov:
        return {}
    names, aliases = _FIT_COV.get(model_name, ((), {}))
    canon = {}
    for key, val in cov.items():
        target = aliases.get(key, key)
        if target not in names:
            raise ValueError(
                f"{model_name} does not accept covariate {key!r}; recognized keys "
                f"for this model: {names or '(none — this model has no covariates)'}"
            )
        canon[target] = val
    return canon


def _resolve_covariates(covariates, n_docs):
    """Normalize ``covariates=`` to a dict of length-n_docs arrays; assert lengths."""
    if covariates is None:
        return {}
    if hasattr(covariates, "columns"):  # a DataFrame bound to a single named kwarg
        raise ValueError(
            "pass covariates as a dict keyed by the model's fit kwarg, e.g. "
            "{'prevalence': X}; a bare DataFrame is ambiguous (STM prevalence is one "
            "design matrix, not one kwarg per column)"
        )
    if not isinstance(covariates, dict):
        raise ValueError("covariates must be a dict {kwarg_name: array}")
    out = {}
    for key, arr in covariates.items():
        # topica.one_hot / design_matrix return (matrix, names); a common first-timer
        # mistake is to pass that tuple straight through. np.asarray on it yields a
        # ragged object array and a cryptic downstream error — catch it here.
        if isinstance(arr, tuple):
            raise ValueError(
                f"covariate {key!r} is a tuple, not an array — did you forget to "
                f"unpack one_hot()/design_matrix()? They return (matrix, names); pass "
                f"the matrix, e.g. `X, names = topica.one_hot(...)` then "
                f"`covariates={{'{key}': X}}`"
            )
        a = np.asarray(arr)
        if a.dtype == object or not np.issubdtype(a.dtype, np.number):
            raise ValueError(
                f"covariate {key!r} must be a numeric array, got dtype {a.dtype}; "
                "encode categoricals with topica.one_hot(...) first"
            )
        if a.shape[0] != n_docs:
            raise ValueError(
                f"covariate {key!r} has length {a.shape[0]}, expected n_docs={n_docs}"
            )
        out[key] = a
    return out


def _slice_cov(cov, idx):
    return {k: v[idx] for k, v in cov.items()}


def _heldout_completion(model, test_docs, cov_slice, seed):
    """Document-completion held-out likelihood on the fold's test docs.

    Mirrors :func:`topica.validation.perplexity` (even tokens infer theta, odd
    tokens are scored) but conditions theta on the doc's test covariates, and keeps
    the covariate rows aligned to the docs it actually scores (Gate A-A7/B3: short
    docs are co-dropped with their covariate rows). Returns a dict with per-fold
    perplexity, scored-token count, and OOV info.
    """
    from .coherence import _as_topic_word

    phi = _as_topic_word(model)
    if phi.shape[0] == 0:
        raise ValueError("the model has no topics (empty topic_word)")
    vocab = {w: i for i, w in enumerate(model.vocabulary)}

    est, ev, kept = [], [], []
    oov_dropped = 0
    for i, d in enumerate(test_docs):
        # Drop OOV *before* the completion split (design §3a), so both the theta
        # estimate and the scored half are over the fold vocabulary, and the
        # 2-token minimum is measured in in-vocabulary tokens.
        in_vocab = [w for w in d if w in vocab]
        oov_dropped += len(d) - len(in_vocab)
        if len(in_vocab) < 2:
            continue
        est.append(in_vocab[0::2])
        ev.append(in_vocab[1::2])
        kept.append(i)
    if not est:
        raise ValueError("no test document had at least 2 tokens")

    kept = np.asarray(kept, dtype=np.int64)
    cov_kept = _slice_cov(cov_slice, kept)
    theta, conditioned = _route_transform(model, est, cov_kept, seed)

    logp, n = 0.0, 0
    for j, evdoc in enumerate(ev):
        ids = [vocab[w] for w in evdoc if w in vocab]
        if not ids:
            continue
        pw = np.clip(theta[j] @ phi[:, ids], 1e-12, None)
        logp += float(np.log(pw).sum())
        n += len(ids)
    if n == 0:
        raise ValueError("no held-out token was in the fold vocabulary")
    return {
        "perplexity": float(np.exp(-logp / n)),
        "heldout_loglik": logp,
        "n_eval_tokens": n,
        "n_scored_docs": int(kept.size),
        "oov_dropped": int(oov_dropped),
        "covariate_conditioned": bool(conditioned),
    }


def _fold_quality(model, train_docs, coherence_type, topn):
    """Coherence + exclusivity on the fold's *training* text (the reference corpus
    the topics were estimated on — Gate A-A12). A metric that raises is dropped from
    the record with a one-time warning, so a broken metric is visible, not silent."""
    import warnings

    out = {}
    try:
        topics = model.top_words(topn)
    except Exception as exc:
        warnings.warn(f"fold quality skipped: top_words failed ({exc})", stacklevel=3)
        return out
    try:
        from .coherence import coherence as _coh

        out["coherence"] = float(
            np.mean(_coh(topics, train_docs, coherence_type=coherence_type, n=topn))
        )
    except Exception as exc:
        warnings.warn(f"fold coherence dropped ({exc})", stacklevel=3)
    try:
        from .coherence import exclusivity as _excl

        out["exclusivity"] = float(np.mean(_excl(model, n=topn)))
    except Exception as exc:
        warnings.warn(f"fold exclusivity dropped ({exc})", stacklevel=3)
    return out


def cross_validate(
    factory,
    docs,
    *,
    y=None,
    covariates=None,
    covariate_names=None,
    folds=5,
    strategy="kfold",
    groups=None,
    times=None,
    window=None,
    vocab="per_fold",
    seed=13,
    fit_kwargs=None,
    metrics=None,
    fit_fn=None,
    score_fn=None,
    coherence_type="c_v",
    topn=10,
    coverage_levels=(0.90, 0.95),
    predict_kwargs=None,
    n_jobs=1,
    manifest=True,
    preprocessing=None,
):
    """Cross-validate a topic model's held-out predictive behavior (#701, PR1).

    Refits a fresh model per fold (via ``factory(seed_fold)``) on the training
    documents, rebuilds the vocabulary per fold by default (leakage-free), and
    scores held-out document-completion on the test documents, conditioning on the
    fold's test covariates where the family supports it.

    Parameters
    ----------
    factory : ``callable(seed_fold) -> unfitted model``. The seed MUST be threaded
        into the constructor, e.g. ``lambda s: topica.STM(10, seed=s)``.
    docs : list[list[str]] or a Corpus.
    covariates : dict ``{fit_kwarg: array}`` of per-document covariates (length
        n_docs), e.g. ``{"prevalence": X}`` for STM. Sub-indexed per fold.
    covariate_names : optional names for the covariate columns, used to label the
        covariate-effect stability table/plot (keyATM/DMR/GDMR). A flat sequence
        (``["Liberal", "day_z"]``) or a dict keyed by the covariate kwarg. Length must
        equal the number of covariate effects (intercept excluded); handy because
        ``one_hot`` returns the names separately from the matrix.
    y : per-document numeric response (length n_docs). When given, runs the
        supervised out-of-fold path (regression): fit on each training fold, predict
        the held-out response, assemble an out-of-fold vector, and report pooled +
        per-fold RMSE/MAE/R2, interval coverage, and calibration. The model must
        expose ``predict(docs, return_std=True)`` (e.g. SupervisedLDA).
    folds, strategy, groups, times, window, seed : see :func:`make_folds`. ``folds``
        may also be a prebuilt :class:`Folds` object (takes precedence).
    vocab : "per_fold" (default, leakage-free; perplexity not pooled) or "fixed"
        (shared vocabulary, comparable perplexity, but feature-selection leakage —
        emits a warning).
    fit_kwargs : static hyperparameters passed to ``model.fit`` every fold.
    fit_fn / score_fn : escape hatch. ``fit_fn(train_docs, train_idx, seed_fold) ->
        model`` and ``score_fn(model, test_docs, test_idx, seed_fold) -> dict`` let
        you own covariate handling the named routing does not cover.
    coherence_type, topn : fold-quality coherence settings.
    manifest : record fold assignments + seeds in an AnalysisManifest.

    Returns
    -------
    CrossValResult
    """
    import warnings

    if metrics is not None:
        raise NotImplementedError(
            "custom metric selection (metrics=) is not implemented. The topic path "
            "reports held-out perplexity + fold coherence/exclusivity/stability; the "
            "supervised path (y=) reports regression RMSE/MAE/R2, interval coverage, "
            "and calibration. Classification metrics (accuracy/F1/log-loss) are not "
            "available: no model in the roster exposes predict_proba."
        )
    if fit_fn is not None and score_fn is None:
        raise ValueError("fit_fn requires a matching score_fn (test-time inference)")
    if score_fn is not None and fit_fn is None:
        raise ValueError("score_fn requires a matching fit_fn (the escape hatch is both)")

    # Normalize documents to token lists.
    if hasattr(docs, "documents"):
        # A pre-built Corpus already had its vocabulary/frequency pruning learned on
        # the WHOLE corpus, which per-fold rebuilding cannot undo — so this is not the
        # leakage-free path the default implies. Warn; pass token lists for a truly
        # per-fold vocabulary.
        if vocab == "per_fold" and getattr(docs, "preprocessing", None):
            warnings.warn(
                "cross_validate received a pre-built Corpus whose vocabulary was "
                "pruned on the full corpus; per-fold vocabulary rebuilding cannot "
                "undo that global feature selection, so this is not fully leakage-"
                "free. Pass raw token lists (list[list[str]]) for a truly per-fold "
                "vocabulary.",
                stacklevel=2,
            )
        doc_lists = [list(d) for d in docs.documents()]
    else:
        doc_lists = [list(d) for d in docs]
    n_docs = len(doc_lists)

    cov = _resolve_covariates(covariates, n_docs)
    fit_kwargs = dict(fit_kwargs or {})
    predict_kwargs = dict(predict_kwargs or {})
    preprocessing = dict(preprocessing or {})
    for k in cov:
        if k in fit_kwargs:
            raise ValueError(f"covariate {k!r} collides with a fit_kwargs key")

    # Validate the supervised response up front (it bypasses _resolve_covariates, so a
    # truncated/off-by-one y would otherwise slip through — the classic covariate-CV
    # bug PR1 guards for covariates).
    if y is not None:
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")
        if y.shape[0] != n_docs:
            raise ValueError(f"y has length {y.shape[0]}, expected n_docs={n_docs}")
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains NaN or inf; the response must be finite")
        if fit_fn is not None:
            raise ValueError("the supervised path (y=) does not use fit_fn/score_fn")

    if vocab not in ("per_fold", "fixed"):
        raise ValueError("vocab must be 'per_fold' or 'fixed'")
    if vocab == "fixed":
        warnings.warn(
            "vocab='fixed': the vocabulary is learned once on the full corpus, so "
            "test-fold word frequencies leak into feature selection. Perplexity is "
            "comparable across folds, but this is not fully leakage-free.",
            stacklevel=2,
        )

    # Build folds (or accept a prebuilt Folds).
    if isinstance(folds, Folds):
        fold_obj = folds
        if fold_obj.n_docs != n_docs:
            raise ValueError(
                f"prebuilt Folds is for {fold_obj.n_docs} docs, got {n_docs}"
            )
        # Re-check the leakage invariants on a user-supplied Folds — never trust it
        # blind (the whole point of the guard is that it always runs).
        _validate_folds_obj(fold_obj, n_docs)
        # A supplied Folds takes precedence; splitter arguments are then inert.
        if any(a is not None for a in (groups, times, window)) or strategy != "kfold":
            warnings.warn(
                "a prebuilt Folds was supplied; strategy/groups/times/window are "
                "ignored (the Folds object already fixes the split).",
                stacklevel=2,
            )
    else:
        fold_obj = make_folds(
            n_docs, strategy=strategy, folds=folds, groups=groups, times=times,
            window=window, seed=seed,
        )

    from . import Corpus  # local import to avoid a cycle at module load

    fixed_corpus = None
    if vocab == "fixed":
        fixed_corpus = Corpus.from_documents(doc_lists, **preprocessing)

    # Supervised out-of-fold path (PR2) — a separate loop + aggregation.
    if y is not None:
        return _supervised_cv(
            factory, doc_lists, y, cov, fold_obj, vocab, fixed_corpus, Corpus,
            fit_kwargs, predict_kwargs, preprocessing, coverage_levels, n_jobs, manifest,
        )

    per_fold = []
    fold_models = []
    for f, ((train_idx, test_idx), seed_fold) in enumerate(
        zip(fold_obj.splits, fold_obj.fold_seeds)
    ):
        train_docs = [doc_lists[i] for i in train_idx]
        test_docs = [doc_lists[i] for i in test_idx]
        model = factory(seed_fold)

        rec = {
            "fold": f,
            "n_train": int(train_idx.size),
            "n_test": int(test_idx.size),
            "seed_fold": int(seed_fold),
        }

        if fit_fn is not None:
            model = fit_fn(train_docs, train_idx, seed_fold)
            rec.update(score_fn(model, test_docs, test_idx, seed_fold) or {})
            per_fold.append(rec)
            fold_models.append(model)
            continue

        # Per-fold (or fixed) vocabulary rebuild.
        if vocab == "per_fold":
            train_corpus = Corpus.from_documents(train_docs, **preprocessing)
        else:
            train_corpus = fixed_corpus.transform(train_docs)
        # Resolve the family's covariate kwargs (aliases -> canonical names) once,
        # hard-failing any key the model does not accept, and use the SAME canonical
        # dict for both fitting and held-out scoring so a covariate passed under an
        # alias still conditions test inference (not just the fit).
        canon = _canonical_family_cov(cov, type(model).__name__)

        # Align train covariates to the corpus rows that survived pruning.
        kept = np.asarray(train_corpus.kept_indices, dtype=np.int64)
        fit_cov = {k: v[train_idx][kept] for k, v in canon.items()}
        rec["vocab_size"] = int(train_corpus.num_words)

        model.fit(train_corpus, **fit_cov, **fit_kwargs)

        # Held-out completion on the test docs, conditioned on test covariates.
        test_cov = {k: v[test_idx] for k, v in canon.items()}
        rec.update(_heldout_completion(model, test_docs, test_cov, seed_fold))
        rec.update(_fold_quality(model, train_docs, coherence_type, topn))
        per_fold.append(rec)
        fold_models.append(model)

    # Aggregate: macro mean/std over folds for genuine quality metrics only, not
    # the bookkeeping fields (sizes, seeds, token counts, flags). heldout_loglik is
    # kept per-fold but NOT aggregated: raw log-likelihood totals scale with each
    # fold's test-token count, so a macro mean mixes denominators (use perplexity).
    _METRICS = ("perplexity", "coherence", "exclusivity")
    aggregate = {}
    for k in _METRICS:
        vals = np.array(
            [r[k] for r in per_fold if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)],
            dtype=np.float64,
        )
        if vals.size:
            aggregate[k] = {"mean": float(vals.mean()), "std": float(vals.std())}

    # If the user supplied covariates but the model's transform ignored them at
    # scoring time (e.g. keyATM), warn loudly — otherwise a marginal held-out score
    # looks identical to a conditioned one and the researcher would over-claim.
    if cov:
        cond_flags = [r.get("covariate_conditioned") for r in per_fold
                      if "covariate_conditioned" in r]
        if cond_flags and not any(cond_flags):
            warnings.warn(
                f"covariates were supplied but {type(fold_models[0]).__name__}'s "
                "transform cannot condition held-out inference on them, so the "
                "held-out scores are MARGINAL (not conditioned). Reported perplexity "
                "does not reflect the covariate. See covariate_conditioned in "
                "result.per_fold / result.summary().",
                stacklevel=2,
            )

    stability = _cross_fold_stability(fold_models)
    covariate_stability = _covariate_effect_stability(fold_models, cov, covariate_names)

    # Capture the fitted model's class + settings for the manifest (constant across
    # folds; the factory itself is an opaque callable).
    model_spec = None
    if fold_models and fold_models[0] is not None:
        m0 = fold_models[0]
        model_spec = {
            "class": type(m0).__name__,
            "settings": getattr(m0, "settings", None),
        }
    mani = (
        _record_cv_manifest(
            fold_obj, doc_lists, cov, fit_kwargs, preprocessing, vocab,
            fit_fn is not None, model_spec, coherence_type, topn,
        )
        if manifest
        else None
    )

    return CrossValResult(
        per_fold=per_fold,
        aggregate=aggregate,
        folds=fold_obj,
        manifest=mani,
        vocab=vocab,
        stability=stability,
        covariate_stability=covariate_stability,
    )


def _cross_fold_stability(models):
    """Mean +/- std matched-topic cosine similarity over all fold pairs, aligned
    vocabulary-aware (fold vocabularies differ under per-fold rebuild — Gate A-A12)."""
    usable = [m for m in models if m is not None and hasattr(m, "topic_word")]
    if len(usable) < 2:
        return None
    from .validation import align_topics

    sims = []
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            try:
                res = align_topics(usable[i], usable[j], metric="cosine")
                # `.matches` yields (topic_a, topic_b, similarity) — the cosine
                # itself (1.0 for a self-match), NOT a distance. Use it directly.
                matches = res.matches
                if len(matches):
                    sims.append(np.mean([s for _, _, s in matches]))
            except Exception as exc:
                import warnings

                warnings.warn(
                    f"cross-fold stability skipped a fold pair ({exc})", stacklevel=3
                )
                continue
    if not sims:
        return None
    sims = np.array(sims, dtype=np.float64)
    return {"mean": float(sims.mean()), "std": float(sims.std()), "n_pairs": int(sims.size)}


def _placeholder_names(cov_names):
    """True when the model handed back positional placeholders (keyATM emits
    ``feature_0``/``feature_1``/... for the covariate columns when the covariate
    arrives as a bare matrix), so the caller should substitute a kwarg label instead.
    ``cov_names`` is the covariate columns only (intercept already dropped)."""
    import re

    return bool(cov_names) and all(
        re.fullmatch(r"feature_\d+", str(nm)) for nm in cov_names
    )


def _resolve_covariate_names(covariate_names, cov, n_feat):
    """Turn the user's ``covariate_names=`` into a length-``n_feat`` list, or None.

    Accepts a flat sequence (``["Liberal", "day_z"]``) or a dict keyed by the covariate
    kwarg (``{"covariates": ["Liberal", "day_z"]}``). Any other length is a hard error —
    a silent mismatch would mislabel effects."""
    if covariate_names is None:
        return None
    names = covariate_names
    if isinstance(covariate_names, dict):
        keys = list(cov.keys()) if cov else []
        if len(keys) == 1 and keys[0] in covariate_names:
            names = covariate_names[keys[0]]
        elif len(covariate_names) == 1:
            names = next(iter(covariate_names.values()))
        else:
            raise ValueError(
                "covariate_names as a dict must key the single covariate kwarg "
                f"(got keys {list(covariate_names)} vs covariates {list(cov or [])})"
            )
    names = list(names)
    if len(names) != n_feat:
        raise ValueError(
            f"covariate_names has {len(names)} names but the model learned {n_feat} "
            "covariate effect(s); they must match (intercept is not named)"
        )
    return [str(x) for x in names]


def _covariate_effect_stability(models, cov=None, covariate_names=None):
    """Covariate-effect fold-stability for models with a learned lambda (keyATM
    covariate, DMR, GDMR).

    For every pair of fold models, align their topics (vocabulary-aware, since fold
    vocabularies differ), reorder the second fold's ``feature_effects`` rows onto the
    first's topics, then compare the matched covariate coefficients two ways:

    - **sign-agreement**: the fraction of (topic, covariate) cells where the two folds
      agree on the sign of the effect, and
    - **magnitude correlation**: the Pearson correlation of the matched effects.

    Both headline numbers are **macro-averaged over features** (mean of the per-feature
    statistics), not pooled over raw cells: pooling one Pearson correlation across
    features of different scales lets the highest-variance covariate dominate and hide
    instability in a smaller or null one (adversarial review, #705). Per-feature detail
    is in ``per_feature``.

    Partial topic alignment is disclosed, not hidden: ``align_topics`` may match fewer
    than K topics (an unstable topic whose word distribution also drifts can fall below
    the alignment threshold), and comparing only the topics that happened to align would
    bias stability upward. ``topics_compared``/``n_topics`` report the coverage so a
    reader can see when a fold pair dropped topics.

    The intercept column (0) is excluded — it is a per-topic baseline, not a covariate
    effect. This is NOT predictive coverage (lambda has no per-doc truth), and the
    caller must never fold it into a ``coverage_*`` field.

    Returns a dict (macro headline + per-feature) or None when fewer than two fold
    models expose ``feature_effects`` or the folds disagree on the number of covariate
    columns.
    """
    import warnings

    usable = []
    for m in models:
        if m is None or not hasattr(m, "topic_word"):
            continue
        try:
            fe = np.asarray(m.feature_effects, dtype=np.float64)
        except Exception:
            continue  # not a covariate model (feature_effects raises without covariates)
        if fe.ndim != 2 or fe.shape[1] < 2:
            continue  # need at least one covariate column beyond the intercept
        names = list(getattr(m, "feature_names", []))
        usable.append((m, fe, names))
    if len(usable) < 2:
        return None

    # All folds must share the covariate design (same coefficient columns); otherwise
    # aligning effects across folds compares different quantities.
    ncols = {fe.shape[1] for _, fe, _ in usable}
    if len(ncols) != 1:
        warnings.warn(
            "covariate-effect stability skipped: fold models have different numbers "
            f"of covariate coefficients ({sorted(ncols)}); the covariate design must "
            "match across folds to compare effects.",
            stacklevel=3,
        )
        return None

    from .validation import align_topics

    # Covariate columns are 1..F (drop the intercept at 0). Prefer the model's own
    # feature names; when it hands back positional placeholders (keyATM with a bare
    # covariate matrix), fall back to the covariate kwarg the user actually passed, so
    # a two-covariate model is not reported as feature_0/feature_1 (usability, #705).
    n_feat = usable[0][1].shape[1] - 1
    ref_names = usable[0][2]
    cov_names = list(ref_names[1:]) if len(ref_names) == n_feat + 1 else []
    user_names = _resolve_covariate_names(covariate_names, cov, n_feat)
    if user_names is not None:
        feat_names = user_names  # explicit names win over anything the model reports
    elif cov_names and not _placeholder_names(cov_names):
        feat_names = cov_names
    else:
        # A single covariate kwarg maps to the whole F-column matrix; label columns
        # kwarg[0..F-1]. With 0 or >1 kwargs, keep the neutral feature_j.
        keys = list(cov.keys()) if cov else []
        prefix = keys[0] if len(keys) == 1 else None
        feat_names = (
            [f"{prefix}[{j}]" for j in range(n_feat)] if prefix is not None
            else [f"feature_{j}" for j in range(n_feat)]
        )

    # GDMR coefficients are Legendre-basis terms whose affine map depends on each fold's
    # own covariate range (metadata_range=None infers it per fold), so column j means a
    # different function in different folds. The diagnostic still runs, but say so.
    caveat = None
    if any(type(m).__name__ == "GDMR" for m, _, _ in usable):
        caveat = (
            "GDMR coefficients are basis-expansion terms on a per-fold covariate range; "
            "aligning column j across folds compares different basis functions unless "
            "metadata_range= is fixed. Interpret magnitude correlation cautiously."
        )
        warnings.warn("covariate-effect stability: " + caveat, stacklevel=3)

    # Accumulate matched (fold_a, fold_b) coefficient values per feature over all pairs,
    # and track how many of K topics each pair actually compared.
    n_topics = int(usable[0][1].shape[0])
    per_feat_a = [[] for _ in range(n_feat)]
    per_feat_b = [[] for _ in range(n_feat)]
    n_pairs = 0
    topics_compared = []
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            (mi, fei, _), (mj, fej, _) = usable[i], usable[j]
            try:
                matches = align_topics(mi, mj, metric="cosine").matches
            except Exception as exc:
                warnings.warn(
                    f"covariate-effect stability skipped a fold pair ({exc})",
                    stacklevel=3,
                )
                continue
            if not matches:
                continue
            n_pairs += 1
            topics_compared.append(len(matches))
            for ta, tb, _sim in matches:
                for f in range(n_feat):
                    per_feat_a[f].append(fei[ta, f + 1])
                    per_feat_b[f].append(fej[tb, f + 1])

    if n_pairs == 0:
        return None

    # "Near zero" tolerance, relative to the largest learned effect anywhere: an
    # unconverged keyATM returns lambda == 0 for every topic (it has not warmed up),
    # and a zero coefficient has no sign to agree on. Scoring sign(0)==sign(0) as
    # agreement would report an all-zero (learned-nothing) model as perfectly stable
    # — the exact trap a first-time user hit (#705). Exclude ~0 cells instead.
    _all = np.array(
        [v for f in range(n_feat) for v in (per_feat_a[f] + per_feat_b[f])],
        dtype=np.float64,
    )
    scale = float(np.max(np.abs(_all))) if _all.size else 0.0
    tol = max(1e-12, 1e-8 * scale)
    effects_near_zero = scale <= tol  # the model learned essentially no covariate effect

    def _sign_agree(a, b):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        # A coefficient at (or near) 0 has no sign to agree on; count agreement only over
        # cells where at least one fold learned a nonzero effect. No such cell -> undefined.
        mask = (np.abs(a) > tol) | (np.abs(b) > tol)
        if not mask.any():
            return float("nan")
        return float(np.mean(np.sign(a[mask]) == np.sign(b[mask])))

    def _corr(a, b):
        a, b = np.asarray(a), np.asarray(b)
        if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
            return float("nan")  # a constant column has no magnitude to correlate
        return float(np.corrcoef(a, b)[0, 1])

    per_feature = {}
    for f in range(n_feat):
        a, b = per_feat_a[f], per_feat_b[f]
        mag = np.abs(np.concatenate([a, b])) if a else np.array([])
        per_feature[feat_names[f]] = {
            "sign_agreement": _sign_agree(a, b),
            "magnitude_correlation": _corr(a, b),
            "effect_magnitude": float(mag.mean()) if mag.size else float("nan"),
            "n": int(len(a)),
        }

    # Macro headline: mean of the per-feature statistics (equal weight per covariate),
    # NOT a single correlation over pooled raw cells. Skip NaN per-feature values
    # (constant/near-zero columns) so one degenerate covariate does not blank the headline.
    sa = np.array([v["sign_agreement"] for v in per_feature.values()], dtype=np.float64)
    mc = np.array([v["magnitude_correlation"] for v in per_feature.values()],
                  dtype=np.float64)
    sa = sa[np.isfinite(sa)]
    mc = mc[np.isfinite(mc)]
    tc = np.array(topics_compared, dtype=np.float64)

    note = ("covariate-effect fold-stability (feature-macro sign-agreement + "
            "magnitude correlation of aligned lambda); NOT predictive coverage")
    if effects_near_zero:
        note += (". WARNING: the learned covariate effects are ~0, so stability is "
                 "undefined — the model likely needs more fit iterations before its "
                 "covariate effects mean anything")

    return {
        "sign_agreement": float(sa.mean()) if sa.size else float("nan"),
        "magnitude_correlation": float(mc.mean()) if mc.size else float("nan"),
        "effects_near_zero": bool(effects_near_zero),
        "max_effect_magnitude": scale,
        "n_pairs": int(n_pairs),
        "n_features": int(n_feat),
        "n_topics": n_topics,
        "topics_compared": {
            "mean": float(tc.mean()), "min": int(tc.min()), "max": int(tc.max()),
        },
        "partial_alignment": bool(tc.min() < n_topics),
        "per_feature": per_feature,
        "caveat": caveat,
        "note": note,
    }


# ---------------------------------------------------------------------------
# Supervised out-of-fold path (PR2)
# ---------------------------------------------------------------------------


def _fit_predict_fold(factory, seed_fold, train_docs, test_docs, y, train_idx,
                      test_idx, cov, vocab, fixed_corpus, Corpus, fit_kwargs,
                      predict_kwargs, preprocessing):
    """One supervised fold: fit on the pruned train corpus, predict on the pruned
    test corpus, return (destination_indices, yhat, std, y_test, per_fold_record).

    Predicting on the TRANSFORMED test corpus (not raw docs) is the correctness
    crux: SupervisedLDA.predict returns a fabricated (0.0, sigma2) for an empty/
    all-OOV doc, so those docs must be dropped BEFORE predict, never scored. The
    survivor map (Corpus.transform kept_indices) gives the scatter destination."""
    model = factory(seed_fold)
    name = type(model).__name__

    if vocab == "per_fold":
        train_corpus = Corpus.from_documents(train_docs, **preprocessing)
    else:
        train_corpus = fixed_corpus.transform(train_docs)
    train_keep = np.asarray(train_corpus.kept_indices, dtype=np.int64)
    y_train = y[train_idx[train_keep]]

    canon = _canonical_family_cov(cov, name)
    fit_cov = {k: v[train_idx][train_keep] for k, v in canon.items()}

    if not hasattr(model, "predict"):
        raise ValueError(
            f"{name} has no predict(); the supervised out-of-fold path needs a "
            "response model such as SupervisedLDA. For unsupervised held-out "
            "evaluation, omit y=."
        )
    model.fit(train_corpus, y_train, **fit_cov, **fit_kwargs)

    # Drop zero-in-vocab test docs via the shared vocabulary, then predict only on
    # survivors so no fabricated 0.0 prediction can enter the OOF vector.
    test_corpus = train_corpus.transform(test_docs)
    test_keep = np.asarray(test_corpus.kept_indices, dtype=np.int64)
    destination = test_idx[test_keep]
    out = model.predict(test_corpus, return_std=True, **predict_kwargs)
    yhat, std = out
    yhat = np.asarray(yhat, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    if not (len(yhat) == len(std) == len(destination)):
        raise ValueError(
            f"supervised fold length mismatch: {len(yhat)} predictions for "
            f"{len(destination)} surviving test docs"
        )
    y_test = y[destination]
    rec = {
        "model_class": name,
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "n_oof": int(destination.size),
        "yhat_std": float(np.std(yhat)) if yhat.size else float("nan"),
    }
    if destination.size >= 1:
        # RMSE/MAE are valid even for a constant response; only R2 is undefined when
        # the response has no variance to explain (ss_tot == 0).
        rec["rmse"] = float(np.sqrt(np.mean((y_test - yhat) ** 2)))
        rec["mae"] = float(np.mean(np.abs(y_test - yhat)))
        ss_res = float(np.sum((y_test - yhat) ** 2))
        ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
        rec["r2"] = float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot
    else:
        rec["rmse"] = rec["mae"] = rec["r2"] = float("nan")
    return destination, yhat, std, rec


def _supervised_cv(factory, doc_lists, y, cov, fold_obj, vocab, fixed_corpus, Corpus,
                   fit_kwargs, predict_kwargs, preprocessing, coverage_levels, n_jobs,
                   manifest):
    import warnings

    n_docs = len(doc_lists)
    oof_pred = np.full(n_docs, np.nan)
    oof_std = np.full(n_docs, np.nan)

    jobs = []
    for (train_idx, test_idx), seed_fold in zip(fold_obj.splits, fold_obj.fold_seeds):
        train_docs = [doc_lists[i] for i in train_idx]
        test_docs = [doc_lists[i] for i in test_idx]
        jobs.append((seed_fold, train_docs, test_docs, train_idx, test_idx))

    def run(job):
        seed_fold, train_docs, test_docs, train_idx, test_idx = job
        return _fit_predict_fold(
            factory, seed_fold, train_docs, test_docs, y, train_idx, test_idx, cov,
            vocab, fixed_corpus, Corpus, fit_kwargs, predict_kwargs, preprocessing,
        )

    # Folds are independent. A thread pool parallelizes them (the Rust fit releases
    # the GIL); results scatter by destination index, so completion order can't change
    # the OOF vector — determinism holds. A process pool is not used: it can't pickle
    # the documented lambda factory.
    if n_jobs and n_jobs != 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            results = list(ex.map(run, jobs))
    else:
        results = [run(job) for job in jobs]

    per_fold = []
    for f, (destination, yhat, std, rec) in enumerate(results):
        oof_pred[destination] = yhat
        oof_std[destination] = std
        per_fold.append({"fold": f, "seed_fold": int(fold_obj.fold_seeds[f]), **rec})

    scored_mask = ~np.isnan(oof_pred)
    aggregate, calibration_table = _supervised_aggregate(
        y, oof_pred, oof_std, scored_mask, per_fold, coverage_levels
    )

    dropped = int(fold_obj.oof_mask.sum() - scored_mask.sum())
    if dropped > 0:
        warnings.warn(
            f"{dropped} tested documents were dropped from out-of-fold scoring "
            "(empty after per-fold vocabulary pruning); they are NaN in "
            "oof_predictions and excluded from every metric.",
            stacklevel=3,
        )

    mani = None
    if manifest:
        # Probe the factory (unfitted) for the full constructor settings, not just the
        # class — so the manifest can actually reconstruct the model.
        model_spec = None
        try:
            probe = factory(int(fold_obj.fold_seeds[0]))
            model_spec = {
                "class": type(probe).__name__,
                "settings": getattr(probe, "settings", None),
            }
        except Exception:
            if per_fold:
                model_spec = {"class": per_fold[0].get("model_class"), "settings": None}
        from statistics import NormalDist

        _nd = NormalDist()
        mani = _record_cv_manifest(
            fold_obj, doc_lists, cov, fit_kwargs, preprocessing, vocab, False,
            model_spec, None, None, supervised={
                "response_hash": _hash_array(y),
                "response_shape": list(np.shape(y)),
                "response_dtype": str(np.asarray(y).dtype),
                "covariate_hashes": {k: _hash_array(v) for k, v in cov.items()},
                "coverage_levels": list(coverage_levels),
                "coverage_z": {str(lvl): _nd.inv_cdf((1.0 + lvl) / 2.0)
                               for lvl in coverage_levels},
                "predict_kwargs": {k: repr(v) for k, v in predict_kwargs.items()},
                "calibration_rule": "up-to-10 quantile bins, deduped edges; "
                                    "OLS intercept/slope when >=10 scored rows",
                "n_jobs": n_jobs,
                "backend": "thread" if (n_jobs and n_jobs != 1) else "serial",
                "scored_mask": scored_mask.tolist(),
            },
        )

    return CrossValResult(
        per_fold=per_fold,
        aggregate=aggregate,
        folds=fold_obj,
        manifest=mani,
        vocab=vocab,
        kind="supervised",
        oof_predictions=oof_pred,
        oof_std=oof_std,
        scored_mask=scored_mask,
        calibration_table=calibration_table,
    )


def _supervised_aggregate(y, oof_pred, oof_std, scored_mask, per_fold, coverage_levels):
    """Pooled + macro regression metrics, calibration table, and interval coverage,
    all over the single `scored_mask` population."""
    yt = y[scored_mask]
    yp = oof_pred[scored_mask]
    ys = oof_std[scored_mask]
    agg = {}
    if yt.size:
        agg["rmse_pooled"] = float(np.sqrt(np.mean((yt - yp) ** 2)))
        agg["mae_pooled"] = float(np.mean(np.abs(yt - yp)))
        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
        ss_res = float(np.sum((yt - yp) ** 2))
        agg["r2_pooled"] = float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot

    # Macro over per-fold values (ddof=1), skipping NaN (constant-y or tiny folds).
    for metric in ("rmse", "mae", "r2"):
        vals = np.array([r[metric] for r in per_fold
                         if isinstance(r.get(metric), float) and np.isfinite(r[metric])])
        if vals.size:
            agg[f"{metric}_macro"] = {
                "mean": float(vals.mean()),
                # ddof=1 sample std; undefined (NaN) for a single valid fold — NOT 0.0,
                # which would read as "zero variance".
                "std": float(vals.std(ddof=1)) if vals.size > 1 else float("nan"),
                "n_valid_folds": int(vals.size),
            }

    # Interval coverage from the predictive std (includes sigma^2). Two-sided
    # Gaussian z for a central level lvl: z = Phi^-1((1 + lvl) / 2).
    from statistics import NormalDist

    _nd = NormalDist()
    for lvl in coverage_levels:
        z = _nd.inv_cdf((1.0 + lvl) / 2.0)
        if yt.size:
            covered = np.abs(yt - yp) <= z * ys
            agg[f"coverage_{int(round(lvl * 100))}"] = float(np.mean(covered))

    # Calibration. The OLS intercept/slope need >= min_calib points and varying
    # predictions to be identifiable; the reliability TABLE is independent of that
    # threshold and is built whenever predictions vary over >= 2 scored rows.
    min_calib = 10
    calibration_table = None
    if yt.size >= 2 and np.var(yp) > 0:
        calibration_table = _calibration_table(yp, yt)
    if yt.size >= min_calib and np.var(yp) > 0:
        b, a = np.polyfit(yp, yt, 1)
        agg["calibration_intercept"] = float(a)
        agg["calibration_slope"] = float(b)
    else:
        # Undefined: too few scored rows or a constant predictor (no identifiable slope).
        agg["calibration_intercept"] = float("nan")
        agg["calibration_slope"] = float("nan")
    return agg, calibration_table


def _calibration_table(yp, yt):
    """Up-to-10 quantile bins of y_hat with deduplicated edges; columns bin, n,
    mean_pred, mean_obs."""
    try:
        import pandas as pd
    except Exception:
        return None
    edges = np.unique(np.quantile(yp, np.linspace(0, 1, 11)))
    if edges.size < 2:
        return None
    bin_idx = np.clip(np.digitize(yp, edges[1:-1]), 0, edges.size - 2)
    rows = []
    for b in range(edges.size - 1):
        m = bin_idx == b
        if not m.any():
            continue
        rows.append({
            "bin": b,
            "n": int(m.sum()),
            "mean_pred": float(yp[m].mean()),
            "mean_obs": float(yt[m].mean()),
        })
    return pd.DataFrame(rows)


@dataclass
class CVManifest:
    """A reproducibility record for one :func:`cross_validate` run (Gate A-A13).

    Records fold assignments, every derived seed, splitter parameters, and the
    callback provenance so a rerun reproduces the same folds and fits. Distinct from
    :class:`~topica.manifest.AnalysisManifest`, which records a *single* fit; a CV
    run has K fits, so it gets its own record.
    """

    cv: dict

    def to_dict(self) -> dict:
        return {"schema": "topica.crossval", "schema_version": 1, "cv": self.cv}

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)

    def save(self, path: str) -> None:
        from pathlib import Path

        Path(path).write_text(self.to_json(), encoding="utf-8")


def _record_cv_manifest(
    fold_obj, doc_lists, cov, fit_kwargs, preprocessing, vocab, callback,
    model_spec, coherence_type, topn, supervised=None,
):
    """Record fold assignments + seeds + splitter params + fit/metric config so a
    rerun reproduces the folds and fits (Gate A-A13)."""
    import hashlib

    # Hash the actual document CONTENT (order-sensitive), not just lengths, so a
    # reordering or a different same-length corpus produces a different identity.
    hasher = hashlib.sha256()
    for d in doc_lists:
        hasher.update(("\x1f".join(d) + "\x1e").encode("utf-8"))
    content_hash = hasher.hexdigest()[:16]

    try:
        from . import __version__ as _ver
    except Exception:
        _ver = None

    cv_section = {
        "n_docs": fold_obj.n_docs,
        "doc_content_hash": content_hash,
        "topica_version": _ver,
        "strategy": fold_obj.strategy,
        "window": fold_obj.window,
        "vocab": vocab,
        "preprocessing": {k: repr(v) for k, v in preprocessing.items()},
        "seed": fold_obj.seed,
        "fold_seeds": list(fold_obj.fold_seeds),
        "splits": [
            {"train": t.tolist(), "test": e.tolist()} for t, e in fold_obj.splits
        ],
        "oof_mask": fold_obj.oof_mask.tolist(),
        "groups_hash": fold_obj.groups_hash,
        "times_hash": fold_obj.times_hash,
        "covariate_kwargs": sorted(cov.keys()),
        "fit_kwargs": {k: repr(v) for k, v in fit_kwargs.items()},
        "model": model_spec,
        "metric_params": {"coherence_type": coherence_type, "topn": topn,
                          "completion_split": "even_odd_deterministic"},
        # Replayable only when there is no opaque user callback AND the model's full
        # constructor settings were captured (so the factory can be reconstructed).
        # A factory whose settings we could not read is recorded not-replayable.
        "replayable": (not callback) and bool(
            model_spec and model_spec.get("settings") is not None
        ),
        "callback": "fit_fn/score_fn" if callback else None,
    }
    if supervised is not None:
        cv_section["path"] = "supervised"
        cv_section["supervised"] = supervised
    else:
        cv_section["path"] = "topic"
    return CVManifest(cv=cv_section)
