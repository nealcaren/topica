"""SITS-specific reporting helpers (Rossiter 2022 agenda-setting tables).

``topica.SITS`` reports one chain. A paper needs several chains pooled, both
per-speaker measures side by side, and the facts that shape them (how many of a
speaker's turns open a conversation, how many are too short to be sampled). This
module builds that table from one or more fitted models.
"""

import warnings as _warnings

import numpy as _np

__all__ = ["speaker_table"]

# Thresholds for speaker_table's warnings.
_RHAT_WARN = 1.01
_FORCED_WARN = 0.5
_RANK_WARN = 0.8


def _as_list(models):
    if isinstance(models, (list, tuple)):
        return list(models)
    return [models]


def _check_poolable(ms):
    """Refuse chains that do not share data and settings (the seed may differ)."""
    first = ms[0]
    # Compare the warm-up actually run, not the constructor value (None and an
    # explicit value that resolves to the same count are the same chain).
    ref = {k: v for k, v in first.settings.items() if k not in ("seed", "warmup")}
    arrays = ("speaker_index", "conversation_index", "eligible")
    for i, m in enumerate(ms[1:], start=1):
        if list(m.speakers) != list(first.speakers):
            raise ValueError(
                f"chain {i} has different speakers from chain 0; speaker_table pools "
                "chains fitted on the same turns only")
        for name in arrays:
            if not _np.array_equal(_np.asarray(getattr(m, name)),
                                   _np.asarray(getattr(first, name))):
                raise ValueError(
                    f"chain {i} differs from chain 0 in {name}; speaker_table pools "
                    "chains fitted on the same turns, speakers and conversations only")
        if m.data_fingerprint != first.data_fingerprint:
            raise ValueError(
                f"chain {i} was fitted on different turns from chain 0 (data_fingerprint "
                f"{m.data_fingerprint} vs {first.data_fingerprint}); speaker_table pools "
                "chains fitted on the same data only")
        other = {k: v for k, v in m.settings.items() if k not in ("seed", "warmup")}
        diff = sorted(k for k in set(ref) | set(other) if ref.get(k) != other.get(k))
        if diff:
            detail = ", ".join(f"{k}: {ref.get(k)!r} vs {other.get(k)!r}" for k in diff)
            raise ValueError(
                f"chain {i} was fitted with different settings from chain 0 ({detail}); "
                "pool only chains that differ in their seed")
        run = ("iters", "burn_in", "warmup_used")
        for name in run:
            if getattr(m, name) != getattr(first, name):
                raise ValueError(
                    f"chain {i} ran with {name}={getattr(m, name)}, chain 0 with "
                    f"{getattr(first, name)}; pool chains of the same length")
        if len(m.shift_propensity_draws) != len(first.shift_propensity_draws):
            raise ValueError(
                f"chain {i} stored {len(m.shift_propensity_draws)} draws, chain 0 "
                f"{len(first.shift_propensity_draws)}; pass the same sample_interval to "
                "every chain so each contributes equally to the pooled intervals")
    seeds = [m.settings["seed"] for m in ms]
    dup = sorted({s for s in seeds if seeds.count(s) > 1})
    if dup:
        _warnings.warn(
            f"speaker_table: several chains share seed {dup}; identical seeds give "
            "identical chains, so they add no information and make R-hat look better "
            "than it is. Fit each chain with its own seed.",
            stacklevel=3)


def _avg_ranks(x):
    """Ranks with ties given their average rank (as scipy.stats.rankdata)."""
    order = _np.argsort(x, kind="mergesort")
    ranks = _np.empty(len(x))
    ranks[order] = _np.arange(len(x), dtype=float)
    _, inv, counts = _np.unique(x, return_inverse=True, return_counts=True)
    sums = _np.bincount(inv, weights=ranks)
    return (sums / counts)[inv]


def _spearman(a, b):
    ok = _np.isfinite(a) & _np.isfinite(b)
    if ok.sum() < 3:
        return _np.nan
    ra, rb = _avg_ranks(a[ok]), _avg_ranks(b[ok])
    if ra.std() == 0 or rb.std() == 0:
        return _np.nan  # a constant measure has no ranking
    return float(_np.corrcoef(ra, rb)[0, 1])


def speaker_table(models, level: float = 0.9):
    """Per-speaker agenda-setting table pooled over one or more SITS chains.

    Parameters
    ----------
    models : SITS or sequence of SITS
        Fitted chains that differ only in their seed. They must share the turns
        (``data_fingerprint``), speakers, conversation layout, eligibility, run
        length (``iters``, ``burn_in``, number of stored draws), and every other
        setting (K, the priors, ``min_shift_tokens``, ``compat``, ``init``,
        ``warmup``); a mismatch in any of these raises.
    level : float, default 0.9
        Coverage of the equal-tailed intervals, taken over the stored posterior
        draws of every chain pooled.

    Returns
    -------
    pandas.DataFrame
        One row per speaker, in ``models[0].speakers`` order. ``turns``,
        ``openers`` (turns that open a conversation, always shifts),
        ``short_turns`` (turns too short to be sampled, never shifts) and
        ``eligible`` count the speaker's turns. ``forced_share`` is
        ``(openers + short_turns) / turns``, the share of the speaker's turns
        whose shift value is fixed rather than inferred.
        ``eligible_shift_rate`` with ``eligible_shift_rate_lo`` /
        ``eligible_shift_rate_hi`` is the share of the speaker's sampled turns
        that shift (a topica addition). ``shift_propensity`` with
        ``shift_propensity_lo`` / ``shift_propensity_hi`` is Rossiter's readSits
        score over all of the speaker's turns. ``rhat`` is the split R-hat across
        chains (NaN with one chain). It is the same for both measures, since each
        is an affine function of the speaker's number of sampled shifts.

    Pooled intervals reflect both within-chain and between-chain variation, unlike
    ``SITS.shift_propensity_interval``, which covers one chain.

    Warns when any ``rhat`` exceeds 1.01 (the chains disagree; on real
    conversations they usually do, and longer runs do not fix it, so pool more
    chains rather than rely on one), when a
    speaker's ``forced_share`` exceeds 0.5 (their ``shift_propensity`` mostly
    reflects how often they open conversations or speak briefly), and when the two
    measures rank speakers differently (Spearman correlation below 0.8).
    """
    import pandas as pd

    from .mcmc import rhat

    ms = _as_list(models)
    if not ms:
        raise ValueError("speaker_table needs at least one fitted SITS model")
    if not (0.0 < level < 1.0):
        raise ValueError("level must be in (0, 1)")
    _check_poolable(ms)
    first = ms[0]
    speakers = list(first.speakers)
    counts = _np.asarray(first.speaker_turn_counts)

    conv = _np.asarray(first.conversation_index)
    opener = _np.r_[True, conv[1:] != conv[:-1]]
    eligible = _np.asarray(first.eligible)
    spk = _np.asarray(first.speaker_index)
    nspk = len(speakers)
    openers = _np.bincount(spk[opener], minlength=nspk)
    short = _np.bincount(spk[~opener & ~eligible], minlength=nspk)
    elig = _np.asarray(first.speaker_eligible_counts)
    forced = (openers + short) / _np.maximum(counts, 1)

    q = [(1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0]
    prop = [_np.asarray(m.shift_propensity_draws) for m in ms]
    prop_all = _np.vstack(prop)
    rate_all = _np.vstack([_np.asarray(m.eligible_shift_rate_draws) for m in ms])

    def _rhat(j):
        if len(prop) < 2:
            return _np.nan
        cols = [d[:, j] for d in prop]
        if any(not _np.all(_np.isfinite(c)) for c in cols):
            return _np.nan
        try:
            return float(rhat(cols))
        except ValueError:
            return _np.nan

    with _np.errstate(all="ignore"), _warnings.catch_warnings():
        _warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
        rate_lo, rate_hi = (_np.nanquantile(rate_all, qq, axis=0) if rate_all.size else
                            _np.full(nspk, _np.nan) for qq in q)
    rate = _np.mean([_np.asarray(m.eligible_shift_rate) for m in ms], axis=0)
    propensity = _np.mean([_np.asarray(m.shift_propensity) for m in ms], axis=0)
    rhats = _np.array([_rhat(j) for j in range(nspk)])
    df = pd.DataFrame({
        "speaker": speakers,
        "turns": counts,
        "openers": openers,
        "short_turns": short,
        "eligible": elig,
        "forced_share": forced,
        "eligible_shift_rate": rate,
        "eligible_shift_rate_lo": rate_lo,
        "eligible_shift_rate_hi": rate_hi,
        "shift_propensity": propensity,
        "shift_propensity_lo": _np.quantile(prop_all, q[0], axis=0),
        "shift_propensity_hi": _np.quantile(prop_all, q[1], axis=0),
        "rhat": rhats,
    })
    rank_corr = _spearman(rate, propensity)
    df.attrs["chains"] = len(ms)
    df.attrs["level"] = level
    df.attrs["short_turn_share"] = float(first.short_turn_share)
    df.attrs["min_shift_tokens"] = first.settings["min_shift_tokens"]
    df.attrs["rank_correlation"] = rank_corr

    bad = [str(s) for s, r in zip(speakers, rhats) if not _np.isnan(r) and r > _RHAT_WARN]
    if bad:
        _warnings.warn(
            f"speaker_table: R-hat above {_RHAT_WARN} for {len(bad)} of {nspk} speakers "
            f"(max {_np.nanmax(rhats):.2f}). SITS chains on real conversations settle at "
            "somewhat different levels, and longer runs do not remove this. The pooled "
            "intervals include that variation, but pooling summarizes the disagreement "
            "rather than resolving it: check that the chains have at least 200,000 "
            "sweeps each, add chains until the pooled means stop moving, and report "
            "the spread between chains.",
            stacklevel=2)
    heavy = [str(s) for s, f in zip(speakers, forced) if f > _FORCED_WARN]
    if heavy:
        _warnings.warn(
            f"speaker_table: more than half of the turns of {', '.join(heavy)} are "
            "openers or too short to be sampled, so their shift_propensity mostly "
            "reflects how often they open conversations or speak briefly. Compare "
            "eligible_shift_rate, and report forced_share.", stacklevel=2)
    if _np.isfinite(rank_corr) and rank_corr < _RANK_WARN:
        _warnings.warn(
            f"speaker_table: eligible_shift_rate and shift_propensity rank the speakers "
            f"differently (Spearman {rank_corr:.2f}). The difference comes from the "
            "forced turns (see forced_share); say which measure your claim rests on.",
            stacklevel=2)
    return df
