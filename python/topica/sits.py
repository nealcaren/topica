"""SITS-specific reporting helpers (Rossiter 2022 agenda-setting tables).

``topica.SITS`` reports one chain. A paper needs several chains pooled, both
per-speaker measures side by side, and the facts that shape them (how many of a
speaker's turns open a conversation, how many are too short to be sampled). This
module builds that table from one or more fitted models.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["speaker_table"]


def _as_list(models):
    if isinstance(models, (list, tuple)):
        return list(models)
    return [models]


def speaker_table(models, level: float = 0.9):
    """Per-speaker agenda-setting table pooled over one or more SITS chains.

    Parameters
    ----------
    models : SITS or sequence of SITS
        Fitted chains on the same data and settings (typically different seeds).
        They must share the speaker labels, turn counts and ``gamma``.
    level : float, default 0.9
        Coverage of the equal-tailed intervals, taken over the stored posterior
        draws of every chain pooled.

    Returns
    -------
    pandas.DataFrame, one row per speaker (in ``models[0].speakers`` order):

    - ``turns``, ``openers`` (turns that open a conversation, always shifts),
      ``short_turns`` (turns too short to be sampled, never shifts), ``eligible``;
    - ``shift_propensity`` with ``shift_propensity_lo`` / ``_hi``: Rossiter's
      readSits score over all of the speaker's turns;
    - ``eligible_shift_rate`` with ``_lo`` / ``_hi``: the share of the speaker's
      sampled turns that shift (openers and short turns left out);
    - ``rhat_propensity`` and ``rhat_eligible``: split R-hat of each measure across
      the chains (NaN with one chain). Values above about 1.01 mean the chains
      disagree and need more sweeps.

    Pooled intervals reflect both within-chain and between-chain variation, unlike
    ``SITS.shift_propensity_interval``, which covers one chain.
    """
    import pandas as pd

    from .mcmc import rhat

    ms = _as_list(models)
    if not ms:
        raise ValueError("speaker_table needs at least one fitted SITS model")
    if not (0.0 < level < 1.0):
        raise ValueError("level must be in (0, 1)")
    first = ms[0]
    speakers = list(first.speakers)
    counts = np.asarray(first.speaker_turn_counts)
    for m in ms[1:]:
        if list(m.speakers) != speakers or not np.array_equal(
                np.asarray(m.speaker_turn_counts), counts):
            raise ValueError("all models must be fitted on the same turns and speakers")
        if m.settings["gamma"] != first.settings["gamma"]:
            raise ValueError("all models must share gamma")

    conv = np.asarray(first.conversation_index)
    opener = np.r_[True, conv[1:] != conv[:-1]]
    eligible = np.asarray(first.eligible)
    spk = np.asarray(first.speaker_index)
    nspk = len(speakers)
    openers = np.bincount(spk[opener], minlength=nspk)
    short = np.bincount(spk[~opener & ~eligible], minlength=nspk)
    elig = np.asarray(first.speaker_eligible_counts)

    q = [(1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0]
    prop = [np.asarray(m.shift_propensity_draws) for m in ms]
    rate = [np.asarray(m.eligible_shift_rate_draws) for m in ms]
    prop_all = np.vstack(prop)
    rate_all = np.vstack(rate)

    def _rhat(draws, j):
        if len(draws) < 2:
            return np.nan
        cols = [d[:, j] for d in draws]
        if any(not np.all(np.isfinite(c)) for c in cols):
            return np.nan
        try:
            return float(rhat(cols))
        except ValueError:
            return np.nan

    with np.errstate(all="ignore"):
        rate_lo, rate_hi = (np.nanquantile(rate_all, qq, axis=0) if rate_all.size else
                            np.full(len(speakers), np.nan) for qq in q)
    rows = {
        "speaker": speakers,
        "turns": counts,
        "openers": openers,
        "short_turns": short,
        "eligible": elig,
        "shift_propensity": np.mean([np.asarray(m.shift_propensity) for m in ms], axis=0),
        "shift_propensity_lo": np.quantile(prop_all, q[0], axis=0),
        "shift_propensity_hi": np.quantile(prop_all, q[1], axis=0),
        "eligible_shift_rate": np.mean([np.asarray(m.eligible_shift_rate) for m in ms], axis=0),
        "eligible_shift_rate_lo": rate_lo,
        "eligible_shift_rate_hi": rate_hi,
        "rhat_propensity": [_rhat(prop, j) for j in range(len(speakers))],
        "rhat_eligible": [_rhat(rate, j) for j in range(len(speakers))],
    }
    df = pd.DataFrame(rows)
    df.attrs["chains"] = len(ms)
    df.attrs["level"] = level
    df.attrs["short_turn_share"] = float(first.short_turn_share)
    df.attrs["min_shift_tokens"] = first.settings["min_shift_tokens"]
    return df
