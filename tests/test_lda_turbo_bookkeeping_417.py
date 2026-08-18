"""#417: in turbo mode (num_threads>1, turbo_merge_every>1) the LDA fit loop
advanced the iteration counter by the merge stride and tested exact divisibility
(iter.is_multiple_of(interval)), so any bookkeeping interval that did not divide
the stride was subsampled or skipped entirely. The fix fires each scheduled event
at the first reconciled window boundary at/after its scheduled iteration."""
import topica


def _fired_iters(merge_every, progress_interval, iters=200, threads=2):
    fired = []
    m = topica.LDA(num_topics=4, seed=0)
    docs = [[f"w{(i + d) % 12}" for i in range(8)] for d in range(60)]
    m.fit(
        docs,
        iters=iters,
        num_samples=1,
        sample_interval=10,
        num_threads=threads,
        turbo_merge_every=merge_every,
        progress=lambda it, total, info: fired.append(it),  # 3-arg contract (#785)
        progress_interval=progress_interval,
    )
    return fired


def test_turbo_does_not_skip_scheduled_progress_callbacks():
    # stride 8 does not divide interval 10. Window ends land on multiples of 8:
    # {8,16,24,...}. The OLD code fired only where that value was also a multiple
    # of 10 -> {40,80,120,160,200} (5 calls). The fix fires once per window that
    # crosses a multiple of 10, so e.g. iter=16 (window (8,16] crosses 10) fires.
    fired = _fired_iters(merge_every=8, progress_interval=10)
    # 16 is NOT a multiple of 10, so the old is_multiple_of(10) path could never
    # report it. Its presence is a fingerprint of the crossed-window fix.
    assert 16 in fired, f"scheduled callback at ~iter=10 was skipped; fired={fired}"
    # Overall the fix must fire far more often than the old lcm(8,10)=40 cadence.
    assert len(fired) >= 15, f"expected ~one call per 10 iters, got {len(fired)}: {fired}"


def test_exact_path_unchanged_when_stride_divides_interval():
    # merge_every=5 divides progress_interval=10: every reported iter is still a
    # multiple of 10, matching the exact per-sweep schedule (no coalescing/skew).
    fired = _fired_iters(merge_every=5, progress_interval=10)
    assert fired == [i for i in range(10, 201, 10)], fired
