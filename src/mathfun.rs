//! Shared special functions.
//!
//! `log_gamma` was copy-pasted, byte-for-byte, into half a dozen model files
//! (`dmr`, `dtm`, `rtm`, `keyatm`, `pt`, `hlda`) — every copy carrying the same
//! latent bug (see below). This module is the single source of truth for that
//! general-purpose Stirling log Γ. (The MALLET-matched variant in
//! [`crate::output`] uses a different shift threshold to reproduce MALLET's
//! `Dirichlet.logGammaStirling` exactly and is intentionally kept separate.)

/// Stirling-series `ln Γ(z)` for `z > 0`, accurate to ~1e-10.
///
/// The argument is shifted up to `x >= 10` for the asymptotic series and the
/// recurrence `ln Γ(z) = ln Γ(z + 1) - ln z` is unwound to recover `ln Γ(z)`.
///
/// The historical per-model copies unwound that recurrence by decrementing the
/// *shifted* value back down (`while shift > 0 { z -= 1.0; result -= z.ln(); }`).
/// For a tiny argument the shift `z += 1.0` rounds to exactly `1.0` (any `z`
/// below ~1e-16), so the reverse pass stepped through `0.0` and evaluated
/// `ln(0.0) = -inf`, returning `+inf` — e.g. when a DMR prior `α = exp(λ·x)` is
/// small. Here we keep that exact algorithm for all normal-range arguments (so
/// results are bit-identical to the previous copies) and only lift a tiny `z`
/// into the safe range first, evaluating `ln z` on the real argument.
pub(crate) fn log_gamma(z: f64) -> f64 {
    // Lift arguments small enough that the shift-then-decrement below would lose
    // them (well above the ~1e-16 danger point). This branch is never taken for
    // ordinary hyperparameters, so the common path is unchanged.
    if z < 1e-10 {
        return log_gamma(z + 1.0) - z.ln();
    }
    const HALF_LOG_TWO_PI: f64 = 0.918_938_533_204_672_7;
    let mut z = z;
    let mut shift = 0i32;
    while z < 10.0 {
        z += 1.0;
        shift += 1;
    }
    let mut result = HALF_LOG_TWO_PI + (z - 0.5) * z.ln() - z + 1.0 / (12.0 * z)
        - 1.0 / (360.0 * z * z * z)
        + 1.0 / (1260.0 * z * z * z * z * z);
    while shift > 0 {
        shift -= 1;
        z -= 1.0;
        result -= z.ln();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::log_gamma;

    /// The exact per-model algorithm before deduplication, for bit-parity checks.
    fn legacy(mut z: f64) -> f64 {
        const HALF_LOG_TWO_PI: f64 = 0.918_938_533_204_672_7;
        let mut shift = 0i32;
        while z < 10.0 {
            z += 1.0;
            shift += 1;
        }
        let mut result = HALF_LOG_TWO_PI + (z - 0.5) * z.ln() - z + 1.0 / (12.0 * z)
            - 1.0 / (360.0 * z * z * z)
            + 1.0 / (1260.0 * z * z * z * z * z);
        while shift > 0 {
            shift -= 1;
            z -= 1.0;
            result -= z.ln();
        }
        result
    }

    #[test]
    fn matches_known_values() {
        let cases = [
            (1.0, 0.0),
            (2.0, 0.0),
            (0.5, std::f64::consts::PI.sqrt().ln()),
            (3.0, 2.0f64.ln()),
            (5.0, 24.0f64.ln()),
            (10.0, 362_880.0f64.ln()),
            (0.1, 9.513_507_698_668_732_f64.ln()),
        ];
        for (z, expected) in cases {
            assert!(
                (log_gamma(z) - expected).abs() < 1e-9,
                "log_gamma({z}) = {}, expected {expected}",
                log_gamma(z)
            );
        }
    }

    #[test]
    fn bit_identical_to_legacy_on_normal_args() {
        // The dedup must not change any model's results: for every normal-range
        // argument the shared helper must be bit-for-bit equal to the algorithm the
        // per-model copies used.
        let mut z = 1e-9f64;
        while z < 60.0 {
            assert_eq!(
                log_gamma(z).to_bits(),
                legacy(z).to_bits(),
                "log_gamma diverged from the legacy copy at z = {z}"
            );
            z += 0.0007;
        }
    }

    #[test]
    fn finite_for_tiny_argument() {
        // Regression: the legacy copies returned +inf here (ln(0) on the reverse
        // pass). Γ(z) ≈ 1/z for small z, so ln Γ(z) ≈ -ln z.
        for &e in &[-20.0f64, -40.0, -80.0, -300.0] {
            let z = e.exp();
            let got = log_gamma(z);
            assert!(got.is_finite(), "log_gamma({z}) not finite: {got}");
            assert!(
                (got - (-z.ln())).abs() < 1e-3,
                "log_gamma({z}) = {got}, expected ≈ {}",
                -z.ln()
            );
        }
    }
}
