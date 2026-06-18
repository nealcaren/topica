//! Shared scaffolding for stochastic variational inference (SVI / minibatch EM)
//! across the logistic-normal family (CTM/STM, ECTM, STS). The per-model fits own
//! their sufficient statistics and (non-conjugate) topic-word M-steps; this module
//! owns only the two pieces that are identical everywhere: the Robbins-Monro
//! learning-rate schedule and the deterministic per-epoch document shuffle. Both
//! draw from the model's own `Rng`, so an SVI fit is seed-reproducible.

use rand::Rng;

/// Robbins-Monro step size at global step `t_step` (1-based): `(tau + t)^(-kappa)`.
/// `tau >= 0` down-weights early, noisy minibatches; `kappa in (0.5, 1]` controls
/// the forgetting rate (the usual SVI requirement for convergence).
#[inline]
pub fn rho(tau: f64, kappa: f64, t_step: usize) -> f64 {
    (tau + t_step as f64).powf(-kappa)
}

/// In-place Fisher-Yates shuffle of `order` using `rng`. Identical draw sequence
/// to the hand-rolled loop the CTM SVI path used, so behaviour is unchanged when
/// callers adopt it: each `rng.gen::<f64>()` maps to one swap in descending index
/// order, keeping the fit deterministic for a fixed seed and thread count.
pub fn shuffle_in_place<R: Rng>(order: &mut [usize], rng: &mut R) {
    let n = order.len();
    for i in (1..n).rev() {
        let j = ((rng.gen::<f64>() * (i as f64 + 1.0)) as usize).min(i);
        order.swap(i, j);
    }
}

/// A fresh shuffled document order `0..d`.
pub fn shuffled_order<R: Rng>(d: usize, rng: &mut R) -> Vec<usize> {
    let mut order: Vec<usize> = (0..d).collect();
    shuffle_in_place(&mut order, rng);
    order
}
