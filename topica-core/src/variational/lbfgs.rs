//! Generic L-BFGS minimizer (relocated from dmr.rs; used by the logistic-normal variational fits and others).

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Minimize `f` (value + gradient) with limited-memory BFGS and a backtracking
/// Armijo line search. Compact by design: DMR re-optimizes frequently between
/// sampling sweeps, so a short history and iteration budget suffice.
pub fn lbfgs_minimize<F>(x0: Vec<f64>, f: F, max_iter: usize, history: usize, tol: f64) -> Vec<f64>
where
    F: FnMut(&[f64]) -> (f64, Vec<f64>),
{
    lbfgs_minimize_status(x0, f, max_iter, history, tol).0
}

/// As [`lbfgs_minimize`], but also reports whether the run reached a stationary
/// point on its own criterion (gradient below `tol`, or a relative function change
/// below `tol`) rather than exhausting `max_iter`. Callers that build asymptotic
/// standard errors from the result need this: the observed-information inverse is
/// only a valid covariance at an optimum, so a run that hit `max_iter` (or was
/// given `max_iter == 0`) must not be treated as converged.
pub fn lbfgs_minimize_status<F>(
    x0: Vec<f64>,
    mut f: F,
    max_iter: usize,
    history: usize,
    tol: f64,
) -> (Vec<f64>, bool)
where
    F: FnMut(&[f64]) -> (f64, Vec<f64>),
{
    let n = x0.len();
    let mut x = x0;
    let (mut fx, mut g) = f(&x);

    let mut s_list: Vec<Vec<f64>> = Vec::new();
    let mut y_list: Vec<Vec<f64>> = Vec::new();
    let mut rho_list: Vec<f64> = Vec::new();
    let mut converged = false;

    for _ in 0..max_iter {
        if g.iter().map(|v| v * v).sum::<f64>().sqrt() < tol {
            converged = true;
            break;
        }

        // Two-loop recursion for the search direction d = -H·g.
        let m = s_list.len();
        let mut q = g.clone();
        let mut alpha = vec![0.0f64; m];
        for i in (0..m).rev() {
            let a = rho_list[i] * dot(&s_list[i], &q);
            alpha[i] = a;
            for j in 0..n {
                q[j] -= a * y_list[i][j];
            }
        }
        let gamma = if m > 0 {
            let yy = dot(&y_list[m - 1], &y_list[m - 1]);
            if yy > 0.0 {
                dot(&s_list[m - 1], &y_list[m - 1]) / yy
            } else {
                1.0
            }
        } else {
            1.0
        };
        for v in q.iter_mut() {
            *v *= gamma;
        }
        for i in 0..m {
            let b = rho_list[i] * dot(&y_list[i], &q);
            for j in 0..n {
                q[j] += (alpha[i] - b) * s_list[i][j];
            }
        }
        let mut d: Vec<f64> = q.iter().map(|v| -v).collect();

        // Fall back to steepest descent if the direction isn't a descent one.
        if dot(&d, &g) >= 0.0 {
            d = g.iter().map(|v| -v).collect();
        }
        let dg = dot(&d, &g);

        // Backtracking Armijo line search.
        let mut step = 1.0;
        let mut x_new = x.clone();
        // Assigned on every loop iteration before they are read; the line search
        // always runs the body at least once, so no initial value is needed.
        let mut fx_new: f64;
        let mut g_new: Vec<f64>;
        let accepted: bool;
        loop {
            for j in 0..n {
                x_new[j] = x[j] + step * d[j];
            }
            let r = f(&x_new);
            fx_new = r.0;
            g_new = r.1;
            // A non-finite trial never satisfies Armijo (NaN comparisons are false),
            // so the shrink loop would otherwise fall through and lock it in.
            if fx_new.is_finite() && fx_new <= fx + 1e-4 * step * dg {
                accepted = true;
                break;
            }
            if step < 1e-12 {
                // Give up shrinking. Accept the exhausted step only if it is at least
                // finite (a negligible move that preserves the prior behaviour);
                // reject a non-finite trial outright (#419).
                accepted = fx_new.is_finite();
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            // No finite decrease found: stop rather than corrupting x/g with a
            // non-finite trial. `converged` stays false, so a caller that gates
            // standard errors on convergence correctly withholds them.
            break;
        }

        // Curvature update (skip if it would break positive-definiteness).
        let s: Vec<f64> = (0..n).map(|j| x_new[j] - x[j]).collect();
        let y: Vec<f64> = (0..n).map(|j| g_new[j] - g[j]).collect();
        let sy = dot(&s, &y);
        if sy > 1e-10 {
            if s_list.len() == history {
                s_list.remove(0);
                y_list.remove(0);
                rho_list.remove(0);
            }
            rho_list.push(1.0 / sy);
            s_list.push(s);
            y_list.push(y);
        }

        let fx_converged = (fx - fx_new).abs() < tol * (1.0 + fx.abs());
        x = x_new;
        fx = fx_new;
        g = g_new;
        if fx_converged {
            converged = true;
            break;
        }
    }
    (x, converged)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Regression: finite problems still converge exactly as before.
    #[test]
    fn minimizes_a_quadratic() {
        // f(x) = (x0-3)^2 + (x1+2)^2, min at (3, -2).
        let f = |x: &[f64]| -> (f64, Vec<f64>) {
            let (a, b) = (x[0] - 3.0, x[1] + 2.0);
            (a * a + b * b, vec![2.0 * a, 2.0 * b])
        };
        let (x, converged) = lbfgs_minimize_status(vec![0.0, 0.0], f, 100, 7, 1e-10);
        assert!(converged);
        assert!(
            (x[0] - 3.0).abs() < 1e-4 && (x[1] + 2.0).abs() < 1e-4,
            "{x:?}"
        );
    }

    // #419: a line search whose every trial is non-finite must not lock the
    // non-finite point into x. f(x0) = x0 for x0 >= 0 (the gradient pushes x0
    // negative), NaN for x0 < 0, so every trial step lands in the NaN region.
    #[test]
    fn rejects_a_nonfinite_trial_and_keeps_the_last_finite_point() {
        let f = |x: &[f64]| -> (f64, Vec<f64>) {
            if x[0] >= 0.0 {
                (x[0], vec![1.0])
            } else {
                (f64::NAN, vec![f64::NAN])
            }
        };
        let (x, converged) = lbfgs_minimize_status(vec![0.0], f, 50, 7, 1e-6);
        assert!(x.iter().all(|v| v.is_finite()), "x must stay finite: {x:?}");
        assert!(!converged, "a failed line search is not convergence");
        assert_eq!(x, vec![0.0], "must keep the last finite point");
    }
}
