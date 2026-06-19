//! Generic L-BFGS minimizer (relocated from dmr.rs; used by the logistic-normal variational fits and others).

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Minimize `f` (value + gradient) with limited-memory BFGS and a backtracking
/// Armijo line search. Compact by design: DMR re-optimizes frequently between
/// sampling sweeps, so a short history and iteration budget suffice.
pub fn lbfgs_minimize<F>(
    x0: Vec<f64>,
    mut f: F,
    max_iter: usize,
    history: usize,
    tol: f64,
) -> Vec<f64>
where
    F: FnMut(&[f64]) -> (f64, Vec<f64>),
{
    let n = x0.len();
    let mut x = x0;
    let (mut fx, mut g) = f(&x);

    let mut s_list: Vec<Vec<f64>> = Vec::new();
    let mut y_list: Vec<Vec<f64>> = Vec::new();
    let mut rho_list: Vec<f64> = Vec::new();

    for _ in 0..max_iter {
        if g.iter().map(|v| v * v).sum::<f64>().sqrt() < tol {
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
        loop {
            for j in 0..n {
                x_new[j] = x[j] + step * d[j];
            }
            let r = f(&x_new);
            fx_new = r.0;
            g_new = r.1;
            if fx_new <= fx + 1e-4 * step * dg || step < 1e-12 {
                break;
            }
            step *= 0.5;
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

        let converged = (fx - fx_new).abs() < tol * (1.0 + fx.abs());
        x = x_new;
        fx = fx_new;
        g = g_new;
        if converged {
            break;
        }
    }
    x
}
