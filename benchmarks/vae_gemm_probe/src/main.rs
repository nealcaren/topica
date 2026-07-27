// Standalone microbenchmark for topica issue #378: does a pure-Rust GEMM
// (matrixmultiply) beat the scalar triple-loop decoder in prodlda.rs, single
// threaded, and what is the numeric delta vs the current scalar path?
//
// Replicates the three O(N*K*V) decoder terms verbatim from prodlda.rs:
//   forward : logit_raw[i][j]   = sum_t theta_do[i][t] * beta[t*V+j]
//   backward: dtheta_do[i][t]   = sum_j dlogit_raw[i][j] * beta[t*V+j]
//   backward: g_beta[t*V+j]    += sum_i theta_do[i][t] * dlogit_raw[i][j]
// beta is flat row-major K x V, theta_do is N x K, logit_raw/dlogit_raw are N x V.

use matrixmultiply::dgemm;
use std::time::Instant;

// Deterministic LCG so runs are reproducible without external crates.
struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn fill(v: &mut [f64], rng: &mut Lcg, scale: f64, center: f64) {
    for x in v.iter_mut() {
        *x = (rng.next_f64() - center) * scale;
    }
}

// ---- scalar (current prodlda.rs) ----

fn scalar_forward(theta_do: &[f64], beta: &[f64], logit: &mut [f64], n: usize, k: usize, v: usize) {
    for x in logit.iter_mut() {
        *x = 0.0;
    }
    for i in 0..n {
        let row = &mut logit[i * v..(i + 1) * v];
        for t in 0..k {
            let w_t = theta_do[i * k + t];
            if w_t != 0.0 {
                let base = t * v;
                for j in 0..v {
                    row[j] += w_t * beta[base + j];
                }
            }
        }
    }
}

fn scalar_backward(
    theta_do: &[f64],
    beta: &[f64],
    dlogit_raw: &[f64],
    dtheta_do: &mut [f64],
    g_beta: &mut [f64],
    n: usize,
    k: usize,
    v: usize,
) {
    for x in g_beta.iter_mut() {
        *x = 0.0;
    }
    for i in 0..n {
        for t in 0..k {
            let base = t * v;
            let mut acc = 0.0;
            for j in 0..v {
                let dl = dlogit_raw[i * v + j];
                acc += dl * beta[base + j];
                g_beta[base + j] += theta_do[i * k + t] * dl;
            }
            dtheta_do[i * k + t] = acc;
        }
    }
}

// ---- GEMM (matrixmultiply, single-threaded, deterministic) ----

fn gemm_forward(theta_do: &[f64], beta: &[f64], logit: &mut [f64], n: usize, k: usize, v: usize) {
    // logit (N x V) = theta_do (N x K) * beta (K x V)
    unsafe {
        dgemm(
            n, k, v, 1.0,
            theta_do.as_ptr(), k as isize, 1,
            beta.as_ptr(), v as isize, 1,
            0.0,
            logit.as_mut_ptr(), v as isize, 1,
        );
    }
}

fn gemm_backward(
    theta_do: &[f64],
    beta: &[f64],
    dlogit_raw: &[f64],
    dtheta_do: &mut [f64],
    g_beta: &mut [f64],
    n: usize,
    k: usize,
    v: usize,
) {
    // dtheta_do (N x K) = dlogit_raw (N x V) * beta^T (V x K)
    // beta is K x V row-major; view its transpose by swapping strides: rsb=1, csb=V.
    unsafe {
        dgemm(
            n, v, k, 1.0,
            dlogit_raw.as_ptr(), v as isize, 1,
            beta.as_ptr(), 1, v as isize,
            0.0,
            dtheta_do.as_mut_ptr(), k as isize, 1,
        );
    }
    // g_beta (K x V) = theta_do^T (K x N) * dlogit_raw (N x V)
    // theta_do is N x K row-major; transpose via strides: rsa=1, csa=K.
    unsafe {
        dgemm(
            k, n, v, 1.0,
            theta_do.as_ptr(), 1, k as isize,
            dlogit_raw.as_ptr(), v as isize, 1,
            0.0,
            g_beta.as_mut_ptr(), v as isize, 1,
        );
    }
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}
fn max_abs(a: &[f64]) -> f64 {
    a.iter().map(|x| x.abs()).fold(0.0, f64::max)
}

fn bench(n: usize, v: usize, k: usize, iters: usize) {
    let mut rng = Lcg(0x1234_5678_9abc_def0 ^ ((n * 131 + k * 17 + v) as u64));
    // theta_do: a softmax-like row (nonneg, ~sums to 1) with dropout zeros, like prodlda.
    let mut theta_do = vec![0.0; n * k];
    for i in 0..n {
        let mut s = 0.0;
        for t in 0..k {
            let val = rng.next_f64();
            theta_do[i * k + t] = val;
            s += val;
        }
        for t in 0..k {
            theta_do[i * k + t] /= s;
            if rng.next_f64() < 0.2 {
                theta_do[i * k + t] = 0.0; // dropout mask
            }
        }
    }
    let mut beta = vec![0.0; k * v];
    fill(&mut beta, &mut rng, 2.0, 0.5);
    let mut dlogit_raw = vec![0.0; n * v];
    fill(&mut dlogit_raw, &mut rng, 0.5, 0.5);

    let mut logit_s = vec![0.0; n * v];
    let mut logit_g = vec![0.0; n * v];
    let mut dtheta_s = vec![0.0; n * k];
    let mut dtheta_g = vec![0.0; n * k];
    let mut gbeta_s = vec![0.0; k * v];
    let mut gbeta_g = vec![0.0; k * v];

    // correctness / numeric-delta pass
    scalar_forward(&theta_do, &beta, &mut logit_s, n, k, v);
    gemm_forward(&theta_do, &beta, &mut logit_g, n, k, v);
    scalar_backward(&theta_do, &beta, &dlogit_raw, &mut dtheta_s, &mut gbeta_s, n, k, v);
    gemm_backward(&theta_do, &beta, &dlogit_raw, &mut dtheta_g, &mut gbeta_g, n, k, v);

    let d_logit = max_abs_diff(&logit_s, &logit_g) / max_abs(&logit_s).max(1e-300);
    let d_dtheta = max_abs_diff(&dtheta_s, &dtheta_g) / max_abs(&dtheta_s).max(1e-300);
    let d_gbeta = max_abs_diff(&gbeta_s, &gbeta_g) / max_abs(&gbeta_s).max(1e-300);

    // timing: forward + backward = one "decoder pass" per iter
    let t0 = Instant::now();
    for _ in 0..iters {
        scalar_forward(&theta_do, &beta, &mut logit_s, n, k, v);
        scalar_backward(&theta_do, &beta, &dlogit_raw, &mut dtheta_s, &mut gbeta_s, n, k, v);
        std::hint::black_box(&logit_s);
        std::hint::black_box(&gbeta_s);
    }
    let t_scalar = t0.elapsed().as_secs_f64() / iters as f64;

    let t1 = Instant::now();
    for _ in 0..iters {
        gemm_forward(&theta_do, &beta, &mut logit_g, n, k, v);
        gemm_backward(&theta_do, &beta, &dlogit_raw, &mut dtheta_g, &mut gbeta_g, n, k, v);
        std::hint::black_box(&logit_g);
        std::hint::black_box(&gbeta_g);
    }
    let t_gemm = t1.elapsed().as_secs_f64() / iters as f64;

    println!(
        "N={:5} V={:5} K={:3} | scalar {:8.2} ms | gemm {:8.2} ms | speedup {:5.2}x | rel-diff logit {:.1e} dtheta {:.1e} gbeta {:.1e}",
        n, v, k,
        t_scalar * 1e3, t_gemm * 1e3, t_scalar / t_gemm,
        d_logit, d_dtheta, d_gbeta
    );
}

fn main() {
    println!("matrixmultiply single-threaded decoder GEMM vs scalar triple-loop (issue #378)\n");
    // issue's benchmark sizes: bench(N, V, K, iters)
    bench(500, 500, 10, 40);
    bench(2000, 2000, 20, 12);
    bench(4000, 3000, 30, 6);
    bench(3000, 3000, 50, 6);
    println!("\n-- K sweep at N=2000 V=2000 (isolates the K-dependent decoder cost) --");
    for &k in &[10usize, 20, 30, 50, 80] {
        bench(2000, 2000, k, 10);
    }
}
