// Determinism probe for issue #378: is matrixmultiply's threaded GEMM
// bit-for-bit identical regardless of thread count? Prints a bitwise checksum of
// the three decoder GEMM outputs. Run under different MATMUL_NUM_THREADS and
// compare the checksums — equal checksums => bit-identical output.

use matrixmultiply::dgemm;

struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

// Order-independent bitwise checksum: XOR-fold the raw f64 bits. Any single flipped
// mantissa bit anywhere changes it.
fn checksum(a: &[f64]) -> u64 {
    a.iter().fold(0u64, |h, &x| h ^ x.to_bits().rotate_left(17)).wrapping_mul(0x9E3779B97F4A7C15)
}

fn main() {
    let (n, v, k) = (2000usize, 2000usize, 50usize);
    let mut rng = Lcg(0xDEADBEEF);
    let mut theta_do = vec![0.0; n * k];
    for x in theta_do.iter_mut() {
        *x = rng.next_f64();
    }
    let mut beta = vec![0.0; k * v];
    for x in beta.iter_mut() {
        *x = (rng.next_f64() - 0.5) * 2.0;
    }
    let mut dlogit = vec![0.0; n * v];
    for x in dlogit.iter_mut() {
        *x = (rng.next_f64() - 0.5) * 0.5;
    }

    let mut logit = vec![0.0; n * v];
    let mut dtheta = vec![0.0; n * k];
    let mut gbeta = vec![0.0; k * v];
    unsafe {
        dgemm(n, k, v, 1.0, theta_do.as_ptr(), k as isize, 1, beta.as_ptr(), v as isize, 1, 0.0, logit.as_mut_ptr(), v as isize, 1);
        dgemm(n, v, k, 1.0, dlogit.as_ptr(), v as isize, 1, beta.as_ptr(), 1, v as isize, 0.0, dtheta.as_mut_ptr(), k as isize, 1);
        dgemm(k, n, v, 1.0, theta_do.as_ptr(), 1, k as isize, dlogit.as_ptr(), v as isize, 1, 0.0, gbeta.as_mut_ptr(), v as isize, 1);
    }

    let threads = std::env::var("MATMUL_NUM_THREADS").unwrap_or_else(|_| "unset".into());
    println!(
        "MATMUL_NUM_THREADS={:>5} | checksum logit={:016x} dtheta={:016x} gbeta={:016x}",
        threads, checksum(&logit), checksum(&dtheta), checksum(&gbeta)
    );
}
