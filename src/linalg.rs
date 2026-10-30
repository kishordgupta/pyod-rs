//! Dense linear algebra and special functions standing in for SciPy / sklearn helpers.

use crate::base::Matrix;
use crate::utils::{sq_euclidean, symmetric_eigen, Rng};

/// Eigenvalues of a symmetric matrix, descending.
pub fn symmetric_eigen_values(a: &[Vec<f64>]) -> Vec<f64> {
    symmetric_eigen(a).0
}

pub type Mat = Vec<Vec<f64>>;

pub fn zeros(r: usize, c: usize) -> Mat {
    vec![vec![0.0; c]; r]
}

pub fn identity(n: usize) -> Mat {
    (0..n)
        .map(|i| (0..n).map(|j| (i == j) as u8 as f64).collect())
        .collect()
}

pub fn transpose(a: &[Vec<f64>]) -> Mat {
    if a.is_empty() {
        return Vec::new();
    }
    (0..a[0].len())
        .map(|j| a.iter().map(|r| r[j]).collect())
        .collect()
}

pub fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Mat {
    let (n, k, m) = (a.len(), b.len(), b.first().map_or(0, Vec::len));
    let mut out = zeros(n, m);
    for i in 0..n {
        for p in 0..k {
            let aip = a[i][p];
            if aip == 0.0 {
                continue;
            }
            for j in 0..m {
                out[i][j] += aip * b[p][j];
            }
        }
    }
    out
}

pub fn matvec(a: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|r| r.iter().zip(v).map(|(x, y)| x * y).sum())
        .collect()
}

/// Column means.
pub fn col_mean(x: &Matrix) -> Vec<f64> {
    let n = x.len() as f64;
    let mut m = vec![0.0; x[0].len()];
    for r in x {
        for (mj, v) in m.iter_mut().zip(r) {
            *mj += v;
        }
    }
    m.iter_mut().for_each(|v| *v /= n);
    m
}

/// Covariance about `center`, divided by `n - ddof` (sklearn's `empirical_covariance` is ddof 0).
pub fn covariance(x: &Matrix, center: &[f64], ddof: usize) -> Mat {
    let d = center.len();
    let mut c = zeros(d, d);
    for r in x {
        for i in 0..d {
            let di = r[i] - center[i];
            for j in i..d {
                c[i][j] += di * (r[j] - center[j]);
            }
        }
    }
    let denom = (x.len().saturating_sub(ddof)).max(1) as f64;
    for i in 0..d {
        for j in i..d {
            c[i][j] /= denom;
            c[j][i] = c[i][j];
        }
    }
    c
}

/// Solve `A X = B` for square `A` by Gaussian elimination with partial pivoting.
/// Returns `None` if `A` is singular.
pub fn solve(a: &[Vec<f64>], b: &[Vec<f64>]) -> Option<Mat> {
    let n = a.len();
    let m = b.first().map_or(0, Vec::len);
    let mut a: Mat = a.to_vec();
    let mut x: Mat = b.to_vec();
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[piv][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, piv);
        x.swap(col, piv);
        for r in col + 1..n {
            let f = a[r][col] / a[col][col];
            if f == 0.0 {
                continue;
            }
            for k in col..n {
                a[r][k] -= f * a[col][k];
            }
            for k in 0..m {
                x[r][k] -= f * x[col][k];
            }
        }
    }
    for col in (0..n).rev() {
        for k in 0..m {
            let s: f64 = (col + 1..n).map(|j| a[col][j] * x[j][k]).sum();
            x[col][k] = (x[col][k] - s) / a[col][col];
        }
    }
    Some(x)
}

/// Lower-triangular Cholesky factor, or `None` if `a` is not positive definite.
pub fn cholesky(a: &[Vec<f64>]) -> Option<Mat> {
    let n = a.len();
    let mut l = zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
            if i == j {
                let v = a[i][i] - s;
                if v <= 0.0 || !v.is_finite() {
                    return None;
                }
                l[i][j] = v.sqrt();
            } else {
                l[i][j] = (a[i][j] - s) / l[j][j];
            }
        }
    }
    Some(l)
}

/// Solve `L y = b` for lower-triangular `L`.
pub fn forward_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; b.len()];
    for i in 0..b.len() {
        let s: f64 = (0..i).map(|k| l[i][k] * y[k]).sum();
        y[i] = (b[i] - s) / l[i][i];
    }
    y
}

/// Pseudo-inverse of a symmetric matrix, like `scipy.linalg.pinvh`.
pub fn pinvh(a: &[Vec<f64>]) -> Mat {
    let n = a.len();
    let (vals, vecs) = symmetric_eigen(a);
    let max_abs = vals.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let cutoff = max_abs * n as f64 * f64::EPSILON;
    let mut out = zeros(n, n);
    for (val, v) in vals.iter().zip(&vecs) {
        if val.abs() <= cutoff {
            continue;
        }
        for i in 0..n {
            for j in 0..n {
                out[i][j] += v[i] * v[j] / val;
            }
        }
    }
    out
}

/// log|det(a)| for a symmetric matrix, or -inf when it is not positive definite
/// (sklearn's `fast_logdet`).
pub fn fast_logdet(a: &[Vec<f64>]) -> f64 {
    match cholesky(a) {
        Some(l) => 2.0 * (0..a.len()).map(|i| l[i][i].ln()).sum::<f64>(),
        None => f64::NEG_INFINITY,
    }
}

/// Squared Mahalanobis distance of each row from `loc` under `precision`.
pub fn mahalanobis_sq(x: &Matrix, loc: &[f64], precision: &[Vec<f64>]) -> Vec<f64> {
    x.iter()
        .map(|r| {
            let c: Vec<f64> = r.iter().zip(loc).map(|(a, b)| a - b).collect();
            matvec(precision, &c)
                .iter()
                .zip(&c)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect()
}

pub fn logsumexp(v: &[f64]) -> f64 {
    let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY {
        return m;
    }
    m + v.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
}

/// ln Γ(x) (Lanczos, g = 7).
pub fn lgamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).abs().ln() - lgamma(1.0 - x);
    }
    let x = x - 1.0;
    let t = x + 7.5;
    let s = G[1..]
        .iter()
        .enumerate()
        .fold(G[0], |s, (i, g)| s + g / (x + i as f64 + 1.0));
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + s.ln()
}

/// Regularised lower incomplete gamma P(a, x).
pub fn gamma_p(a: f64, x: f64) -> f64 {
    let (p, _) = gamma_pq(a, x);
    p
}

/// Regularised upper incomplete gamma Q(a, x) = 1 - P(a, x), without cancellation.
pub fn gamma_q(a: f64, x: f64) -> f64 {
    let (_, q) = gamma_pq(a, x);
    q
}

fn gamma_pq(a: f64, x: f64) -> (f64, f64) {
    if x <= 0.0 {
        return (0.0, 1.0);
    }
    let gln = lgamma(a);
    if x < a + 1.0 {
        let (mut sum, mut del, mut ap) = (1.0 / a, 1.0 / a, a);
        for _ in 0..1000 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-16 {
                break;
            }
        }
        let p = (sum.ln() - x + a * x.ln() - gln).exp();
        (p, 1.0 - p)
    } else {
        // Continued fraction for Q(a, x) (modified Lentz).
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..1000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-16 {
                break;
            }
        }
        let q = (-x + a * x.ln() - gln).exp() * h;
        (1.0 - q, q)
    }
}

/// Chi-squared CDF.
pub fn chi2_cdf(x: f64, df: f64) -> f64 {
    gamma_p(df / 2.0, x / 2.0)
}

/// Chi-squared quantile (`scipy.stats.chi2(df).ppf(q)`), by bisection.
pub fn chi2_ppf(q: f64, df: f64) -> f64 {
    let (mut lo, mut hi) = (0.0, df.max(1.0));
    while chi2_cdf(hi, df) < q {
        hi *= 2.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if chi2_cdf(mid, df) < q {
            lo = mid
        } else {
            hi = mid
        }
    }
    0.5 * (lo + hi)
}

/// Inverse of the standard normal CDF (Acklam's rational approximation, refined once).
pub fn norm_ppf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let pl = 0.024_25;
    let x = if p < pl {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - pl {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    // One Halley step against erfc for full precision.
    let e = 0.5 * crate::utils::erfc(-x / std::f64::consts::SQRT_2) - p;
    let u = e * (2.0 * std::f64::consts::PI).sqrt() * (x * x / 2.0).exp();
    x - u / (1.0 + x * u / 2.0)
}

/// k-means with k-means++ seeding and Lloyd iterations. Returns (centers, labels).
pub fn kmeans(
    x: &Matrix,
    k: usize,
    n_init: usize,
    max_iter: usize,
    rng: &mut Rng,
) -> (Mat, Vec<usize>) {
    let n = x.len();
    let mut best: Option<(f64, Mat, Vec<usize>)> = None;
    for _ in 0..n_init.max(1) {
        let mut centers = vec![x[rng.below(n)].clone()];
        let mut d2: Vec<f64> = x.iter().map(|r| sq_euclidean(r, &centers[0])).collect();
        while centers.len() < k {
            let total: f64 = d2.iter().sum();
            let next = if total <= 0.0 {
                rng.below(n)
            } else {
                let mut t = rng.next_f64() * total;
                let mut idx = n - 1;
                for (i, d) in d2.iter().enumerate() {
                    t -= d;
                    if t <= 0.0 {
                        idx = i;
                        break;
                    }
                }
                idx
            };
            centers.push(x[next].clone());
            for (i, r) in x.iter().enumerate() {
                d2[i] = d2[i].min(sq_euclidean(r, centers.last().unwrap()));
            }
        }
        let mut labels = vec![0; n];
        for _ in 0..max_iter {
            let mut changed = false;
            for (i, r) in x.iter().enumerate() {
                let l = nearest(&centers, r).0;
                changed |= l != labels[i];
                labels[i] = l;
            }
            let d = x[0].len();
            let mut sums = zeros(k, d);
            let mut counts = vec![0usize; k];
            for (r, &l) in x.iter().zip(&labels) {
                counts[l] += 1;
                for j in 0..d {
                    sums[l][j] += r[j];
                }
            }
            for c in 0..k {
                if counts[c] > 0 {
                    centers[c] = sums[c].iter().map(|s| s / counts[c] as f64).collect();
                }
            }
            if !changed {
                break;
            }
        }
        let inertia: f64 = x
            .iter()
            .zip(&labels)
            .map(|(r, &l)| sq_euclidean(r, &centers[l]))
            .sum();
        if best.as_ref().is_none_or(|b| inertia < b.0) {
            best = Some((inertia, centers, labels));
        }
    }
    let (_, c, l) = best.unwrap();
    (c, l)
}

/// Index and squared distance of the nearest center.
pub fn nearest(centers: &[Vec<f64>], r: &[f64]) -> (usize, f64) {
    centers
        .iter()
        .enumerate()
        .map(|(i, c)| (i, sq_euclidean(r, c)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chi2_quantiles_match_scipy() {
        assert!((chi2_ppf(0.5, 2.0) - 1.386_294_361_119_890_6).abs() < 1e-10);
        assert!((chi2_ppf(0.975, 4.0) - 11.143_286_781_877_8).abs() < 1e-9);
    }

    #[test]
    fn lgamma_matches_known_values() {
        assert!((lgamma(5.0) - 24f64.ln()).abs() < 1e-12);
        assert!((lgamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-12);
    }

    #[test]
    fn norm_ppf_matches_scipy() {
        assert!((norm_ppf(0.975) - 1.959_963_984_540_054).abs() < 1e-12);
        assert!((norm_ppf(0.001) + 3.090_232_306_167_813_5).abs() < 1e-10);
    }

    #[test]
    fn solve_matches_known_system() {
        let a = vec![
            vec![0.0, 2.0, 1.0],
            vec![1.0, 1.0, 0.0],
            vec![3.0, 0.0, 1.0],
        ];
        let b = vec![vec![5.0], vec![3.0], vec![6.0]];
        let x = solve(&a, &b).unwrap();
        for (r, want) in x.iter().zip([1.4, 1.6, 1.8]) {
            assert!((r[0] - want).abs() < 1e-12);
        }
        assert!(solve(&[vec![1.0, 2.0], vec![2.0, 4.0]], &[vec![1.0], vec![2.0]]).is_none());
    }

    #[test]
    fn pinvh_inverts() {
        let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let p = matmul(&a, &pinvh(&a));
        assert!((p[0][0] - 1.0).abs() < 1e-12 && p[0][1].abs() < 1e-12);
    }
}
