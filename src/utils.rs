//! Numeric helpers standing in for the NumPy / SciPy calls PyOD relies on.

use crate::base::Matrix;

pub fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// Population standard deviation (NumPy's default `ddof=0`).
pub fn std_dev(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

/// Same as `np.percentile` with linear interpolation; `q` in [0, 100].
pub fn percentile(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let pos = (q / 100.0).clamp(0.0, 1.0) * (s.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    s[lo] + (s[hi] - s[lo]) * (pos - lo as f64)
}

pub fn median(v: &[f64]) -> f64 {
    percentile(v, 50.0)
}

/// Biased sample skewness, as `scipy.stats.skew` computes it. Constant input gives 0.
pub fn skew(v: &[f64]) -> f64 {
    let m = mean(v);
    let n = v.len() as f64;
    let m2 = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
    let m3 = v.iter().map(|x| (x - m).powi(3)).sum::<f64>() / n;
    if m2 == 0.0 {
        0.0
    } else {
        m3 / m2.powf(1.5)
    }
}

/// `np.sign`, which (unlike `f64::signum`) maps 0 to 0.
pub fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Empirical CDF of each value within its own column: P(X <= x).
/// Port of `pyod.utils.stat_models.column_ecdf`; tied values share the highest rank.
pub fn column_ecdf(x: &Matrix) -> Vec<Vec<f64>> {
    let n = x.len();
    let d = x[0].len();
    let mut out = vec![vec![0.0; d]; n];
    for j in 0..d {
        let mut col: Vec<f64> = x.iter().map(|r| r[j]).collect();
        col.sort_by(f64::total_cmp);
        for i in 0..n {
            let rank = col.partition_point(|&v| v <= x[i][j]);
            out[i][j] = rank as f64 / n as f64;
        }
    }
    out
}

pub fn sq_euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

pub fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    sq_euclidean(a, b).sqrt()
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Indices and distances of the `k` nearest rows of `train` to `query`, nearest first.
/// `skip` excludes one training index (the query itself when scoring training data).
pub fn k_nearest(
    train: &Matrix,
    query: &[f64],
    k: usize,
    skip: Option<usize>,
) -> Vec<(usize, f64)> {
    let mut d: Vec<(usize, f64)> = train
        .iter()
        .enumerate()
        .filter(|(i, _)| Some(*i) != skip)
        .map(|(i, r)| (i, euclidean(r, query)))
        .collect();
    let k = k.min(d.len());
    if k < d.len() {
        d.select_nth_unstable_by(k, |a, b| a.1.total_cmp(&b.1));
        d.truncate(k);
    }
    d.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    d
}

/// Argsort with numba's default quicksort, so ties break exactly as in PyOD's
/// numba-compiled kernels (median-of-three partitioning, insertion sort below 15).
pub fn numba_argsort<T: PartialOrd + Copy>(a: &[T]) -> Vec<usize> {
    let n = a.len();
    let mut r: Vec<usize> = (0..n).collect();
    if n < 2 {
        return r;
    }
    let lt = |x: T, y: T| x < y;
    let insertion = |r: &mut [usize], low: usize, high: usize| {
        for i in low + 1..=high {
            let k = r[i];
            let v = a[k];
            let mut j = i;
            while j > low && lt(v, a[r[j - 1]]) {
                r[j] = r[j - 1];
                j -= 1;
            }
            r[j] = k;
        }
    };
    let partition = |r: &mut [usize], low: usize, high: usize| -> usize {
        let mid = (low + high) >> 1;
        if lt(a[r[mid]], a[r[low]]) {
            r.swap(low, mid);
        }
        if lt(a[r[high]], a[r[mid]]) {
            r.swap(high, mid);
        }
        if lt(a[r[mid]], a[r[low]]) {
            r.swap(low, mid);
        }
        let pivot = a[r[mid]];
        r.swap(high, mid);
        let (mut i, mut j) = (low as isize, high as isize - 1);
        loop {
            while (i as usize) < high && lt(a[r[i as usize]], pivot) {
                i += 1;
            }
            while j >= low as isize && lt(pivot, a[r[j as usize]]) {
                j -= 1;
            }
            if i >= j {
                break;
            }
            r.swap(i as usize, j as usize);
            i += 1;
            j -= 1;
        }
        r.swap(i as usize, high);
        i as usize
    };
    // Signed bounds, as in numba: `high = i - 1` reaches -1 when the pivot
    // lands at index 0.
    let mut stack = vec![(0isize, n as isize - 1)];
    while let Some((mut low, mut high)) = stack.pop() {
        while high - low >= 15 {
            let i = partition(&mut r, low as usize, high as usize) as isize;
            if high - i > i - low {
                if high > i {
                    stack.push((i + 1, high));
                }
                high = i - 1;
            } else {
                if i > low {
                    stack.push((low, i - 1));
                }
                low = i + 1;
            }
        }
        if high > low {
            insertion(&mut r, low as usize, high as usize);
        }
    }
    r
}

/// Column-wise z-score scaler, like sklearn's `StandardScaler`.
#[derive(Debug, Clone)]
pub struct StandardScaler {
    pub mean: Vec<f64>,
    pub scale: Vec<f64>,
}

impl StandardScaler {
    pub fn fit(x: &Matrix) -> Self {
        let d = x[0].len();
        let mut mean_ = vec![0.0; d];
        let mut scale = vec![0.0; d];
        for j in 0..d {
            let col: Vec<f64> = x.iter().map(|r| r[j]).collect();
            mean_[j] = mean(&col);
            let s = std_dev(&col);
            scale[j] = if s == 0.0 { 1.0 } else { s };
        }
        StandardScaler { mean: mean_, scale }
    }

    pub fn transform(&self, x: &Matrix) -> Vec<Vec<f64>> {
        x.iter()
            .map(|r| {
                r.iter()
                    .enumerate()
                    .map(|(j, v)| (v - self.mean[j]) / self.scale[j])
                    .collect()
            })
            .collect()
    }
}

/// Error function, via the regularised incomplete gamma function (~1e-15 accuracy).
pub fn erf(x: f64) -> f64 {
    sign(x) * crate::linalg::gamma_p(0.5, x * x)
}

/// Complementary error function, accurate in the far tail.
pub fn erfc(x: f64) -> f64 {
    if x >= 0.0 {
        crate::linalg::gamma_q(0.5, x * x)
    } else {
        1.0 + crate::linalg::gamma_p(0.5, x * x)
    }
}

/// Standard normal CDF.
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// Eigen-decomposition of a symmetric matrix: Householder tridiagonalisation then
/// implicit QL (the EISPACK `tred2` / `tql2` routines, as in JAMA).
/// Returns eigenvalues in descending order and matching unit eigenvectors.
pub fn symmetric_eigen(a: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut v: Vec<Vec<f64>> = a.to_vec();
    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n];
    // tred2
    d.copy_from_slice(&v[n - 1]);
    for i in (1..n).rev() {
        let mut scale = 0.0;
        let mut h = 0.0;
        for k in 0..i {
            scale += d[k].abs();
        }
        if scale == 0.0 {
            e[i] = d[i - 1];
            for j in 0..i {
                d[j] = v[i - 1][j];
                v[i][j] = 0.0;
                v[j][i] = 0.0;
            }
        } else {
            for k in 0..i {
                d[k] /= scale;
                h += d[k] * d[k];
            }
            let mut f = d[i - 1];
            let mut g = h.sqrt();
            if f > 0.0 {
                g = -g;
            }
            e[i] = scale * g;
            h -= f * g;
            d[i - 1] = f - g;
            for j in 0..i {
                e[j] = 0.0;
            }
            for j in 0..i {
                f = d[j];
                v[j][i] = f;
                g = e[j] + v[j][j] * f;
                for k in j + 1..i {
                    g += v[k][j] * d[k];
                    e[k] += v[k][j] * f;
                }
                e[j] = g;
            }
            f = 0.0;
            for j in 0..i {
                e[j] /= h;
                f += e[j] * d[j];
            }
            let hh = f / (h + h);
            for j in 0..i {
                e[j] -= hh * d[j];
            }
            for j in 0..i {
                f = d[j];
                g = e[j];
                for k in j..i {
                    v[k][j] -= f * e[k] + g * d[k];
                }
                d[j] = v[i - 1][j];
                v[i][j] = 0.0;
            }
        }
        d[i] = h;
    }
    for i in 0..n - 1 {
        v[n - 1][i] = v[i][i];
        v[i][i] = 1.0;
        let h = d[i + 1];
        if h != 0.0 {
            for k in 0..=i {
                d[k] = v[k][i + 1] / h;
            }
            for j in 0..=i {
                let mut g = 0.0;
                for k in 0..=i {
                    g += v[k][i + 1] * v[k][j];
                }
                for k in 0..=i {
                    v[k][j] -= g * d[k];
                }
            }
        }
        for k in 0..=i {
            v[k][i + 1] = 0.0;
        }
    }
    for j in 0..n {
        d[j] = v[n - 1][j];
        v[n - 1][j] = 0.0;
    }
    v[n - 1][n - 1] = 1.0;
    e[0] = 0.0;
    // tql2
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;
    let mut f = 0.0;
    let mut tst1 = 0.0f64;
    let eps = f64::EPSILON;
    for l in 0..n {
        tst1 = tst1.max(d[l].abs() + e[l].abs());
        let mut m = l;
        while m < n {
            if e[m].abs() <= eps * tst1 {
                break;
            }
            m += 1;
        }
        let m = m.min(n - 1);
        if m > l {
            loop {
                let mut g = d[l];
                let mut p = (d[l + 1] - g) / (2.0 * e[l]);
                let mut r = p.hypot(1.0);
                if p < 0.0 {
                    r = -r;
                }
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                let dl1 = d[l + 1];
                let mut h = g - d[l];
                for di in d.iter_mut().take(n).skip(l + 2) {
                    *di -= h;
                }
                f += h;
                p = d[m];
                let mut c = 1.0;
                let mut c2 = c;
                let mut c3 = c;
                let el1 = e[l + 1];
                let mut s = 0.0;
                let mut s2 = 0.0;
                for i in (l..m).rev() {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[i];
                    h = c * p;
                    r = p.hypot(e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    for row in v.iter_mut() {
                        h = row[i + 1];
                        row[i + 1] = s * row[i] + c * h;
                        row[i] = c * row[i] - s * h;
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;
                if e[l].abs() <= eps * tst1 {
                    break;
                }
            }
        }
        d[l] += f;
        e[l] = 0.0;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| d[j].total_cmp(&d[i]));
    let values = order.iter().map(|&i| d[i]).collect();
    let vectors = order
        .iter()
        .map(|&i| (0..n).map(|k| v[k][i]).collect())
        .collect();
    (values, vectors)
}

/// Small deterministic PRNG (SplitMix64) so the crate needs no dependencies.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform float in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform integer in [0, n).
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// `k` distinct indices from [0, n) (partial Fisher–Yates).
    pub fn sample_indices(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..n).collect();
        for i in 0..k.min(n) {
            let j = i + self.below(n - i);
            idx.swap(i, j);
        }
        idx.truncate(k.min(n));
        idx
    }

    /// Standard normal sample (Box–Muller).
    pub fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_matches_numpy() {
        let v = [1.0, 2.0, 3.0, 4.0];
        assert!((percentile(&v, 50.0) - 2.5).abs() < 1e-12);
        assert!((percentile(&v, 90.0) - 3.7).abs() < 1e-12);
    }

    #[test]
    fn ecdf_handles_ties() {
        let x = vec![vec![1.0], vec![0.0], vec![1.0], vec![2.0]];
        let e = column_ecdf(&x);
        assert_eq!(
            e.iter().map(|r| r[0]).collect::<Vec<_>>(),
            vec![0.75, 0.25, 0.75, 1.0]
        );
    }

    #[test]
    fn eigen_reconstructs_random_symmetric() {
        let mut rng = Rng::new(3);
        for n in [1, 2, 5, 30] {
            let mut a = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in i..n {
                    a[i][j] = rng.normal();
                    a[j][i] = a[i][j];
                }
            }
            let (vals, vecs) = symmetric_eigen(&a);
            assert!(vals.windows(2).all(|w| w[0] >= w[1]));
            for (l, v) in vals.iter().zip(&vecs) {
                for i in 0..n {
                    let av: f64 = (0..n).map(|k| a[i][k] * v[k]).sum();
                    assert!((av - l * v[i]).abs() < 1e-9, "n={n}");
                }
                assert!((dot(v, v) - 1.0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn jacobi_recovers_eigenvalues() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let (vals, vecs) = symmetric_eigen(&a);
        assert!((vals[0] - 3.0).abs() < 1e-10 && (vals[1] - 1.0).abs() < 1e-10);
        assert!((vecs[0][0].abs() - vecs[0][1].abs()).abs() < 1e-10);
    }

    #[test]
    fn numba_argsort_pivot_at_start() {
        // The first partition leaves the pivot at index 0; order from numba 0.60.
        let a = [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1];
        assert_eq!(
            numba_argsort(&a),
            vec![7, 0, 1, 2, 3, 4, 5, 6, 15, 8, 9, 10, 11, 12, 13, 14]
        );
        let mut rng = Rng::new(5);
        for n in [16, 17, 40, 100, 1000] {
            let v: Vec<u32> = (0..n).map(|_| (rng.next_f64() * 3.0) as u32).collect();
            let r = numba_argsort(&v);
            assert!(r.windows(2).all(|w| v[w[0]] <= v[w[1]]), "n={n}");
        }
    }

    #[test]
    fn erf_is_accurate() {
        assert!((erf(1.0) - 0.842_700_792_949_714_9).abs() < 1e-14);
        assert!((erf(-0.5) + 0.520_499_877_813_046_5).abs() < 1e-14);
        assert!((erfc(5.0) - 1.537_459_794_428_034_8e-12).abs() < 1e-24);
    }
}
