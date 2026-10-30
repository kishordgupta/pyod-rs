//! Time-series helpers shared by the `ts_*` detectors. Port of `pyod.models._ts_utils`
//! plus the FFT and shape-based distance they rely on.
//!
//! A time series is a [`Matrix`] with one row per timestamp and one column per channel.

use crate::base::{FitState, Matrix};
use crate::utils::{mean, percentile, std_dev};

/// Flattened sliding windows (rows `t..t+window`, channels interleaved per timestamp).
pub fn sliding_windows(x: &Matrix, window: usize, step: usize) -> Vec<Vec<f64>> {
    let n = x.len();
    if n < window {
        return Vec::new();
    }
    (0..=(n - window) / step)
        .map(|i| {
            x[i * step..i * step + window]
                .iter()
                .flatten()
                .copied()
                .collect()
        })
        .collect()
}

/// How overlapping window scores combine at each timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Aggregation {
    #[default]
    Max,
    Mean,
}

/// Map window scores onto timestamps. `None` marks timestamps no window covers.
pub fn map_scores_to_timestamps(
    window_scores: &[f64],
    window: usize,
    step: usize,
    n: usize,
    agg: Aggregation,
) -> Vec<Option<f64>> {
    let mut scores: Vec<Option<f64>> = vec![None; n];
    let mut counts = vec![0usize; n];
    for (i, &s) in window_scores.iter().enumerate() {
        for j in i * step..(i * step + window).min(n) {
            scores[j] = Some(match (agg, scores[j]) {
                (_, None) => s,
                (Aggregation::Max, Some(v)) => v.max(s),
                (Aggregation::Mean, Some(v)) => v + s,
            });
            counts[j] += 1;
        }
    }
    if agg == Aggregation::Mean {
        for (s, c) in scores.iter_mut().zip(&counts) {
            if let Some(v) = s {
                *v /= *c as f64;
            }
        }
    }
    scores
}

/// Z-normalise each channel's scores, then take the max or mean across channels.
pub fn aggregate_channel_scores(per_channel: &[Vec<f64>], agg: Aggregation) -> Vec<f64> {
    let normalized: Vec<Vec<f64>> = per_channel
        .iter()
        .map(|c| {
            let (mu, sigma) = (mean(c), std_dev(c));
            c.iter()
                .map(|v| {
                    if sigma > 0.0 {
                        (v - mu) / sigma
                    } else {
                        v - mu
                    }
                })
                .collect()
        })
        .collect();
    (0..normalized[0].len())
        .map(|i| {
            let vals = normalized.iter().map(|c| c[i]);
            match agg {
                Aggregation::Max => vals.fold(f64::NEG_INFINITY, f64::max),
                Aggregation::Mean => vals.sum::<f64>() / normalized.len() as f64,
            }
        })
        .collect()
}

/// Combine per-channel timestamp scores; a timestamp is valid only if every channel covers it.
pub fn combine_channels(per_channel: Vec<Vec<Option<f64>>>, agg: Aggregation) -> Vec<Option<f64>> {
    if per_channel.len() == 1 {
        return per_channel.into_iter().next().unwrap();
    }
    let n = per_channel[0].len();
    let valid: Vec<bool> = (0..n)
        .map(|i| per_channel.iter().all(|c| c[i].is_some()))
        .collect();
    let filled: Vec<Vec<f64>> = per_channel
        .iter()
        .map(|c| c.iter().map(|v| v.unwrap_or(0.0)).collect())
        .collect();
    aggregate_channel_scores(&filled, agg)
        .into_iter()
        .zip(valid)
        .map(|(s, v)| v.then_some(s))
        .collect()
}

/// Threshold on the covered timestamps, then fill uncovered ones with the threshold (PyOD's TS convention).
pub fn ts_fit_state(scores: &[Option<f64>], contamination: f64) -> FitState {
    let valid: Vec<f64> = scores.iter().flatten().copied().collect();
    let threshold = percentile(&valid, 100.0 * (1.0 - contamination));
    let (mu, sigma) = (mean(&valid), std_dev(&valid));
    let full: Vec<f64> = scores.iter().map(|s| s.unwrap_or(threshold)).collect();
    let mut state = FitState::from_threshold(full, threshold);
    (state.mu, state.sigma, state.contamination) = (mu, sigma, contamination);
    state
}

pub fn fill_with(scores: Vec<Option<f64>>, value: f64) -> Vec<f64> {
    scores.into_iter().map(|s| s.unwrap_or(value)).collect()
}

/// (x - mean) / std, or zeros for a (near-)constant sequence.
pub fn znormalize(x: &[f64]) -> Vec<f64> {
    let s = std_dev(x);
    if s < 1e-10 {
        return vec![0.0; x.len()];
    }
    let m = mean(x);
    x.iter().map(|v| (v - m) / s).collect()
}

/// Shape-based distance (Paparrizos & Gravano, 2015): 1 - max normalised
/// cross-correlation, and the shift that attains it.
pub fn sbd(x: &[f64], y: &[f64]) -> (f64, isize) {
    let m = x.len();
    let norm =
        x.iter().map(|v| v * v).sum::<f64>().sqrt() * y.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm < 1e-10 {
        return (2.0, 0);
    }
    let mut best = (f64::NEG_INFINITY, 0isize);
    for s in -(m as isize - 1)..m as isize {
        let cc: f64 = (0..m as isize)
            .filter(|&i| i + s >= 0 && i + s < m as isize)
            .map(|i| x[(i + s) as usize] * y[i as usize])
            .sum();
        if cc / norm > best.0 {
            best = (cc / norm, s);
        }
    }
    ((1.0 - best.0).clamp(0.0, 2.0), best.1)
}

/// A complex number as (re, im).
pub type Complex = (f64, f64);

fn cmul(a: Complex, b: Complex) -> Complex {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// In-place iterative radix-2 FFT; `inverse` uses the conjugate twiddles (unscaled).
fn fft_pow2(a: &mut [Complex], inverse: bool) {
    let n = a.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            a.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = 2.0 * std::f64::consts::PI / len as f64 * if inverse { 1.0 } else { -1.0 };
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let w = ((ang * k as f64).cos(), (ang * k as f64).sin());
                let u = a[start + k];
                let v = cmul(a[start + k + len / 2], w);
                a[start + k] = (u.0 + v.0, u.1 + v.1);
                a[start + k + len / 2] = (u.0 - v.0, u.1 - v.1);
            }
        }
        len <<= 1;
    }
}

/// Discrete Fourier transform of any length (Bluestein's algorithm for non-powers of two),
/// matching `np.fft.fft` / `np.fft.ifft`.
pub fn dft(x: &[Complex], inverse: bool) -> Vec<Complex> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let mut out = if n.is_power_of_two() {
        let mut a = x.to_vec();
        fft_pow2(&mut a, inverse);
        a
    } else {
        let sign = if inverse { 1.0 } else { -1.0 };
        let w: Vec<Complex> = (0..n)
            .map(|k| {
                let ang = sign * std::f64::consts::PI * ((k * k) % (2 * n)) as f64 / n as f64;
                (ang.cos(), ang.sin())
            })
            .collect();
        let m = (2 * n - 1).next_power_of_two();
        let mut a = vec![(0.0, 0.0); m];
        for k in 0..n {
            a[k] = cmul(x[k], w[k]);
        }
        let mut b = vec![(0.0, 0.0); m];
        b[0] = (w[0].0, -w[0].1);
        for k in 1..n {
            b[k] = (w[k].0, -w[k].1);
            b[m - k] = b[k];
        }
        fft_pow2(&mut a, false);
        fft_pow2(&mut b, false);
        let mut c: Vec<Complex> = a.iter().zip(&b).map(|(p, q)| cmul(*p, *q)).collect();
        fft_pow2(&mut c, true);
        (0..n)
            .map(|k| cmul((c[k].0 / m as f64, c[k].1 / m as f64), w[k]))
            .collect()
    };
    if inverse {
        out.iter_mut()
            .for_each(|v| *v = (v.0 / n as f64, v.1 / n as f64));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(x: &[Complex]) -> Vec<Complex> {
        let n = x.len();
        (0..n)
            .map(|k| {
                x.iter().enumerate().fold((0.0, 0.0), |acc, (t, v)| {
                    let ang = -2.0 * std::f64::consts::PI * (k * t) as f64 / n as f64;
                    let p = cmul(*v, (ang.cos(), ang.sin()));
                    (acc.0 + p.0, acc.1 + p.1)
                })
            })
            .collect()
    }

    #[test]
    fn dft_matches_naive_and_inverts() {
        for n in [1, 2, 7, 8, 30, 64] {
            let x: Vec<Complex> = (0..n)
                .map(|i| ((i as f64 * 0.7).sin(), (i as f64).cos() * 0.3))
                .collect();
            let f = dft(&x, false);
            for (a, b) in f.iter().zip(naive(&x)) {
                assert!(
                    (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9,
                    "n={n}"
                );
            }
            let back = dft(&f, true);
            for (a, b) in back.iter().zip(&x) {
                assert!((a.0 - b.0).abs() < 1e-12 && (a.1 - b.1).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn sbd_finds_shift() {
        let x: Vec<f64> = (0..20).map(|i| ((i as f64) / 3.0).sin()).collect();
        let (d, _) = sbd(&x, &x);
        assert!(d.abs() < 1e-12);
    }
}
