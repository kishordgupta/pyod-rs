//! Score combination across detectors. Port of `pyod.models.combination`.
//!
//! `scores[i][j]` is detector `j`'s score for sample `i`; standardize each column
//! first (e.g. with `utils::StandardScaler`) so detectors are on one scale.

use crate::utils::median as median_of;

/// Mean score per sample, optionally weighted per detector.
pub fn average(scores: &[Vec<f64>], weights: Option<&[f64]>) -> Vec<f64> {
    scores
        .iter()
        .map(|row| match weights {
            Some(w) => row.iter().zip(w).map(|(s, w)| s * w).sum::<f64>() / w.iter().sum::<f64>(),
            None => row.iter().sum::<f64>() / row.len() as f64,
        })
        .collect()
}

/// Maximum score per sample.
pub fn maximization(scores: &[Vec<f64>]) -> Vec<f64> {
    scores
        .iter()
        .map(|row| row.iter().copied().fold(f64::NEG_INFINITY, f64::max))
        .collect()
}

/// Median score per sample.
pub fn median(scores: &[Vec<f64>]) -> Vec<f64> {
    scores.iter().map(|row| median_of(row)).collect()
}

/// Majority vote over binary labels (ties go to inlier).
pub fn majority_vote(labels: &[Vec<u8>]) -> Vec<u8> {
    labels
        .iter()
        .map(|row| (row.iter().map(|&l| l as usize).sum::<usize>() * 2 > row.len()) as u8)
        .collect()
}

/// Average of Maximum: split detectors into `n_buckets` consecutive groups,
/// take the max within each group, then average the groups (static AOM).
pub fn aom(scores: &[Vec<f64>], n_buckets: usize) -> Vec<f64> {
    bucketed(scores, n_buckets, maximization, |b| average(b, None))
}

/// Maximum of Average: average within each group, then take the max (static MOA).
pub fn moa(scores: &[Vec<f64>], n_buckets: usize) -> Vec<f64> {
    bucketed(scores, n_buckets, |b| average(b, None), maximization)
}

fn bucketed(
    scores: &[Vec<f64>],
    n_buckets: usize,
    inner: impl Fn(&[Vec<f64>]) -> Vec<f64>,
    outer: impl Fn(&[Vec<f64>]) -> Vec<f64>,
) -> Vec<f64> {
    let n_det = scores.first().map_or(0, Vec::len);
    assert!(
        n_buckets >= 1 && n_buckets <= n_det,
        "n_buckets must be in [1, n_detectors]"
    );
    let per = n_det / n_buckets;
    let mut combined = vec![Vec::with_capacity(n_buckets); scores.len()];
    for b in 0..n_buckets {
        let end = if b == n_buckets - 1 {
            n_det
        } else {
            (b + 1) * per
        };
        let sub: Vec<Vec<f64>> = scores.iter().map(|r| r[b * per..end].to_vec()).collect();
        for (row, v) in combined.iter_mut().zip(inner(&sub)) {
            row.push(v);
        }
    }
    outer(&combined)
}
