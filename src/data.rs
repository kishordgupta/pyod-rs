//! Synthetic data and metrics, after `pyod.utils.data` and `pyod.utils.utility`.

use crate::utils::Rng;

/// Gaussian inliers around the origin plus uniform outliers, like `generate_data`.
/// Returns `(x, y)` with `y[i] == 1` for outliers, which come last.
pub fn generate_data(
    n: usize,
    n_features: usize,
    contamination: f64,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<u8>) {
    let mut rng = Rng::new(seed);
    let n_out = (n as f64 * contamination) as usize;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n - n_out {
        x.push(
            (0..n_features)
                .map(|_| 0.3 * rng.normal() + 2.0 * (rng.below(2) as f64 * 2.0 - 1.0))
                .collect(),
        );
        y.push(0);
    }
    for _ in 0..n_out {
        x.push(
            (0..n_features)
                .map(|_| rng.next_f64() * 12.0 - 6.0)
                .collect(),
        );
        y.push(1);
    }
    (x, y)
}

/// ROC AUC via the Mann–Whitney U statistic (ties count half).
pub fn roc_auc(y_true: &[u8], scores: &[f64]) -> f64 {
    let pos: Vec<f64> = y_true
        .iter()
        .zip(scores)
        .filter(|(y, _)| **y == 1)
        .map(|(_, s)| *s)
        .collect();
    let neg: Vec<f64> = y_true
        .iter()
        .zip(scores)
        .filter(|(y, _)| **y == 0)
        .map(|(_, s)| *s)
        .collect();
    let mut wins = 0.0;
    for p in &pos {
        for q in &neg {
            wins += if p > q {
                1.0
            } else if p == q {
                0.5
            } else {
                0.0
            };
        }
    }
    wins / (pos.len() * neg.len()) as f64
}

/// Precision among the top-n scores, n = number of true outliers.
pub fn precision_n_scores(y_true: &[u8], scores: &[f64]) -> f64 {
    let n = y_true.iter().filter(|&&y| y == 1).count();
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    idx[..n].iter().filter(|&&i| y_true[i] == 1).count() as f64 / n as f64
}
