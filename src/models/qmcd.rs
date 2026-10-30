//! QMCD: Quasi-Monte Carlo Discrepancy outlier detection (Fang & Ma, 2001).
//! Port of `pyod.models.qmcd`.

use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, Result,
};
use crate::utils::{mean, skew};

#[derive(Debug, Clone)]
pub struct Qmcd {
    pub contamination: f64,
    min: Vec<f64>,
    range: Vec<f64>,
    data: Vec<Vec<f64>>,
    flipped: bool,
    state: Option<FitState>,
}

impl Default for Qmcd {
    fn default() -> Self {
        Self::new(0.1)
    }
}

/// Wrap-around L2 discrepancy of each `check` row against `data`.
fn wrap_around_discrepancy(data: &Matrix, check: &Matrix) -> Vec<f64> {
    let n = data.len() as f64;
    let d = data[0].len() as i32;
    check
        .iter()
        .map(|c| {
            let dc: f64 = data
                .iter()
                .map(|r| {
                    c.iter()
                        .zip(r)
                        .map(|(a, b)| {
                            let v = (a - b).abs();
                            1.5 - v + v * v
                        })
                        .product::<f64>()
                })
                .sum();
            -(4.0f64 / 3.0).powi(d) + dc / (n * n)
        })
        .collect()
}

/// Excess kurtosis (biased), as `scipy.stats.kurtosis` computes it.
fn kurtosis(v: &[f64]) -> f64 {
    let m = mean(v);
    let n = v.len() as f64;
    let m2 = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
    let m4 = v.iter().map(|x| (x - m).powi(4)).sum::<f64>() / n;
    if m2 == 0.0 {
        -3.0
    } else {
        m4 / (m2 * m2) - 3.0
    }
}

impl Qmcd {
    pub fn new(contamination: f64) -> Self {
        Qmcd {
            contamination,
            min: Vec::new(),
            range: Vec::new(),
            data: Vec::new(),
            flipped: false,
            state: None,
        }
    }

    fn normalize(&self, x: &Matrix) -> Vec<Vec<f64>> {
        x.iter()
            .map(|r| {
                r.iter()
                    .enumerate()
                    .map(|(j, v)| (v - self.min[j]) / self.range[j])
                    .collect()
            })
            .collect()
    }
}

impl Detector for Qmcd {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        let d = check_array(x)?;
        self.min = (0..d)
            .map(|j| x.iter().map(|r| r[j]).fold(f64::INFINITY, f64::min))
            .collect();
        self.range = (0..d)
            .map(|j| {
                let r = x.iter().map(|r| r[j]).fold(f64::NEG_INFINITY, f64::max) - self.min[j];
                if r == 0.0 {
                    1.0
                } else {
                    r
                }
            })
            .collect();
        self.data = self.normalize(x);
        let mut scores = wrap_around_discrepancy(&self.data, &self.data);
        let (s, k) = (skew(&scores), kurtosis(&scores));
        self.flipped = s < 0.0 || k < 0.0;
        if self.flipped {
            let (mn, mx) = scores
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                    (a.min(v), b.max(v))
                });
            scores.iter_mut().for_each(|v| *v = mx + mn - *v);
        }
        self.state = Some(FitState::from_contamination(scores, self.contamination));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        let state = self.fitted()?;
        check_features(x, self.min.len())?;
        let mut scores = wrap_around_discrepancy(&self.data, &self.normalize(x));
        if self.flipped {
            let s = &state.decision_scores;
            let (mn, mx) = s
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                    (a.min(v), b.max(v))
                });
            scores.iter_mut().for_each(|v| *v = mx + mn - *v);
        }
        Ok(scores)
    }

    crate::state_accessor!();
}
