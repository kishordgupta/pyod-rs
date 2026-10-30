//! Matrix Profile discord detection (Yeh et al., ICDM 2016; STOMP).
//! Port of `pyod.models.ts_matrix_profile`. Transductive: only training scores exist.

use crate::base::{check_array, check_contamination, Detector, FitState, Matrix, OdError, Result};
use crate::ts::{combine_channels, map_scores_to_timestamps, ts_fit_state, Aggregation};

#[derive(Debug, Clone)]
pub struct MatrixProfile {
    pub window_size: usize,
    pub channel_aggregation: Aggregation,
    pub contamination: f64,
    state: Option<FitState>,
}

impl Default for MatrixProfile {
    fn default() -> Self {
        MatrixProfile {
            window_size: 50,
            channel_aggregation: Aggregation::Max,
            contamination: 0.1,
            state: None,
        }
    }
}

impl MatrixProfile {
    pub fn new(window_size: usize, contamination: f64) -> Self {
        MatrixProfile {
            window_size,
            contamination,
            ..Self::default()
        }
    }
}

/// z-normalised Euclidean distance from each subsequence to its nearest non-trivial match (STOMP).
pub(crate) fn matrix_profile(t: &[f64], m: usize) -> Vec<f64> {
    let n = t.len();
    let ns = n - m + 1;
    let ez = m / 4;
    let mut mu = vec![0.0; ns];
    let mut sigma = vec![0.0; ns];
    for i in 0..ns {
        let w = &t[i..i + m];
        mu[i] = w.iter().sum::<f64>() / m as f64;
        let var = w.iter().map(|v| v * v).sum::<f64>() / m as f64 - mu[i] * mu[i];
        sigma[i] = var.max(0.0).sqrt();
    }
    let constant: Vec<bool> = sigma.iter().map(|s| *s < 1e-10).collect();
    let mut mp = vec![f64::INFINITY; ns];
    let mut qt: Vec<f64> = (0..ns)
        .map(|i| (0..m).map(|k| t[k] * t[i + k]).sum())
        .collect();
    for j in 0..ns {
        if j > 0 {
            for i in (1..ns).rev() {
                qt[i] = qt[i - 1] - t[i - 1] * t[j - 1] + t[i + m - 1] * t[j + m - 1];
            }
            qt[0] = (0..m).map(|k| t[k] * t[j + k]).sum();
        }
        if constant[j] {
            continue;
        }
        let (lo, hi) = (j.saturating_sub(ez), (j + ez + 1).min(ns));
        for i in 0..ns {
            if constant[i] || (lo..hi).contains(&i) {
                continue;
            }
            let denom = m as f64 * sigma[i] * sigma[j];
            let corr = if denom > 0.0 {
                (qt[i] - m as f64 * mu[i] * mu[j]) / denom
            } else {
                0.0
            };
            let d = (2.0 * m as f64 * (1.0 - corr)).max(0.0).sqrt();
            if d < mp[i] {
                mp[i] = d;
            }
        }
    }
    mp
}

impl Detector for MatrixProfile {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        let d = check_array(x)?;
        let m = self.window_size;
        if m < 2 || x.len() < m + 1 {
            return Err(OdError::InvalidParameter(format!(
                "need at least window_size + 1 = {} timestamps",
                m + 1
            )));
        }
        let per: Vec<Vec<Option<f64>>> = (0..d)
            .map(|c| {
                let mp = matrix_profile(&x.iter().map(|r| r[c]).collect::<Vec<_>>(), m);
                map_scores_to_timestamps(&mp, m, 1, x.len(), self.channel_aggregation)
            })
            .collect();
        self.state = Some(ts_fit_state(
            &combine_channels(per, self.channel_aggregation),
            self.contamination,
        ));
        Ok(())
    }

    fn decision_function(&self, _x: &Matrix) -> Result<Vec<f64>> {
        Err(OdError::Unsupported(
            "MatrixProfile is transductive; use decision_scores() after fit".into(),
        ))
    }

    crate::state_accessor!();
}
