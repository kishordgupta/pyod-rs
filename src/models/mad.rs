//! MAD: Median Absolute Deviation for univariate data (Iglewicz & Hoaglin, 1993).
//! Port of `pyod.models.mad`. Labels come from a fixed modified-z-score threshold,
//! not from contamination.

use crate::base::{check_features, Detector, FitState, Matrix, Result};
use crate::utils::median;

#[derive(Debug, Clone)]
pub struct Mad {
    /// Modified z-score above which a sample is an outlier.
    pub threshold: f64,
    median: f64,
    median_diff: f64,
    state: Option<FitState>,
}

impl Default for Mad {
    fn default() -> Self {
        Self::new(3.5)
    }
}

impl Mad {
    pub fn new(threshold: f64) -> Self {
        Mad {
            threshold,
            median: 0.0,
            median_diff: 0.0,
            state: None,
        }
    }

    fn score(&self, x: &Matrix) -> Vec<f64> {
        x.iter()
            .map(|r| {
                let z = 0.6745 * (r[0] - self.median).abs() / self.median_diff;
                if z.is_finite() {
                    z
                } else if z.is_nan() {
                    0.0
                } else {
                    f64::MAX
                }
            })
            .collect()
    }
}

impl Detector for Mad {
    /// `x` must have exactly one feature.
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_features(x, 1)?;
        let col: Vec<f64> = x.iter().map(|r| r[0]).collect();
        self.median = median(&col);
        let diffs: Vec<f64> = col.iter().map(|v| (v - self.median).abs()).collect();
        self.median_diff = median(&diffs);
        self.state = Some(FitState::from_threshold(self.score(x), self.threshold));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, 1)?;
        Ok(self.score(x))
    }

    fn state(&self) -> Option<&FitState> {
        self.state.as_ref()
    }
}
