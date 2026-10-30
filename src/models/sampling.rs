//! Rapid distance-based outlier detection via sampling (Sugiyama & Borgwardt, 2013).
//! Port of `pyod.models.sampling`: the score is the distance to the nearest of a
//! random subset of training points.

use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};
use crate::utils::{euclidean, Rng};

#[derive(Debug, Clone)]
pub struct Sampling {
    pub subset_size: usize,
    pub contamination: f64,
    pub random_state: u64,
    subset: Vec<Vec<f64>>,
    state: Option<FitState>,
}

impl Default for Sampling {
    fn default() -> Self {
        Self::new(20, 0.1)
    }
}

impl Sampling {
    pub fn new(subset_size: usize, contamination: f64) -> Self {
        Sampling {
            subset_size,
            contamination,
            random_state: 0,
            subset: Vec::new(),
            state: None,
        }
    }

    fn score(&self, x: &Matrix) -> Vec<f64> {
        x.iter()
            .map(|r| {
                self.subset
                    .iter()
                    .map(|s| euclidean(r, s))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect()
    }
}

impl Detector for Sampling {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        check_array(x)?;
        if self.subset_size == 0 || self.subset_size > x.len() {
            return Err(OdError::InvalidParameter(
                "subset_size must be in [1, n_samples]".into(),
            ));
        }
        let mut rng = Rng::new(self.random_state);
        self.subset = rng
            .sample_indices(x.len(), self.subset_size)
            .iter()
            .map(|&i| x[i].clone())
            .collect();
        self.state = Some(FitState::from_contamination(
            self.score(x),
            self.contamination,
        ));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.subset[0].len())?;
        Ok(self.score(x))
    }

    crate::state_accessor!();
}
