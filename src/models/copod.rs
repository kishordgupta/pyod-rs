//! COPOD: Copula-Based Outlier Detection (Li et al., ICDM 2020).
//! Port of `pyod.models.copod`.

use super::ecod::{score_with_train, tail_scores, TailAggregation};
use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, Result,
};

#[derive(Debug, Clone)]
pub struct Copod {
    pub contamination: f64,
    x_train: Vec<Vec<f64>>,
    state: Option<FitState>,
}

impl Default for Copod {
    fn default() -> Self {
        Self::new(0.1)
    }
}

impl Copod {
    pub fn new(contamination: f64) -> Self {
        Copod {
            contamination,
            x_train: Vec::new(),
            state: None,
        }
    }
}

impl Detector for Copod {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        check_array(x)?;
        let scores = tail_scores(x, x.len(), TailAggregation::SkewOrMean);
        self.x_train = x.to_vec();
        self.state = Some(FitState::from_contamination(scores, self.contamination));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.x_train[0].len())?;
        Ok(score_with_train(
            &self.x_train,
            x,
            TailAggregation::SkewOrMean,
        ))
    }

    fn state(&self) -> Option<&FitState> {
        self.state.as_ref()
    }
}
