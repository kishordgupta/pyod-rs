//! Feature Bagging (Lazarevic & Kumar, KDD 2005). Port of `pyod.models.feature_bagging`.
//!
//! Each member is fitted on a random subset of features; member scores are
//! averaged (or maximised) without normalisation, as in PyOD.

use crate::base::{
    check_array, check_contamination, check_features, Detector, DetectorFactory, FitState, Matrix,
    OdError, Result,
};
use crate::models::combination::{average, maximization};
use crate::models::lof::Lof;
use crate::utils::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Combination {
    #[default]
    Average,
    Max,
}

pub struct FeatureBagging {
    pub base_estimator: DetectorFactory,
    pub n_estimators: usize,
    /// Upper bound on features per member, as a fraction of all features.
    pub max_features: f64,
    /// Draw features with replacement.
    pub bootstrap_features: bool,
    pub combination: Combination,
    pub contamination: f64,
    pub random_state: u64,
    estimators: Vec<(Box<dyn Detector>, Vec<usize>)>,
    n_features: usize,
    state: Option<FitState>,
}

impl Default for FeatureBagging {
    fn default() -> Self {
        Self::new(Box::new(|_| Box::new(Lof::default())), 10, 0.1)
    }
}

/// Port of `generate_bagging_indices`: draw a feature count in `[min, max)`, then the features.
pub(crate) fn bagging_indices(
    rng: &mut Rng,
    bootstrap: bool,
    n_features: usize,
    min: usize,
    max: usize,
) -> Vec<usize> {
    let n_drawn = min + rng.below(max.saturating_sub(min).max(1));
    if bootstrap {
        (0..n_drawn).map(|_| rng.below(n_features)).collect()
    } else {
        rng.sample_indices(n_features, n_drawn)
    }
}

fn select(x: &Matrix, features: &[usize]) -> Vec<Vec<f64>> {
    x.iter()
        .map(|r| features.iter().map(|&f| r[f]).collect())
        .collect()
}

impl FeatureBagging {
    pub fn new(base_estimator: DetectorFactory, n_estimators: usize, contamination: f64) -> Self {
        FeatureBagging {
            base_estimator,
            n_estimators,
            max_features: 1.0,
            bootstrap_features: false,
            combination: Combination::Average,
            contamination,
            random_state: 0,
            estimators: Vec::new(),
            n_features: 0,
            state: None,
        }
    }

    fn combine(&self, per_sample: &[Vec<f64>]) -> Vec<f64> {
        match self.combination {
            Combination::Average => average(per_sample, None),
            Combination::Max => maximization(per_sample),
        }
    }
}

impl Detector for FeatureBagging {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        let d = check_array(x)?;
        self.n_features = d;
        let min = d / 2;
        let max = (self.max_features * d as f64) as usize;
        if min == 0 || max < min || self.n_estimators == 0 {
            return Err(OdError::InvalidParameter(
                "FeatureBagging needs >= 2 features, max_features >= 0.5 and n_estimators >= 1"
                    .into(),
            ));
        }
        let mut rng = Rng::new(self.random_state);
        self.estimators.clear();
        for _ in 0..self.n_estimators {
            let seed = rng.next_u64();
            let features = bagging_indices(
                &mut Rng::new(seed),
                self.bootstrap_features,
                d,
                min,
                max + 1,
            );
            let mut det = (self.base_estimator)(seed);
            det.fit(&select(x, &features))?;
            self.estimators.push((det, features));
        }
        let cols: Vec<&[f64]> = self
            .estimators
            .iter()
            .map(|(e, _)| e.decision_scores())
            .collect::<Result<_>>()?;
        let per_sample: Vec<Vec<f64>> = (0..x.len())
            .map(|i| cols.iter().map(|c| c[i]).collect())
            .collect();
        self.state = Some(FitState::from_contamination(
            self.combine(&per_sample),
            self.contamination,
        ));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.n_features)?;
        let cols: Vec<Vec<f64>> = self
            .estimators
            .iter()
            .map(|(e, f)| e.decision_function(&select(x, f)))
            .collect::<Result<_>>()?;
        let per_sample: Vec<Vec<f64>> = (0..x.len())
            .map(|i| cols.iter().map(|c| c[i]).collect())
            .collect();
        Ok(self.combine(&per_sample))
    }

    crate::state_accessor!();
}
