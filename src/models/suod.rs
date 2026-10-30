//! SUOD: accelerating large-scale unsupervised outlier detection (Zhao et al., MLSys 2021).
//! Port of `pyod.models.suod` and the `suod` package it wraps (without the
//! cost-based parallel scheduler, which only affects speed).
//!
//! Members can be fitted on a Johnson–Lindenstrauss random projection of the
//! data, and expensive members can be replaced at prediction time by a random
//! forest regressor trained on their training scores. Member scores are
//! standardised and averaged (or maximised).

use super::feature_bagging::Combination;
use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};
use crate::linalg::matmul;
use crate::models::combination::{average, maximization};
use crate::models::{copod::Copod, hbos::Hbos, iforest::IForest, lof::Lof};
use crate::trees::{RandomForestRegressor, Regressor};
use crate::utils::{Rng, StandardScaler};

/// One SUOD member and how to accelerate it.
pub struct SuodMember {
    pub detector: Box<dyn Detector>,
    /// Fit on a JL random projection (suod's `rp_clf_list`: LOF, KNN, ABOD, COF).
    pub project: bool,
    /// Replace predictions with a regressor (suod's `approx_clf_list`: LOF, KNN, CBLOF, OCSVM).
    pub approximate: bool,
}

impl SuodMember {
    pub fn new(detector: Box<dyn Detector>, project: bool, approximate: bool) -> Self {
        SuodMember {
            detector,
            project,
            approximate,
        }
    }
}

pub struct Suod {
    pub members: Vec<SuodMember>,
    /// Projected dimension as a fraction of the original.
    pub target_dim_frac: f64,
    pub combination: Combination,
    pub contamination: f64,
    pub random_state: u64,
    projections: Vec<Option<Vec<Vec<f64>>>>,
    approximators: Vec<Option<RandomForestRegressor>>,
    scaler: Option<StandardScaler>,
    n_features: usize,
    state: Option<FitState>,
}

impl Default for Suod {
    /// PyOD's default members, with suod's default acceleration flags.
    fn default() -> Self {
        let members = vec![
            SuodMember::new(Box::new(Lof::new(15, 0.1)), true, true),
            SuodMember::new(Box::new(Lof::new(20, 0.1)), true, true),
            SuodMember::new(Box::new(Hbos::new(10, 0.1)), false, false),
            SuodMember::new(Box::new(Hbos::new(20, 0.1)), false, false),
            SuodMember::new(Box::new(Copod::default()), false, false),
            SuodMember::new(Box::new(IForest::new(50, 0.1, 1)), false, false),
            SuodMember::new(Box::new(IForest::new(100, 0.1, 2)), false, false),
            SuodMember::new(Box::new(IForest::new(150, 0.1, 3)), false, false),
        ];
        Self::new(members, 0.1)
    }
}

impl Suod {
    pub fn new(members: Vec<SuodMember>, contamination: f64) -> Self {
        Suod {
            members,
            target_dim_frac: 0.5,
            combination: Combination::Average,
            contamination,
            random_state: 0,
            projections: Vec::new(),
            approximators: Vec::new(),
            scaler: None,
            n_features: 0,
            state: None,
        }
    }

    fn project(x: &Matrix, p: &Option<Vec<Vec<f64>>>) -> Vec<Vec<f64>> {
        match p {
            Some(m) => matmul(x, m),
            None => x.to_vec(),
        }
    }

    fn combine(&self, per_sample: &[Vec<f64>]) -> Vec<f64> {
        match self.combination {
            Combination::Average => average(per_sample, None),
            Combination::Max => maximization(per_sample),
        }
    }
}

impl Detector for Suod {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        let d = check_array(x)?;
        if self.members.is_empty() || !(self.target_dim_frac > 0.0 && self.target_dim_frac <= 1.0) {
            return Err(OdError::InvalidParameter(
                "need members and target_dim_frac in (0, 1]".into(),
            ));
        }
        self.n_features = d;
        let k = ((self.target_dim_frac * d as f64) as usize).max(1);
        let mut rng = Rng::new(self.random_state);
        self.projections.clear();
        self.approximators.clear();
        let mut cols = Vec::new();
        for m in &mut self.members {
            // JL "basic" transform: N(0, 1) / sqrt(k), shape (d, k).
            let proj = m.project.then(|| {
                (0..d)
                    .map(|_| (0..k).map(|_| rng.normal() / (k as f64).sqrt()).collect())
                    .collect()
            });
            let xp = Self::project(x, &proj);
            m.detector.fit(&xp)?;
            let scores = m.detector.decision_scores()?.to_vec();
            self.approximators.push(m.approximate.then(|| {
                let mut rf = RandomForestRegressor::new(50, rng.next_u64());
                rf.fit(&xp, &scores);
                rf
            }));
            self.projections.push(proj);
            cols.push(scores);
        }
        let per_sample: Vec<Vec<f64>> = (0..x.len())
            .map(|i| cols.iter().map(|c| c[i]).collect())
            .collect();
        let scaler = StandardScaler::fit(&per_sample);
        let combined = self.combine(&scaler.transform(&per_sample));
        self.scaler = Some(scaler);
        self.state = Some(FitState::from_contamination(combined, self.contamination));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.n_features)?;
        let mut cols = Vec::new();
        for ((m, p), a) in self
            .members
            .iter()
            .zip(&self.projections)
            .zip(&self.approximators)
        {
            let xp = Self::project(x, p);
            cols.push(match a {
                Some(rf) => rf.predict(&xp),
                None => m.detector.decision_function(&xp)?,
            });
        }
        let per_sample: Vec<Vec<f64>> = (0..x.len())
            .map(|i| cols.iter().map(|c| c[i]).collect())
            .collect();
        Ok(self.combine(&self.scaler.as_ref().unwrap().transform(&per_sample)))
    }

    crate::state_accessor!();
}
