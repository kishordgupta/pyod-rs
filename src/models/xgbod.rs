//! XGBOD: improving supervised outlier detection with unsupervised representation
//! learning (Zhao & Hryniewicki, IJCNN 2018). Port of `pyod.models.xgbod`.
//!
//! A bank of unsupervised detectors produces extra features (outlier scores),
//! which are appended to the data before training a supervised classifier.
//! The classifier is pluggable via [`Classifier`]; the default is the crate's
//! XGBoost-style [`GradientBoostingClassifier`] with XGBOD's defaults.

use crate::base::{check_array, check_features, Detector, FitState, Matrix, OdError, Result};
use crate::models::{
    hbos::Hbos,
    iforest::IForest,
    knn::{Knn, KnnMethod},
    lof::Lof,
    ocsvm::OcSvm,
};
use crate::trees::{Classifier, GradientBoostingClassifier};
use crate::utils::StandardScaler;

pub struct Xgbod {
    /// Unsupervised detectors and whether each is fitted on standardised data.
    /// `None` uses PyOD's default bank (KNN, LOF, HBOS, OCSVM and IForest variants).
    pub estimator_list: Option<Vec<(Box<dyn Detector>, bool)>>,
    pub classifier: Box<dyn Classifier>,
    pub random_state: u64,
    estimators: Vec<(Box<dyn Detector>, bool)>,
    scaler: Option<StandardScaler>,
    n_features: usize,
    state: Option<FitState>,
}

impl Default for Xgbod {
    fn default() -> Self {
        let mut gb = GradientBoostingClassifier::default();
        gb.max_depth = 3;
        gb.learning_rate = 0.1;
        Xgbod::new(Box::new(gb))
    }
}

impl Xgbod {
    pub fn new(classifier: Box<dyn Classifier>) -> Self {
        Xgbod {
            estimator_list: None,
            classifier,
            random_state: 0,
            estimators: Vec::new(),
            scaler: None,
            n_features: 0,
            state: None,
        }
    }

    /// Port of `_init_detectors`.
    fn default_bank(n: usize, seed: u64) -> Vec<(Box<dyn Detector>, bool)> {
        let mut bank: Vec<(Box<dyn Detector>, bool)> = Vec::new();
        for k in [1, 3, 5, 10, 20, 30, 40, 50].into_iter().filter(|&k| k < n) {
            bank.push((Box::new(Knn::new(k, KnnMethod::Largest, 0.1)), true));
            bank.push((Box::new(Lof::new(k, 0.1)), true));
        }
        for b in [5, 10, 15, 20, 25, 30, 50] {
            bank.push((Box::new(Hbos::new(b, 0.1)), false));
        }
        for nu in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99] {
            bank.push((Box::new(OcSvm::new(nu, 0.1)), true));
        }
        for t in [10, 20, 50, 70, 100, 150, 200] {
            bank.push((Box::new(IForest::new(t, 0.1, seed)), false));
        }
        bank
    }

    fn augment(&self, x: &Matrix) -> Result<Vec<Vec<f64>>> {
        let xn = self.scaler.as_ref().unwrap().transform(x);
        let cols: Vec<Vec<f64>> = self
            .estimators
            .iter()
            .map(|(e, std)| e.decision_function(if *std { &xn } else { x }))
            .collect::<Result<_>>()?;
        Ok(x.iter()
            .enumerate()
            .map(|(i, r)| r.iter().copied().chain(cols.iter().map(|c| c[i])).collect())
            .collect())
    }

    /// Fit on labelled data (`y[i] == 1` for outliers).
    pub fn fit_supervised(&mut self, x: &Matrix, y: &[u8]) -> Result<()> {
        self.n_features = check_array(x)?;
        if y.len() != x.len() {
            return Err(OdError::InvalidParameter(
                "y must have one label per sample".into(),
            ));
        }
        self.estimators = self
            .estimator_list
            .take()
            .unwrap_or_else(|| Self::default_bank(x.len(), self.random_state));
        if self.estimators.is_empty() {
            return Err(OdError::InvalidParameter(
                "need at least one unsupervised estimator".into(),
            ));
        }
        let scaler = StandardScaler::fit(x);
        let xn = scaler.transform(x);
        self.scaler = Some(scaler);
        for (e, std) in &mut self.estimators {
            e.fit(if *std { &xn } else { x })?;
        }
        let xa = self.augment(x)?;
        self.classifier.fit(&xa, y);
        let scores = self.classifier.predict_proba(&xa);
        self.state = Some(FitState::from_threshold(scores, 0.5));
        Ok(())
    }
}

impl Detector for Xgbod {
    /// XGBOD is supervised; use [`Xgbod::fit_supervised`].
    fn fit(&mut self, _x: &Matrix) -> Result<()> {
        Err(OdError::InvalidParameter(
            "XGBOD needs labels: call fit_supervised(x, y)".into(),
        ))
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.n_features)?;
        Ok(self.classifier.predict_proba(&self.augment(x)?))
    }

    crate::state_accessor!();
}
