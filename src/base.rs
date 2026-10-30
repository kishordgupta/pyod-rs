//! Port of `pyod.models.base.BaseDetector`.

use crate::utils::{erf, mean, percentile, std_dev};
use std::fmt;

/// Row-major data matrix: one `Vec<f64>` per sample.
pub type Matrix = [Vec<f64>];

#[derive(Debug, Clone, PartialEq)]
pub enum OdError {
    NotFitted,
    EmptyInput,
    RaggedInput,
    DimensionMismatch {
        expected: usize,
        got: usize,
    },
    InvalidParameter(String),
    /// The operation is not available for this detector (e.g. scoring new data
    /// with a transductive method).
    Unsupported(String),
}

impl fmt::Display for OdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OdError::NotFitted => write!(f, "detector has not been fitted"),
            OdError::EmptyInput => write!(f, "input contains no samples"),
            OdError::RaggedInput => write!(f, "rows have different lengths"),
            OdError::DimensionMismatch { expected, got } => {
                write!(f, "expected {expected} features, got {got}")
            }
            OdError::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            OdError::Unsupported(msg) => write!(f, "unsupported: {msg}"),
        }
    }
}

impl std::error::Error for OdError {}

pub type Result<T> = std::result::Result<T, OdError>;

/// Attributes PyOD sets in `_process_decision_scores`.
#[derive(Debug, Clone)]
pub struct FitState {
    /// Outlier scores of the training data (`decision_scores_`).
    pub decision_scores: Vec<f64>,
    /// Score above which a sample is labelled an outlier (`threshold_`).
    pub threshold: f64,
    /// Binary labels of the training data, 1 = outlier (`labels_`).
    pub labels: Vec<u8>,
    pub mu: f64,
    pub sigma: f64,
    /// Expected outlier fraction used for confidence estimates.
    pub contamination: f64,
}

impl FitState {
    /// Threshold at the `(1 - contamination)` percentile of the training scores.
    pub fn from_contamination(scores: Vec<f64>, contamination: f64) -> Self {
        let threshold = percentile(&scores, 100.0 * (1.0 - contamination));
        let mut state = Self::from_threshold(scores, threshold);
        state.contamination = contamination;
        state
    }

    pub fn from_threshold(scores: Vec<f64>, threshold: f64) -> Self {
        let labels: Vec<u8> = scores.iter().map(|&s| (s > threshold) as u8).collect();
        let mu = mean(&scores);
        let sigma = std_dev(&scores);
        let contamination =
            labels.iter().map(|&l: &u8| l as f64).sum::<f64>() / scores.len() as f64;
        FitState {
            decision_scores: scores,
            threshold,
            labels,
            mu,
            sigma,
            contamination,
        }
    }
}

/// How `predict_proba` maps raw scores into [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProbaMethod {
    /// Min-max scaling against the training scores.
    #[default]
    Linear,
    /// Gaussian scaling (Kriegel et al., 2011).
    Unify,
}

/// Common interface for all outlier detectors. Larger scores mean more outlying.
pub trait Detector {
    /// Fit the detector on unlabelled data.
    fn fit(&mut self, x: &Matrix) -> Result<()>;

    /// Raw outlier scores for `x` using the fitted detector.
    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>>;

    /// Training state, or `None` before `fit`.
    fn state(&self) -> Option<&FitState>;

    fn fitted(&self) -> Result<&FitState> {
        self.state().ok_or(OdError::NotFitted)
    }

    fn decision_scores(&self) -> Result<&[f64]> {
        Ok(&self.fitted()?.decision_scores)
    }

    fn threshold(&self) -> Result<f64> {
        Ok(self.fitted()?.threshold)
    }

    fn labels(&self) -> Result<&[u8]> {
        Ok(&self.fitted()?.labels)
    }

    /// Fit, then return the training labels.
    fn fit_predict(&mut self, x: &Matrix) -> Result<Vec<u8>> {
        self.fit(x)?;
        Ok(self.fitted()?.labels.clone())
    }

    /// Binary labels (1 = outlier) for `x`.
    fn predict(&self, x: &Matrix) -> Result<Vec<u8>> {
        let threshold = self.fitted()?.threshold;
        Ok(self
            .decision_function(x)?
            .into_iter()
            .map(|s| (s > threshold) as u8)
            .collect())
    }

    /// Probability of each sample being an outlier.
    fn predict_proba(&self, x: &Matrix, method: ProbaMethod) -> Result<Vec<f64>> {
        let state = self.fitted()?;
        let scores = self.decision_function(x)?;
        let probs = match method {
            ProbaMethod::Linear => {
                let lo = state
                    .decision_scores
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min);
                let hi = state
                    .decision_scores
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                let range = if hi - lo == 0.0 { 1.0 } else { hi - lo };
                scores
                    .iter()
                    .map(|s| ((s - lo) / range).clamp(0.0, 1.0))
                    .collect()
            }
            ProbaMethod::Unify => scores
                .iter()
                .map(|s| erf((s - state.mu) / (state.sigma * 2f64.sqrt())).clamp(0.0, 1.0))
                .collect(),
        };
        Ok(probs)
    }

    /// Confidence of each prediction (Perini et al., 2020), in [0, 1].
    fn predict_confidence(&self, x: &Matrix) -> Result<Vec<f64>> {
        let state = self.fitted()?;
        let n = state.decision_scores.len() as f64;
        let mut train = state.decision_scores.clone();
        train.sort_by(f64::total_cmp);
        let threshold = state.threshold;
        let scores = self.decision_function(x)?;
        Ok(scores
            .iter()
            .map(|&s| {
                let count = train.partition_point(|&t| t <= s) as f64;
                let p = (1.0 + count) / (2.0 + n);
                let k = n as u64 - (n * state.contamination) as u64;
                let conf = 1.0 - binom_cdf(k, n as u64, p);
                if s > threshold {
                    conf
                } else {
                    1.0 - conf
                }
            })
            .collect())
    }
}

/// Binomial CDF P(X <= k) for X ~ Bin(n, p), evaluated in log space.
fn binom_cdf(k: u64, n: u64, p: f64) -> f64 {
    if p <= 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return if k >= n { 1.0 } else { 0.0 };
    }
    let (lp, lq) = (p.ln(), (1.0 - p).ln());
    let mut log_coef = 0.0; // ln C(n, 0)
    let mut total = 0.0;
    for i in 0..=k.min(n) {
        if i > 0 {
            log_coef += ((n - i + 1) as f64).ln() - (i as f64).ln();
        }
        total += (log_coef + i as f64 * lp + (n - i) as f64 * lq).exp();
    }
    total.min(1.0)
}

/// Validate a data matrix and return its number of features.
pub fn check_array(x: &Matrix) -> Result<usize> {
    let first = x.first().ok_or(OdError::EmptyInput)?;
    let d = first.len();
    if d == 0 {
        return Err(OdError::EmptyInput);
    }
    if x.iter().any(|row| row.len() != d) {
        return Err(OdError::RaggedInput);
    }
    Ok(d)
}

/// Validate that `x` has the `expected` number of features.
pub fn check_features(x: &Matrix, expected: usize) -> Result<()> {
    let got = check_array(x)?;
    if got != expected {
        return Err(OdError::DimensionMismatch { expected, got });
    }
    Ok(())
}

pub fn check_contamination(c: f64) -> Result<()> {
    if !(c > 0.0 && c <= 0.5) {
        return Err(OdError::InvalidParameter(format!(
            "contamination must be in (0, 0.5], got {c}"
        )));
    }
    Ok(())
}

/// Implements `Detector::state` for a struct with a `state: Option<FitState>` field.
#[macro_export]
macro_rules! state_accessor {
    () => {
        fn state(&self) -> Option<&$crate::base::FitState> {
            self.state.as_ref()
        }
    };
}

/// Builds a fresh, unfitted detector from a random seed. Ensembles use this to
/// create their members, so any `Detector`, including user-defined ones, can be plugged in.
pub type DetectorFactory = Box<dyn Fn(u64) -> Box<dyn Detector>>;

/// Chooses a threshold from training scores, as PyOD's PyThresh integration does.
/// Implement this to plug in your own thresholding method.
pub trait Thresholder {
    fn threshold(&self, scores: &[f64]) -> f64;
}

impl<F: Fn(&[f64]) -> f64> Thresholder for F {
    fn threshold(&self, scores: &[f64]) -> f64 {
        self(scores)
    }
}

/// Re-threshold a fitted detector's training scores with `t`; returns the new
/// threshold and training labels. Label new data with
/// `decision_function(x)` scores `> threshold`.
pub fn apply_thresholder(det: &dyn Detector, t: &dyn Thresholder) -> Result<(f64, Vec<u8>)> {
    let scores = det.decision_scores()?;
    let th = t.threshold(scores);
    Ok((th, scores.iter().map(|&s| (s > th) as u8).collect()))
}
