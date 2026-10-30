//! ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution
//! Functions (Li et al., TKDE 2022). Port of `pyod.models.ecod`.

use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, Result,
};
use crate::utils::{column_ecdf, sign, skew};

/// How the left, right and skew-corrected tail probabilities are aggregated.
#[derive(Clone, Copy)]
pub(crate) enum TailAggregation {
    /// max(U_l, U_r, U_skew) — ECOD.
    Max,
    /// max(U_skew, (U_l + U_r) / 2) — COPOD.
    SkewOrMean,
}

/// Score the last `n_tail` rows of `x`, using every row of `x` to build the ECDFs.
pub(crate) fn tail_scores(x: &Matrix, n_tail: usize, agg: TailAggregation) -> Vec<f64> {
    let neg: Vec<Vec<f64>> = x.iter().map(|r| r.iter().map(|v| -v).collect()).collect();
    let left = column_ecdf(x);
    let right = column_ecdf(&neg);
    let d = x[0].len();
    let skew_sign: Vec<f64> = (0..d)
        .map(|j| sign(skew(&x.iter().map(|r| r[j]).collect::<Vec<_>>())))
        .collect();
    let start = x.len() - n_tail;
    (start..x.len())
        .map(|i| {
            (0..d)
                .map(|j| {
                    let ul = -left[i][j].ln();
                    let ur = -right[i][j].ln();
                    let s = skew_sign[j];
                    let uskew = ul * -sign(s - 1.0) + ur * sign(s + 1.0);
                    match agg {
                        TailAggregation::Max => ul.max(ur).max(uskew),
                        TailAggregation::SkewOrMean => uskew.max((ul + ur) / 2.0),
                    }
                })
                .sum()
        })
        .collect()
}

/// Score new rows against stored training rows, as PyOD does by concatenating them.
pub(crate) fn score_with_train(train: &Matrix, x: &Matrix, agg: TailAggregation) -> Vec<f64> {
    let all: Vec<Vec<f64>> = train.iter().chain(x.iter()).cloned().collect();
    tail_scores(&all, x.len(), agg)
}

#[derive(Debug, Clone)]
pub struct Ecod {
    pub contamination: f64,
    x_train: Vec<Vec<f64>>,
    state: Option<FitState>,
}

impl Default for Ecod {
    fn default() -> Self {
        Self::new(0.1)
    }
}

impl Ecod {
    pub fn new(contamination: f64) -> Self {
        Ecod {
            contamination,
            x_train: Vec::new(),
            state: None,
        }
    }
}

impl Detector for Ecod {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        check_array(x)?;
        let scores = tail_scores(x, x.len(), TailAggregation::Max);
        self.x_train = x.to_vec();
        self.state = Some(FitState::from_contamination(scores, self.contamination));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.x_train[0].len())?;
        Ok(score_with_train(&self.x_train, x, TailAggregation::Max))
    }

    fn state(&self) -> Option<&FitState> {
        self.state.as_ref()
    }
}
