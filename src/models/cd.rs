//! CD: Cook's Distance outlier detection (Cook, 1977). Port of `pyod.models.cd`.
//!
//! Each feature is regressed on the others with ordinary least squares; a
//! sample's score is its min-max-normalised Cook's distance averaged over features.
//! As in PyOD, `decision_function` keeps the fitted regressions but measures
//! leverage and residual variance on the data it is given.

use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};
use crate::linalg::{col_mean, matmul, pinvh, symmetric_eigen_values, transpose};

#[derive(Debug, Clone)]
struct Ols {
    coef: Vec<f64>,
    intercept: f64,
}

#[derive(Debug, Clone)]
pub struct Cd {
    pub contamination: f64,
    models: Vec<Ols>,
    state: Option<FitState>,
}

impl Default for Cd {
    fn default() -> Self {
        Self::new(0.1)
    }
}

fn drop_col(x: &Matrix, i: usize) -> Vec<Vec<f64>> {
    x.iter()
        .map(|r| {
            r.iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, v)| *v)
                .collect()
        })
        .collect()
}

/// Least squares with intercept (minimum-norm solution, like sklearn's `LinearRegression`).
fn ols(x: &Matrix, y: &[f64]) -> Ols {
    let xm = col_mean(x);
    let ym = y.iter().sum::<f64>() / y.len() as f64;
    let xc: Vec<Vec<f64>> = x
        .iter()
        .map(|r| r.iter().zip(&xm).map(|(a, b)| a - b).collect())
        .collect();
    let xt = transpose(&xc);
    let xtx = matmul(&xt, &xc);
    let xty: Vec<f64> = xt
        .iter()
        .map(|c| c.iter().zip(y).map(|(a, b)| a * (b - ym)).sum())
        .collect();
    let coef = crate::linalg::matvec(&pinvh(&xtx), &xty);
    let intercept = ym - coef.iter().zip(&xm).map(|(a, b)| a * b).sum::<f64>();
    Ols { coef, intercept }
}

/// Port of `_Cooks_dist`; `None` when the residual variance is zero or undefined.
fn cooks(x: &Matrix, y: &[f64], m: &Ols) -> Option<Vec<f64>> {
    let (n, p) = (x.len(), x[0].len());
    let xt = transpose(x);
    let gram = matmul(&xt, x);
    // leverage_i = x_i^T (X^T X)^+ x_i, i.e. diag(X pinv(X)).
    let ginv = pinvh(&gram);
    let leverage: Vec<f64> = x
        .iter()
        .map(|r| {
            crate::linalg::matvec(&ginv, r)
                .iter()
                .zip(r)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect();
    let sv = symmetric_eigen_values(&gram);
    let smax = sv.iter().fold(0.0f64, |m, v| m.max(v.max(0.0).sqrt()));
    let rank = sv
        .iter()
        .filter(|v| v.max(0.0).sqrt() > smax * n.max(p) as f64 * f64::EPSILON)
        .count();
    let resid: Vec<f64> = x
        .iter()
        .zip(y)
        .map(|(r, yi)| yi - (m.intercept + r.iter().zip(&m.coef).map(|(a, b)| a * b).sum::<f64>()))
        .collect();
    let mse = resid.iter().map(|r| r * r).sum::<f64>() / (n as f64 - rank as f64);
    if mse == 0.0 || !mse.is_finite() {
        return None;
    }
    let dist: Vec<f64> = resid
        .iter()
        .zip(&leverage)
        .map(|(r, h)| {
            let stud = r / mse.sqrt() / (1.0 - h).sqrt();
            stud * stud / p as f64 * (h / (1.0 - h))
        })
        .collect();
    let (lo, hi) = dist
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
            (a.min(v), b.max(v))
        });
    Some(dist.iter().map(|d| (d - lo) / (hi - lo)).collect())
}

impl Cd {
    pub fn new(contamination: f64) -> Self {
        Cd {
            contamination,
            models: Vec::new(),
            state: None,
        }
    }

    fn score(&self, x: &Matrix) -> Vec<f64> {
        let per_feature: Vec<Vec<f64>> = self
            .models
            .iter()
            .enumerate()
            .filter_map(|(i, m)| {
                cooks(
                    &drop_col(x, i),
                    &x.iter().map(|r| r[i]).collect::<Vec<_>>(),
                    m,
                )
            })
            .collect();
        (0..x.len())
            .map(|s| {
                let vals: Vec<f64> = per_feature
                    .iter()
                    .map(|f| f[s])
                    .filter(|v| !v.is_nan())
                    .collect();
                if vals.is_empty() {
                    f64::NAN
                } else {
                    vals.iter().sum::<f64>() / vals.len() as f64
                }
            })
            .collect()
    }
}

impl Detector for Cd {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        let d = check_array(x)?;
        if d < 2 {
            return Err(OdError::InvalidParameter(
                "CD needs at least 2 features".into(),
            ));
        }
        self.models = (0..d)
            .map(|i| ols(&drop_col(x, i), &x.iter().map(|r| r[i]).collect::<Vec<_>>()))
            .collect();
        self.state = Some(FitState::from_contamination(
            self.score(x),
            self.contamination,
        ));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.models.len())?;
        Ok(self.score(x))
    }

    crate::state_accessor!();
}
