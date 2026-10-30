//! Kernel Density Estimation outlier detection (Latecki et al., 2007).
//! Port of `pyod.models.kde` (sklearn `KernelDensity`, Euclidean metric).
//! The score is the negative log density.

use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};
use crate::linalg::{lgamma, logsumexp};
use crate::utils::euclidean;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Kernel {
    #[default]
    Gaussian,
    Tophat,
    Epanechnikov,
    Exponential,
    Linear,
}

#[derive(Debug, Clone)]
pub struct Kde {
    pub bandwidth: f64,
    pub kernel: Kernel,
    pub contamination: f64,
    x_train: Vec<Vec<f64>>,
    state: Option<FitState>,
}

impl Default for Kde {
    fn default() -> Self {
        Self::new(1.0, 0.1)
    }
}

/// log of the volume of the unit n-ball.
fn log_vn(n: f64) -> f64 {
    0.5 * n * std::f64::consts::PI.ln() - lgamma(0.5 * n + 1.0)
}

impl Kde {
    pub fn new(bandwidth: f64, contamination: f64) -> Self {
        Kde {
            bandwidth,
            kernel: Kernel::Gaussian,
            contamination,
            x_train: Vec::new(),
            state: None,
        }
    }

    /// Port of sklearn's `_log_kernel_norm`.
    fn log_norm(&self, d: f64) -> f64 {
        let factor = match self.kernel {
            Kernel::Gaussian => 0.5 * d * (2.0 * std::f64::consts::PI).ln(),
            Kernel::Tophat => log_vn(d),
            Kernel::Epanechnikov => log_vn(d) + (2.0 / (d + 2.0)).ln(),
            Kernel::Exponential => (2.0 * std::f64::consts::PI).ln() + log_vn(d - 1.0) + lgamma(d),
            Kernel::Linear => log_vn(d) - (d + 1.0).ln(),
        };
        -factor - d * self.bandwidth.ln()
    }

    fn log_kernel(&self, dist: f64) -> f64 {
        let u = dist / self.bandwidth;
        match self.kernel {
            Kernel::Gaussian => -0.5 * u * u,
            Kernel::Tophat => {
                if u < 1.0 {
                    0.0
                } else {
                    f64::NEG_INFINITY
                }
            }
            Kernel::Epanechnikov => {
                if u < 1.0 {
                    (1.0 - u * u).ln()
                } else {
                    f64::NEG_INFINITY
                }
            }
            Kernel::Exponential => -u,
            Kernel::Linear => {
                if u < 1.0 {
                    (1.0 - u).ln()
                } else {
                    f64::NEG_INFINITY
                }
            }
        }
    }

    fn score(&self, x: &Matrix) -> Vec<f64> {
        let n = self.x_train.len() as f64;
        let norm = self.log_norm(self.x_train[0].len() as f64);
        x.iter()
            .map(|q| {
                let logs: Vec<f64> = self
                    .x_train
                    .iter()
                    .map(|t| self.log_kernel(euclidean(q, t)))
                    .collect();
                -(logsumexp(&logs) + norm - n.ln())
            })
            .collect()
    }
}

impl Detector for Kde {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        check_array(x)?;
        if self.bandwidth <= 0.0 {
            return Err(OdError::InvalidParameter(
                "bandwidth must be positive".into(),
            ));
        }
        self.x_train = x.to_vec();
        self.state = Some(FitState::from_contamination(
            self.score(x),
            self.contamination,
        ));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.x_train[0].len())?;
        Ok(self.score(x))
    }

    crate::state_accessor!();
}
