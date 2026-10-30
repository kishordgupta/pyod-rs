//! SOS: Stochastic Outlier Selection (Janssens et al., 2012). Port of `pyod.models.sos`.
//!
//! As in PyOD, `decision_function` scores the given samples against each other.

use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, Result,
};
use crate::utils::euclidean;

#[derive(Debug, Clone)]
pub struct Sos {
    pub perplexity: f64,
    /// Tolerance of the binary search for each point's precision.
    pub eps: f64,
    pub contamination: f64,
    n_features: usize,
    state: Option<FitState>,
}

impl Default for Sos {
    fn default() -> Self {
        Sos {
            perplexity: 4.5,
            eps: 1e-5,
            contamination: 0.1,
            n_features: 0,
            state: None,
        }
    }
}

fn perplexity(d: &[f64], beta: f64) -> (f64, Vec<f64>) {
    let a: Vec<f64> = d.iter().map(|v| (-v * beta).exp()).collect();
    let sum: f64 = a.iter().sum();
    let h = sum.ln() + beta * d.iter().zip(&a).map(|(x, y)| x * y).sum::<f64>() / sum;
    (h, a)
}

impl Sos {
    pub fn new(perplexity: f64, contamination: f64) -> Self {
        Sos {
            perplexity,
            contamination,
            ..Self::default()
        }
    }

    fn sos(&self, x: &Matrix) -> Vec<f64> {
        let n = x.len();
        let log_u = self.perplexity.ln();
        let mut outlier = vec![1.0; n];
        for i in 0..n {
            let di: Vec<f64> = (0..n)
                .filter(|&j| j != i)
                .map(|j| euclidean(&x[i], &x[j]))
                .collect();
            let (mut beta, mut bmin, mut bmax) = (1.0, f64::NEG_INFINITY, f64::INFINITY);
            let (mut h, mut a) = perplexity(&di, beta);
            let mut diff = h - log_u;
            let mut tries = 0;
            while (diff.is_nan() || diff.abs() > self.eps) && tries < 5000 {
                if diff.is_nan() {
                    beta /= 10.0;
                } else if diff > 0.0 {
                    bmin = beta;
                    beta = if bmax.is_infinite() {
                        beta * 2.0
                    } else {
                        (beta + bmax) / 2.0
                    };
                } else {
                    bmax = beta;
                    beta = if bmin.is_infinite() {
                        beta / 2.0
                    } else {
                        (beta + bmin) / 2.0
                    };
                }
                (h, a) = perplexity(&di, beta);
                diff = h - log_u;
                tries += 1;
            }
            // Binding probabilities from i to every j; outlier prob = prod_i (1 - b_ij).
            let sum: f64 = a.iter().sum();
            for (k, j) in (0..n).filter(|&j| j != i).enumerate() {
                outlier[j] *= 1.0 - a[k] / sum;
            }
        }
        outlier
    }
}

impl Detector for Sos {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        self.n_features = check_array(x)?;
        self.state = Some(FitState::from_contamination(
            self.sos(x),
            self.contamination,
        ));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.n_features)?;
        Ok(self.sos(x))
    }

    crate::state_accessor!();
}
