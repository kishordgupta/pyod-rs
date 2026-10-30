//! Gaussian Mixture Model outlier detection (Aggarwal, 2015).
//! Port of `pyod.models.gmm` (sklearn `GaussianMixture`, full covariances,
//! k-means initialisation). The score is the negative log likelihood.

use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};
use crate::linalg::{cholesky, forward_solve, kmeans, logsumexp, Mat};
use crate::utils::Rng;

#[derive(Debug, Clone)]
pub struct Gmm {
    pub n_components: usize,
    pub tol: f64,
    /// Added to covariance diagonals for stability.
    pub reg_covar: f64,
    pub max_iter: usize,
    pub n_init: usize,
    pub random_state: u64,
    pub contamination: f64,
    pub weights: Vec<f64>,
    pub means: Mat,
    pub covariances: Vec<Mat>,
    pub converged: bool,
    chol: Vec<Mat>,
    state: Option<FitState>,
}

impl Default for Gmm {
    fn default() -> Self {
        Self::new(1, 0.1)
    }
}

impl Gmm {
    pub fn new(n_components: usize, contamination: f64) -> Self {
        Gmm {
            n_components,
            tol: 1e-3,
            reg_covar: 1e-6,
            max_iter: 100,
            n_init: 1,
            random_state: 0,
            contamination,
            weights: Vec::new(),
            means: Vec::new(),
            covariances: Vec::new(),
            converged: false,
            chol: Vec::new(),
            state: None,
        }
    }

    /// log(w_k) + log N(x | mu_k, Sigma_k) for each component.
    fn weighted_log_prob(&self, r: &[f64]) -> Vec<f64> {
        let d = r.len() as f64;
        (0..self.weights.len())
            .map(|k| {
                let l = &self.chol[k];
                let diff: Vec<f64> = r.iter().zip(&self.means[k]).map(|(a, b)| a - b).collect();
                let z = forward_solve(l, &diff);
                let maha: f64 = z.iter().map(|v| v * v).sum();
                let log_det: f64 = (0..l.len()).map(|i| l[i][i].ln()).sum();
                self.weights[k].ln()
                    - 0.5 * (d * (2.0 * std::f64::consts::PI).ln() + maha)
                    - log_det
            })
            .collect()
    }

    fn m_step(&mut self, x: &Matrix, resp: &[Vec<f64>]) -> Result<()> {
        let (n, d, k) = (x.len(), x[0].len(), self.n_components);
        let nk: Vec<f64> = (0..k)
            .map(|c| resp.iter().map(|r| r[c]).sum::<f64>() + 10.0 * f64::EPSILON)
            .collect();
        self.weights = nk.iter().map(|v| v / n as f64).collect();
        self.means = (0..k)
            .map(|c| {
                (0..d)
                    .map(|j| x.iter().zip(resp).map(|(r, p)| p[c] * r[j]).sum::<f64>() / nk[c])
                    .collect()
            })
            .collect();
        self.covariances = (0..k)
            .map(|c| {
                let mut cov = vec![vec![0.0; d]; d];
                for (r, p) in x.iter().zip(resp) {
                    for i in 0..d {
                        let di = r[i] - self.means[c][i];
                        for j in i..d {
                            cov[i][j] += p[c] * di * (r[j] - self.means[c][j]);
                        }
                    }
                }
                for i in 0..d {
                    for j in i..d {
                        cov[i][j] /= nk[c];
                        cov[j][i] = cov[i][j];
                    }
                    cov[i][i] += self.reg_covar;
                }
                cov
            })
            .collect();
        self.chol = self
            .covariances
            .iter()
            .map(|c| {
                cholesky(c).ok_or_else(|| {
                    OdError::InvalidParameter("ill-defined covariance; increase reg_covar".into())
                })
            })
            .collect::<Result<_>>()?;
        Ok(())
    }

    /// One EM run from a k-means initialisation; returns the final mean log likelihood.
    fn em(&mut self, x: &Matrix, rng: &mut Rng) -> Result<f64> {
        let (_, labels) = kmeans(x, self.n_components, 1, 300, rng);
        let mut resp: Vec<Vec<f64>> = labels
            .iter()
            .map(|&l| {
                (0..self.n_components)
                    .map(|c| (c == l) as u8 as f64)
                    .collect()
            })
            .collect();
        self.m_step(x, &resp)?;
        let mut lower = f64::NEG_INFINITY;
        self.converged = false;
        for _ in 0..self.max_iter {
            let prev = lower;
            let mut total = 0.0;
            for (i, r) in x.iter().enumerate() {
                let wlp = self.weighted_log_prob(r);
                let norm = logsumexp(&wlp);
                total += norm;
                resp[i] = wlp.iter().map(|v| (v - norm).exp()).collect();
            }
            lower = total / x.len() as f64;
            self.m_step(x, &resp)?;
            if (lower - prev).abs() < self.tol {
                self.converged = true;
                break;
            }
        }
        Ok(lower)
    }

    fn score(&self, x: &Matrix) -> Vec<f64> {
        x.iter()
            .map(|r| -logsumexp(&self.weighted_log_prob(r)))
            .collect()
    }
}

impl Detector for Gmm {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        check_array(x)?;
        if self.n_components == 0 || self.n_components > x.len() {
            return Err(OdError::InvalidParameter(
                "n_components must be in [1, n_samples]".into(),
            ));
        }
        let mut rng = Rng::new(self.random_state);
        let mut best: Option<(f64, Gmm)> = None;
        for _ in 0..self.n_init.max(1) {
            let ll = self.em(x, &mut rng)?;
            if best.as_ref().is_none_or(|b| ll > b.0) {
                best = Some((ll, self.clone()));
            }
        }
        *self = best.unwrap().1;
        self.state = Some(FitState::from_contamination(
            self.score(x),
            self.contamination,
        ));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.means[0].len())?;
        Ok(self.score(x))
    }

    crate::state_accessor!();
}
