//! Kernel PCA outlier detection (Hoffmann, 2007). Port of `pyod.models.kpca`.
//!
//! The score is the reconstruction error in feature space: the squared distance
//! of the centred feature vector minus its squared projection on the kept components.

use super::ocsvm::SvmKernel;
use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};
use crate::utils::{symmetric_eigen, Rng};

#[derive(Debug, Clone)]
pub struct Kpca {
    /// Components kept by the kernel PCA; `None` keeps every positive eigenvalue.
    pub n_components: Option<usize>,
    /// Leading components used for the projection; `None` uses all kept.
    pub n_selected_components: Option<usize>,
    pub kernel: SvmKernel,
    /// Kernel coefficient; `None` means `1 / n_features`.
    pub gamma: Option<f64>,
    /// Fit on a random subset of the training data.
    pub sampling: bool,
    pub subset_size: usize,
    pub contamination: f64,
    pub random_state: u64,
    x_fit: Vec<Vec<f64>>,
    gamma_: f64,
    /// Column means of the training kernel matrix (`K_fit_rows_`) and its overall mean.
    k_fit_rows: Vec<f64>,
    k_fit_all: f64,
    /// Eigenvectors scaled by 1/sqrt(lambda), one per selected component.
    scaled_alphas: Vec<Vec<f64>>,
    state: Option<FitState>,
}

impl Default for Kpca {
    fn default() -> Self {
        Kpca {
            n_components: None,
            n_selected_components: None,
            kernel: SvmKernel::Rbf,
            gamma: None,
            sampling: false,
            subset_size: 20,
            contamination: 0.1,
            random_state: 0,
            x_fit: Vec::new(),
            gamma_: 0.0,
            k_fit_rows: Vec::new(),
            k_fit_all: 0.0,
            scaled_alphas: Vec::new(),
            state: None,
        }
    }
}

impl Kpca {
    pub fn new(contamination: f64) -> Self {
        Kpca {
            contamination,
            ..Self::default()
        }
    }

    fn score(&self, x: &Matrix) -> Vec<f64> {
        let m = self.x_fit.len() as f64;
        x.iter()
            .map(|r| {
                let k: Vec<f64> = self
                    .x_fit
                    .iter()
                    .map(|t| self.kernel.eval(self.gamma_, r, t))
                    .collect();
                let row_mean = k.iter().sum::<f64>() / m;
                let kc: Vec<f64> = k
                    .iter()
                    .zip(&self.k_fit_rows)
                    .map(|(v, c)| v - c - row_mean + self.k_fit_all)
                    .collect();
                let proj: f64 = self
                    .scaled_alphas
                    .iter()
                    .map(|a| a.iter().zip(&kc).map(|(x, y)| x * y).sum::<f64>().powi(2))
                    .sum();
                let potential =
                    self.kernel.eval(self.gamma_, r, r) - 2.0 * row_mean + self.k_fit_all;
                potential - proj
            })
            .collect()
    }
}

impl Detector for Kpca {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        let d = check_array(x)?;
        self.x_fit = if self.sampling {
            if self.subset_size == 0 || self.subset_size > x.len() {
                return Err(OdError::InvalidParameter(
                    "subset_size must be in [1, n_samples]".into(),
                ));
            }
            Rng::new(self.random_state)
                .sample_indices(x.len(), self.subset_size)
                .iter()
                .map(|&i| x[i].clone())
                .collect()
        } else {
            x.to_vec()
        };
        self.gamma_ = self.gamma.unwrap_or(1.0 / d as f64);
        let xf = &self.x_fit;
        let n = xf.len();
        let k: Vec<Vec<f64>> = xf
            .iter()
            .map(|a| {
                xf.iter()
                    .map(|b| self.kernel.eval(self.gamma_, a, b))
                    .collect()
            })
            .collect();
        self.k_fit_rows = (0..n)
            .map(|j| k.iter().map(|r| r[j]).sum::<f64>() / n as f64)
            .collect();
        self.k_fit_all = self.k_fit_rows.iter().sum::<f64>() / n as f64;
        let kc: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| k[i][j] - self.k_fit_rows[j] - self.k_fit_rows[i] + self.k_fit_all)
                    .collect()
            })
            .collect();
        let (vals, vecs) = symmetric_eigen(&kc);
        // sklearn's `_check_psd_eigenvalues`: drop non-positive and negligible eigenvalues.
        let max = vals.first().copied().unwrap_or(0.0).max(0.0);
        let n_comp = self.n_components.unwrap_or(n).min(n);
        let kept: Vec<(f64, Vec<f64>)> = vals
            .into_iter()
            .zip(vecs)
            .take(n_comp)
            .filter(|(l, _)| *l > 1e-12 * max && *l > 0.0)
            .collect();
        let n_sel = self.n_selected_components.unwrap_or(n_comp).min(kept.len());
        self.scaled_alphas = kept[..n_sel]
            .iter()
            .map(|(l, v)| v.iter().map(|a| a / l.sqrt()).collect())
            .collect();
        let scores = self.score(x);
        self.state = Some(FitState::from_contamination(scores, self.contamination));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.x_fit[0].len())?;
        Ok(self.score(x))
    }

    crate::state_accessor!();
}
