//! PCA-based outlier detection (Shyu et al., 2003). Port of `pyod.models.pca`.
//!
//! The score is the sum, over the selected (lowest-variance) eigenvectors, of each
//! sample's distance to the eigenvector divided by its explained-variance ratio.

use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};
use crate::utils::{euclidean, symmetric_eigen, StandardScaler};

#[derive(Debug, Clone)]
pub struct Pca {
    /// Components to keep; `None` keeps `min(n_samples, n_features)`.
    pub n_components: Option<usize>,
    /// Components used for scoring, taken from the low-variance end; `None` uses all kept.
    pub n_selected_components: Option<usize>,
    /// Weight each component's distance by its explained-variance ratio.
    pub weighted: bool,
    pub standardization: bool,
    pub contamination: f64,
    scaler: Option<StandardScaler>,
    /// Principal axes, highest variance first (`components_`).
    pub components: Vec<Vec<f64>>,
    pub explained_variance_ratio: Vec<f64>,
    selected: Vec<(Vec<f64>, f64)>,
    n_features: usize,
    state: Option<FitState>,
}

impl Default for Pca {
    fn default() -> Self {
        Pca {
            n_components: None,
            n_selected_components: None,
            weighted: true,
            standardization: true,
            contamination: 0.1,
            scaler: None,
            components: Vec::new(),
            explained_variance_ratio: Vec::new(),
            selected: Vec::new(),
            n_features: 0,
            state: None,
        }
    }
}

impl Pca {
    pub fn new(contamination: f64) -> Self {
        Pca {
            contamination,
            ..Self::default()
        }
    }

    fn score(&self, x: &Matrix) -> Vec<f64> {
        let x = match &self.scaler {
            Some(s) => s.transform(x),
            None => x.to_vec(),
        };
        x.iter()
            .map(|r| self.selected.iter().map(|(v, w)| euclidean(r, v) / w).sum())
            .collect()
    }
}

impl Detector for Pca {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        let d = check_array(x)?;
        self.n_features = d;
        let n = x.len();
        let scaler = self.standardization.then(|| StandardScaler::fit(x));
        let xs = match &scaler {
            Some(s) => s.transform(x),
            None => x.to_vec(),
        };
        self.scaler = scaler;

        let mean: Vec<f64> = (0..d)
            .map(|j| xs.iter().map(|r| r[j]).sum::<f64>() / n as f64)
            .collect();
        let mut cov = vec![vec![0.0; d]; d];
        for r in &xs {
            for i in 0..d {
                for j in i..d {
                    cov[i][j] += (r[i] - mean[i]) * (r[j] - mean[j]);
                }
            }
        }
        for i in 0..d {
            for j in i..d {
                cov[i][j] /= (n.max(2) - 1) as f64;
                cov[j][i] = cov[i][j];
            }
        }
        let (values, mut vectors) = symmetric_eigen(&cov);
        // sklearn's svd_flip: make the largest-magnitude entry of each axis positive.
        for v in &mut vectors {
            let big = v
                .iter()
                .copied()
                .fold(0.0f64, |m, e| if e.abs() > m.abs() { e } else { m });
            if big < 0.0 {
                v.iter_mut().for_each(|e| *e = -*e);
            }
        }
        let total: f64 = values.iter().map(|v| v.max(0.0)).sum();
        let k = self.n_components.unwrap_or(n.min(d));
        if k == 0 || k > d.min(n) {
            return Err(OdError::InvalidParameter(format!(
                "n_components must be in [1, {}]",
                d.min(n)
            )));
        }
        let s = self.n_selected_components.unwrap_or(k);
        if s == 0 || s > k {
            return Err(OdError::InvalidParameter(format!(
                "n_selected_components must be in [1, {k}]"
            )));
        }
        self.components = vectors[..k].to_vec();
        self.explained_variance_ratio = values[..k].iter().map(|v| v.max(0.0) / total).collect();
        // Guard against zero-variance axes, which make PyOD divide by zero.
        self.selected = (k - s..k)
            .map(|i| {
                let w = if self.weighted {
                    self.explained_variance_ratio[i].max(f64::EPSILON)
                } else {
                    1.0
                };
                (self.components[i].clone(), w)
            })
            .collect();
        self.state = Some(FitState::from_contamination(
            self.score(x),
            self.contamination,
        ));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.n_features)?;
        Ok(self.score(x))
    }

    fn state(&self) -> Option<&FitState> {
        self.state.as_ref()
    }
}
