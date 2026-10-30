//! SOD: Subspace Outlier Detection (Kriegel et al., 2009). Port of `pyod.models.sod`.
//!
//! As in PyOD, `decision_function` scores the given samples against each other.

use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};
use crate::utils::{k_nearest, numba_argsort};

#[derive(Debug, Clone)]
pub struct Sod {
    pub n_neighbors: usize,
    /// Size of the shared-nearest-neighbour reference set.
    pub ref_set: usize,
    /// Fraction of the average variance below which a dimension is "relevant".
    pub alpha: f64,
    pub contamination: f64,
    n_features: usize,
    state: Option<FitState>,
}

impl Default for Sod {
    fn default() -> Self {
        Sod {
            n_neighbors: 20,
            ref_set: 10,
            alpha: 0.8,
            contamination: 0.1,
            n_features: 0,
            state: None,
        }
    }
}

impl Sod {
    pub fn new(n_neighbors: usize, ref_set: usize, alpha: f64, contamination: f64) -> Self {
        Sod {
            n_neighbors,
            ref_set,
            alpha,
            contamination,
            ..Self::default()
        }
    }

    fn sod(&self, x: &Matrix) -> Vec<f64> {
        let n = x.len();
        let d = x[0].len();
        let knn: Vec<Vec<bool>> = (0..n)
            .map(|i| {
                let mut m = vec![false; n];
                k_nearest(x, &x[i], self.n_neighbors, Some(i))
                    .iter()
                    .for_each(|&(j, _)| m[j] = true);
                m
            })
            .collect();
        let knn_idx: Vec<Vec<usize>> = knn
            .iter()
            .map(|m| (0..n).filter(|&j| m[j]).collect())
            .collect();
        (0..n)
            .map(|i| {
                // Reference set: points sharing the most nearest neighbours with i,
                // ranked with PyOD's numba argsort so ties resolve identically.
                let mut shared: Vec<u32> = (0..n)
                    .map(|j| knn_idx[j].iter().filter(|&&p| knn[i][p]).count() as u32)
                    .collect();
                shared[i] = u32::MAX;
                let order = numba_argsort(&shared);
                let refs: Vec<&Vec<f64>> = order
                    .iter()
                    .rev()
                    .skip(1)
                    .take(self.ref_set)
                    .map(|&j| &x[j])
                    .collect();
                let m = refs.len() as f64;
                let means: Vec<f64> = (0..d)
                    .map(|k| refs.iter().map(|r| r[k]).sum::<f64>() / m)
                    .collect();
                let var: Vec<f64> = (0..d)
                    .map(|k| refs.iter().map(|r| (r[k] - means[k]).powi(2)).sum::<f64>() / m)
                    .collect();
                let var_total: f64 = var.iter().sum::<f64>() * m / self.ref_set as f64;
                let expect = self.alpha * var_total / d as f64;
                let relevant: Vec<usize> = (0..d).filter(|&k| var[k] < expect).collect();
                if relevant.is_empty() {
                    0.0
                } else {
                    (relevant
                        .iter()
                        .map(|&k| (x[i][k] - means[k]).powi(2))
                        .sum::<f64>()
                        / relevant.len() as f64)
                        .sqrt()
                }
            })
            .collect()
    }
}

impl Detector for Sod {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        self.n_features = check_array(x)?;
        if self.ref_set == 0 || self.ref_set > self.n_neighbors || self.n_neighbors >= x.len() {
            return Err(OdError::InvalidParameter(
                "need 1 <= ref_set <= n_neighbors < n_samples".into(),
            ));
        }
        self.state = Some(FitState::from_contamination(
            self.sod(x),
            self.contamination,
        ));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.n_features)?;
        Ok(self.sod(x))
    }

    crate::state_accessor!();
}
