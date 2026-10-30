//! k Nearest Neighbors outlier detection (Ramaswamy et al., 2000; Angiulli & Pizzuti, 2002).
//! Port of `pyod.models.knn` with Euclidean distance and brute-force search.

use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};
use crate::utils::{k_nearest, mean, median};

/// How the distances to the k neighbours become one score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KnnMethod {
    /// Distance to the k-th neighbour.
    #[default]
    Largest,
    Mean,
    Median,
}

#[derive(Debug, Clone)]
pub struct Knn {
    pub n_neighbors: usize,
    pub method: KnnMethod,
    pub contamination: f64,
    x_train: Vec<Vec<f64>>,
    state: Option<FitState>,
}

impl Default for Knn {
    fn default() -> Self {
        Knn {
            n_neighbors: 5,
            method: KnnMethod::Largest,
            contamination: 0.1,
            x_train: Vec::new(),
            state: None,
        }
    }
}

impl Knn {
    pub fn new(n_neighbors: usize, method: KnnMethod, contamination: f64) -> Self {
        Knn {
            n_neighbors,
            method,
            contamination,
            ..Self::default()
        }
    }

    fn aggregate(&self, neighbours: &[(usize, f64)]) -> f64 {
        let d: Vec<f64> = neighbours.iter().map(|n| n.1).collect();
        match self.method {
            KnnMethod::Largest => *d.last().unwrap(),
            KnnMethod::Mean => mean(&d),
            KnnMethod::Median => median(&d),
        }
    }
}

impl Detector for Knn {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        check_array(x)?;
        if self.n_neighbors == 0 || self.n_neighbors >= x.len() {
            return Err(OdError::InvalidParameter(format!(
                "n_neighbors must be in [1, n_samples - 1], got {}",
                self.n_neighbors
            )));
        }
        let scores = x
            .iter()
            .enumerate()
            .map(|(i, r)| self.aggregate(&k_nearest(x, r, self.n_neighbors, Some(i))))
            .collect();
        self.x_train = x.to_vec();
        self.state = Some(FitState::from_contamination(scores, self.contamination));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.x_train[0].len())?;
        Ok(x.iter()
            .map(|r| self.aggregate(&k_nearest(&self.x_train, r, self.n_neighbors, None)))
            .collect())
    }

    fn state(&self) -> Option<&FitState> {
        self.state.as_ref()
    }
}
