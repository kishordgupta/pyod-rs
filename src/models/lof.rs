//! LOF: Local Outlier Factor (Breunig et al., 2000).
//! Port of `pyod.models.lof` (which wraps sklearn's `LocalOutlierFactor`).

use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, Result,
};
use crate::utils::{euclidean, k_nearest};

#[derive(Debug, Clone)]
pub struct Lof {
    pub n_neighbors: usize,
    pub contamination: f64,
    x_train: Vec<Vec<f64>>,
    /// Distance from each training point to its k-th neighbour.
    k_distance: Vec<f64>,
    /// Local reachability density of each training point.
    lrd: Vec<f64>,
    k: usize,
    state: Option<FitState>,
}

impl Default for Lof {
    fn default() -> Self {
        Self::new(20, 0.1)
    }
}

impl Lof {
    pub fn new(n_neighbors: usize, contamination: f64) -> Self {
        Lof {
            n_neighbors,
            contamination,
            x_train: Vec::new(),
            k_distance: Vec::new(),
            lrd: Vec::new(),
            k: 0,
            state: None,
        }
    }

    fn local_reachability_density(&self, query: &[f64], neighbours: &[(usize, f64)]) -> f64 {
        let reach: f64 = neighbours
            .iter()
            .map(|&(j, _)| self.k_distance[j].max(euclidean(query, &self.x_train[j])))
            .sum();
        1.0 / (reach / neighbours.len() as f64 + 1e-10)
    }

    fn lof(&self, lrd: f64, neighbours: &[(usize, f64)]) -> f64 {
        neighbours.iter().map(|&(j, _)| self.lrd[j]).sum::<f64>() / neighbours.len() as f64 / lrd
    }
}

impl Detector for Lof {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        check_array(x)?;
        self.k = self.n_neighbors.clamp(1, x.len().saturating_sub(1).max(1));
        self.x_train = x.to_vec();
        let neighbours: Vec<Vec<(usize, f64)>> = (0..x.len())
            .map(|i| k_nearest(x, &x[i], self.k, Some(i)))
            .collect();
        self.k_distance = neighbours
            .iter()
            .map(|n| n.last().map_or(0.0, |l| l.1))
            .collect();
        self.lrd = (0..x.len())
            .map(|i| self.local_reachability_density(&x[i], &neighbours[i]))
            .collect();
        let scores = (0..x.len())
            .map(|i| self.lof(self.lrd[i], &neighbours[i]))
            .collect();
        self.state = Some(FitState::from_contamination(scores, self.contamination));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.x_train[0].len())?;
        Ok(x.iter()
            .map(|r| {
                let n = k_nearest(&self.x_train, r, self.k, None);
                self.lof(self.local_reachability_density(r, &n), &n)
            })
            .collect())
    }

    fn state(&self) -> Option<&FitState> {
        self.state.as_ref()
    }
}
