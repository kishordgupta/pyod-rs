//! COF: Connectivity-Based Outlier Factor (Tang et al., 2002). Port of `pyod.models.cof`.
//!
//! As in PyOD, `decision_function` scores the given samples against each other
//! (not against the training set).

use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, Result,
};
use crate::utils::euclidean;

#[derive(Debug, Clone)]
pub struct Cof {
    pub n_neighbors: usize,
    pub contamination: f64,
    n_features: usize,
    k: usize,
    state: Option<FitState>,
}

impl Default for Cof {
    fn default() -> Self {
        Self::new(20, 0.1)
    }
}

impl Cof {
    pub fn new(n_neighbors: usize, contamination: f64) -> Self {
        Cof {
            n_neighbors,
            contamination,
            n_features: 0,
            k: 0,
            state: None,
        }
    }

    /// Port of `_cof_fast`.
    fn cof(x: &Matrix, k: usize) -> Vec<f64> {
        let n = x.len();
        let k = k.min(n - 1).max(1);
        let dist: Vec<Vec<f64>> = x
            .iter()
            .map(|a| x.iter().map(|b| euclidean(a, b)).collect())
            .collect();
        let mut paths = Vec::with_capacity(n);
        let mut ac_dist = vec![0.0; n];
        for i in 0..n {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| dist[i][a].total_cmp(&dist[i][b]));
            // Set-based nearest path: the cost of each step is the shortest link
            // from the new point back to any point already on the path.
            let mut acd = 0.0;
            for j in 0..k {
                let cost = order[..=j]
                    .iter()
                    .map(|&p| dist[order[j + 1]][p])
                    .fold(f64::INFINITY, f64::min);
                acd += 2.0 * (k + 1 - (j + 1)) as f64 / ((k + 1) * k) as f64 * cost;
            }
            ac_dist[i] = acd;
            paths.push(order[1..=k].to_vec());
        }
        (0..n)
            .map(|g| {
                let v = ac_dist[g] * k as f64 / paths[g].iter().map(|&p| ac_dist[p]).sum::<f64>();
                if v.is_finite() {
                    v
                } else if v.is_nan() {
                    0.0
                } else {
                    f64::MAX
                }
            })
            .collect()
    }
}

impl Detector for Cof {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        self.n_features = check_array(x)?;
        self.k = self.n_neighbors;
        self.state = Some(FitState::from_contamination(
            Self::cof(x, self.k),
            self.contamination,
        ));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.n_features)?;
        Ok(Self::cof(x, self.k))
    }

    crate::state_accessor!();
}
