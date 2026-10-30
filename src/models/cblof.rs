//! CBLOF: Clustering-Based Local Outlier Factor (He et al., 2003).
//! Port of `pyod.models.cblof` with k-means clustering.

use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};
use crate::linalg::{kmeans, nearest, Mat};
use crate::utils::{euclidean, Rng};

#[derive(Debug, Clone)]
pub struct Cblof {
    pub n_clusters: usize,
    /// Fraction of samples the large clusters must cover.
    pub alpha: f64,
    /// Minimum size ratio between consecutive large and small clusters.
    pub beta: f64,
    /// Multiply scores by the size of the sample's cluster.
    pub use_weights: bool,
    pub contamination: f64,
    pub random_state: u64,
    pub cluster_centers: Mat,
    pub cluster_sizes: Vec<usize>,
    large: Vec<bool>,
    state: Option<FitState>,
}

impl Default for Cblof {
    fn default() -> Self {
        Self::new(8, 0.1)
    }
}

impl Cblof {
    pub fn new(n_clusters: usize, contamination: f64) -> Self {
        Cblof {
            n_clusters,
            alpha: 0.9,
            beta: 5.0,
            use_weights: false,
            contamination,
            random_state: 0,
            cluster_centers: Vec::new(),
            cluster_sizes: Vec::new(),
            large: Vec::new(),
            state: None,
        }
    }

    /// Port of `_set_small_large_clusters`.
    fn split_clusters(&mut self, n: usize) -> Result<()> {
        let sizes = &self.cluster_sizes;
        let mut order: Vec<usize> = (0..sizes.len()).collect();
        order.sort_by(|&a, &b| sizes[b].cmp(&sizes[a]));
        let (mut alpha_list, mut beta_list) = (Vec::new(), Vec::new());
        for i in 1..sizes.len() {
            let covered: usize = order[..i].iter().map(|&c| sizes[c]).sum();
            if covered as f64 >= n as f64 * self.alpha {
                alpha_list.push(i);
            }
            if sizes[order[i - 1]] as f64 / sizes[order[i]] as f64 >= self.beta {
                beta_list.push(i);
            }
        }
        let threshold = alpha_list
            .iter()
            .find(|i| beta_list.contains(i))
            .or(alpha_list.first())
            .or(beta_list.first())
            .copied()
            .ok_or_else(|| {
                OdError::InvalidParameter(
                    "could not separate small and large clusters; change n_clusters".into(),
                )
            })?;
        self.large = vec![false; sizes.len()];
        order[..threshold]
            .iter()
            .for_each(|&c| self.large[c] = true);
        Ok(())
    }

    fn score(&self, x: &Matrix, labels: &[usize]) -> Vec<f64> {
        x.iter()
            .zip(labels)
            .map(|(r, &l)| {
                let s = if self.large[l] {
                    euclidean(r, &self.cluster_centers[l])
                } else {
                    (0..self.cluster_centers.len())
                        .filter(|&c| self.large[c])
                        .map(|c| euclidean(r, &self.cluster_centers[c]))
                        .fold(f64::INFINITY, f64::min)
                };
                if self.use_weights {
                    s * self.cluster_sizes[l] as f64
                } else {
                    s
                }
            })
            .collect()
    }
}

impl Detector for Cblof {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        check_array(x)?;
        if !(self.alpha > 0.0 && self.alpha < 1.0) || self.beta <= 1.0 {
            return Err(OdError::InvalidParameter(
                "alpha must be in (0, 1) and beta > 1".into(),
            ));
        }
        if self.n_clusters < 2 || self.n_clusters > x.len() {
            return Err(OdError::InvalidParameter(
                "n_clusters must be in [2, n_samples]".into(),
            ));
        }
        let mut rng = Rng::new(self.random_state);
        let (centers, labels) = kmeans(x, self.n_clusters, 1, 300, &mut rng);
        self.cluster_centers = centers;
        self.cluster_sizes = vec![0; self.n_clusters];
        labels.iter().for_each(|&l| self.cluster_sizes[l] += 1);
        self.split_clusters(x.len())?;
        self.state = Some(FitState::from_contamination(
            self.score(x, &labels),
            self.contamination,
        ));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.cluster_centers[0].len())?;
        let labels: Vec<usize> = x
            .iter()
            .map(|r| nearest(&self.cluster_centers, r).0)
            .collect();
        Ok(self.score(x, &labels))
    }

    crate::state_accessor!();
}
