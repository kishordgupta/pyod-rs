//! INNE: Isolation-based anomaly detection using Nearest-Neighbor Ensembles
//! (Bandaragoda et al., 2018). Port of `pyod.models.inne`.

use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};
use crate::utils::{sq_euclidean, Rng};

#[derive(Debug, Clone)]
struct Ensemble {
    centroids: Vec<Vec<f64>>,
    /// Squared distance from each centroid to its nearest other centroid.
    radius: Vec<f64>,
    ratio: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct Inne {
    pub n_estimators: usize,
    /// Hyperspheres per ensemble member; `None` means `min(8, n_samples)`.
    pub max_samples: Option<usize>,
    pub contamination: f64,
    pub random_state: u64,
    ensembles: Vec<Ensemble>,
    n_features: usize,
    state: Option<FitState>,
}

impl Default for Inne {
    fn default() -> Self {
        Self::new(200, 0.1)
    }
}

impl Inne {
    pub fn new(n_estimators: usize, contamination: f64) -> Self {
        Inne {
            n_estimators,
            max_samples: None,
            contamination,
            random_state: 0,
            ensembles: Vec::new(),
            n_features: 0,
            state: None,
        }
    }

    fn score(&self, x: &Matrix) -> Vec<f64> {
        x.iter()
            .map(|r| {
                self.ensembles
                    .iter()
                    .map(|e| {
                        // Smallest covering hypersphere; uncovered samples score 1.
                        let mut best: Option<(f64, usize)> = None;
                        for (c, (cent, &rad)) in e.centroids.iter().zip(&e.radius).enumerate() {
                            if sq_euclidean(r, cent) <= rad && best.is_none_or(|b| rad < b.0) {
                                best = Some((rad, c));
                            }
                        }
                        best.map_or(1.0, |(_, c)| e.ratio[c])
                    })
                    .sum::<f64>()
                    / self.ensembles.len() as f64
            })
            .collect()
    }
}

impl Detector for Inne {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        self.n_features = check_array(x)?;
        let psi = self.max_samples.unwrap_or(8).min(x.len());
        if psi < 2 {
            return Err(OdError::InvalidParameter(
                "INNE needs at least 2 samples per ensemble".into(),
            ));
        }
        let mut rng = Rng::new(self.random_state);
        self.ensembles = (0..self.n_estimators)
            .map(|_| {
                let centroids: Vec<Vec<f64>> = rng
                    .sample_indices(x.len(), psi)
                    .iter()
                    .map(|&i| x[i].clone())
                    .collect();
                let nn: Vec<(usize, f64)> = (0..psi)
                    .map(|i| {
                        (0..psi)
                            .filter(|&j| j != i)
                            .map(|j| (j, sq_euclidean(&centroids[i], &centroids[j])))
                            .min_by(|a, b| a.1.total_cmp(&b.1))
                            .unwrap()
                    })
                    .collect();
                let radius: Vec<f64> = nn.iter().map(|n| n.1).collect();
                let ratio = (0..psi)
                    .map(|i| 1.0 - (radius[nn[i].0] + f64::EPSILON) / (radius[i] + f64::EPSILON))
                    .collect();
                Ensemble {
                    centroids,
                    radius,
                    ratio,
                }
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

    crate::state_accessor!();
}
