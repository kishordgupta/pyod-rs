//! ABOD: Angle-Based Outlier Detection, "fast" variant (Kriegel et al., KDD 2008).
//! Port of `pyod.models.abod` with `method='fast'`.

use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};
use crate::utils::{dot, k_nearest};

#[derive(Debug, Clone)]
pub struct Abod {
    pub n_neighbors: usize,
    pub contamination: f64,
    x_train: Vec<Vec<f64>>,
    state: Option<FitState>,
}

impl Default for Abod {
    fn default() -> Self {
        Self::new(5, 0.1)
    }
}

impl Abod {
    pub fn new(n_neighbors: usize, contamination: f64) -> Self {
        Abod {
            n_neighbors,
            contamination,
            x_train: Vec::new(),
            state: None,
        }
    }

    /// Negated variance of the distance-weighted cosines between pairs of neighbours,
    /// so that outliers (small variance) get large scores.
    fn score(&self, point: &[f64], neighbours: &[(usize, f64)]) -> f64 {
        let mut wcos = Vec::new();
        for (a_pos, &(a, _)) in neighbours.iter().enumerate() {
            for &(b, _) in &neighbours[a_pos + 1..] {
                let (pa, pb) = (&self.x_train[a], &self.x_train[b]);
                if pa.as_slice() == point || pb.as_slice() == point {
                    continue;
                }
                let va: Vec<f64> = pa.iter().zip(point).map(|(x, p)| x - p).collect();
                let vb: Vec<f64> = pb.iter().zip(point).map(|(x, p)| x - p).collect();
                wcos.push(dot(&va, &vb) / dot(&va, &va) / dot(&vb, &vb));
            }
        }
        if wcos.is_empty() {
            return 0.0;
        }
        let m = wcos.iter().sum::<f64>() / wcos.len() as f64;
        -(wcos.iter().map(|w| (w - m).powi(2)).sum::<f64>() / wcos.len() as f64)
    }
}

impl Detector for Abod {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        check_array(x)?;
        if self.n_neighbors < 2 || self.n_neighbors >= x.len() {
            return Err(OdError::InvalidParameter(
                "n_neighbors must be in [2, n_samples - 1]".into(),
            ));
        }
        self.x_train = x.to_vec();
        // Like PyOD, the training query includes the point itself; `score` skips
        // pairs containing it.
        let scores = x
            .iter()
            .map(|r| self.score(r, &k_nearest(x, r, self.n_neighbors, None)))
            .collect();
        self.state = Some(FitState::from_contamination(scores, self.contamination));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.x_train[0].len())?;
        Ok(x.iter()
            .map(|r| self.score(r, &k_nearest(&self.x_train, r, self.n_neighbors, None)))
            .collect())
    }

    fn state(&self) -> Option<&FitState> {
        self.state.as_ref()
    }
}
