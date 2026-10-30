//! LOCI: Local Correlation Integral (Papadimitriou et al., 2003). Port of `pyod.models.loci`.
//!
//! O(n^3); as in PyOD, `decision_function` scores the given samples against each other.

use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, Result,
};
use crate::utils::euclidean;

#[derive(Debug, Clone)]
pub struct Loci {
    /// Ratio of the counting radius to the sampling radius.
    pub alpha: f64,
    /// Number of standard deviations of MDEF beyond which a point is flagged.
    pub k: f64,
    pub contamination: f64,
    n_features: usize,
    state: Option<FitState>,
}

impl Default for Loci {
    fn default() -> Self {
        Loci {
            alpha: 0.5,
            k: 3.0,
            contamination: 0.1,
            n_features: 0,
            state: None,
        }
    }
}

impl Loci {
    fn loci(&self, x: &Matrix) -> Vec<f64> {
        let n = x.len();
        let dist: Vec<Vec<f64>> = x
            .iter()
            .map(|a| x.iter().map(|b| euclidean(a, b)).collect())
            .collect();
        let max_dist = dist.iter().flatten().copied().fold(0.0, f64::max);
        let r_max = max_dist / self.alpha;
        let alpha_n =
            |i: usize, r: f64| dist[i].iter().filter(|&&d| d < r * self.alpha).count() as f64;
        (0..n)
            .map(|p| {
                let mut critical: Vec<f64> = dist[p]
                    .iter()
                    .filter(|&&d| d > 0.0 && d <= r_max)
                    .flat_map(|&d| [d, d / self.alpha])
                    .collect();
                critical.sort_by(f64::total_cmp);
                let mut score = 0.0;
                for r in critical {
                    let counts: Vec<f64> = (0..n)
                        .filter(|&j| dist[p][j] <= r)
                        .map(|j| alpha_n(j, r))
                        .collect();
                    let n_hat = counts.iter().sum::<f64>() / counts.len() as f64;
                    let mdef = 1.0 - alpha_n(p, r) / n_hat;
                    let sigma = (counts.iter().map(|c| (c - n_hat).powi(2)).sum::<f64>()
                        / counts.len() as f64)
                        .sqrt()
                        / n_hat;
                    if n_hat >= 20.0 {
                        score = mdef / sigma;
                        if mdef > self.k * sigma {
                            break;
                        }
                    }
                }
                score
            })
            .collect()
    }
}

impl Detector for Loci {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        self.n_features = check_array(x)?;
        self.state = Some(FitState::from_contamination(
            self.loci(x),
            self.contamination,
        ));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.n_features)?;
        Ok(self.loci(x))
    }

    crate::state_accessor!();
}
