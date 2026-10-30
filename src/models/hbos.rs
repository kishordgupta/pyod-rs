//! HBOS: Histogram-based Outlier Score (Goldstein & Dengel, 2012).
//! Port of `pyod.models.hbos` with a fixed number of bins.

use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};

#[derive(Debug, Clone)]
pub struct Hbos {
    pub n_bins: usize,
    /// Regulariser added to densities before taking the log.
    pub alpha: f64,
    /// How far (in bin widths) a value may fall outside the edges and still use the edge bin.
    pub tol: f64,
    pub contamination: f64,
    /// Per feature: bin densities (`hist_`) and bin edges (`bin_edges_`).
    hist: Vec<Vec<f64>>,
    edges: Vec<Vec<f64>>,
    state: Option<FitState>,
}

impl Default for Hbos {
    fn default() -> Self {
        Hbos {
            n_bins: 10,
            alpha: 0.1,
            tol: 0.5,
            contamination: 0.1,
            hist: Vec::new(),
            edges: Vec::new(),
            state: None,
        }
    }
}

impl Hbos {
    pub fn new(n_bins: usize, contamination: f64) -> Self {
        Hbos {
            n_bins,
            contamination,
            ..Self::default()
        }
    }

    /// Equivalent of `np.histogram(col, bins=n_bins, density=True)`.
    fn histogram(col: &[f64], n_bins: usize) -> (Vec<f64>, Vec<f64>) {
        let mut lo = col.iter().copied().fold(f64::INFINITY, f64::min);
        let mut hi = col.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if lo == hi {
            lo -= 0.5;
            hi += 0.5;
        }
        let edges: Vec<f64> = (0..=n_bins)
            .map(|i| lo + (hi - lo) * i as f64 / n_bins as f64)
            .collect();
        let mut counts = vec![0.0; n_bins];
        for &v in col {
            let b = (((v - lo) / (hi - lo)) * n_bins as f64) as usize;
            counts[b.min(n_bins - 1)] += 1.0;
        }
        let n = col.len() as f64;
        let hist = counts
            .iter()
            .enumerate()
            .map(|(i, c)| c / (n * (edges[i + 1] - edges[i])))
            .collect();
        (hist, edges)
    }

    /// Port of `_calculate_outlier_scores`: summed log2 densities (higher = more normal).
    fn log_density(&self, x: &Matrix) -> Vec<f64> {
        let nb = self.n_bins;
        let mut out = vec![0.0; x.len()];
        for (j, (hist, edges)) in self.hist.iter().zip(&self.edges).enumerate() {
            let score: Vec<f64> = hist.iter().map(|h| (h + self.alpha).log2()).collect();
            let min_score = score.iter().copied().fold(f64::INFINITY, f64::min);
            for (i, row) in x.iter().enumerate() {
                let v = row[j];
                // np.digitize(right=True): first index with v <= edges[idx]
                let idx = edges.partition_point(|&e| e < v);
                out[i] += if idx == 0 {
                    if edges[0] - v <= (edges[1] - edges[0]) * self.tol {
                        score[0]
                    } else {
                        min_score
                    }
                } else if idx == nb + 1 {
                    if v - edges[nb] <= (edges[nb] - edges[nb - 1]) * self.tol {
                        score[nb - 1]
                    } else {
                        min_score
                    }
                } else {
                    score[idx - 1]
                };
            }
        }
        out
    }
}

impl Detector for Hbos {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        if self.n_bins < 2 {
            return Err(OdError::InvalidParameter("n_bins must be >= 2".into()));
        }
        let d = check_array(x)?;
        (self.hist, self.edges) = (0..d)
            .map(|j| Self::histogram(&x.iter().map(|r| r[j]).collect::<Vec<_>>(), self.n_bins))
            .unzip();
        let scores = self.log_density(x).into_iter().map(|s| -s).collect();
        self.state = Some(FitState::from_contamination(scores, self.contamination));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.hist.len())?;
        Ok(self.log_density(x).into_iter().map(|s| -s).collect())
    }

    fn state(&self) -> Option<&FitState> {
        self.state.as_ref()
    }
}
