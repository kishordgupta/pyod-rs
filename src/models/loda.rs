//! LODA: Lightweight On-line Detector of Anomalies (Pevný, 2016). Port of `pyod.models.loda`.

use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, Result,
};
use crate::utils::{dot, Rng};

#[derive(Debug, Clone)]
pub struct Loda {
    pub n_bins: usize,
    pub n_random_cuts: usize,
    pub contamination: f64,
    pub random_state: u64,
    projections: Vec<Vec<f64>>,
    histograms: Vec<Vec<f64>>,
    limits: Vec<Vec<f64>>,
    state: Option<FitState>,
}

impl Default for Loda {
    fn default() -> Self {
        Self::new(10, 100, 0.1)
    }
}

impl Loda {
    pub fn new(n_bins: usize, n_random_cuts: usize, contamination: f64) -> Self {
        Loda {
            n_bins,
            n_random_cuts,
            contamination,
            random_state: 0,
            projections: Vec::new(),
            histograms: Vec::new(),
            limits: Vec::new(),
            state: None,
        }
    }

    fn score(&self, x: &Matrix) -> Vec<f64> {
        let cuts = self.n_random_cuts as f64;
        x.iter()
            .map(|r| {
                let mut s = 0.0;
                for ((p, h), lim) in self
                    .projections
                    .iter()
                    .zip(&self.histograms)
                    .zip(&self.limits)
                {
                    let v = dot(p, r);
                    // np.searchsorted(limits[:n_bins - 1], v, side='left')
                    let idx = lim[..self.n_bins - 1].partition_point(|&l| l < v);
                    s += -(1.0 / cuts) * h[idx].ln();
                }
                s / cuts
            })
            .collect()
    }
}

impl Detector for Loda {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        let d = check_array(x)?;
        let mut rng = Rng::new(self.random_state);
        let n_zero = d - (d as f64).sqrt() as usize;
        self.projections.clear();
        self.histograms.clear();
        self.limits.clear();
        for _ in 0..self.n_random_cuts {
            let mut p: Vec<f64> = (0..d).map(|_| rng.normal()).collect();
            for i in rng.sample_indices(d, n_zero) {
                p[i] = 0.0;
            }
            let proj: Vec<f64> = x.iter().map(|r| dot(&p, r)).collect();
            let (mut lo, mut hi) = proj
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                    (a.min(v), b.max(v))
                });
            if lo == hi {
                lo -= 0.5;
                hi += 0.5;
            }
            let nb = self.n_bins;
            let limits: Vec<f64> = (0..=nb)
                .map(|i| lo + (hi - lo) * i as f64 / nb as f64)
                .collect();
            let mut hist = vec![1e-12; nb];
            for v in &proj {
                hist[((((v - lo) / (hi - lo)) * nb as f64) as usize).min(nb - 1)] += 1.0;
            }
            let total: f64 = hist.iter().sum();
            hist.iter_mut().for_each(|h| *h /= total);
            self.projections.push(p);
            self.histograms.push(hist);
            self.limits.push(limits);
        }
        self.state = Some(FitState::from_contamination(
            self.score(x),
            self.contamination,
        ));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.projections[0].len())?;
        Ok(self.score(x))
    }

    crate::state_accessor!();
}
