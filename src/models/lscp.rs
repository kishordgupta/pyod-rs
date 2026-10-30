//! LSCP: Locally Selective Combination of Parallel outlier ensembles
//! (Zhao et al., SDM 2019). Port of `pyod.models.lscp`.
//!
//! For each sample, LSCP finds its local region in the training data, measures
//! how well each detector agrees with a pseudo ground truth there (Pearson
//! correlation), and averages the scores of the most competent detectors.

use super::feature_bagging::bagging_indices;
use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};
use crate::utils::{k_nearest, Rng, StandardScaler};
use std::collections::HashMap;

pub struct Lscp {
    pub detector_list: Vec<Box<dyn Detector>>,
    pub local_region_size: usize,
    pub local_max_features: f64,
    pub n_bins: usize,
    pub contamination: f64,
    pub random_state: u64,
    x_train: Vec<Vec<f64>>,
    train_scores: Vec<Vec<f64>>,
    state: Option<FitState>,
}

const LOCAL_REGION_ITERATIONS: usize = 20;
const LOCAL_REGION_THRESHOLD: usize = LOCAL_REGION_ITERATIONS / 2;
const LOCAL_MIN_FEATURES: f64 = 0.5;

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f64 = a.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f64 = b.iter().map(|y| (y - mb).powi(2)).sum();
    cov / (va * vb).sqrt()
}

impl Lscp {
    pub fn new(detector_list: Vec<Box<dyn Detector>>, contamination: f64) -> Self {
        Lscp {
            detector_list,
            local_region_size: 30,
            local_max_features: 1.0,
            n_bins: 10,
            contamination,
            random_state: 0,
            x_train: Vec::new(),
            train_scores: Vec::new(),
            state: None,
        }
    }

    /// Port of `_get_local_region`: training points that are among the
    /// nearest neighbours in more than half of 20 random feature subspaces.
    ///
    /// PyOD draws the subspaces from one random state shared across calls, so
    /// scoring the same data twice gives different results; we reseed from
    /// `random_state` on every call so scores are reproducible.
    fn local_regions(&self, x: &Matrix) -> Vec<Vec<usize>> {
        let mut rng = Rng::new(self.random_state);
        let d = self.x_train[0].len();
        let max_f = self.local_max_features.min(1.0);
        let min_f = if d as f64 * LOCAL_MIN_FEATURES < 1.0 {
            1.0
        } else {
            LOCAL_MIN_FEATURES
        };
        let mut counts: Vec<HashMap<usize, usize>> = vec![HashMap::new(); x.len()];
        let mut order: Vec<Vec<usize>> = vec![Vec::new(); x.len()];
        for _ in 0..LOCAL_REGION_ITERATIONS {
            let features: Vec<usize> = if max_f == min_f {
                (0..d).collect()
            } else {
                bagging_indices(
                    &mut rng,
                    false,
                    d,
                    (d as f64 * min_f) as usize,
                    (d as f64 * max_f) as usize,
                )
            };
            let proj = |r: &Vec<f64>| features.iter().map(|&f| r[f]).collect::<Vec<f64>>();
            let train: Vec<Vec<f64>> = self.x_train.iter().map(proj).collect();
            for (j, r) in x.iter().enumerate() {
                for (i, _) in k_nearest(&train, &proj(r), self.local_region_size, None) {
                    let c = counts[j].entry(i).or_insert(0);
                    if *c == 0 {
                        order[j].push(i);
                    }
                    *c += 1;
                }
            }
        }
        (0..x.len())
            .map(|j| {
                let mut threshold = LOCAL_REGION_THRESHOLD;
                loop {
                    let region: Vec<usize> = order[j]
                        .iter()
                        .copied()
                        .filter(|i| counts[j][i] > threshold)
                        .collect();
                    if region.len() >= 2 || threshold == 0 {
                        return region;
                    }
                    threshold -= 1;
                }
            })
            .collect()
    }

    /// Port of `_get_competent_detectors`: detectors in the most populated histogram bin of correlations.
    fn competent(&self, corr: &[f64]) -> Vec<usize> {
        let corr: Vec<f64> = corr
            .iter()
            .map(|c| if c.is_nan() { 0.0 } else { *c })
            .collect();
        let n_bins = self.n_bins.min(corr.len()).max(1);
        let (mut lo, mut hi) = corr
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                (a.min(v), b.max(v))
            });
        if lo == hi {
            lo -= 0.5;
            hi += 0.5;
        }
        let edges: Vec<f64> = (0..=n_bins)
            .map(|i| lo + (hi - lo) * i as f64 / n_bins as f64)
            .collect();
        let mut hist = vec![0usize; n_bins];
        for &c in &corr {
            hist[(((c - lo) / (hi - lo) * n_bins as f64) as usize).min(n_bins - 1)] += 1;
        }
        let best = (0..n_bins)
            .max_by(|&a, &b| hist[a].cmp(&hist[b]).then(b.cmp(&a)))
            .unwrap();
        (0..corr.len())
            .filter(|&k| corr[k] >= edges[best] && corr[k] <= edges[best + 1])
            .collect()
    }

    fn scores(&self, x: &Matrix) -> Result<Vec<f64>> {
        let regions = self.local_regions(x);
        let n_clf = self.detector_list.len();
        let test_cols: Vec<Vec<f64>> = self
            .detector_list
            .iter()
            .map(|d| d.decision_function(x))
            .collect::<Result<_>>()?;
        let test: Vec<Vec<f64>> = (0..x.len())
            .map(|i| test_cols.iter().map(|c| c[i]).collect())
            .collect();
        let scaler = StandardScaler::fit(&self.train_scores);
        let train_norm = scaler.transform(&self.train_scores);
        let test_norm = scaler.transform(&test);
        let pseudo: Vec<f64> = train_norm
            .iter()
            .map(|r| r.iter().copied().fold(f64::NEG_INFINITY, f64::max))
            .collect();
        Ok(regions
            .iter()
            .enumerate()
            .map(|(i, region)| {
                let truth: Vec<f64> = region.iter().map(|&r| pseudo[r]).collect();
                let corr: Vec<f64> = (0..n_clf)
                    .map(|k| {
                        pearson(
                            &truth,
                            &region.iter().map(|&r| train_norm[r][k]).collect::<Vec<_>>(),
                        )
                    })
                    .collect();
                let chosen = self.competent(&corr);
                chosen.iter().map(|&k| test_norm[i][k]).sum::<f64>() / chosen.len() as f64
            })
            .collect())
    }
}

impl Detector for Lscp {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        check_array(x)?;
        if self.detector_list.len() < 2 {
            return Err(OdError::InvalidParameter(
                "LSCP needs at least 2 detectors".into(),
            ));
        }
        self.x_train = x.to_vec();
        let mut cols = Vec::new();
        for det in &mut self.detector_list {
            det.fit(x)?;
            cols.push(det.decision_scores()?.to_vec());
        }
        self.train_scores = (0..x.len())
            .map(|i| cols.iter().map(|c| c[i]).collect())
            .collect();
        let scores = self.scores(x)?;
        self.state = Some(FitState::from_contamination(scores, self.contamination));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.x_train[0].len())?;
        self.scores(x)
    }

    crate::state_accessor!();
}
