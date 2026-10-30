//! SAND: streaming subsequence anomaly detection (Boniol et al., VLDB 2021).
//! Port of `pyod.models.ts_sand`: k-Shape centroids are refreshed batch by batch
//! with exponential smoothing, and windows are scored by distance to them.

use super::ts_kshape::{kshape, nearest_sbd, znorm_windows};
use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};
use crate::ts::{
    combine_channels, fill_with, map_scores_to_timestamps, ts_fit_state, znormalize, Aggregation,
};
use crate::utils::Rng;

#[derive(Debug, Clone)]
pub struct Sand {
    pub window_size: usize,
    pub n_clusters: usize,
    /// Weight of each new batch's centroids in the running centroids.
    pub alpha: f64,
    pub batch_size: usize,
    pub max_iter: usize,
    pub channel_aggregation: Aggregation,
    pub contamination: f64,
    pub random_state: u64,
    centroids: Vec<Vec<Vec<f64>>>,
    state: Option<FitState>,
}

impl Default for Sand {
    fn default() -> Self {
        Sand {
            window_size: 50,
            n_clusters: 5,
            alpha: 0.5,
            batch_size: 100,
            max_iter: 50,
            channel_aggregation: Aggregation::Max,
            contamination: 0.1,
            random_state: 42,
            centroids: Vec::new(),
            state: None,
        }
    }
}

impl Sand {
    pub fn new(window_size: usize, contamination: f64) -> Self {
        Sand {
            window_size,
            contamination,
            ..Self::default()
        }
    }

    /// Port of `_streaming_fit_channel`.
    fn stream(&self, w: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
        let n = w.len();
        let bs = self.batch_size.min(n);
        let (mut centroids, _) = kshape(
            &w[..bs],
            self.n_clusters,
            self.max_iter,
            &mut Rng::new(self.random_state),
        );
        let mut scores = Vec::with_capacity(n);
        let mut pos = 0;
        while pos < n {
            let end = (pos + bs).min(n);
            let batch = &w[pos..end];
            scores.extend(batch.iter().map(|s| nearest_sbd(s, &centroids)));
            if end < n && batch.len() >= self.n_clusters {
                let (bc, _) = kshape(
                    batch,
                    self.n_clusters,
                    self.max_iter,
                    &mut Rng::new(self.random_state),
                );
                for (c, b) in centroids.iter_mut().zip(bc) {
                    let upd: Vec<f64> = b
                        .iter()
                        .zip(c.iter())
                        .map(|(nb, old)| self.alpha * nb + (1.0 - self.alpha) * old)
                        .collect();
                    *c = znormalize(&upd);
                }
            }
            pos = end;
        }
        (scores, centroids)
    }
}

impl Detector for Sand {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        let d = check_array(x)?;
        if self.window_size < 2
            || x.len() < self.window_size + 1
            || (x.len() - self.window_size + 1).min(self.batch_size) < self.n_clusters
        {
            return Err(OdError::InvalidParameter(
                "series too short for window_size, batch_size and n_clusters".into(),
            ));
        }
        self.centroids.clear();
        let mut per = Vec::new();
        for c in 0..d {
            let w = znorm_windows(
                &x.iter().map(|r| r[c]).collect::<Vec<_>>(),
                self.window_size,
            );
            let (scores, cent) = self.stream(&w);
            per.push(map_scores_to_timestamps(
                &scores,
                self.window_size,
                1,
                x.len(),
                Aggregation::Max,
            ));
            self.centroids.push(cent);
        }
        self.state = Some(ts_fit_state(
            &combine_channels(per, self.channel_aggregation),
            self.contamination,
        ));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        let state = self.fitted()?;
        check_features(x, self.centroids.len())?;
        let per = (0..self.centroids.len())
            .map(|c| {
                let w = znorm_windows(
                    &x.iter().map(|r| r[c]).collect::<Vec<_>>(),
                    self.window_size,
                );
                let d: Vec<f64> = w
                    .iter()
                    .map(|s| nearest_sbd(s, &self.centroids[c]))
                    .collect();
                map_scores_to_timestamps(&d, self.window_size, 1, x.len(), Aggregation::Max)
            })
            .collect();
        Ok(fill_with(
            combine_channels(per, self.channel_aggregation),
            state.threshold,
        ))
    }

    crate::state_accessor!();
}
