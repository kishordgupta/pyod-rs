//! k-Shape clustering for time-series subsequence anomalies (Paparrizos & Gravano,
//! SIGMOD 2015). Port of `pyod.models.ts_kshape`: windows far (in shape-based
//! distance) from every cluster centroid are anomalous.

use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};
use crate::ts::{
    combine_channels, fill_with, map_scores_to_timestamps, sbd, sliding_windows, ts_fit_state,
    znormalize, Aggregation,
};
use crate::utils::{symmetric_eigen, Rng};

/// Shift `x` by `shift` positions, zero-filling.
fn shift_align(x: &[f64], shift: isize) -> Vec<f64> {
    let m = x.len() as isize;
    (0..m)
        .map(|i| {
            let j = i - shift;
            if (0..m).contains(&j) {
                x[j as usize]
            } else {
                0.0
            }
        })
        .collect()
}

/// Shape-extraction centroid: the eigenvector maximising agreement with the aligned members.
fn centroid(members: &[Vec<f64>]) -> Vec<f64> {
    let (n, m) = (members.len(), members[0].len());
    if n == 1 {
        return znormalize(&members[0]);
    }
    let mut mat = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..m {
            let q: f64 = members.iter().map(|r| r[i] * r[j]).sum();
            mat[i][j] = (i == j) as u8 as f64 - q / n as f64;
        }
    }
    let (_, vecs) = symmetric_eigen(&mat);
    let mut c = vecs.last().unwrap().clone();
    let mean: Vec<f64> = (0..m)
        .map(|j| members.iter().map(|r| r[j]).sum::<f64>() / n as f64)
        .collect();
    if mean.iter().zip(&c).map(|(a, b)| a * b).sum::<f64>() < 0.0 {
        c.iter_mut().for_each(|v| *v = -*v);
    }
    znormalize(&c)
}

pub(crate) fn nearest_sbd(w: &[f64], centroids: &[Vec<f64>]) -> f64 {
    centroids
        .iter()
        .map(|c| sbd(w, c).0)
        .fold(f64::INFINITY, f64::min)
}

/// Port of `_kshape`. Returns the centroids and each window's distance to its nearest one.
pub(crate) fn kshape(
    windows: &[Vec<f64>],
    k: usize,
    max_iter: usize,
    rng: &mut Rng,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = windows.len();
    let mut centroids: Vec<Vec<f64>> = rng
        .sample_indices(n, k)
        .iter()
        .map(|&i| znormalize(&windows[i]))
        .collect();
    let mut labels = vec![usize::MAX; n];
    for _ in 0..max_iter {
        let new: Vec<usize> = windows
            .iter()
            .map(|w| {
                (0..k)
                    .map(|c| (c, sbd(w, &centroids[c]).0))
                    .fold((0, f64::INFINITY), |b, x| if x.1 < b.1 { x } else { b })
                    .0
            })
            .collect();
        if new == labels {
            break;
        }
        labels = new;
        for c in 0..k {
            let idx: Vec<usize> = (0..n).filter(|&i| labels[i] == c).collect();
            if idx.is_empty() {
                centroids[c] = znormalize(&windows[rng.below(n)]);
                continue;
            }
            let aligned: Vec<Vec<f64>> = idx
                .iter()
                .map(|&i| {
                    znormalize(&shift_align(
                        &windows[i],
                        -sbd(&windows[i], &centroids[c]).1,
                    ))
                })
                .collect();
            centroids[c] = centroid(&aligned);
        }
    }
    let dist = windows.iter().map(|w| nearest_sbd(w, &centroids)).collect();
    (centroids, dist)
}

pub(crate) fn znorm_windows(series: &[f64], window: usize) -> Vec<Vec<f64>> {
    let col: Vec<Vec<f64>> = series.iter().map(|v| vec![*v]).collect();
    sliding_windows(&col, window, 1)
        .iter()
        .map(|w| znormalize(w))
        .collect()
}

#[derive(Debug, Clone)]
pub struct KShape {
    pub n_clusters: usize,
    pub window_size: usize,
    pub max_iter: usize,
    pub channel_aggregation: Aggregation,
    pub contamination: f64,
    pub random_state: u64,
    centroids: Vec<Vec<Vec<f64>>>,
    state: Option<FitState>,
}

impl Default for KShape {
    fn default() -> Self {
        KShape {
            n_clusters: 3,
            window_size: 50,
            max_iter: 100,
            channel_aggregation: Aggregation::Max,
            contamination: 0.1,
            random_state: 42,
            centroids: Vec::new(),
            state: None,
        }
    }
}

impl KShape {
    pub fn new(window_size: usize, contamination: f64) -> Self {
        KShape {
            window_size,
            contamination,
            ..Self::default()
        }
    }

    fn channel_scores(&self, x: &Matrix, c: usize, centroids: &[Vec<f64>]) -> Vec<Option<f64>> {
        let w = znorm_windows(
            &x.iter().map(|r| r[c]).collect::<Vec<_>>(),
            self.window_size,
        );
        let d: Vec<f64> = w.iter().map(|w| nearest_sbd(w, centroids)).collect();
        map_scores_to_timestamps(&d, self.window_size, 1, x.len(), Aggregation::Max)
    }
}

impl Detector for KShape {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        let d = check_array(x)?;
        if self.window_size < 2
            || x.len() < self.window_size
            || x.len() - self.window_size + 1 < self.n_clusters
        {
            return Err(OdError::InvalidParameter(
                "need at least n_clusters windows of window_size".into(),
            ));
        }
        self.centroids.clear();
        let mut per = Vec::new();
        for c in 0..d {
            let w = znorm_windows(
                &x.iter().map(|r| r[c]).collect::<Vec<_>>(),
                self.window_size,
            );
            let (cent, dist) = kshape(
                &w,
                self.n_clusters,
                self.max_iter,
                &mut Rng::new(self.random_state),
            );
            per.push(map_scores_to_timestamps(
                &dist,
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
            .map(|c| self.channel_scores(x, c, &self.centroids[c]))
            .collect();
        Ok(fill_with(
            combine_channels(per, self.channel_aggregation),
            state.threshold,
        ))
    }

    crate::state_accessor!();
}
