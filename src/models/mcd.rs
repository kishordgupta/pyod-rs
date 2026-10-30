//! Minimum Covariance Determinant outlier detection (Rousseeuw & Van Driessen, 1999).
//! Port of `pyod.models.mcd` (sklearn `MinCovDet` / FastMCD). The score is the
//! squared robust Mahalanobis distance.

use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};
use crate::linalg::{chi2_ppf, col_mean, covariance, fast_logdet, mahalanobis_sq, pinvh, Mat};
use crate::utils::{median, Rng};

#[derive(Debug, Clone)]
struct Estimate {
    location: Vec<f64>,
    cov: Mat,
    det: f64,
    support: Vec<usize>,
    dist: Vec<f64>,
}

/// Indices of the `h` smallest values.
fn smallest(dist: &[f64], h: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..dist.len()).collect();
    idx.select_nth_unstable_by(h - 1, |&a, &b| dist[a].total_cmp(&dist[b]));
    idx.truncate(h);
    idx
}

fn rows(x: &Matrix, idx: &[usize]) -> Vec<Vec<f64>> {
    idx.iter().map(|&i| x[i].clone()).collect()
}

/// Port of sklearn's `_c_step`: concentrate the support until the determinant stops shrinking.
fn c_step(
    x: &Matrix,
    h: usize,
    iters: usize,
    init: Option<(&[f64], &Mat)>,
    rng: &mut Rng,
) -> Estimate {
    let mut support = match init {
        None => rng.sample_indices(x.len(), h),
        Some((loc, cov)) => smallest(&mahalanobis_sq(x, loc, &pinvh(cov)), h),
    };
    let sub = rows(x, &support);
    let mut location = col_mean(&sub);
    let mut cov = covariance(&sub, &location, 0);
    let mut det = fast_logdet(&cov);
    let mut precision = pinvh(&cov);
    let mut previous: Option<Estimate> = None;
    let mut dist = vec![f64::INFINITY; x.len()];
    let mut remaining = iters;
    let mut prev_det = f64::INFINITY;
    while det < prev_det && remaining > 0 && det.is_finite() {
        previous = Some(Estimate {
            location: location.clone(),
            cov: cov.clone(),
            det,
            support: support.clone(),
            dist: dist.clone(),
        });
        prev_det = det;
        precision = pinvh(&cov);
        dist = mahalanobis_sq(x, &location, &precision);
        support = smallest(&dist, h);
        let sub = rows(x, &support);
        location = col_mean(&sub);
        cov = covariance(&sub, &location, 0);
        det = fast_logdet(&cov);
        remaining -= 1;
    }
    let dist = mahalanobis_sq(x, &location, &precision);
    let current = Estimate {
        location,
        cov,
        det,
        support,
        dist,
    };
    let close = (det - prev_det).abs() <= 1e-8 + 1e-5 * prev_det.abs();
    if det.is_infinite() || close || remaining == 0 {
        return current;
    }
    if det > prev_det {
        if let Some(p) = previous {
            return p;
        }
    }
    current
}

/// Port of sklearn's `select_candidates`: run C-steps from many starts and keep the best.
fn select(
    x: &Matrix,
    h: usize,
    starts: Starts,
    keep: usize,
    iters: usize,
    rng: &mut Rng,
) -> Vec<Estimate> {
    let mut all: Vec<Estimate> = match starts {
        Starts::Random(n) => (0..n).map(|_| c_step(x, h, iters, None, rng)).collect(),
        Starts::From(est) => est
            .iter()
            .map(|e| c_step(x, h, iters, Some((&e.location, &e.cov)), rng))
            .collect(),
    };
    all.sort_by(|a, b| a.det.total_cmp(&b.det));
    all.truncate(keep);
    all
}

enum Starts<'a> {
    Random(usize),
    From(&'a [Estimate]),
}

/// Port of sklearn's `fast_mcd`. Returns the raw (location, covariance, support mask, distances).
fn fast_mcd(x: &Matrix, h: usize, rng: &mut Rng) -> (Vec<f64>, Mat, Vec<bool>, Vec<f64>) {
    let (n, d) = (x.len(), x[0].len());
    if d == 1 {
        let col: Vec<f64> = x.iter().map(|r| r[0]).collect();
        let (location, support) = if h < n {
            let mut s = col.clone();
            s.sort_by(f64::total_cmp);
            let diffs: Vec<f64> = (0..n - h).map(|i| s[i + h] - s[i]).collect();
            let min = diffs.iter().copied().fold(f64::INFINITY, f64::min);
            let starts: Vec<usize> = (0..diffs.len()).filter(|&i| diffs[i] == min).collect();
            let loc =
                0.5 * starts.iter().map(|&i| s[i + h] + s[i]).sum::<f64>() / starts.len() as f64;
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&a, &b| (col[a] - loc).abs().total_cmp(&(col[b] - loc).abs()));
            let mut mask = vec![false; n];
            idx[..h].iter().for_each(|&i| mask[i] = true);
            (loc, mask)
        } else {
            (col.iter().sum::<f64>() / n as f64, vec![true; n])
        };
        let sup: Vec<f64> = col
            .iter()
            .zip(&support)
            .filter(|(_, &m)| m)
            .map(|(v, _)| *v)
            .collect();
        let m = sup.iter().sum::<f64>() / sup.len() as f64;
        let var = sup.iter().map(|v| (v - m).powi(2)).sum::<f64>() / sup.len() as f64;
        let cov = vec![vec![var]];
        let dist = mahalanobis_sq(x, &[location], &pinvh(&cov));
        return (vec![location], cov, support, dist);
    }
    let best = if n > 500 {
        let n_subsets = n / 300;
        let per = n / n_subsets;
        let shuffle = rng.sample_indices(n, n);
        let h_sub = (per as f64 * (h as f64 / n as f64)).ceil() as usize;
        let n_trials = (500 / n_subsets).max(10);
        let mut pooled = Vec::new();
        for i in 0..n_subsets {
            let sub = rows(x, &shuffle[i * per..(i + 1) * per]);
            pooled.extend(select(&sub, h_sub, Starts::Random(n_trials), 10, 2, rng));
        }
        let n_merged = n.min(1500);
        let h_merged = (n_merged as f64 * (h as f64 / n as f64)).ceil() as usize;
        let selection = rng.sample_indices(n, n_merged);
        let merged = select(
            &rows(x, &selection),
            h_merged,
            Starts::From(&pooled),
            if n > 1500 { 10 } else { 1 },
            30,
            rng,
        );
        if n < 1500 {
            // Map the merged estimate's support and distances back to the full data.
            let e = &merged[0];
            let mut support = vec![false; n];
            let mut dist = vec![0.0; n];
            for &s in &e.support {
                support[selection[s]] = true;
            }
            for (k, &i) in selection.iter().enumerate() {
                dist[i] = e.dist[k];
            }
            return (e.location.clone(), e.cov.clone(), support, dist);
        }
        select(x, h, Starts::From(&merged), 1, 30, rng).remove(0)
    } else {
        let candidates = select(x, h, Starts::Random(30), 10, 2, rng);
        select(x, h, Starts::From(&candidates), 1, 30, rng).remove(0)
    };
    let mut support = vec![false; n];
    best.support.iter().for_each(|&i| support[i] = true);
    (best.location, best.cov, support, best.dist)
}

#[derive(Debug, Clone)]
pub struct Mcd {
    /// Fraction of points in the raw MCD support; `None` uses `(n + d + 1) / 2`.
    pub support_fraction: Option<f64>,
    pub contamination: f64,
    pub random_state: u64,
    pub raw_location: Vec<f64>,
    pub raw_covariance: Mat,
    pub raw_support: Vec<bool>,
    pub location: Vec<f64>,
    pub covariance: Mat,
    pub support: Vec<bool>,
    precision: Mat,
    state: Option<FitState>,
}

impl Default for Mcd {
    fn default() -> Self {
        Self::new(0.1)
    }
}

impl Mcd {
    pub fn new(contamination: f64) -> Self {
        Mcd {
            support_fraction: None,
            contamination,
            random_state: 0,
            raw_location: Vec::new(),
            raw_covariance: Vec::new(),
            raw_support: Vec::new(),
            location: Vec::new(),
            covariance: Vec::new(),
            support: Vec::new(),
            precision: Vec::new(),
            state: None,
        }
    }
}

impl Detector for Mcd {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        let d = check_array(x)?;
        let n = x.len();
        if n < 2 {
            return Err(OdError::InvalidParameter(
                "MCD needs at least 2 samples".into(),
            ));
        }
        let h = match self.support_fraction {
            None => (0.5 * (n + d + 1) as f64).ceil() as usize,
            Some(f) if f > 0.0 && f <= 1.0 => ((f * n as f64) as usize).max(1),
            Some(f) => {
                return Err(OdError::InvalidParameter(format!(
                    "support_fraction must be in (0, 1], got {f}"
                )))
            }
        };
        let mut rng = Rng::new(self.random_state);
        let (loc, cov, support, mut dist) = fast_mcd(x, h.min(n), &mut rng);

        // Consistency correction (sklearn `correct_covariance`).
        let correction = median(&dist) / chi2_ppf(0.5, d as f64);
        let corrected: Mat = cov
            .iter()
            .map(|r| r.iter().map(|v| v * correction).collect())
            .collect();
        dist.iter_mut().for_each(|v| *v /= correction);
        self.raw_location = loc;
        self.raw_covariance = corrected;
        self.raw_support = support;

        // Reweighting (sklearn `reweight_covariance`).
        let cut = chi2_ppf(0.975, d as f64);
        let keep: Vec<Vec<f64>> = x
            .iter()
            .zip(&dist)
            .filter(|(_, &v)| v < cut)
            .map(|(r, _)| r.clone())
            .collect();
        self.support = dist.iter().map(|&v| v < cut).collect();
        self.location = col_mean(&keep);
        self.covariance = covariance(&keep, &self.location, 0);
        self.precision = pinvh(&self.covariance);
        let scores = mahalanobis_sq(x, &self.location, &self.precision);
        self.state = Some(FitState::from_contamination(scores, self.contamination));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.location.len())?;
        Ok(mahalanobis_sq(x, &self.location, &self.precision))
    }

    crate::state_accessor!();
}
