//! ROD: Rotation-based Outlier Detection (Almardeny et al., 2020). Port of `pyod.models.rod`.
//!
//! Data with more than 3 features is robust-scaled and split into every 3-D
//! subspace; each subspace is scored by the volume of rotating its points about
//! the geometric median, and the sigmoid-squashed subspace scores are averaged.

use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, Result,
};
use crate::utils::{dot, euclidean, median, percentile};
use std::f64::consts::FRAC_PI_2;

/// MinMaxScaler fitted on `values + [lo, hi]` with feature range `[lo, hi]`.
#[derive(Debug, Clone, Copy)]
struct AngleScaler {
    data_min: f64,
    data_max: f64,
    lo: f64,
    hi: f64,
}

impl AngleScaler {
    fn fit(values: &[f64], lo: f64, hi: f64) -> Self {
        let (mut mn, mut mx) = (lo.min(hi), lo.max(hi));
        for &v in values {
            mn = mn.min(v);
            mx = mx.max(v);
        }
        AngleScaler {
            data_min: mn,
            data_max: mx,
            lo,
            hi,
        }
    }

    fn transform(&self, v: f64) -> f64 {
        let range = if self.data_max - self.data_min == 0.0 {
            1.0
        } else {
            self.data_max - self.data_min
        };
        (v - self.data_min) / range * (self.hi - self.lo) + self.lo
    }
}

/// Everything learned from one 3-D subspace during `fit`.
#[derive(Debug, Clone)]
struct SubspaceModel {
    gm: Vec<f64>,
    median: f64,
    s1: AngleScaler,
    s2: AngleScaler,
}

fn geometric_median(x: &Matrix, eps: f64) -> Vec<f64> {
    let mut pts: Vec<Vec<f64>> = x.to_vec();
    pts.sort_by(|a, b| {
        a.iter()
            .zip(b)
            .map(|(p, q)| p.total_cmp(q))
            .find(|o| o.is_ne())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    pts.dedup();
    let d = pts[0].len();
    let mut gm: Vec<f64> = (0..d)
        .map(|j| pts.iter().map(|p| p[j]).sum::<f64>() / pts.len() as f64)
        .collect();
    loop {
        let dists: Vec<f64> = pts.iter().map(|p| euclidean(p, &gm)).collect();
        let nz: Vec<usize> = (0..pts.len()).filter(|&i| dists[i] != 0.0).collect();
        if nz.is_empty() {
            return gm;
        }
        let dinvs: f64 = nz.iter().map(|&i| 1.0 / dists[i]).sum();
        let t: Vec<f64> = (0..d)
            .map(|j| nz.iter().map(|&i| pts[i][j] / dists[i]).sum::<f64>() / dinvs)
            .collect();
        let n_zero = pts.len() - nz.len();
        let next = if n_zero == 0 {
            t
        } else {
            let r: f64 = t
                .iter()
                .zip(&gm)
                .map(|(a, b)| ((a - b) * dinvs).powi(2))
                .sum::<f64>()
                .sqrt();
            let r_inv = if r == 0.0 { 0.0 } else { n_zero as f64 / r };
            t.iter()
                .zip(&gm)
                .map(|(a, b)| (1.0 - r_inv).max(0.0) * a + r_inv.min(1.0) * b)
                .collect()
        };
        if euclidean(&gm, &next) < eps {
            return next;
        }
        gm = next;
    }
}

/// Port of `rod_3D`; fits the subspace model when `model` is `None`.
fn rod_3d(x: &Matrix, model: Option<&SubspaceModel>) -> (Vec<f64>, SubspaceModel) {
    let gm = model.map_or_else(|| geometric_median(x, 1e-5), |m| m.gm.clone());
    let norm = dot(&gm, &gm).sqrt();
    let centered: Vec<Vec<f64>> = x
        .iter()
        .map(|r| r.iter().zip(&gm).map(|(a, b)| a - b).collect())
        .collect();
    let v_norm: Vec<f64> = centered.iter().map(|c| dot(c, c).sqrt()).collect();
    let raw: Vec<Option<f64>> = centered
        .iter()
        .zip(&v_norm)
        .map(|(c, &vn)| {
            if norm == 0.0 {
                Some(FRAC_PI_2)
            } else if vn * norm > 0.0 {
                Some((dot(c, &gm) / (vn * norm)).clamp(-1.0, 1.0).acos())
            } else {
                None
            }
        })
        .collect();
    let (s1, s2) = match model {
        Some(m) => (m.s1, m.s2),
        None => {
            let first: Vec<f64> = raw
                .iter()
                .flatten()
                .copied()
                .filter(|&g| g <= FRAC_PI_2)
                .collect();
            let second: Vec<f64> = raw
                .iter()
                .flatten()
                .copied()
                .filter(|&g| g > FRAC_PI_2)
                .collect();
            (
                AngleScaler::fit(&first, 0.001, 0.955),
                AngleScaler::fit(&second, FRAC_PI_2 + 0.001, 2.186),
            )
        }
    };
    let costs: Vec<f64> = raw
        .iter()
        .zip(&v_norm)
        .map(|(g, vn)| match g {
            Some(g) => {
                let g = if *g <= FRAC_PI_2 {
                    s1.transform(*g)
                } else {
                    s2.transform(*g)
                };
                vn.powi(3) * g.cos() * g.sin().powi(2)
            }
            None => 0.0,
        })
        .collect();
    let med = model.map_or_else(|| median(&costs), |m| m.median);
    let diffs: Vec<f64> = costs.iter().map(|c| (c - med).abs()).collect();
    let mad = median(&diffs);
    let scores = diffs.iter().map(|d| 0.6745 * d / mad).collect();
    (
        scores,
        SubspaceModel {
            gm,
            median: med,
            s1,
            s2,
        },
    )
}

fn sigmoid_nan_to_num(v: f64) -> f64 {
    let v = if v.is_nan() {
        0.0
    } else if v.is_infinite() {
        v.signum() * f64::MAX
    } else {
        v
    };
    1.0 / (1.0 + (-v).exp())
}

/// sklearn `RobustScaler`: centre on the median, scale by the interquartile range.
#[derive(Debug, Clone)]
struct RobustScaler {
    center: Vec<f64>,
    scale: Vec<f64>,
}

impl RobustScaler {
    fn fit(x: &Matrix) -> Self {
        let d = x[0].len();
        let cols: Vec<Vec<f64>> = (0..d).map(|j| x.iter().map(|r| r[j]).collect()).collect();
        RobustScaler {
            center: cols.iter().map(|c| median(c)).collect(),
            scale: cols
                .iter()
                .map(|c| {
                    let iqr = percentile(c, 75.0) - percentile(c, 25.0);
                    if iqr == 0.0 {
                        1.0
                    } else {
                        iqr
                    }
                })
                .collect(),
        }
    }

    fn transform(&self, x: &Matrix) -> Vec<Vec<f64>> {
        x.iter()
            .map(|r| {
                r.iter()
                    .enumerate()
                    .map(|(j, v)| (v - self.center[j]) / self.scale[j])
                    .collect()
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct Rod {
    pub contamination: f64,
    scaler: Option<RobustScaler>,
    subspaces: Vec<SubspaceModel>,
    n_features: usize,
    state: Option<FitState>,
}

impl Default for Rod {
    fn default() -> Self {
        Self::new(0.1)
    }
}

impl Rod {
    pub fn new(contamination: f64) -> Self {
        Rod {
            contamination,
            scaler: None,
            subspaces: Vec::new(),
            n_features: 0,
            state: None,
        }
    }

    /// Scores `x`, fitting the per-subspace models first when `fit` is true.
    fn run(&mut self, x: &Matrix, fit: bool) -> Vec<f64> {
        let x: Vec<Vec<f64>> = x
            .iter()
            .map(|r| {
                let mut r = r.clone();
                r.resize(r.len().max(3), 0.0);
                r
            })
            .collect();
        let d = x[0].len();
        if d == 3 {
            let (s, m) = rod_3d(&x, if fit { None } else { self.subspaces.first() });
            if fit {
                self.subspaces = vec![m];
            }
            return s;
        }
        if fit {
            self.scaler = Some(RobustScaler::fit(&x));
        }
        let xs = self.scaler.as_ref().unwrap().transform(&x);
        let mut total = vec![0.0; xs.len()];
        let mut fitted = Vec::new();
        let mut count = 0;
        for a in 0..d {
            for b in a + 1..d {
                for c in b + 1..d {
                    let sub: Vec<Vec<f64>> = xs.iter().map(|r| vec![r[a], r[b], r[c]]).collect();
                    let (s, m) = rod_3d(
                        &sub,
                        if fit {
                            None
                        } else {
                            Some(&self.subspaces[count])
                        },
                    );
                    for (t, v) in total.iter_mut().zip(s) {
                        *t += sigmoid_nan_to_num(v);
                    }
                    fitted.push(m);
                    count += 1;
                }
            }
        }
        if fit {
            self.subspaces = fitted;
        }
        total.iter().map(|t| t / count as f64).collect()
    }
}

impl Detector for Rod {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        self.n_features = check_array(x)?;
        let scores = self.run(x, true);
        self.state = Some(FitState::from_contamination(scores, self.contamination));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.n_features)?;
        Ok(self.clone().run(x, false))
    }

    crate::state_accessor!();
}
