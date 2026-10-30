//! R-Graph: outlier detection by random walks on an elastic-net self-representation
//! graph (You et al., CVPR 2017). Port of `pyod.models.rgraph`.
//!
//! Each sample is written as a sparse combination of the others (lasso, solved
//! with the active-support method); the absolute coefficients form a transition
//! matrix, and samples that a random walk rarely visits are outliers. PyOD solves
//! the lasso with LARS; we use coordinate descent to the same optimum.

use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, Result,
};
use crate::utils::{dot, StandardScaler};

#[derive(Debug, Clone)]
pub struct RGraph {
    pub transition_steps: usize,
    /// Maximum non-zero coefficients kept per sample.
    pub n_nonzero: usize,
    /// Regularisation: alpha = max coherence / gamma.
    pub gamma: f64,
    pub support_size: usize,
    pub maxiter: usize,
    pub maxiter_lasso: usize,
    pub active_support: bool,
    pub preprocessing: bool,
    /// Test samples are scored together with the training data in blocks of this size.
    pub blocksize_test_data: usize,
    pub contamination: f64,
    scaler: Option<StandardScaler>,
    x_train: Vec<Vec<f64>>,
    state: Option<FitState>,
}

impl Default for RGraph {
    fn default() -> Self {
        RGraph {
            transition_steps: 10,
            n_nonzero: 10,
            gamma: 50.0,
            support_size: 100,
            maxiter: 40,
            maxiter_lasso: 1000,
            active_support: true,
            preprocessing: true,
            blocksize_test_data: 10,
            contamination: 0.1,
            scaler: None,
            x_train: Vec::new(),
            state: None,
        }
    }
}

/// min_c 0.5 ||y - sum_j c_j atoms_j||^2 + alpha ||c||_1 by cyclic coordinate descent.
fn lasso(atoms: &[&Vec<f64>], y: &[f64], alpha: f64, max_iter: usize) -> Vec<f64> {
    let m = atoms.len();
    let gram: Vec<Vec<f64>> = atoms
        .iter()
        .map(|a| atoms.iter().map(|b| dot(a, b)).collect())
        .collect();
    let b: Vec<f64> = atoms.iter().map(|a| dot(a, y)).collect();
    let mut c = vec![0.0; m];
    // Coordinate descent needs many more sweeps than LARS needs steps to reach the same optimum.
    for _ in 0..max_iter.max(1) * 10 {
        let mut max_delta = 0.0f64;
        for j in 0..m {
            if gram[j][j] == 0.0 {
                continue;
            }
            let rho = b[j]
                - (0..m)
                    .filter(|&k| k != j)
                    .map(|k| gram[j][k] * c[k])
                    .sum::<f64>();
            let new = rho.signum() * (rho.abs() - alpha).max(0.0) / gram[j][j];
            max_delta = max_delta.max((new - c[j]).abs());
            c[j] = new;
        }
        if max_delta < 1e-12 {
            break;
        }
    }
    c
}

impl RGraph {
    pub fn new(contamination: f64) -> Self {
        RGraph {
            contamination,
            ..Self::default()
        }
    }

    /// Port of `active_support_elastic_net` (tau = 1, i.e. lasso).
    fn active_support_lasso(&self, x: &Matrix, y: &[f64], alpha: f64) -> Vec<f64> {
        let n = x.len();
        let mut support_size = self.support_size;
        let mut supp: Vec<usize> = if n <= support_size {
            (0..n).collect()
        } else {
            // L2 initialisation: ridge solution in feature space, projected back on the samples.
            let d = y.len();
            let mut a = vec![vec![0.0; d]; d];
            for r in x {
                for i in 0..d {
                    for j in 0..d {
                        a[i][j] += r[i] * r[j];
                    }
                }
            }
            for (i, row) in a.iter_mut().enumerate() {
                row[i] += alpha;
            }
            let w = crate::linalg::matvec(&crate::linalg::pinvh(&a), y);
            let c0: Vec<f64> = x.iter().map(|r| dot(r, &w)).collect();
            top_k(
                &c0.iter().map(|v| v.abs()).collect::<Vec<_>>(),
                support_size,
            )
        };
        let mut cs = Vec::new();
        let mut curr_obj = f64::INFINITY;
        for _ in 0..self.maxiter {
            let atoms: Vec<&Vec<f64>> = supp.iter().map(|&i| &x[i]).collect();
            cs = lasso(&atoms, y, alpha, self.maxiter_lasso);
            let recon: Vec<f64> = (0..y.len())
                .map(|k| atoms.iter().zip(&cs).map(|(a, c)| a[k] * c).sum())
                .collect();
            let delta: Vec<f64> = y.iter().zip(&recon).map(|(a, b)| (a - b) / alpha).collect();
            let obj = cs.iter().map(|c| c.abs()).sum::<f64>()
                + alpha / 2.0 * delta.iter().map(|d| d * d).sum::<f64>();
            if curr_obj - obj < 1e-10 * curr_obj {
                break;
            }
            curr_obj = obj;
            let mut coherence: Vec<f64> = x.iter().map(|r| dot(&delta, r).abs()).collect();
            supp.iter().for_each(|&s| coherence[s] = 0.0);
            let mut added: Vec<usize> = (0..n).filter(|&i| coherence[i] > 1.0 + 1e-10).collect();
            if added.is_empty() {
                break;
            }
            let active: Vec<usize> = supp
                .iter()
                .zip(&cs)
                .filter(|(_, c)| c.abs() > 1e-10)
                .map(|(&s, _)| s)
                .collect();
            if active.len() as f64 > 0.8 * support_size as f64 {
                support_size =
                    ((active.len().max(support_size) as f64 * 1.1).round() as usize).min(n);
            }
            if added.len() + active.len() > support_size {
                let keep = support_size - active.len();
                let coh: Vec<f64> = added.iter().map(|&i| coherence[i]).collect();
                added = top_k(&coh, keep).into_iter().map(|k| added[k]).collect();
            }
            supp = active.into_iter().chain(added).collect();
        }
        let mut c = vec![0.0; n];
        for (&s, v) in supp.iter().zip(&cs) {
            c[s] = *v;
        }
        c
    }

    /// Port of `_decision_function`: negative mean visiting probability of a random walk.
    fn scores(&self, xn: &Matrix) -> Vec<f64> {
        let n = xn.len();
        let mut x: Vec<Vec<f64>> = xn.to_vec();
        let mut trans = vec![vec![0.0; n]; n];
        for i in 0..n {
            let y = std::mem::replace(&mut x[i], vec![0.0; xn[0].len()]);
            let alpha = (0..n)
                .filter(|&j| j != i)
                .map(|j| dot(&x[j], &y).abs())
                .fold(0.0, f64::max)
                / self.gamma;
            let c = if alpha == 0.0 {
                vec![0.0; n]
            } else if self.active_support {
                self.active_support_lasso(&x, &y, alpha)
            } else {
                lasso(&x.iter().collect::<Vec<_>>(), &y, alpha, self.maxiter_lasso)
            };
            let mut nz: Vec<usize> = (0..n).filter(|&j| c[j] != 0.0).collect();
            if nz.len() > self.n_nonzero {
                nz.sort_by(|&a, &b| c[b].abs().total_cmp(&c[a].abs()));
                nz.truncate(self.n_nonzero);
            }
            let total: f64 = nz.iter().map(|&j| c[j].abs()).sum();
            if total > 0.0 {
                nz.iter().for_each(|&j| trans[i][j] = c[j].abs() / total);
            }
            x[i] = y;
        }
        let mut pi = vec![1.0 / n as f64; n];
        let mut pi_bar = vec![0.0; n];
        for _ in 0..self.transition_steps {
            let mut next = vec![0.0; n];
            for i in 0..n {
                if pi[i] != 0.0 {
                    for (nj, t) in next.iter_mut().zip(&trans[i]) {
                        *nj += pi[i] * t;
                    }
                }
            }
            pi = next;
            pi_bar.iter_mut().zip(&pi).for_each(|(b, p)| *b += p);
        }
        pi_bar
            .iter()
            .map(|p| -p / self.transition_steps as f64)
            .collect()
    }

    fn normalize(&self, x: &Matrix) -> Vec<Vec<f64>> {
        match &self.scaler {
            Some(s) => s.transform(x),
            None => x.to_vec(),
        }
    }
}

/// Indices of the `k` largest values.
fn top_k(v: &[f64], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]));
    idx.truncate(k);
    idx
}

impl Detector for RGraph {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        check_array(x)?;
        self.scaler = self.preprocessing.then(|| StandardScaler::fit(x));
        self.x_train = Vec::new();
        let scores = self.scores(&self.normalize(x));
        self.x_train = x.to_vec();
        self.state = Some(FitState::from_contamination(scores, self.contamination));
        Ok(())
    }

    /// Each block of test samples is scored jointly with the training data.
    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.x_train[0].len())?;
        let mut out = Vec::with_capacity(x.len());
        for block in x.chunks(self.blocksize_test_data.max(1)) {
            let all: Vec<Vec<f64>> = self.x_train.iter().chain(block).cloned().collect();
            let s = self.scores(&self.normalize(&all));
            out.extend_from_slice(&s[s.len() - block.len()..]);
        }
        Ok(out)
    }

    crate::state_accessor!();
}
