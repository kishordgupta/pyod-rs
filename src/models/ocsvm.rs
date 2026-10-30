//! One-Class SVM (Schölkopf et al., 2001). Port of `pyod.models.ocsvm`
//! (sklearn `OneClassSVM`, i.e. libsvm). The dual is solved with libsvm's SMO
//! using second-order working-set selection; the score is the negated decision value.

use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};
use crate::utils::{dot, sq_euclidean};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SvmKernel {
    Linear,
    /// `(gamma * <x, y> + coef0)^degree`
    Poly {
        degree: i32,
        coef0: f64,
    },
    Rbf,
    /// `tanh(gamma * <x, y> + coef0)`
    Sigmoid {
        coef0: f64,
    },
}

impl SvmKernel {
    pub(crate) fn eval(&self, gamma: f64, a: &[f64], b: &[f64]) -> f64 {
        match *self {
            SvmKernel::Linear => dot(a, b),
            SvmKernel::Poly { degree, coef0 } => (gamma * dot(a, b) + coef0).powi(degree),
            SvmKernel::Rbf => (-gamma * sq_euclidean(a, b)).exp(),
            SvmKernel::Sigmoid { coef0 } => (gamma * dot(a, b) + coef0).tanh(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OcSvm {
    pub kernel: SvmKernel,
    /// Kernel coefficient; `None` means `1 / n_features` (sklearn's `gamma='auto'`).
    pub gamma: Option<f64>,
    /// Upper bound on the training error fraction and lower bound on the support-vector fraction.
    pub nu: f64,
    /// Stopping tolerance.
    pub tol: f64,
    pub max_iter: Option<usize>,
    pub contamination: f64,
    pub support_vectors: Vec<Vec<f64>>,
    /// `alpha_i` for each support vector (`dual_coef_`).
    pub dual_coef: Vec<f64>,
    /// Offset of the decision function (`-intercept_`).
    pub rho: f64,
    gamma_: f64,
    state: Option<FitState>,
}

impl Default for OcSvm {
    fn default() -> Self {
        OcSvm {
            kernel: SvmKernel::Rbf,
            gamma: None,
            nu: 0.5,
            tol: 1e-3,
            max_iter: None,
            contamination: 0.1,
            support_vectors: Vec::new(),
            dual_coef: Vec::new(),
            rho: 0.0,
            gamma_: 0.0,
            state: None,
        }
    }
}

const TAU: f64 = 1e-12;

/// Kernel rows, precomputed up to a size limit and computed on demand beyond it.
struct Q<'a> {
    x: &'a Matrix,
    kernel: SvmKernel,
    gamma: f64,
    full: Option<Vec<Vec<f64>>>,
}

impl Q<'_> {
    fn row(&self, i: usize) -> std::borrow::Cow<'_, [f64]> {
        match &self.full {
            Some(m) => std::borrow::Cow::Borrowed(&m[i]),
            None => std::borrow::Cow::Owned(
                self.x
                    .iter()
                    .map(|r| self.kernel.eval(self.gamma, &self.x[i], r))
                    .collect(),
            ),
        }
    }
}

impl OcSvm {
    pub fn new(nu: f64, contamination: f64) -> Self {
        OcSvm {
            nu,
            contamination,
            ..Self::default()
        }
    }

    /// libsvm `solve_one_class`: min 0.5 a'Qa  s.t. 0 <= a_i <= 1, sum a = nu * l.
    fn solve(&self, x: &Matrix) -> (Vec<f64>, f64) {
        let l = x.len();
        let q = Q {
            x,
            kernel: self.kernel,
            gamma: self.gamma_,
            full: (l <= 6000).then(|| {
                (0..l)
                    .map(|i| {
                        (0..l)
                            .map(|j| self.kernel.eval(self.gamma_, &x[i], &x[j]))
                            .collect()
                    })
                    .collect()
            }),
        };
        let qd: Vec<f64> = (0..l)
            .map(|i| self.kernel.eval(self.gamma_, &x[i], &x[i]))
            .collect();
        let total = self.nu * l as f64;
        let n_full = total as usize;
        let mut alpha = vec![0.0; l];
        for a in alpha.iter_mut().take(n_full.min(l)) {
            *a = 1.0;
        }
        if n_full < l {
            alpha[n_full] = total - n_full as f64;
        }
        // Gradient G = Q alpha (the linear term is zero).
        let mut g = vec![0.0; l];
        for i in 0..l {
            if alpha[i] > 0.0 {
                let qi = q.row(i);
                for k in 0..l {
                    g[k] += alpha[i] * qi[k];
                }
            }
        }
        let max_iter = self.max_iter.unwrap_or((100 * l).max(10_000_000));
        for _ in 0..max_iter {
            // Working-set selection (WSS 2, Fan et al. 2005), all labels +1.
            let mut gmax = f64::NEG_INFINITY;
            let mut i_sel = usize::MAX;
            for t in 0..l {
                if alpha[t] < 1.0 && -g[t] >= gmax {
                    gmax = -g[t];
                    i_sel = t;
                }
            }
            if i_sel == usize::MAX {
                break;
            }
            let qi = q.row(i_sel);
            let mut gmax2 = f64::NEG_INFINITY;
            let mut j_sel = usize::MAX;
            let mut obj_min = f64::INFINITY;
            for t in 0..l {
                if alpha[t] > 0.0 {
                    let grad_diff = gmax + g[t];
                    if g[t] >= gmax2 {
                        gmax2 = g[t];
                    }
                    if grad_diff > 0.0 {
                        let quad = qd[i_sel] + qd[t] - 2.0 * qi[t];
                        let obj = -(grad_diff * grad_diff) / if quad > 0.0 { quad } else { TAU };
                        if obj <= obj_min {
                            j_sel = t;
                            obj_min = obj;
                        }
                    }
                }
            }
            if gmax + gmax2 < self.tol || j_sel == usize::MAX {
                break;
            }
            let (i, j) = (i_sel, j_sel);
            let qj = q.row(j);
            let (old_i, old_j) = (alpha[i], alpha[j]);
            let mut quad = qd[i] + qd[j] - 2.0 * qi[j];
            if quad <= 0.0 {
                quad = TAU;
            }
            let delta = (g[i] - g[j]) / quad;
            let sum = alpha[i] + alpha[j];
            alpha[i] -= delta;
            alpha[j] += delta;
            if sum > 1.0 {
                if alpha[i] > 1.0 {
                    alpha[i] = 1.0;
                    alpha[j] = sum - 1.0;
                }
            } else if alpha[j] < 0.0 {
                alpha[j] = 0.0;
                alpha[i] = sum;
            }
            if sum > 1.0 {
                if alpha[j] > 1.0 {
                    alpha[j] = 1.0;
                    alpha[i] = sum - 1.0;
                }
            } else if alpha[i] < 0.0 {
                alpha[i] = 0.0;
                alpha[j] = sum;
            }
            let (di, dj) = (alpha[i] - old_i, alpha[j] - old_j);
            for k in 0..l {
                g[k] += qi[k] * di + qj[k] * dj;
            }
        }
        // libsvm `calculate_rho`.
        let (mut ub, mut lb, mut sum_free, mut n_free) =
            (f64::INFINITY, f64::NEG_INFINITY, 0.0, 0usize);
        for t in 0..l {
            if alpha[t] >= 1.0 {
                lb = lb.max(g[t]);
            } else if alpha[t] <= 0.0 {
                ub = ub.min(g[t]);
            } else {
                n_free += 1;
                sum_free += g[t];
            }
        }
        let rho = if n_free > 0 {
            sum_free / n_free as f64
        } else {
            (ub + lb) / 2.0
        };
        (alpha, rho)
    }

    fn score(&self, x: &Matrix) -> Vec<f64> {
        x.iter()
            .map(|r| {
                let f: f64 = self
                    .support_vectors
                    .iter()
                    .zip(&self.dual_coef)
                    .map(|(sv, a)| a * self.kernel.eval(self.gamma_, sv, r))
                    .sum();
                -(f - self.rho)
            })
            .collect()
    }
}

impl Detector for OcSvm {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        let d = check_array(x)?;
        if !(self.nu > 0.0 && self.nu <= 1.0) {
            return Err(OdError::InvalidParameter(format!(
                "nu must be in (0, 1], got {}",
                self.nu
            )));
        }
        self.gamma_ = self.gamma.unwrap_or(1.0 / d as f64);
        let (alpha, rho) = self.solve(x);
        self.rho = rho;
        (self.support_vectors, self.dual_coef) = x
            .iter()
            .zip(&alpha)
            .filter(|(_, &a)| a > 0.0)
            .map(|(r, &a)| (r.clone(), a))
            .unzip();
        self.state = Some(FitState::from_contamination(
            self.score(x),
            self.contamination,
        ));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.support_vectors[0].len())?;
        Ok(self.score(x))
    }

    crate::state_accessor!();
}
