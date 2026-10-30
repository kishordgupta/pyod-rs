//! LMDD: Linear Method for Deviation Detection (Arning et al., 1996). Port of `pyod.models.lmdd`.
//!
//! As in PyOD, `decision_function` scores the given samples against each other.

use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};
use crate::utils::{percentile, Rng};

/// Dissimilarity of a set of rows, computed over all their entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Dissimilarity {
    /// Mean absolute deviation.
    #[default]
    Aad,
    Var,
    /// Interquartile range.
    Iqr,
}

impl Dissimilarity {
    fn eval<'a>(&self, rows: impl Iterator<Item = &'a Vec<f64>>) -> f64 {
        let v: Vec<f64> = rows.flatten().copied().collect();
        let m = v.iter().sum::<f64>() / v.len() as f64;
        match self {
            Dissimilarity::Aad => v.iter().map(|x| (x - m).abs()).sum::<f64>() / v.len() as f64,
            Dissimilarity::Var => v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64,
            Dissimilarity::Iqr => percentile(&v, 75.0) - percentile(&v, 25.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Lmdd {
    pub n_iter: usize,
    pub dis_measure: Dissimilarity,
    pub contamination: f64,
    pub random_state: u64,
    n_features: usize,
    state: Option<FitState>,
}

impl Default for Lmdd {
    fn default() -> Self {
        Lmdd {
            n_iter: 50,
            dis_measure: Dissimilarity::Aad,
            contamination: 0.1,
            random_state: 0,
            n_features: 0,
            state: None,
        }
    }
}

impl Lmdd {
    pub fn new(n_iter: usize, dis_measure: Dissimilarity, contamination: f64) -> Self {
        Lmdd {
            n_iter,
            dis_measure,
            contamination,
            ..Self::default()
        }
    }

    /// Port of `__dis`: smoothing factor of each row in this ordering.
    fn dis(&self, x: &[&Vec<f64>]) -> Vec<f64> {
        let n = x.len();
        let mut res = vec![0.0; n];
        let f = |rows: &[&Vec<f64>]| self.dis_measure.eval(rows.iter().copied());
        let (mut var_max, mut j) = (f64::NEG_INFINITY, 0);
        for i in 1..n {
            let v = f(&x[..=i]) - f(&x[..i]);
            if v > var_max {
                var_max = v;
                j = i;
            }
        }
        if var_max > res[j] {
            res[j] = var_max;
            for k in j + 1..n {
                let mut a: Vec<&Vec<f64>> = x[..j].to_vec();
                a.push(x[k]);
                let mut b: Vec<&Vec<f64>> = x[..=j].to_vec();
                b.push(x[k]);
                let diff = f(&a) - f(&b);
                if diff >= 0.0 {
                    res[k] = diff + var_max;
                }
            }
        }
        res
    }

    fn lmdd(&self, x: &Matrix) -> Vec<f64> {
        let n = x.len();
        let mut dis = vec![0.0; n];
        let mut card = vec![0.0; n];
        let mut rng = Rng::new(self.random_state);
        for _ in 0..self.n_iter {
            let perm = rng.sample_indices(n, n);
            let shuffled: Vec<&Vec<f64>> = perm.iter().map(|&i| &x[i]).collect();
            let res = self.dis(&shuffled);
            let current_card = (n - res.iter().filter(|&&v| v > 0.0).count()) as f64;
            for (pos, &orig) in perm.iter().enumerate() {
                if res[pos] > dis[orig] {
                    dis[orig] = res[pos];
                    card[orig] = current_card;
                }
            }
        }
        dis.iter().zip(&card).map(|(d, c)| d * c).collect()
    }
}

impl Detector for Lmdd {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        self.n_features = check_array(x)?;
        if self.n_iter == 0 {
            return Err(OdError::InvalidParameter("n_iter must be >= 1".into()));
        }
        self.state = Some(FitState::from_contamination(
            self.lmdd(x),
            self.contamination,
        ));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.n_features)?;
        Ok(self.lmdd(x))
    }

    crate::state_accessor!();
}
