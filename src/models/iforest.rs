//! Isolation Forest (Liu et al., ICDM 2008). Port of `pyod.models.iforest`.
//!
//! Scores are the paper's anomaly score `2^(-E[h(x)] / c(psi))` in (0, 1).
//! PyOD's scores differ from these only by a constant offset, so rankings,
//! thresholds and labels agree.

use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};
use crate::utils::Rng;

#[derive(Debug, Clone)]
pub(crate) enum Node {
    Split {
        feature: usize,
        value: f64,
        left: Box<Node>,
        right: Box<Node>,
    },
    Leaf {
        size: usize,
    },
}

/// Average path length of an unsuccessful BST search over `n` points.
pub(crate) fn c(n: usize) -> f64 {
    match n {
        0 | 1 => 0.0,
        2 => 1.0,
        _ => {
            let n = n as f64;
            2.0 * ((n - 1.0).ln() + std::f64::consts::EULER_GAMMA) - 2.0 * (n - 1.0) / n
        }
    }
}

pub(crate) fn build(
    x: &Matrix,
    idx: &mut [usize],
    depth: usize,
    max_depth: usize,
    rng: &mut Rng,
) -> Node {
    if depth >= max_depth || idx.len() <= 1 {
        return Node::Leaf { size: idx.len() };
    }
    let d = x[0].len();
    // Only features that still vary within this node can split it.
    let candidates: Vec<(usize, f64, f64)> = (0..d)
        .filter_map(|f| {
            let (lo, hi) = idx
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &i| {
                    (lo.min(x[i][f]), hi.max(x[i][f]))
                });
            (hi > lo).then_some((f, lo, hi))
        })
        .collect();
    if candidates.is_empty() {
        return Node::Leaf { size: idx.len() };
    }
    let (feature, lo, hi) = candidates[rng.below(candidates.len())];
    let value = lo + rng.next_f64() * (hi - lo);
    let mut mid = 0;
    for k in 0..idx.len() {
        if x[idx[k]][feature] < value {
            idx.swap(k, mid);
            mid += 1;
        }
    }
    let (l, r) = idx.split_at_mut(mid);
    Node::Split {
        feature,
        value,
        left: Box::new(build(x, l, depth + 1, max_depth, rng)),
        right: Box::new(build(x, r, depth + 1, max_depth, rng)),
    }
}

fn path_length(node: &Node, row: &[f64], depth: f64) -> f64 {
    match node {
        Node::Leaf { size } => depth + c(*size),
        Node::Split {
            feature,
            value,
            left,
            right,
        } => {
            let next = if row[*feature] < *value { left } else { right };
            path_length(next, row, depth + 1.0)
        }
    }
}

#[derive(Debug, Clone)]
pub struct IForest {
    pub n_estimators: usize,
    /// Sub-sample size per tree; `None` means `min(256, n_samples)`.
    pub max_samples: Option<usize>,
    pub contamination: f64,
    pub random_state: u64,
    trees: Vec<Node>,
    psi: usize,
    n_features: usize,
    state: Option<FitState>,
}

impl Default for IForest {
    fn default() -> Self {
        IForest {
            n_estimators: 100,
            max_samples: None,
            contamination: 0.1,
            random_state: 42,
            trees: Vec::new(),
            psi: 0,
            n_features: 0,
            state: None,
        }
    }
}

impl IForest {
    pub fn new(n_estimators: usize, contamination: f64, random_state: u64) -> Self {
        IForest {
            n_estimators,
            contamination,
            random_state,
            ..Self::default()
        }
    }

    fn score(&self, x: &Matrix) -> Vec<f64> {
        let cn = c(self.psi);
        x.iter()
            .map(|r| {
                let mean_h = self
                    .trees
                    .iter()
                    .map(|t| path_length(t, r, 0.0))
                    .sum::<f64>()
                    / self.trees.len() as f64;
                2f64.powf(-mean_h / cn)
            })
            .collect()
    }
}

impl Detector for IForest {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        self.n_features = check_array(x)?;
        if self.n_estimators == 0 {
            return Err(OdError::InvalidParameter(
                "n_estimators must be >= 1".into(),
            ));
        }
        self.psi = self
            .max_samples
            .unwrap_or(256)
            .clamp(2, x.len().max(2))
            .min(x.len());
        let max_depth = (self.psi as f64).log2().ceil() as usize;
        let mut rng = Rng::new(self.random_state);
        self.trees = (0..self.n_estimators)
            .map(|_| {
                let mut idx = rng.sample_indices(x.len(), self.psi);
                build(x, &mut idx, 0, max_depth, &mut rng)
            })
            .collect();
        self.state = Some(FitState::from_contamination(
            self.score(x),
            self.contamination,
        ));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.n_features)?;
        Ok(self.score(x))
    }

    fn state(&self) -> Option<&FitState> {
        self.state.as_ref()
    }
}
