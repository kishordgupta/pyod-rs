//! DIF: Deep Isolation Forest (Xu et al., TKDE 2023). Port of `pyod.models.dif`.
//!
//! Each ensemble member projects the min-max-scaled data through a randomly
//! initialised (never trained) bias-free tanh network, standardises and
//! squashes the representation, and fits a small isolation forest on it. The
//! score combines isolation depth with how far samples fall from the split
//! thresholds along their paths. No neural-network training is involved, so
//! this detector does not need the `deep` feature.
//!
//! As in PyOD, the representation of new data is standardised with its own
//! statistics (not the training data's).

use super::iforest::{build, c, Node};
use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};
use crate::linalg::matmul;
use crate::utils::{Rng, StandardScaler};

#[derive(Debug, Clone)]
struct Member {
    /// Weight matrices (input x output) of the random network.
    weights: Vec<Vec<Vec<f64>>>,
    trees: Vec<Node>,
}

#[derive(Debug, Clone)]
pub struct Dif {
    pub representation_dim: usize,
    pub hidden_neurons: Vec<usize>,
    /// Feed each layer's input forward alongside its output (dense skip connections).
    pub skip_connection: bool,
    pub n_ensemble: usize,
    pub n_estimators: usize,
    pub max_samples: usize,
    pub random_state: u64,
    pub contamination: f64,
    min: Vec<f64>,
    range: Vec<f64>,
    members: Vec<Member>,
    psi: usize,
    state: Option<FitState>,
}

impl Default for Dif {
    fn default() -> Self {
        Dif {
            representation_dim: 20,
            hidden_neurons: vec![500, 100],
            skip_connection: false,
            n_ensemble: 50,
            n_estimators: 6,
            max_samples: 256,
            random_state: 0,
            contamination: 0.1,
            min: Vec::new(),
            range: Vec::new(),
            members: Vec::new(),
            psi: 0,
            state: None,
        }
    }
}

/// Isolation depth (path length + c(leaf size)) and mean |x_f - threshold| over the path's splits.
fn path_stats(node: &Node, r: &[f64]) -> (f64, f64) {
    let (mut depth, mut dev_sum, mut dev_n, mut n) = (0.0, 0.0, 0usize, node);
    loop {
        match n {
            Node::Leaf { size } => {
                let dev = dev_sum / (dev_n as f64 + 1e-6);
                return (depth + c(*size), dev);
            }
            Node::Split {
                feature,
                value,
                left,
                right,
            } => {
                let diff = (r[*feature] - value).abs();
                if diff != 0.0 {
                    dev_sum += diff;
                    dev_n += 1;
                }
                depth += 1.0;
                n = if r[*feature] < *value { left } else { right };
            }
        }
    }
}

impl Dif {
    pub fn new(contamination: f64) -> Self {
        Dif {
            contamination,
            ..Self::default()
        }
    }

    fn represent(&self, w: &[Vec<Vec<f64>>], x: &Matrix) -> Vec<Vec<f64>> {
        let mut h: Vec<Vec<f64>> = x.to_vec();
        let last = w.len() - 1;
        for (i, wi) in w.iter().enumerate() {
            let mut out = matmul(&h, wi);
            if i < last {
                out.iter_mut().flatten().for_each(|v| *v = v.tanh());
                if self.skip_connection {
                    out = h
                        .iter()
                        .zip(out)
                        .map(|(a, b)| a.iter().copied().chain(b).collect())
                        .collect();
                }
            }
            h = out;
        }
        let z = StandardScaler::fit(&h).transform(&h);
        z.into_iter()
            .map(|r| r.into_iter().map(f64::tanh).collect())
            .collect()
    }

    fn scaled(&self, x: &Matrix) -> Vec<Vec<f64>> {
        x.iter()
            .map(|r| {
                r.iter()
                    .enumerate()
                    .map(|(j, v)| (v - self.min[j]) / self.range[j])
                    .collect()
            })
            .collect()
    }

    fn score(&self, x: &Matrix) -> Vec<f64> {
        let xs = self.scaled(x);
        let cn = c(self.psi);
        let mut total = vec![0.0; x.len()];
        for m in &self.members {
            let z = self.represent(&m.weights, &xs);
            for (t, r) in total.iter_mut().zip(&z) {
                let (mut depth_sum, mut dev_sum) = (0.0, 0.0);
                for tree in &m.trees {
                    let (d, dev) = path_stats(tree, r);
                    depth_sum += d;
                    dev_sum += dev;
                }
                let k = m.trees.len() as f64;
                *t += 2f64.powf(-depth_sum / (k * cn)) * (dev_sum / k);
            }
        }
        total
            .iter()
            .map(|t| t / self.members.len() as f64)
            .collect()
    }
}

impl Detector for Dif {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        let d = check_array(x)?;
        if self.n_ensemble == 0 || self.n_estimators == 0 || self.representation_dim == 0 {
            return Err(OdError::InvalidParameter(
                "n_ensemble, n_estimators and representation_dim must be >= 1".into(),
            ));
        }
        self.min = (0..d)
            .map(|j| x.iter().map(|r| r[j]).fold(f64::INFINITY, f64::min))
            .collect();
        self.range = (0..d)
            .map(|j| {
                let r = x.iter().map(|r| r[j]).fold(f64::NEG_INFINITY, f64::max) - self.min[j];
                if r == 0.0 {
                    1.0
                } else {
                    r
                }
            })
            .collect();
        let xs = self.scaled(x);
        self.psi = self.max_samples.min(x.len()).max(2).min(x.len());
        let max_depth = (self.psi as f64).log2().ceil() as usize;
        let mut rng = Rng::new(self.random_state);
        self.members.clear();
        for _ in 0..self.n_ensemble {
            let mut in_dim = d;
            let mut weights = Vec::new();
            for &h in self
                .hidden_neurons
                .iter()
                .chain(std::iter::once(&self.representation_dim))
            {
                weights.push(
                    (0..in_dim)
                        .map(|_| (0..h).map(|_| rng.normal()).collect())
                        .collect(),
                );
                in_dim = if self.skip_connection { in_dim + h } else { h };
            }
            let z = self.represent(&weights, &xs);
            let trees = (0..self.n_estimators)
                .map(|_| {
                    let mut idx = rng.sample_indices(z.len(), self.psi);
                    build(&z, &mut idx, 0, max_depth, &mut rng)
                })
                .collect();
            self.members.push(Member { weights, trees });
        }
        self.state = Some(FitState::from_contamination(
            self.score(x),
            self.contamination,
        ));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.min.len())?;
        Ok(self.score(x))
    }

    crate::state_accessor!();
}
