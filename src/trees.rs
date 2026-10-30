//! Decision-tree learners used by the ensembles: a random-forest regressor
//! (SUOD's default approximator) and an XGBoost-style gradient-boosted
//! classifier (XGBOD's default supervised model).

use crate::base::Matrix;
use crate::utils::Rng;

/// A regressor that can stand in for an unsupervised detector's scores (SUOD).
pub trait Regressor {
    fn fit(&mut self, x: &Matrix, y: &[f64]);
    fn predict(&self, x: &Matrix) -> Vec<f64>;
}

/// A binary classifier returning outlier probabilities (XGBOD).
pub trait Classifier {
    fn fit(&mut self, x: &Matrix, y: &[u8]);
    fn predict_proba(&self, x: &Matrix) -> Vec<f64>;
}

#[derive(Debug, Clone)]
enum Node {
    Split {
        feature: usize,
        threshold: f64,
        left: usize,
        right: usize,
    },
    Leaf(f64),
}

/// A binary tree split greedily on per-sample gradient statistics.
/// With `g = -y`, `h = 1` and `lambda = 0` this is a CART regression tree on squared error.
#[derive(Debug, Clone, Default)]
struct Tree {
    nodes: Vec<Node>,
}

struct TreeParams {
    max_depth: usize,
    min_samples_split: usize,
    max_features: usize,
    lambda: f64,
    min_child_weight: f64,
}

impl Tree {
    fn fit(
        x: &Matrix,
        g: &[f64],
        h: &[f64],
        idx: Vec<usize>,
        p: &TreeParams,
        rng: &mut Rng,
    ) -> Self {
        let mut t = Tree { nodes: Vec::new() };
        t.grow(x, g, h, idx, 0, p, rng);
        t
    }

    fn leaf_value(g: f64, h: f64, lambda: f64) -> f64 {
        -g / (h + lambda)
    }

    #[allow(clippy::too_many_arguments)]
    fn grow(
        &mut self,
        x: &Matrix,
        g: &[f64],
        h: &[f64],
        mut idx: Vec<usize>,
        depth: usize,
        p: &TreeParams,
        rng: &mut Rng,
    ) -> usize {
        let (gs, hs): (f64, f64) = idx
            .iter()
            .fold((0.0, 0.0), |(a, b), &i| (a + g[i], b + h[i]));
        let me = self.nodes.len();
        self.nodes
            .push(Node::Leaf(Self::leaf_value(gs, hs, p.lambda)));
        if depth >= p.max_depth || idx.len() < p.min_samples_split {
            return me;
        }
        let d = x[0].len();
        let features = rng.sample_indices(d, p.max_features.clamp(1, d));
        let parent_score = gs * gs / (hs + p.lambda);
        let mut best: Option<(f64, usize, f64)> = None;
        for &f in &features {
            idx.sort_by(|&a, &b| x[a][f].total_cmp(&x[b][f]));
            let (mut gl, mut hl) = (0.0, 0.0);
            for k in 0..idx.len() - 1 {
                let i = idx[k];
                gl += g[i];
                hl += h[i];
                let (v, next) = (x[i][f], x[idx[k + 1]][f]);
                if v == next {
                    continue;
                }
                let (gr, hr) = (gs - gl, hs - hl);
                if hl < p.min_child_weight || hr < p.min_child_weight {
                    continue;
                }
                let gain = gl * gl / (hl + p.lambda) + gr * gr / (hr + p.lambda) - parent_score;
                if gain > 1e-12 && best.is_none_or(|b| gain > b.0) {
                    best = Some((gain, f, 0.5 * (v + next)));
                }
            }
        }
        let Some((_, feature, threshold)) = best else {
            return me;
        };
        let (l, r): (Vec<usize>, Vec<usize>) =
            idx.into_iter().partition(|&i| x[i][feature] <= threshold);
        let left = self.grow(x, g, h, l, depth + 1, p, rng);
        let right = self.grow(x, g, h, r, depth + 1, p, rng);
        self.nodes[me] = Node::Split {
            feature,
            threshold,
            left,
            right,
        };
        me
    }

    fn predict(&self, r: &[f64]) -> f64 {
        let mut n = 0;
        loop {
            match self.nodes[n] {
                Node::Leaf(v) => return v,
                Node::Split {
                    feature,
                    threshold,
                    left,
                    right,
                } => n = if r[feature] <= threshold { left } else { right },
            }
        }
    }
}

/// Random-forest regressor (bootstrap samples, fully grown CART trees),
/// like sklearn's `RandomForestRegressor`.
#[derive(Debug, Clone)]
pub struct RandomForestRegressor {
    pub n_estimators: usize,
    pub max_depth: Option<usize>,
    /// Fraction of features considered at each split.
    pub max_features: f64,
    pub random_state: u64,
    trees: Vec<Tree>,
}

impl RandomForestRegressor {
    pub fn new(n_estimators: usize, random_state: u64) -> Self {
        RandomForestRegressor {
            n_estimators,
            max_depth: None,
            max_features: 1.0,
            random_state,
            trees: Vec::new(),
        }
    }
}

impl Regressor for RandomForestRegressor {
    fn fit(&mut self, x: &Matrix, y: &[f64]) {
        let n = x.len();
        let mut rng = Rng::new(self.random_state);
        let g: Vec<f64> = y.iter().map(|v| -v).collect();
        let h = vec![1.0; n];
        let p = TreeParams {
            max_depth: self.max_depth.unwrap_or(usize::MAX),
            min_samples_split: 2,
            max_features: ((self.max_features * x[0].len() as f64).ceil() as usize).max(1),
            lambda: 0.0,
            min_child_weight: 1.0,
        };
        self.trees = (0..self.n_estimators)
            .map(|_| {
                let idx: Vec<usize> = (0..n).map(|_| rng.below(n)).collect();
                Tree::fit(x, &g, &h, idx, &p, &mut rng)
            })
            .collect();
    }

    fn predict(&self, x: &Matrix) -> Vec<f64> {
        x.iter()
            .map(|r| self.trees.iter().map(|t| t.predict(r)).sum::<f64>() / self.trees.len() as f64)
            .collect()
    }
}

/// Gradient-boosted trees on logistic loss with Newton leaf weights and
/// L2-regularised split gains, following XGBoost's defaults.
#[derive(Debug, Clone)]
pub struct GradientBoostingClassifier {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub max_depth: usize,
    /// L2 regularisation on leaf weights (`reg_lambda`).
    pub lambda: f64,
    pub min_child_weight: f64,
    /// Initial prediction (probability) before any tree, as XGBoost's `base_score`.
    pub base_score: f64,
    pub random_state: u64,
    base: f64,
    trees: Vec<Tree>,
}

impl Default for GradientBoostingClassifier {
    fn default() -> Self {
        GradientBoostingClassifier {
            n_estimators: 100,
            learning_rate: 0.3,
            max_depth: 6,
            lambda: 1.0,
            min_child_weight: 1.0,
            base_score: 0.5,
            random_state: 0,
            base: 0.0,
            trees: Vec::new(),
        }
    }
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

impl GradientBoostingClassifier {
    fn margin(&self, r: &[f64]) -> f64 {
        self.base
            + self
                .trees
                .iter()
                .map(|t| self.learning_rate * t.predict(r))
                .sum::<f64>()
    }
}

impl Classifier for GradientBoostingClassifier {
    fn fit(&mut self, x: &Matrix, y: &[u8]) {
        let n = x.len();
        let mut rng = Rng::new(self.random_state);
        let b = self.base_score.clamp(1e-6, 1.0 - 1e-6);
        self.base = (b / (1.0 - b)).ln();
        self.trees.clear();
        let mut margin = vec![self.base; n];
        let p = TreeParams {
            max_depth: self.max_depth,
            min_samples_split: 2,
            max_features: x[0].len(),
            lambda: self.lambda,
            min_child_weight: self.min_child_weight,
        };
        for _ in 0..self.n_estimators {
            let prob: Vec<f64> = margin.iter().map(|&m| sigmoid(m)).collect();
            let g: Vec<f64> = prob.iter().zip(y).map(|(p, &t)| p - t as f64).collect();
            let h: Vec<f64> = prob.iter().map(|p| (p * (1.0 - p)).max(1e-16)).collect();
            let tree = Tree::fit(x, &g, &h, (0..n).collect(), &p, &mut rng);
            for (m, r) in margin.iter_mut().zip(x) {
                *m += self.learning_rate * tree.predict(r);
            }
            self.trees.push(tree);
        }
    }

    fn predict_proba(&self, x: &Matrix) -> Vec<f64> {
        x.iter().map(|r| sigmoid(self.margin(r))).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forest_fits_a_step() {
        let x: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..100).map(|i| if i < 50 { 0.0 } else { 10.0 }).collect();
        let mut rf = RandomForestRegressor::new(20, 1);
        rf.fit(&x, &y);
        let p = rf.predict(&[vec![10.0], vec![90.0]]);
        assert!(p[0] < 1.0 && p[1] > 9.0, "{p:?}");
    }

    #[test]
    fn boosting_separates_classes() {
        let x: Vec<Vec<f64>> = (0..200)
            .map(|i| vec![(i % 20) as f64, (i / 20) as f64])
            .collect();
        let y: Vec<u8> = x.iter().map(|r| (r[0] + r[1] > 20.0) as u8).collect();
        let mut gb = GradientBoostingClassifier::default();
        gb.fit(&x, &y);
        let p = gb.predict_proba(&x);
        let acc = p
            .iter()
            .zip(&y)
            .filter(|(p, &t)| (**p > 0.5) as u8 == t)
            .count();
        assert!(acc >= 195, "accuracy {acc}/200");
    }
}
