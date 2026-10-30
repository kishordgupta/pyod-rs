//! LUNAR: unifying local outlier detection methods via graph neural networks
//! (Goodge et al., AAAI 2022). Port of `pyod.models.lunar`.
//!
//! Each sample is represented by the distances to its k nearest training
//! neighbours. A network learns to tell real samples from generated negatives
//! (uniform noise and/or perturbed copies) using those distances, keeping the
//! epoch with the best validation AUC.

use super::{adam, tensor1, tensor2, to_vec, train_device};
use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};
use crate::data::roc_auc;
use crate::utils::{k_nearest, Rng};
use burn::module::Module;
use burn::nn::{BatchNorm, BatchNormConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::optim::GradientsParams;
use burn::tensor::{activation, Device, Tensor};

/// Which network maps the k distances to a score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LunarModel {
    /// Learns softmax weights over the neighbour distances (default).
    #[default]
    Weight,
    /// Maps the distances straight to a score.
    Score,
}

/// How negative (outlier-like) training samples are generated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NegativeSampling {
    Uniform,
    Subspace,
    #[default]
    Mixed,
}

#[derive(Module, Debug)]
struct Net {
    layers: Vec<Linear>,
    norm: Option<LayerNorm>,
    out: Linear,
    final_norm: Option<BatchNorm>,
    #[module(skip)]
    weight_model: bool,
}

impl Net {
    fn new(k: usize, weight_model: bool, dev: &Device) -> Self {
        let h = 256;
        Net {
            layers: vec![
                LinearConfig::new(k, h).init(dev),
                LinearConfig::new(h, h).init(dev),
                LinearConfig::new(h, h).init(dev),
            ],
            norm: weight_model.then(|| LayerNormConfig::new(h).init(dev)),
            out: LinearConfig::new(h, if weight_model { k } else { 1 }).init(dev),
            final_norm: weight_model.then(|| BatchNormConfig::new(1).init(dev)),
            weight_model,
        }
    }

    fn forward(&self, d: Tensor<2>) -> Tensor<1> {
        let act = |x: Tensor<2>| {
            if self.weight_model {
                activation::relu(x)
            } else {
                activation::tanh(x)
            }
        };
        let mut h = d.clone();
        for l in &self.layers {
            h = act(l.forward(h));
        }
        if let Some(n) = &self.norm {
            h = n.forward(h);
        }
        let out = self.out.forward(h);
        if self.weight_model {
            let alpha = activation::softmax(out, 1);
            let pooled = (alpha * d).sum_dim(1);
            self.final_norm
                .as_ref()
                .unwrap()
                .forward(pooled)
                .squeeze_dim(1)
        } else {
            activation::sigmoid(out).squeeze_dim(1)
        }
    }
}

#[derive(Debug, Clone)]
pub struct Lunar {
    pub model_type: LunarModel,
    pub n_neighbours: usize,
    pub negative_sampling: NegativeSampling,
    pub val_size: f64,
    /// Perturbation size for subspace negatives.
    pub epsilon: f64,
    /// Negatives generated per real sample.
    pub proportion: f64,
    pub n_epochs: usize,
    pub lr: f64,
    pub wd: f64,
    /// Min-max scale the data (PyOD's default scaler).
    pub scale: bool,
    pub random_state: u64,
    pub contamination: f64,
    net: Option<Net>,
    dev: Option<Device>,
    min: Vec<f64>,
    range: Vec<f64>,
    reference: Vec<Vec<f64>>,
    state: Option<FitState>,
}

impl Default for Lunar {
    fn default() -> Self {
        Lunar {
            model_type: LunarModel::Weight,
            n_neighbours: 5,
            negative_sampling: NegativeSampling::Mixed,
            val_size: 0.1,
            epsilon: 0.1,
            proportion: 1.0,
            n_epochs: 200,
            lr: 1e-3,
            wd: 0.1,
            scale: true,
            random_state: 42,
            contamination: 0.1,
            net: None,
            dev: None,
            min: Vec::new(),
            range: Vec::new(),
            reference: Vec::new(),
            state: None,
        }
    }
}

/// Port of `generate_negative_samples`.
fn negatives(
    x: &[Vec<f64>],
    kind: NegativeSampling,
    proportion: f64,
    eps: f64,
    rng: &mut Rng,
) -> Vec<Vec<f64>> {
    let n = (proportion * x.len() as f64) as usize;
    let d = x[0].len();
    let (lo, hi) = x
        .iter()
        .flatten()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
            (a.min(v), b.max(v))
        });
    let uniform: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..d).map(|_| lo + (hi - lo) * rng.next_f64()).collect())
        .collect();
    let subspace: Vec<Vec<f64>> = (0..n)
        .map(|_| {
            let base = &x[rng.below(x.len())];
            base.iter()
                .map(|&v| {
                    if rng.next_f64() < 0.3 {
                        v + eps * rng.normal()
                    } else {
                        v
                    }
                })
                .collect()
        })
        .collect();
    match kind {
        NegativeSampling::Uniform => uniform,
        NegativeSampling::Subspace => subspace,
        NegativeSampling::Mixed => {
            let pool: Vec<Vec<f64>> = uniform.into_iter().chain(subspace).collect();
            (0..n)
                .map(|_| pool[rng.below(pool.len())].clone())
                .collect()
        }
    }
}

impl Lunar {
    pub fn new(contamination: f64) -> Self {
        Lunar {
            contamination,
            ..Self::default()
        }
    }

    fn scaled(&self, x: &Matrix) -> Vec<Vec<f64>> {
        if self.min.is_empty() {
            return x.to_vec();
        }
        x.iter()
            .map(|r| {
                r.iter()
                    .enumerate()
                    .map(|(j, v)| (v - self.min[j]) / self.range[j])
                    .collect()
            })
            .collect()
    }

    /// Distances to the k nearest reference points; `skip_self` drops the closest one.
    fn distances(&self, x: &Matrix, skip_self: bool) -> Vec<Vec<f64>> {
        let k = self.n_neighbours;
        x.iter()
            .map(|r| {
                let nn = k_nearest(&self.reference, r, k + skip_self as usize, None);
                nn.iter().skip(skip_self as usize).map(|p| p.1).collect()
            })
            .collect()
    }

    fn score(&self, x: &Matrix) -> Vec<f64> {
        let d = self.distances(&self.scaled(x), false);
        to_vec(
            self.net
                .as_ref()
                .unwrap()
                .forward(tensor2(&d, self.dev.as_ref().unwrap())),
        )
    }
}

impl Detector for Lunar {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        let d = check_array(x)?;
        let n_val = ((x.len() as f64 * self.val_size).ceil() as usize).max(1);
        if self.n_neighbours == 0 || x.len() < n_val + self.n_neighbours + 2 {
            return Err(OdError::InvalidParameter(
                "need n_neighbours >= 1 and more samples than n_neighbours + validation".into(),
            ));
        }
        let mut rng = Rng::new(self.random_state);
        let perm = rng.sample_indices(x.len(), x.len());
        let (val_idx, train_idx) = perm.split_at(n_val);
        let train_raw: Vec<Vec<f64>> = train_idx.iter().map(|&i| x[i].clone()).collect();
        (self.min, self.range) = (Vec::new(), Vec::new());
        if self.scale {
            self.min = (0..d)
                .map(|j| train_raw.iter().map(|r| r[j]).fold(f64::INFINITY, f64::min))
                .collect();
            self.range = (0..d)
                .map(|j| {
                    let r = train_raw
                        .iter()
                        .map(|r| r[j])
                        .fold(f64::NEG_INFINITY, f64::max)
                        - self.min[j];
                    if r == 0.0 {
                        1.0
                    } else {
                        r
                    }
                })
                .collect();
        }
        let train = self.scaled(&train_raw);
        let val = self.scaled(&val_idx.iter().map(|&i| x[i].clone()).collect::<Vec<_>>());
        let neg_train = negatives(
            &train,
            self.negative_sampling,
            self.proportion,
            self.epsilon,
            &mut rng,
        );
        let neg_val = negatives(
            &val,
            self.negative_sampling,
            self.proportion,
            self.epsilon,
            &mut rng,
        );
        // Neighbours are searched among training samples and their negatives.
        self.reference = train.iter().chain(&neg_train).cloned().collect();
        let mut train_d = self.distances(&train, true);
        train_d.extend(self.distances(&neg_train, false));
        let train_y: Vec<f64> = std::iter::repeat_n(0.0, train.len())
            .chain(std::iter::repeat_n(1.0, neg_train.len()))
            .collect();
        let val_all: Vec<Vec<f64>> = val.iter().chain(&neg_val).cloned().collect();
        let val_d = self.distances(&val_all, false);
        let val_y: Vec<u8> = std::iter::repeat_n(0, val.len())
            .chain(std::iter::repeat_n(1, neg_val.len()))
            .collect();

        let (dev, _lock) = train_device(self.random_state);
        let mut net = Net::new(
            self.n_neighbours,
            self.model_type == LunarModel::Weight,
            &dev,
        );
        let mut opt = adam(self.wd);
        let (td, ty) = (tensor2(&train_d, &dev), tensor1(&train_y, &dev));
        let eval_dev = dev.clone().without_autodiff();
        let vd = tensor2(&val_d, &eval_dev);
        let (mut best_auc, mut best) = (f64::NEG_INFINITY, net.valid());
        for _ in 0..self.n_epochs {
            let snapshot = net.valid();
            let auc = roc_auc(&val_y, &to_vec(snapshot.forward(vd.clone())));
            if auc >= best_auc {
                (best_auc, best) = (auc, snapshot);
            }
            let loss = (net.forward(td.clone()) - ty.clone())
                .powf_scalar(2.0)
                .sum();
            let grads = GradientsParams::from_grads(loss.backward(), &net);
            net = opt.step(self.lr, net, grads);
        }
        self.net = Some(best);
        self.dev = Some(eval_dev);
        let scores = self.score(x);
        self.state = Some(FitState::from_contamination(scores, self.contamination));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.reference[0].len())?;
        Ok(self.score(x))
    }

    crate::state_accessor!();
}
