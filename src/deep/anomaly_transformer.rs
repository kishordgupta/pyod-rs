//! Anomaly Transformer (Xu et al., ICLR 2022). Port of
//! `pyod.models.ts_anomaly_transformer`.
//!
//! Each attention layer produces a *series association* (ordinary attention) and
//! a *prior association* (a learnable Gaussian over time distance). Training
//! alternates a minimise phase (reconstruction, pushing series away from the
//! prior) and a maximise phase (pulling the prior toward the series). A
//! timestamp's score is its reconstruction error weighted by softmax(-discrepancy).

use super::{minibatches, to_vec, train_device};
use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};
use crate::ts::ts_fit_state;
use crate::utils::Rng;
use burn::module::{Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{
    Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig1d,
};
use burn::optim::GradientsParams;
use burn::tensor::{activation, Device, Distribution, Tensor, TensorData};

#[derive(Module, Debug)]
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    sigma: Param<Tensor<1>>,
    #[module(skip)]
    heads: usize,
}

impl Attention {
    /// Returns (output, series association, prior association), associations as `[B, H, L, L]`.
    fn forward(&self, x: Tensor<3>) -> (Tensor<3>, Tensor<4>, Tensor<4>) {
        let [b, l, d] = x.dims();
        let (h, dk) = (self.heads, d / self.heads);
        let split = |t: Tensor<3>| t.reshape([b, l, h, dk]).swap_dims(1, 2);
        let (q, k, v) = (
            split(self.q.forward(x.clone())),
            split(self.k.forward(x.clone())),
            split(self.v.forward(x)),
        );
        let logits = q.matmul(k.swap_dims(2, 3)) / (dk as f64).sqrt();
        let series = activation::softmax(logits, 3);
        let dev = series.device();
        let pos: Vec<f32> = (0..l)
            .flat_map(|i| (0..l).map(move |j| ((i as f32) - (j as f32)).powi(2)))
            .collect();
        let dist: Tensor<4> =
            Tensor::<2>::from_data(TensorData::new(pos, [l, l]), &dev).reshape([1, 1, l, l]);
        let sigma_sq = self
            .sigma
            .val()
            .clamp_min(1e-4)
            .powf_scalar(2.0)
            .reshape([1, h, 1, 1]);
        let prior = activation::softmax(dist.neg() / (sigma_sq * 2.0), 3).repeat_dim(0, b);
        let out = series.clone().matmul(v).swap_dims(1, 2).reshape([b, l, d]);
        (self.out.forward(out), series, prior)
    }
}

#[derive(Module, Debug)]
struct Layer {
    attn: Attention,
    norm1: LayerNorm,
    norm2: LayerNorm,
    ff1: Linear,
    ff2: Linear,
    dropout: Dropout,
}

impl Layer {
    fn forward(&self, x: Tensor<3>) -> (Tensor<3>, Tensor<4>, Tensor<4>) {
        let (a, s, p) = self.attn.forward(x.clone());
        let x = self.norm1.forward(x + self.dropout.forward(a));
        let ff = self.dropout.forward(
            self.ff2.forward(
                self.dropout
                    .forward(activation::gelu(self.ff1.forward(x.clone()))),
            ),
        );
        (self.norm2.forward(x + ff), s, p)
    }
}

#[derive(Module, Debug)]
struct Net {
    embed: Conv1d,
    pos: Param<Tensor<3>>,
    layers: Vec<Layer>,
    out: Linear,
}

impl Net {
    fn forward(&self, x: Tensor<3>) -> (Tensor<3>, Vec<(Tensor<4>, Tensor<4>)>) {
        let mut h = self.embed.forward(x.swap_dims(1, 2)).swap_dims(1, 2) + self.pos.val();
        let mut assoc = Vec::new();
        for layer in &self.layers {
            let (next, s, p) = layer.forward(h);
            h = next;
            assoc.push((s, p));
        }
        (self.out.forward(h), assoc)
    }
}

/// Symmetric KL between series and prior associations, summed over layers: `[B, L]`.
fn discrepancy(
    assoc: &[(Tensor<4>, Tensor<4>)],
    detach_prior: bool,
    detach_series: bool,
) -> Tensor<2> {
    let eps = 1e-8;
    assoc
        .iter()
        .map(|(s, p)| {
            let s = if detach_series {
                s.clone().detach()
            } else {
                s.clone()
            };
            let p = if detach_prior {
                p.clone().detach()
            } else {
                p.clone()
            };
            let (ls, lp) = ((s.clone() + eps).log(), (p.clone() + eps).log());
            let kl_ps = (p * (lp.clone() - ls.clone())).sum_dim(3);
            let kl_sp = (s * (ls - lp)).sum_dim(3);
            let [b, _, l, _] = kl_ps.dims();
            ((kl_ps + kl_sp) / 2.0).mean_dim(1).reshape([b, l])
        })
        .reduce(|a, b| a + b)
        .unwrap()
}

#[derive(Debug, Clone)]
pub struct AnomalyTransformer {
    pub window_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub epochs: usize,
    pub lr: f64,
    pub batch_size: usize,
    /// Weight of the association discrepancy in the minimise phase.
    pub lambda: f64,
    pub step: usize,
    pub dropout: f64,
    pub random_state: u64,
    pub contamination: f64,
    net: Option<Net>,
    dev: Option<Device>,
    n_channels: usize,
    state: Option<FitState>,
}

impl Default for AnomalyTransformer {
    fn default() -> Self {
        AnomalyTransformer {
            window_size: 100,
            d_model: 512,
            n_heads: 8,
            n_layers: 3,
            epochs: 10,
            lr: 1e-4,
            batch_size: 32,
            lambda: 3.0,
            step: 1,
            dropout: 0.1,
            random_state: 42,
            contamination: 0.1,
            net: None,
            dev: None,
            n_channels: 0,
            state: None,
        }
    }
}

fn windows(x: &Matrix, starts: &[usize], w: usize, dev: &Device) -> Tensor<3> {
    let c = x[0].len();
    let flat: Vec<f32> = starts
        .iter()
        .flat_map(|&s| x[s..s + w].iter().flatten().map(|&v| v as f32))
        .collect();
    Tensor::from_data(TensorData::new(flat, [starts.len(), w, c]), dev)
}

impl AnomalyTransformer {
    pub fn new(window_size: usize, contamination: f64) -> Self {
        AnomalyTransformer {
            window_size,
            contamination,
            ..Self::default()
        }
    }

    fn starts(&self, n: usize) -> Vec<usize> {
        (0..=(n - self.window_size) / self.step)
            .map(|i| i * self.step)
            .collect()
    }

    /// Per-timestamp scores averaged over covering windows; `None` where uncovered.
    fn scores(&self, x: &Matrix) -> Vec<Option<f64>> {
        let (net, dev) = (self.net.as_ref().unwrap(), self.dev.as_ref().unwrap());
        let w = self.window_size;
        let (mut acc, mut cnt) = (vec![0.0; x.len()], vec![0usize; x.len()]);
        let starts = self.starts(x.len());
        for chunk in starts.chunks(64) {
            let batch = windows(x, chunk, w, dev);
            let (recon, assoc) = net.forward(batch.clone());
            let [b, l, _] = recon.dims();
            let err = (batch - recon).powf_scalar(2.0).mean_dim(2).reshape([b, l]);
            let weights = activation::softmax(discrepancy(&assoc, false, false).neg(), 1);
            let s = to_vec(weights * err);
            for (k, &start) in chunk.iter().enumerate() {
                for t in 0..w {
                    acc[start + t] += s[k * w + t];
                    cnt[start + t] += 1;
                }
            }
        }
        acc.iter()
            .zip(&cnt)
            .map(|(a, &c)| (c > 0).then(|| a / c as f64))
            .collect()
    }
}

impl Detector for AnomalyTransformer {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        let c = check_array(x)?;
        self.n_channels = c;
        let (w, d, h) = (self.window_size, self.d_model, self.n_heads);
        if w == 0 || self.step == 0 || x.len() < w || h == 0 || d % h != 0 {
            return Err(OdError::InvalidParameter(
                "need series >= window_size and d_model divisible by n_heads".into(),
            ));
        }
        let (dev, _lock) = train_device(self.random_state);
        let mut net = Net {
            embed: Conv1dConfig::new(c, d, 3)
                .with_padding(PaddingConfig1d::Explicit(1, 1))
                .init(&dev),
            pos: Param::from_tensor(
                Tensor::<3>::random([1, w, d], Distribution::Normal(0.0, 1.0), &dev) * 0.02,
            ),
            layers: (0..self.n_layers)
                .map(|_| Layer {
                    attn: Attention {
                        q: LinearConfig::new(d, d).init(&dev),
                        k: LinearConfig::new(d, d).init(&dev),
                        v: LinearConfig::new(d, d).init(&dev),
                        out: LinearConfig::new(d, d).init(&dev),
                        sigma: Param::from_tensor(Tensor::ones([h], &dev)),
                        heads: h,
                    },
                    norm1: LayerNormConfig::new(d).init(&dev),
                    norm2: LayerNormConfig::new(d).init(&dev),
                    ff1: LinearConfig::new(d, 4 * d).init(&dev),
                    ff2: LinearConfig::new(4 * d, d).init(&dev),
                    dropout: DropoutConfig::new(self.dropout).init(),
                })
                .collect(),
            out: LinearConfig::new(d, c).init(&dev),
        };
        let mut opt = super::adam(0.0);
        let mut rng = Rng::new(self.random_state);
        let starts = self.starts(x.len());
        for _ in 0..self.epochs {
            for idx in minibatches(starts.len(), self.batch_size, &mut rng, false) {
                let batch = windows(
                    x,
                    &idx.iter().map(|&i| starts[i]).collect::<Vec<_>>(),
                    w,
                    &dev,
                );
                // Minimise phase: reconstruct, and enlarge the discrepancy against a fixed prior.
                let (recon, assoc) = net.forward(batch.clone());
                let loss = (recon - batch.clone()).powf_scalar(2.0).mean()
                    - discrepancy(&assoc, true, false).mean() * self.lambda;
                let grads = GradientsParams::from_grads(loss.backward(), &net);
                net = opt.step(self.lr, net, grads);
                // Maximise phase: move the prior toward the (fixed) series association.
                let (_, assoc) = net.forward(batch);
                let loss = discrepancy(&assoc, false, true).mean().neg();
                let grads = GradientsParams::from_grads(loss.backward(), &net);
                net = opt.step(self.lr, net, grads);
            }
        }
        self.net = Some(net.valid());
        self.dev = Some(dev.without_autodiff());
        let scores = self.scores(x);
        self.state = Some(ts_fit_state(&scores, self.contamination));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        let state = self.fitted()?;
        check_features(x, self.n_channels)?;
        if x.len() < self.window_size {
            return Err(OdError::InvalidParameter(
                "series is shorter than window_size".into(),
            ));
        }
        Ok(self
            .scores(x)
            .into_iter()
            .map(|s| s.unwrap_or(state.threshold))
            .collect())
    }

    crate::state_accessor!();
}
