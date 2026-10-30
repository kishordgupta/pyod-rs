//! AE-1SVM: autoencoder with a jointly trained one-class SVM on random Fourier
//! features of the code (Nguyen & Vien, 2019). Port of `pyod.models.ae1svm`;
//! the score is the reconstruction error.

use super::{
    minibatches, pick, row_distances, tensor2, train_device, Activation, Block, Mlp, TrainConfig,
};
use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};
use crate::utils::Rng;
use burn::module::{Module, Param};
use burn::optim::GradientsParams;
use burn::tensor::{Device, Distribution, Tensor};
use std::f64::consts::PI;

#[derive(Module, Debug)]
struct Net {
    encoder: Mlp,
    decoder: Mlp,
    rff_w: Param<Tensor<2>>,
    rff_b: Param<Tensor<1>>,
    svm_w: Param<Tensor<1>>,
    svm_b: Param<Tensor<1>>,
}

impl Net {
    fn forward(&self, x: Tensor<2>) -> (Tensor<2>, Tensor<1>) {
        let code = self.encoder.forward(x);
        let rff = (code.clone().matmul(self.rff_w.val()) + self.rff_b.val().unsqueeze_dim(0)).cos();
        let svm = rff
            .matmul(self.svm_w.val().unsqueeze_dim(1))
            .squeeze_dim::<1>(1)
            + self.svm_b.val();
        (self.decoder.forward(code), svm)
    }
}

#[derive(Debug, Clone)]
pub struct Ae1Svm {
    pub train: TrainConfig,
    pub hidden_neurons: Vec<usize>,
    pub hidden_activation: Activation,
    pub batch_norm: bool,
    pub dropout_rate: f64,
    /// Weight of the reconstruction loss relative to the SVM hinge loss.
    pub alpha: f64,
    /// Scale of the random Fourier feature weights.
    pub sigma: f64,
    pub kernel_approx_features: usize,
    pub encoding_dim: usize,
    pub contamination: f64,
    net: Option<Net>,
    dev: Option<Device>,
    mean: Vec<f64>,
    std: Vec<f64>,
    n_features: usize,
    state: Option<FitState>,
}

impl Default for Ae1Svm {
    fn default() -> Self {
        let mut train = TrainConfig::new(50, 32, 1e-3);
        train.weight_decay = 1e-5;
        Ae1Svm {
            train,
            hidden_neurons: vec![64, 32],
            hidden_activation: Activation::Relu,
            batch_norm: true,
            dropout_rate: 0.2,
            alpha: 1.0,
            sigma: 1.0,
            kernel_approx_features: 1000,
            encoding_dim: 32,
            contamination: 0.1,
            net: None,
            dev: None,
            mean: Vec::new(),
            std: Vec::new(),
            n_features: 0,
            state: None,
        }
    }
}

impl Ae1Svm {
    pub fn new(contamination: f64) -> Self {
        Ae1Svm {
            contamination,
            ..Self::default()
        }
    }

    /// `(x - mean) / (std + 1e-8)`, with zero stds replaced by 1e-6 as in PyOD.
    fn normalize(&self, x: &Matrix) -> Vec<Vec<f64>> {
        if self.mean.is_empty() {
            return x.to_vec();
        }
        x.iter()
            .map(|r| {
                r.iter()
                    .enumerate()
                    .map(|(j, v)| (v - self.mean[j]) / (self.std[j] + 1e-8))
                    .collect()
            })
            .collect()
    }

    fn score(&self, x: &Matrix) -> Vec<f64> {
        let x = tensor2(&self.normalize(x), self.dev.as_ref().unwrap());
        let (recon, _) = self.net.as_ref().unwrap().forward(x.clone());
        row_distances(x, recon)
    }
}

impl Detector for Ae1Svm {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        let d = check_array(x)?;
        self.n_features = d;
        if self.encoding_dim == 0 || self.kernel_approx_features == 0 {
            return Err(OdError::InvalidParameter(
                "encoding_dim and kernel_approx_features must be >= 1".into(),
            ));
        }
        (self.mean, self.std) = (Vec::new(), Vec::new());
        if self.train.preprocessing {
            let n = x.len() as f64;
            self.mean = (0..d)
                .map(|j| x.iter().map(|r| r[j]).sum::<f64>() / n)
                .collect();
            self.std = (0..d)
                .map(|j| {
                    let s =
                        (x.iter().map(|r| (r[j] - self.mean[j]).powi(2)).sum::<f64>() / n).sqrt();
                    if s == 0.0 {
                        1e-6
                    } else {
                        s
                    }
                })
                .collect();
        }
        let (dev, _lock) = train_device(self.train.random_state);
        let mut sizes = vec![d];
        sizes.extend(&self.hidden_neurons);
        sizes.push(self.encoding_dim);
        let (act, bn, p) = (
            Some(self.hidden_activation),
            self.batch_norm,
            self.dropout_rate,
        );
        let encoder = Mlp::new(
            sizes
                .windows(2)
                .map(|w| Block::new(w[0], w[1], act, bn, p, true, false, &dev))
                .collect(),
        );
        let rev: Vec<usize> = sizes.iter().rev().copied().collect();
        let last = rev.len() - 2;
        // The decoder's last layer keeps the activation but has no batch norm or dropout.
        let decoder = Mlp::new(
            rev.windows(2)
                .enumerate()
                .map(|(i, w)| {
                    if i < last {
                        Block::new(w[0], w[1], act, bn, p, true, false, &dev)
                    } else {
                        Block::new(w[0], w[1], act, false, 0.0, true, false, &dev)
                    }
                })
                .collect(),
        );
        let (e, k) = (self.encoding_dim, self.kernel_approx_features);
        let normal = Distribution::Normal(0.0, 1.0);
        let mut net = Net {
            encoder,
            decoder,
            rff_w: Param::from_tensor(Tensor::<2>::random([e, k], normal, &dev) * self.sigma),
            rff_b: Param::from_tensor(Tensor::<1>::random([k], normal, &dev) * (2.0 * PI)),
            svm_w: Param::from_tensor(Tensor::<1>::random([k], normal, &dev)),
            svm_b: Param::from_tensor(Tensor::<1>::random([1], normal, &dev)),
        };
        let xn = self.normalize(x);
        let mut opt = self.train.adam();
        let mut rng = Rng::new(self.train.random_state);
        for _ in 0..self.train.epochs {
            for idx in minibatches(xn.len(), self.train.batch_size, &mut rng, false) {
                let xb = tensor2(&pick(&xn, &idx), &dev);
                let (recon, svm) = net.forward(xb.clone());
                let recon_loss = (xb - recon).powf_scalar(2.0).mean();
                let hinge = (svm.neg() + 1.0).clamp_min(0.0).mean();
                let loss = recon_loss * self.alpha + hinge;
                let grads = GradientsParams::from_grads(loss.backward(), &net);
                net = opt.step(self.train.lr, net, grads);
            }
        }
        self.net = Some(net.valid());
        self.dev = Some(dev.without_autodiff());
        let scores = self.score(x);
        self.state = Some(FitState::from_contamination(scores, self.contamination));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.n_features)?;
        Ok(self.score(x))
    }

    crate::state_accessor!();
}
