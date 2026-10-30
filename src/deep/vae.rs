//! Variational AutoEncoder (Kingma & Welling, 2014); the score is the
//! reconstruction error of a sampled reconstruction. Port of `pyod.models.vae`.
//!
//! As in PyOD, the KL term is clamped to `[0, capacity]`, so with the default
//! `capacity = 0` training minimises reconstruction error only.

use super::{
    minibatches, pick, row_distances, tensor2, train_device, Activation, Block, Mlp, Normalizer,
    TrainConfig,
};
use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};
use crate::utils::Rng;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::GradientsParams;
use burn::tensor::{Device, Distribution, Tensor};

#[derive(Module, Debug)]
struct Net {
    encoder: Mlp,
    mu: Linear,
    logvar: Linear,
    decoder: Mlp,
    #[module(skip)]
    clip: (f64, f64),
}

impl Net {
    fn forward(&self, x: Tensor<2>) -> (Tensor<2>, Tensor<2>, Tensor<2>) {
        let h = self.encoder.forward(x);
        let (mu, logvar) = (self.mu.forward(h.clone()), self.logvar.forward(h));
        let lv = logvar.clone().clamp(self.clip.0, self.clip.1);
        let std = (lv * 0.5).exp();
        let eps = std.random_like(Distribution::Normal(0.0, 1.0));
        (self.decoder.forward(mu.clone() + eps * std), mu, logvar)
    }
}

#[derive(Debug, Clone)]
pub struct Vae {
    pub train: TrainConfig,
    pub encoder_neurons: Vec<usize>,
    pub decoder_neurons: Vec<usize>,
    pub latent_dim: usize,
    pub hidden_activation: Activation,
    pub output_activation: Activation,
    pub batch_norm: bool,
    pub dropout_rate: f64,
    /// Weight of the KL term.
    pub beta: f64,
    /// Upper clamp on the KL term.
    pub capacity: f64,
    pub logvar_clip: (f64, f64),
    pub contamination: f64,
    net: Option<Net>,
    dev: Option<Device>,
    n_features: usize,
    norm: Normalizer,
    state: Option<FitState>,
}

impl Default for Vae {
    fn default() -> Self {
        let mut train = TrainConfig::new(30, 32, 1e-3);
        train.weight_decay = 1e-5;
        Vae {
            train,
            encoder_neurons: vec![128, 64, 32],
            decoder_neurons: vec![32, 64, 128],
            latent_dim: 2,
            hidden_activation: Activation::Relu,
            output_activation: Activation::Identity,
            batch_norm: false,
            dropout_rate: 0.2,
            beta: 1.0,
            capacity: 0.0,
            logvar_clip: (-30.0, 20.0),
            contamination: 0.1,
            net: None,
            dev: None,
            n_features: 0,
            norm: Normalizer::default(),
            state: None,
        }
    }
}

impl Vae {
    pub fn new(contamination: f64) -> Self {
        Vae {
            contamination,
            ..Self::default()
        }
    }

    fn score(&self, x: &Matrix) -> Vec<f64> {
        let x = tensor2(&self.norm.apply(x), self.dev.as_ref().unwrap());
        let (recon, _, _) = self.net.as_ref().unwrap().forward(x.clone());
        row_distances(x, recon)
    }
}

impl Detector for Vae {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        let d = check_array(x)?;
        self.n_features = d;
        if self.encoder_neurons.is_empty()
            || self.decoder_neurons.is_empty()
            || self.latent_dim == 0
            || self.logvar_clip.0 >= self.logvar_clip.1
        {
            return Err(OdError::InvalidParameter(
                "need non-empty layers, latent_dim >= 1 and a valid logvar_clip".into(),
            ));
        }
        let (dev, _lock) = train_device(self.train.random_state);
        let mut enc = vec![d];
        enc.extend(&self.encoder_neurons);
        let mut dec = vec![self.latent_dim];
        dec.extend(&self.decoder_neurons);
        let h = *self.encoder_neurons.last().unwrap();
        let mut net = Net {
            encoder: Mlp::hidden(
                &enc,
                self.hidden_activation,
                self.batch_norm,
                self.dropout_rate,
                true,
                &dev,
            ),
            mu: LinearConfig::new(h, self.latent_dim).init(&dev),
            logvar: LinearConfig::new(h, self.latent_dim).init(&dev),
            decoder: Mlp::hidden(
                &dec,
                self.hidden_activation,
                self.batch_norm,
                self.dropout_rate,
                true,
                &dev,
            )
            .push(Block::new(
                *dec.last().unwrap(),
                d,
                Some(self.output_activation),
                false,
                0.0,
                true,
                true,
                &dev,
            )),
            clip: self.logvar_clip,
        };
        self.norm = Normalizer::fit(x, self.train.preprocessing);
        let xn = self.norm.apply(x);
        let mut opt = self.train.adam();
        let mut rng = Rng::new(self.train.random_state);
        for _ in 0..self.train.epochs {
            for idx in minibatches(xn.len(), self.train.batch_size, &mut rng, true) {
                let xb = tensor2(&pick(&xn, &idx), &dev);
                let (recon, mu, logvar) = net.forward(xb.clone());
                let recon_loss = (recon - xb).powf_scalar(2.0).mean();
                let lv = logvar.clamp(self.logvar_clip.0, self.logvar_clip.1);
                let kl =
                    ((lv.clone() + 1.0 - mu.powf_scalar(2.0) - lv.exp()).sum_dim(1) * -0.5).mean();
                let loss = recon_loss + kl.clamp(0.0, self.capacity) * self.beta;
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

    /// Scores use a sampled reconstruction (as in PyOD), reseeded from
    /// `random_state` so repeated calls agree.
    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.n_features)?;
        let _lock = super::rng_lock();
        self.dev.as_ref().unwrap().seed(self.train.random_state);
        Ok(self.score(x))
    }

    crate::state_accessor!();
}
