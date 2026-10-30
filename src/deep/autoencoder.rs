//! Fully-connected AutoEncoder; the score is the reconstruction error.
//! Port of `pyod.models.auto_encoder`.

use super::{
    minibatches, pick, row_distances, tensor2, train_device, Activation, Block, Mlp, Normalizer,
    TrainConfig,
};
use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};
use crate::utils::Rng;
use burn::module::Module;
use burn::optim::GradientsParams;
use burn::tensor::{Device, Tensor};

#[derive(Module, Debug)]
struct Net {
    encoder: Mlp,
    decoder: Mlp,
}

impl Net {
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.decoder.forward(self.encoder.forward(x))
    }
}

#[derive(Debug, Clone)]
pub struct AutoEncoder {
    pub train: TrainConfig,
    pub hidden_neurons: Vec<usize>,
    pub hidden_activation: Activation,
    pub batch_norm: bool,
    pub dropout_rate: f64,
    pub contamination: f64,
    net: Option<Net>,
    dev: Option<Device>,
    n_features: usize,
    norm: Normalizer,
    state: Option<FitState>,
}

impl Default for AutoEncoder {
    fn default() -> Self {
        let mut train = TrainConfig::new(10, 32, 1e-3);
        train.weight_decay = 1e-5;
        AutoEncoder {
            train,
            hidden_neurons: vec![64, 32],
            hidden_activation: Activation::Relu,
            batch_norm: true,
            dropout_rate: 0.2,
            contamination: 0.1,
            net: None,
            dev: None,
            n_features: 0,
            norm: Normalizer::default(),
            state: None,
        }
    }
}

impl AutoEncoder {
    pub fn new(contamination: f64) -> Self {
        AutoEncoder {
            contamination,
            ..Self::default()
        }
    }

    fn score(&self, x: &Matrix) -> Vec<f64> {
        let (net, dev) = (self.net.as_ref().unwrap(), self.dev.as_ref().unwrap());
        let x = tensor2(&self.norm.apply(x), dev);
        row_distances(x.clone(), net.forward(x))
    }
}

impl Detector for AutoEncoder {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        let d = check_array(x)?;
        self.n_features = d;
        if self.hidden_neurons.is_empty() {
            return Err(OdError::InvalidParameter(
                "hidden_neurons must not be empty".into(),
            ));
        }
        let (dev, _lock) = train_device(self.train.random_state);
        let mut enc_sizes = vec![d];
        enc_sizes.extend(&self.hidden_neurons);
        let encoder = Mlp::hidden(
            &enc_sizes,
            self.hidden_activation,
            self.batch_norm,
            self.dropout_rate,
            true,
            &dev,
        );
        let dec_sizes: Vec<usize> = self.hidden_neurons.iter().rev().copied().collect();
        let last = *dec_sizes.last().unwrap();
        let decoder = Mlp::hidden(
            &dec_sizes,
            self.hidden_activation,
            self.batch_norm,
            self.dropout_rate,
            true,
            &dev,
        )
        .push(Block::new(last, d, None, false, 0.0, true, false, &dev));
        let mut net = Net { encoder, decoder };
        self.norm = Normalizer::fit(x, self.train.preprocessing);
        let xn = self.norm.apply(x);
        let mut opt = self.train.adam();
        let mut rng = Rng::new(self.train.random_state);
        for _ in 0..self.train.epochs {
            for idx in minibatches(xn.len(), self.train.batch_size, &mut rng, true) {
                let xb = tensor2(&pick(&xn, &idx), &dev);
                let loss = (net.forward(xb.clone()) - xb).powf_scalar(2.0).mean();
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
