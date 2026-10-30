//! Deep Support Vector Data Description (Ruff et al., ICML 2018). Port of
//! `pyod.models.deep_svdd`: a bias-free network maps data near a fixed centre;
//! the score is the squared distance to it.

use super::{
    minibatches, pick, tensor2, to_vec, train_device, Activation, Block, Mlp, TrainConfig,
};
use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};
use crate::utils::{Rng, StandardScaler};
use burn::module::Module;
use burn::optim::GradientsParams;
use burn::tensor::{Device, Tensor};

#[derive(Debug, Clone)]
pub struct DeepSvdd {
    pub train: TrainConfig,
    /// Also reconstruct the input (adds an MSE term to the loss).
    pub use_ae: bool,
    pub hidden_neurons: Vec<usize>,
    pub hidden_activation: Activation,
    pub output_activation: Activation,
    pub dropout_rate: f64,
    /// Fixed hypersphere centre; `None` initialises it from the data.
    pub c: Option<Vec<f64>>,
    pub contamination: f64,
    net: Option<Mlp>,
    center: Vec<f64>,
    dev: Option<Device>,
    scaler: Option<StandardScaler>,
    n_features: usize,
    state: Option<FitState>,
}

impl Default for DeepSvdd {
    fn default() -> Self {
        let mut train = TrainConfig::new(100, 32, 1e-4);
        train.weight_decay = 5e-7;
        DeepSvdd {
            train,
            use_ae: false,
            hidden_neurons: vec![64, 32],
            hidden_activation: Activation::Relu,
            output_activation: Activation::Sigmoid,
            dropout_rate: 0.2,
            c: None,
            contamination: 0.1,
            net: None,
            center: Vec::new(),
            dev: None,
            scaler: None,
            n_features: 0,
            state: None,
        }
    }
}

impl DeepSvdd {
    pub fn new(contamination: f64) -> Self {
        DeepSvdd {
            contamination,
            ..Self::default()
        }
    }

    /// Port of `InnerDeepSVDD._build_model` (no biases; dropout only on inner layers).
    fn build(&self, d: usize, dev: &Device) -> Mlp {
        let h = &self.hidden_neurons;
        let mut blocks = vec![Block::new(
            d,
            h[0],
            Some(self.hidden_activation),
            false,
            0.0,
            false,
            false,
            dev,
        )];
        for i in 1..h.len() - 1 {
            blocks.push(Block::new(
                h[i - 1],
                h[i],
                Some(self.hidden_activation),
                false,
                self.dropout_rate,
                false,
                false,
                dev,
            ));
        }
        blocks.push(Block::new(
            h[h.len() - 2],
            h[h.len() - 1],
            Some(self.hidden_activation),
            false,
            0.0,
            false,
            false,
            dev,
        ));
        if self.use_ae {
            for j in (1..h.len()).rev() {
                blocks.push(Block::new(
                    h[j],
                    h[j - 1],
                    Some(self.hidden_activation),
                    false,
                    self.dropout_rate,
                    false,
                    false,
                    dev,
                ));
            }
            blocks.push(Block::new(
                h[0],
                d,
                Some(self.output_activation),
                false,
                0.0,
                false,
                false,
                dev,
            ));
        }
        Mlp::new(blocks)
    }

    fn normalize(&self, x: &Matrix) -> Vec<Vec<f64>> {
        match &self.scaler {
            Some(s) => s.transform(x),
            None => x.to_vec(),
        }
    }

    fn score(&self, x: &Matrix) -> Vec<f64> {
        let dev = self.dev.as_ref().unwrap();
        let out = self
            .net
            .as_ref()
            .unwrap()
            .forward(tensor2(&self.normalize(x), dev));
        let c = super::tensor1(&self.center, dev).unsqueeze_dim::<2>(0);
        to_vec((out - c).powf_scalar(2.0).sum_dim(1))
    }
}

impl Detector for DeepSvdd {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        let d = check_array(x)?;
        self.n_features = d;
        if self.hidden_neurons.len() < 2 {
            return Err(OdError::InvalidParameter(
                "hidden_neurons needs at least 2 layers".into(),
            ));
        }
        self.scaler = self.train.preprocessing.then(|| StandardScaler::fit(x));
        let xn = self.normalize(x);
        let (dev, _lock) = train_device(self.train.random_state);
        let mut net = self.build(d, &dev);
        let out_dim = if self.use_ae {
            d
        } else {
            *self.hidden_neurons.last().unwrap()
        };
        self.center = match &self.c {
            Some(c) if c.len() == 1 => vec![c[0]; out_dim],
            Some(c) if c.len() == out_dim => c.clone(),
            Some(_) => {
                return Err(OdError::InvalidParameter(format!(
                    "c must have 1 or {out_dim} values"
                )))
            }
            None => {
                let out = net
                    .valid()
                    .forward(tensor2(&xn, &dev.clone().without_autodiff()));
                to_vec(out.mean_dim(0))
                    .into_iter()
                    .map(|v| {
                        if v.abs() < 0.1 {
                            if v < 0.0 {
                                -0.1
                            } else if v > 0.0 {
                                0.1
                            } else {
                                v
                            }
                        } else {
                            v
                        }
                    })
                    .collect()
            }
        };
        if self.center.iter().all(|&v| v == 0.0) || self.center.iter().any(|v| !v.is_finite()) {
            return Err(OdError::InvalidParameter(
                "hypersphere centre must be finite and non-zero".into(),
            ));
        }
        let c: Tensor<2> = super::tensor1(&self.center, &dev).unsqueeze_dim(0);
        let mut opt = self.train.adam();
        let mut rng = Rng::new(self.train.random_state);
        let (mut best_loss, mut best) = (f64::INFINITY, net.clone());
        for _ in 0..self.train.epochs {
            let mut epoch_loss = 0.0;
            for idx in minibatches(xn.len(), self.train.batch_size, &mut rng, false) {
                let xb = tensor2(&pick(&xn, &idx), &dev);
                let out = net.forward(xb.clone());
                let dist = (out.clone() - c.clone()).powf_scalar(2.0).sum_dim(1).mean();
                // 1e-6 * sum of Frobenius norms of all weights.
                let wd = net.weight_norms().into_iter().reduce(|a, b| a + b).unwrap() * 1e-6;
                let mut loss = dist + wd;
                if self.use_ae {
                    loss = loss + (out - xb).powf_scalar(2.0).mean();
                }
                epoch_loss += super::scalar(loss.clone());
                let grads = GradientsParams::from_grads(loss.backward(), &net);
                net = opt.step(self.train.lr, net, grads);
            }
            if epoch_loss < best_loss {
                best_loss = epoch_loss;
                best = net.clone();
            }
        }
        self.net = Some(best.valid());
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
