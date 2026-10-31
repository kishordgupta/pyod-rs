//! DevNet: deep anomaly detection with deviation networks (Pang et al., KDD 2019).
//! Port of `pyod.models.devnet`. Weakly supervised: train with
//! [`DevNet::fit_supervised`] on data where some outliers are labelled.
//!
//! **Difference from PyOD:** PyOD calls `deviation_loss(outputs, labels)` with its
//! arguments swapped (and with shapes that broadcast to a batch x batch matrix).
//! We use the deviation loss as defined in the paper: inliers are pulled to the
//! reference score distribution N(0, 1), labelled outliers pushed at least
//! `confidence_margin` standard deviations above it.
//!
//! When labelled outliers are present, each training batch alternates inliers
//! and labelled outliers, as in the paper and PyOD's own (unused)
//! `input_batch_generation_sup_sparse`, with `nb_batch` batches per epoch. With
//! few labels, plain shuffled batches let the inlier term dominate and
//! training can collapse.

use super::{minibatches, pick, tensor1, tensor2, to_vec, train_device};
use crate::base::{check_array, check_features, Detector, FitState, Matrix, OdError, Result};
use crate::utils::Rng;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{decay::WeightDecayConfig, GradientsParams, RmsPropConfig};
use burn::tensor::{activation, Device, Distribution, Tensor};

#[derive(Module, Debug)]
struct Net {
    hidden: Vec<Linear>,
    score: Linear,
}

impl Net {
    fn forward(&self, x: Tensor<2>) -> Tensor<1> {
        let h = self
            .hidden
            .iter()
            .fold(x, |x, l| activation::relu(l.forward(x)));
        self.score.forward(h).squeeze_dim(1)
    }
}

#[derive(Debug, Clone)]
pub struct DevNet {
    /// 1 (linear), 2 (one hidden layer of 1000) or 4 (1000-250-20).
    pub network_depth: usize,
    pub batch_size: usize,
    pub epochs: usize,
    /// Balanced batches per epoch (used when labelled outliers are present).
    pub nb_batch: usize,
    pub confidence_margin: f64,
    pub lr: f64,
    pub random_state: u64,
    pub contamination: f64,
    net: Option<Net>,
    dev: Option<Device>,
    n_features: usize,
    state: Option<FitState>,
}

impl Default for DevNet {
    fn default() -> Self {
        DevNet {
            network_depth: 2,
            batch_size: 512,
            epochs: 50,
            nb_batch: 20,
            confidence_margin: 5.0,
            lr: 1e-3,
            random_state: 42,
            contamination: 0.1,
            net: None,
            dev: None,
            n_features: 0,
            state: None,
        }
    }
}

impl DevNet {
    pub fn new(network_depth: usize, contamination: f64) -> Self {
        DevNet {
            network_depth,
            contamination,
            ..Self::default()
        }
    }

    fn score(&self, x: &Matrix) -> Vec<f64> {
        to_vec(
            self.net
                .as_ref()
                .unwrap()
                .forward(tensor2(x, self.dev.as_ref().unwrap())),
        )
    }

    /// Fit on data with partial labels (`y[i] == 1` for known outliers, 0 otherwise).
    pub fn fit_supervised(&mut self, x: &Matrix, y: &[u8]) -> Result<()> {
        let d = check_array(x)?;
        self.n_features = d;
        if y.len() != x.len() {
            return Err(OdError::InvalidParameter(
                "y must have one label per sample".into(),
            ));
        }
        let widths: &[usize] = match self.network_depth {
            1 => &[],
            2 => &[1000],
            4 => &[1000, 250, 20],
            _ => {
                return Err(OdError::InvalidParameter(
                    "network_depth must be 1, 2 or 4".into(),
                ))
            }
        };
        let (dev, _lock) = train_device(self.random_state);
        let mut sizes = vec![d];
        sizes.extend(widths);
        let mut net = Net {
            hidden: sizes
                .windows(2)
                .map(|w| LinearConfig::new(w[0], w[1]).init(&dev))
                .collect(),
            score: LinearConfig::new(*sizes.last().unwrap(), 1).init(&dev),
        };
        let mut opt = RmsPropConfig::new()
            .with_momentum(0.0)
            .with_epsilon(1e-8)
            .with_weight_decay(Some(WeightDecayConfig::new(1e-6)))
            .init();
        let mut rng = Rng::new(self.random_state);
        let labels: Vec<f64> = y.iter().map(|&v| v as f64).collect();
        let (outliers, inliers): (Vec<usize>, Vec<usize>) = (0..x.len()).partition(|&i| y[i] != 0);
        let balanced = !outliers.is_empty() && !inliers.is_empty();
        let bs = self.batch_size.max(1);
        for _ in 0..self.epochs {
            let batches: Vec<Vec<usize>> = if balanced {
                (0..self.nb_batch.max(1))
                    .map(|_| {
                        (0..bs)
                            .map(|i| {
                                let pool = if i % 2 == 0 { &inliers } else { &outliers };
                                pool[rng.below(pool.len())]
                            })
                            .collect()
                    })
                    .collect()
            } else {
                minibatches(x.len(), bs, &mut rng, false)
            };
            for idx in batches {
                let pred = net.forward(tensor2(&pick(x, &idx), &dev));
                let yb = tensor1(&idx.iter().map(|&i| labels[i]).collect::<Vec<_>>(), &dev);
                // Reference scores from the N(0, 1) prior.
                let reference = Tensor::<1>::random([5000], Distribution::Normal(0.0, 1.0), &dev);
                let (mu, sigma) = (reference.clone().mean(), reference.var(0).sqrt());
                let dev_score = (pred - mu) / sigma;
                let inlier = dev_score.clone().abs();
                let outlier = (dev_score.neg() + self.confidence_margin).clamp_min(0.0);
                let loss = ((yb.clone().neg() + 1.0) * inlier + yb * outlier).mean();
                let grads = GradientsParams::from_grads(loss.backward(), &net);
                net = opt.step(self.lr, net, grads);
            }
        }
        self.net = Some(net.valid());
        self.dev = Some(dev.without_autodiff());
        let scores = self.score(x);
        self.state = Some(FitState::from_contamination(
            scores,
            self.contamination.clamp(1e-9, 0.5),
        ));
        Ok(())
    }
}

impl Detector for DevNet {
    /// DevNet needs labels; use [`DevNet::fit_supervised`].
    fn fit(&mut self, _x: &Matrix) -> Result<()> {
        Err(OdError::InvalidParameter(
            "DevNet needs labels: call fit_supervised(x, y)".into(),
        ))
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.n_features)?;
        Ok(self.score(x))
    }

    crate::state_accessor!();
}
