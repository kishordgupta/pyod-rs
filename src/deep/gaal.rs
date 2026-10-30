//! Generative Adversarial Active Learning (Liu et al., TKDE 2019): SO-GAAL and
//! MO-GAAL. Ports of `pyod.models.so_gaal` and `pyod.models.mo_gaal`.
//!
//! Generators synthesise potential outliers around the data and a discriminator
//! learns to tell real samples from them, so samples it judges unlikely to be
//! real are outliers. **Difference from PyOD:** PyOD returns the discriminator's
//! probability of "real", D(x), as the anomaly score; we return `1 - D(x)`,
//! which matches that design. GAAL is sensitive to data scale (the generators
//! start from noise in [0, 1]^d): on PyOD's `generate_data`, `1 - D(x)` scores
//! AUC 0.75-0.93 where D(x) scores 0.24-0.36, but on other data the ranking can flip.

use super::{bce_logits as bce, minibatches, pick, sgd, tensor2, to_vec, train_device};
use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};
use crate::utils::{percentile, Rng};
use burn::module::{Module, Param};
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::optim::GradientsParams;
use burn::tensor::{activation, Device, Distribution, Tensor};

#[derive(Module, Debug)]
struct Generator {
    l1: Linear,
    l2: Linear,
}

impl Generator {
    /// Two ReLU layers initialised to the identity.
    fn new(d: usize, dev: &Device) -> Self {
        let eye = || {
            let mut l = LinearConfig::new(d, d).init(dev);
            l.weight = Param::from_tensor(Tensor::eye(d, dev));
            l
        };
        Generator {
            l1: eye(),
            l2: eye(),
        }
    }

    fn forward(&self, z: Tensor<2>) -> Tensor<2> {
        activation::relu(self.l2.forward(activation::relu(self.l1.forward(z))))
    }
}

#[derive(Module, Debug)]
struct Discriminator {
    l1: Linear,
    l2: Linear,
}

impl Discriminator {
    fn new(d: usize, n: usize, dev: &Device) -> Self {
        let h = (n as f64).sqrt().ceil() as usize;
        Discriminator {
            l1: LinearConfig::new(d, h)
                .with_initializer(Initializer::KaimingNormal {
                    gain: 2f64.sqrt(),
                    fan_out_only: false,
                })
                .init(dev),
            l2: LinearConfig::new(h, 1)
                .with_initializer(Initializer::KaimingNormal {
                    gain: 1.0,
                    fan_out_only: false,
                })
                .init(dev),
        }
    }

    /// Pre-sigmoid output; D(x) = sigmoid(logits).
    fn logits(&self, x: Tensor<2>) -> Tensor<2> {
        self.l2.forward(activation::relu(self.l1.forward(x)))
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        activation::sigmoid(self.logits(x))
    }
}

/// Training settings shared by SO-GAAL and MO-GAAL.
#[derive(Debug, Clone)]
pub struct GaalParams {
    /// Generators stop training after this many epochs; training runs for 3x as many.
    pub stop_epochs: usize,
    pub lr_d: f64,
    pub lr_g: f64,
    /// SGD momentum for both networks.
    pub momentum: f64,
    pub random_state: u64,
}

impl Default for GaalParams {
    fn default() -> Self {
        GaalParams {
            stop_epochs: 20,
            lr_d: 0.01,
            lr_g: 1e-4,
            momentum: 0.9,
            random_state: 42,
        }
    }
}

/// Fitted discriminator.
#[derive(Debug, Clone, Default)]
struct Gaal {
    disc: Option<Discriminator>,
    dev: Option<Device>,
    n_features: usize,
}

impl Gaal {
    fn score(&self, x: &Matrix) -> Vec<f64> {
        let d = self
            .disc
            .as_ref()
            .unwrap()
            .forward(tensor2(x, self.dev.as_ref().unwrap()));
        to_vec(d).into_iter().map(|p| 1.0 - p).collect()
    }

    /// Trains with `k` sub-generators; `k == 1` with trick label 1 is SO-GAAL.
    fn train(&mut self, p: &GaalParams, x: &Matrix, k: usize, multi: bool) {
        let (n, d) = (x.len(), x[0].len());
        self.n_features = d;
        let (dev, _lock) = train_device(p.random_state);
        let mut disc = Discriminator::new(d, n, &dev);
        let mut gens: Vec<Generator> = (0..k).map(|_| Generator::new(d, &dev)).collect();
        let mut opt_d = sgd(p.momentum);
        let mut opt_g: Vec<_> = (0..k).map(|_| sgd(p.momentum)).collect();
        let mut rng = Rng::new(p.random_state);
        let all = tensor2(x, &dev);
        let mut stop = false;
        for epoch in 0..p.stop_epochs * 3 {
            for idx in minibatches(n, n.min(500), &mut rng, false) {
                let bs = idx.len();
                let real = tensor2(&pick(x, &idx), &dev);
                let noise = Tensor::<2>::random([bs, d], Distribution::Default, &dev);
                // Split the noise among sub-generators in proportions k, k-1, ..., 1 (MO-GAAL).
                let block = (1 + k) * k / 2;
                let parts: Vec<(usize, usize)> = (0..k)
                    .map(|i| {
                        if !multi {
                            return (0, bs);
                        }
                        let start = (k + (k - i + 1)) * i / 2 * (bs / block);
                        let end = if i == k - 1 {
                            bs
                        } else {
                            (k + (k - i)) * (i + 1) / 2 * (bs / block)
                        };
                        (start.min(bs), end.min(bs))
                    })
                    .collect();
                let noises: Vec<Tensor<2>> = parts
                    .iter()
                    .map(|&(s, e)| noise.clone().narrow(0, s, e - s))
                    .collect();
                let fakes: Vec<Tensor<2>> = gens
                    .iter()
                    .zip(&noises)
                    .map(|(g, z)| g.forward(z.clone()).detach())
                    .collect();
                let n_fake: usize = parts.iter().map(|(s, e)| e - s).sum();
                let d_loss = if multi {
                    let mut batch = vec![real];
                    batch.extend(fakes);
                    let labels = Tensor::cat(
                        vec![
                            Tensor::<2>::ones([bs, 1], &dev),
                            Tensor::<2>::zeros([n_fake, 1], &dev),
                        ],
                        0,
                    );
                    bce(disc.logits(Tensor::cat(batch, 0)), labels)
                } else {
                    bce(disc.logits(real), Tensor::ones([bs, 1], &dev))
                        + bce(disc.logits(fakes[0].clone()), Tensor::zeros([bs, 1], &dev))
                };
                let grads = GradientsParams::from_grads(d_loss.backward(), &disc);
                disc = opt_d.step(p.lr_d, disc, grads);
                if !stop {
                    // SO-GAAL pushes generated data toward "real"; MO-GAAL sub-generator i
                    // targets the i/k quantile of the discriminator's scores on the data.
                    let targets: Vec<f64> = if multi {
                        let pred = to_vec(disc.valid().forward(all.clone().detach().inner()));
                        (0..k)
                            .map(|i| percentile(&pred, i as f64 / k as f64 * 100.0))
                            .collect()
                    } else {
                        vec![1.0]
                    };
                    for i in 0..k {
                        let m = parts[i].1 - parts[i].0;
                        if m == 0 {
                            continue;
                        }
                        let out = disc.logits(gens[i].forward(noises[i].clone()));
                        let loss = bce(out, Tensor::<2>::ones([m, 1], &dev) * targets[i]);
                        let grads = GradientsParams::from_grads(loss.backward(), &gens[i]);
                        gens[i] = opt_g[i].step(p.lr_g, gens[i].clone(), grads);
                    }
                }
            }
            if epoch + 1 > p.stop_epochs {
                stop = true;
            }
        }
        self.disc = Some(disc.valid());
        self.dev = Some(dev.without_autodiff());
    }
}

fn fit_gaal(
    params: &GaalParams,
    x: &Matrix,
    k: usize,
    multi: bool,
    contamination: f64,
) -> Result<(Gaal, FitState)> {
    check_contamination(contamination)?;
    check_array(x)?;
    if params.stop_epochs == 0 || k == 0 {
        return Err(OdError::InvalidParameter(
            "stop_epochs and k must be >= 1".into(),
        ));
    }
    let mut g = Gaal::default();
    g.train(params, x, k, multi);
    let state = FitState::from_contamination(g.score(x), contamination);
    Ok((g, state))
}

/// Single-Objective GAAL: one generator, trained to fool the discriminator.
#[derive(Debug, Clone)]
pub struct SoGaal {
    pub params: GaalParams,
    pub contamination: f64,
    inner: Gaal,
    state: Option<FitState>,
}

impl Default for SoGaal {
    fn default() -> Self {
        SoGaal {
            params: GaalParams::default(),
            contamination: 0.1,
            inner: Gaal::default(),
            state: None,
        }
    }
}

impl Detector for SoGaal {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        let (g, s) = fit_gaal(&self.params, x, 1, false, self.contamination)?;
        (self.inner, self.state) = (g, Some(s));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.inner.n_features)?;
        Ok(self.inner.score(x))
    }

    crate::state_accessor!();
}

/// Multiple-Objective GAAL: `k` sub-generators, each targeting a different
/// quantile of the discriminator's scores.
#[derive(Debug, Clone)]
pub struct MoGaal {
    pub k: usize,
    pub params: GaalParams,
    pub contamination: f64,
    inner: Gaal,
    state: Option<FitState>,
}

impl Default for MoGaal {
    fn default() -> Self {
        MoGaal {
            k: 10,
            params: GaalParams::default(),
            contamination: 0.1,
            inner: Gaal::default(),
            state: None,
        }
    }
}

impl Detector for MoGaal {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        let (g, s) = fit_gaal(&self.params, x, self.k, true, self.contamination)?;
        (self.inner, self.state) = (g, Some(s));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.inner.n_features)?;
        Ok(self.inner.score(x))
    }

    crate::state_accessor!();
}
