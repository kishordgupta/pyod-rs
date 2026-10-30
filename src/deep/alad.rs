//! ALAD: Adversarially Learned Anomaly Detection (Zenati et al., ICDM 2018).
//! Port of `pyod.models.alad`.
//!
//! A BiGAN (encoder + decoder) is trained against three discriminators on
//! (x, z), (x, x) and (z, z) pairs. The score is how differently the (x, x)
//! discriminator treats `(x, x)` and `(x, G(E(x)))`. Unlike PyOD, which leaves
//! dropout active while scoring, we score in evaluation mode, so scores are deterministic.

use super::{
    adam, bce_logits, pick, tensor2, to_vec, train_device, Activation, DenseNet, DropoutAt,
};
use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};
use crate::utils::{Rng, StandardScaler};
use burn::module::Module;
use burn::optim::GradientsParams;
use burn::tensor::{activation, Device, Distribution, Tensor};

#[derive(Module, Debug)]
struct Gen {
    enc: DenseNet,
    dec: DenseNet,
}

#[derive(Module, Debug)]
struct Discs {
    xx: DenseNet,
    zz: DenseNet,
    xz: DenseNet,
}

#[derive(Debug, Clone)]
pub struct Alad {
    pub activation_hidden_gen: Activation,
    pub activation_hidden_disc: Activation,
    pub output_activation: Option<Activation>,
    pub dropout_rate: f64,
    pub latent_dim: usize,
    pub dec_layers: Vec<usize>,
    pub enc_layers: Vec<usize>,
    pub disc_xx_layers: Vec<usize>,
    pub disc_zz_layers: Vec<usize>,
    pub disc_xz_layers: Vec<usize>,
    pub learning_rate_gen: f64,
    pub learning_rate_disc: f64,
    pub add_recon_loss: bool,
    pub lambda_recon_loss: f64,
    pub add_disc_zz_loss: bool,
    /// One training step per epoch, on a random batch.
    pub epochs: usize,
    pub batch_size: usize,
    pub preprocessing: bool,
    /// Std of the Gaussian noise added to each training batch.
    pub noise_std: f64,
    pub random_state: u64,
    pub contamination: f64,
    nets: Option<(Gen, Discs)>,
    dev: Option<Device>,
    scaler: Option<StandardScaler>,
    n_features: usize,
    state: Option<FitState>,
}

impl Default for Alad {
    fn default() -> Self {
        Alad {
            activation_hidden_gen: Activation::Tanh,
            activation_hidden_disc: Activation::Tanh,
            output_activation: None,
            dropout_rate: 0.2,
            latent_dim: 2,
            dec_layers: vec![5, 10, 25],
            enc_layers: vec![25, 10, 5],
            disc_xx_layers: vec![25, 10, 5],
            disc_zz_layers: vec![25, 10, 5],
            disc_xz_layers: vec![25, 10, 5],
            learning_rate_gen: 1e-4,
            learning_rate_disc: 1e-4,
            add_recon_loss: false,
            lambda_recon_loss: 0.1,
            add_disc_zz_loss: true,
            epochs: 200,
            batch_size: 32,
            preprocessing: false,
            noise_std: 0.1,
            random_state: 42,
            contamination: 0.1,
            nets: None,
            dev: None,
            scaler: None,
            n_features: 0,
            state: None,
        }
    }
}

fn sizes(input: usize, hidden: &[usize], output: usize) -> Vec<usize> {
    std::iter::once(input)
        .chain(hidden.iter().copied())
        .chain(std::iter::once(output))
        .collect()
}

fn cat(a: Tensor<2>, b: Tensor<2>) -> Tensor<2> {
    Tensor::cat(vec![a, b], 1)
}

impl Alad {
    pub fn new(contamination: f64) -> Self {
        Alad {
            contamination,
            ..Self::default()
        }
    }

    fn normalize(&self, x: &Matrix) -> Vec<Vec<f64>> {
        match &self.scaler {
            Some(s) => s.transform(x),
            None => x.to_vec(),
        }
    }

    fn score(&self, x: &Matrix) -> Vec<f64> {
        let ((g, d), dev) = (self.nets.as_ref().unwrap(), self.dev.as_ref().unwrap());
        let x = tensor2(&self.normalize(x), dev);
        let recon = g.dec.forward(g.enc.forward(x.clone()));
        let t = activation::sigmoid(d.xx.forward(cat(x.clone(), x.clone())));
        let f = activation::sigmoid(d.xx.forward(cat(x, recon)));
        to_vec((t - f).powf_scalar(2.0).mean_dim(1))
    }
}

impl Detector for Alad {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        let d = check_array(x)?;
        self.n_features = d;
        if self.latent_dim == 0 || self.batch_size == 0 {
            return Err(OdError::InvalidParameter(
                "latent_dim and batch_size must be >= 1".into(),
            ));
        }
        let (dev, _lock) = train_device(self.random_state);
        let (ag, ad, z, p) = (
            self.activation_hidden_gen,
            self.activation_hidden_disc,
            self.latent_dim,
            self.dropout_rate,
        );
        let mut g = Gen {
            enc: DenseNet::new(
                &sizes(d, &self.enc_layers, z),
                ag,
                self.output_activation,
                p,
                DropoutAt::AfterLinear,
                &dev,
            ),
            dec: DenseNet::new(
                &sizes(z, &self.dec_layers, d),
                ag,
                self.output_activation,
                p,
                DropoutAt::AfterLinear,
                &dev,
            ),
        };
        let mut disc = Discs {
            xx: DenseNet::new(
                &sizes(2 * d, &self.disc_xx_layers, 1),
                ad,
                None,
                p,
                DropoutAt::AfterLinear,
                &dev,
            ),
            zz: DenseNet::new(
                &sizes(2 * z, &self.disc_zz_layers, 1),
                ad,
                None,
                p,
                DropoutAt::AfterLinear,
                &dev,
            ),
            xz: DenseNet::new(
                &sizes(d + z, &self.disc_xz_layers, 1),
                ad,
                None,
                p,
                DropoutAt::AfterLinear,
                &dev,
            ),
        };
        self.scaler = self.preprocessing.then(|| StandardScaler::fit(x));
        let xn = self.normalize(x);
        let (mut opt_g, mut opt_d) = (adam(0.0), adam(0.0));
        let mut rng = Rng::new(self.random_state);
        let bs = self.batch_size.min(xn.len());
        let (ones, zeros) = (
            Tensor::<2>::ones([bs, 1], &dev),
            Tensor::<2>::zeros([bs, 1], &dev),
        );
        for _ in 0..self.epochs {
            let idx = rng.sample_indices(xn.len(), bs);
            let x_real = tensor2(&pick(&xn, &idx), &dev);
            let x_real =
                x_real.clone() + x_real.random_like(Distribution::Normal(0.0, self.noise_std));
            let z_real = Tensor::<2>::random([bs, z], Distribution::Normal(0.0, 1.0), &dev);

            // Discriminator step (generator outputs detached).
            let x_gen = g.dec.forward(z_real.clone()).detach();
            let z_gen = g.enc.forward(x_real.clone()).detach();
            let mut loss_d =
                bce_logits(
                    disc.xz.forward(cat(x_real.clone(), z_gen.clone())),
                    ones.clone(),
                ) + bce_logits(
                    disc.xz.forward(cat(x_gen.clone(), z_real.clone())),
                    zeros.clone(),
                ) + bce_logits(
                    disc.xx.forward(cat(x_real.clone(), x_real.clone())),
                    ones.clone(),
                ) + bce_logits(disc.xx.forward(cat(x_real.clone(), x_gen)), zeros.clone());
            if self.add_disc_zz_loss {
                loss_d = loss_d
                    + bce_logits(
                        disc.zz.forward(cat(z_real.clone(), z_real.clone())),
                        ones.clone(),
                    )
                    + bce_logits(disc.zz.forward(cat(z_real.clone(), z_gen)), zeros.clone());
            }
            let grads = GradientsParams::from_grads(loss_d.backward(), &disc);
            disc = opt_d.step(self.learning_rate_disc, disc, grads);

            // Generator step: flipped labels.
            let x_gen = g.dec.forward(z_real.clone());
            let z_gen = g.enc.forward(x_real.clone());
            let mut loss_g =
                bce_logits(
                    disc.xz.forward(cat(x_gen.clone(), z_real.clone())),
                    ones.clone(),
                ) + bce_logits(
                    disc.xz.forward(cat(x_real.clone(), z_gen.clone())),
                    zeros.clone(),
                ) + bce_logits(disc.xx.forward(cat(x_real.clone(), x_gen)), ones.clone())
                    + bce_logits(
                        disc.xx.forward(cat(x_real.clone(), x_real.clone())),
                        zeros.clone(),
                    );
            if self.add_disc_zz_loss {
                loss_g = loss_g
                    + bce_logits(disc.zz.forward(cat(z_real.clone(), z_gen)), ones.clone())
                    + bce_logits(disc.zz.forward(cat(z_real.clone(), z_real)), zeros.clone());
            }
            if self.add_recon_loss {
                let recon = g.dec.forward(g.enc.forward(x_real.clone()));
                loss_g = loss_g + (x_real - recon).powf_scalar(2.0).mean() * self.lambda_recon_loss;
            }
            let grads = GradientsParams::from_grads(loss_g.backward(), &g);
            g = opt_g.step(self.learning_rate_gen, g, grads);
        }
        self.nets = Some((g.valid(), disc.valid()));
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
