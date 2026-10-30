//! AnoGAN (Schlegl et al., IPMI 2017). Port of `pyod.models.anogan`.
//!
//! A GAN is trained on the data; then, for each sample in turn, a small query
//! layer is optimised so the generator reproduces it, and the score is the final
//! reconstruction + discriminator-output loss. The query layer is warm-started
//! from one sample to the next, as in PyOD.
//!
//! **Difference from PyOD:** PyOD scores new data with the query layer frozen at
//! its state after the last training sample, so every test score measures the
//! distance to that one sample. We run the same per-sample optimisation on new
//! data (with the GAN frozen), as the AnoGAN paper does.

use super::{
    adam, bce_logits, minibatches, pick, scalar, tensor2, train_device, Activation, DenseNet,
    DropoutAt,
};
use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};
use crate::utils::{Rng, StandardScaler};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::GradientsParams;
use burn::tensor::{activation, Device, Distribution, Tensor};

#[derive(Debug, Clone)]
pub struct AnoGan {
    pub activation_hidden: Activation,
    pub output_activation: Option<Activation>,
    pub dropout_rate: f64,
    pub latent_dim_g: usize,
    pub g_layers: Vec<usize>,
    pub d_layers: Vec<usize>,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub learning_rate_query: f64,
    pub epochs_query: usize,
    pub preprocessing: bool,
    pub random_state: u64,
    pub contamination: f64,
    /// Generator and discriminator frozen on the training device, so gradients
    /// still flow through them to the query layer.
    gen: Option<DenseNet>,
    disc: Option<DenseNet>,
    query: Option<Linear>,
    dev: Option<Device>,
    scaler: Option<StandardScaler>,
    n_features: usize,
    state: Option<FitState>,
}

impl Default for AnoGan {
    fn default() -> Self {
        AnoGan {
            activation_hidden: Activation::Tanh,
            output_activation: None,
            dropout_rate: 0.2,
            latent_dim_g: 2,
            g_layers: vec![20, 10, 3, 10, 20],
            d_layers: vec![20, 10, 5],
            epochs: 500,
            batch_size: 32,
            learning_rate: 1e-3,
            learning_rate_query: 1e-2,
            epochs_query: 20,
            preprocessing: false,
            random_state: 42,
            contamination: 0.1,
            gen: None,
            disc: None,
            query: None,
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

/// Mean absolute reconstruction error plus discriminator-output error for one query.
fn query_loss(
    gen: &DenseNet,
    disc: &DenseNet,
    query: &Linear,
    sample: Tensor<2>,
    latent: usize,
) -> Tensor<1> {
    let dev = sample.device();
    let z = query.forward(Tensor::<2>::zeros([1, latent], &dev));
    let generated = gen.forward(z);
    let d_gen = activation::sigmoid(disc.forward(generated.clone()));
    let d_orig = activation::sigmoid(disc.forward(sample.clone())).detach();
    (sample - generated).abs().mean() + (d_orig - d_gen).abs().mean()
}

impl AnoGan {
    pub fn new(contamination: f64) -> Self {
        AnoGan {
            contamination,
            ..Self::default()
        }
    }

    /// Optimise `query` on each row in turn; each score is the last loss before the final step.
    fn query_scores(
        &self,
        gen: &DenseNet,
        disc: &DenseNet,
        query: &mut Linear,
        rows: &[Vec<f64>],
        dev: &Device,
    ) -> Vec<f64> {
        let mut opt = adam(0.0);
        rows.iter()
            .map(|row| {
                let sample = tensor2(std::slice::from_ref(row), dev);
                let mut last = 0.0;
                for _ in 0..self.epochs_query {
                    let loss = query_loss(gen, disc, query, sample.clone(), self.latent_dim_g);
                    last = scalar(loss.clone());
                    let grads = GradientsParams::from_grads(loss.backward(), &*query);
                    *query = opt.step(self.learning_rate_query, query.clone(), grads);
                }
                last
            })
            .collect()
    }

    fn normalize(&self, x: &Matrix) -> Vec<Vec<f64>> {
        match &self.scaler {
            Some(s) => s.transform(x),
            None => x.to_vec(),
        }
    }
}

impl Detector for AnoGan {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        let d = check_array(x)?;
        self.n_features = d;
        if self.g_layers.is_empty() || self.d_layers.is_empty() || self.latent_dim_g == 0 {
            return Err(OdError::InvalidParameter(
                "need non-empty layers and latent_dim_g >= 1".into(),
            ));
        }
        let (dev, _lock) = train_device(self.random_state);
        let (act, p) = (self.activation_hidden, self.dropout_rate);
        let mut gen = DenseNet::new(
            &sizes(self.latent_dim_g, &self.g_layers, d),
            act,
            self.output_activation,
            p,
            DropoutAt::Input,
            &dev,
        );
        let mut disc = DenseNet::new(
            &sizes(d, &self.d_layers, 1),
            act,
            None,
            p,
            DropoutAt::Input,
            &dev,
        );
        self.scaler = self.preprocessing.then(|| StandardScaler::fit(x));
        let xn = self.normalize(x);
        let (mut opt_g, mut opt_d) = (adam(0.0), adam(0.0));
        let mut rng = Rng::new(self.random_state);
        for _ in 0..self.epochs {
            for idx in minibatches(xn.len(), self.batch_size, &mut rng, false) {
                let real = tensor2(&pick(&xn, &idx), &dev);
                let n = idx.len();
                let noise =
                    Tensor::<2>::random([n, self.latent_dim_g], Distribution::Default, &dev);
                let fake = gen.forward(noise);
                // One-sided label smoothing (0.9) for real samples.
                let loss_d = bce_logits(disc.forward(real), Tensor::<2>::ones([n, 1], &dev) * 0.9)
                    + bce_logits(
                        disc.forward(fake.clone().detach()),
                        Tensor::zeros([n, 1], &dev),
                    );
                let grads = GradientsParams::from_grads(loss_d.backward(), &disc);
                disc = opt_d.step(self.learning_rate, disc, grads);
                let loss_g = bce_logits(disc.forward(fake), Tensor::ones([n, 1], &dev));
                let grads = GradientsParams::from_grads(loss_g.backward(), &gen);
                gen = opt_g.step(self.learning_rate, gen, grads);
            }
        }
        // Query phase on the training data: G and D are not updated, but (as in PyOD)
        // they stay in training mode, so their dropout is active.
        let mut query = LinearConfig::new(self.latent_dim_g, self.latent_dim_g).init(&dev);
        let (gen, disc) = (gen.no_grad(), disc.no_grad());
        let scores = self.query_scores(&gen, &disc, &mut query, &xn, &dev);
        self.gen = Some(gen.freeze());
        self.disc = Some(disc.freeze());
        self.query = Some(query);
        self.dev = Some(dev);
        self.state = Some(FitState::from_contamination(scores, self.contamination));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_features(x, self.n_features)?;
        let mut query = self.query.clone().unwrap();
        let (g, d) = (self.gen.as_ref().unwrap(), self.disc.as_ref().unwrap());
        Ok(self.query_scores(
            g,
            d,
            &mut query,
            &self.normalize(x),
            self.dev.as_ref().unwrap(),
        ))
    }

    crate::state_accessor!();
}
