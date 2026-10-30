//! Neural-network detectors, built on [burn](https://burn.dev) (enable the `deep` feature).
//!
//! Shared pieces mirror PyOD's `base_dl.py` and `utils/torch_utility.py`:
//! standardisation with `eps = 1e-8`, shuffled mini-batches (dropping the last
//! partial batch), PyOD's `LinearBlock`, and Adam with PyTorch's epsilon.

pub mod ae1svm;
pub mod alad;
pub mod anogan;
pub mod anomaly_transformer;
pub mod audio_ae;
pub mod autoencoder;
pub mod deep_svdd;
pub mod devnet;
pub mod gaal;
pub mod graph;
pub mod lstm;
pub mod lunar;
pub mod vae;

use crate::base::Matrix;
use crate::utils::Rng;
use burn::module::Module;
use burn::nn::{
    BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Initializer, Linear, LinearConfig,
};
use burn::optim::{
    decay::WeightDecayConfig, momentum::MomentumConfig, AdamConfig, ModuleOptimizer, SgdConfig,
};
use burn::tensor::{activation, Device, Tensor, TensorData};

pub use burn::tensor::Device as BurnDevice;

/// Training hyper-parameters shared by the deep detectors.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    /// L2 penalty added to the gradients (PyTorch-style `weight_decay`).
    pub weight_decay: f64,
    /// Standardise inputs with training mean / std before training and scoring.
    pub preprocessing: bool,
    pub random_state: u64,
}

impl TrainConfig {
    pub fn new(epochs: usize, batch_size: usize, lr: f64) -> Self {
        TrainConfig {
            epochs,
            batch_size,
            lr,
            weight_decay: 0.0,
            preprocessing: true,
            random_state: 42,
        }
    }

    pub fn adam(&self) -> ModuleOptimizer {
        adam(self.weight_decay)
    }
}

pub(crate) fn adam(weight_decay: f64) -> ModuleOptimizer {
    let wd = (weight_decay > 0.0).then(|| WeightDecayConfig::new(weight_decay as f32));
    AdamConfig::new()
        .with_epsilon(1e-8)
        .with_weight_decay(wd)
        .init()
}

pub(crate) fn sgd(momentum: f64) -> ModuleOptimizer {
    let m = MomentumConfig::new()
        .with_momentum(momentum)
        .with_dampening(0.0);
    SgdConfig::new().with_momentum(Some(m)).init()
}

/// burn's random seed is process-wide, so concurrent training would make results
/// depend on thread timing. Training (and any stochastic scoring) holds this lock.
static RNG_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub(crate) fn rng_lock() -> std::sync::MutexGuard<'static, ()> {
    RNG_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A seeded CPU device with autodiff, for training, plus the RNG lock to hold
/// for as long as training runs.
pub(crate) fn train_device(seed: u64) -> (Device, std::sync::MutexGuard<'static, ()>) {
    let guard = rng_lock();
    let d = Device::flex().autodiff();
    d.seed(seed);
    (d, guard)
}

pub(crate) fn tensor2(rows: &[Vec<f64>], dev: &Device) -> Tensor<2> {
    let (n, d) = (rows.len(), rows.first().map_or(0, Vec::len));
    let flat: Vec<f32> = rows.iter().flatten().map(|&v| v as f32).collect();
    Tensor::from_data(TensorData::new(flat, [n, d]), dev)
}

pub(crate) fn tensor1(v: &[f64], dev: &Device) -> Tensor<1> {
    let flat: Vec<f32> = v.iter().map(|&x| x as f32).collect();
    Tensor::from_data(TensorData::new(flat, [v.len()]), dev)
}

pub(crate) fn to_vec<const D: usize>(t: Tensor<D>) -> Vec<f64> {
    t.into_data()
        .convert::<f32>()
        .try_into_vec::<f32>()
        .expect("f32 tensor")
        .into_iter()
        .map(f64::from)
        .collect()
}

pub(crate) fn to_rows(t: Tensor<2>) -> Vec<Vec<f64>> {
    let [_, d] = t.dims();
    to_vec(t).chunks(d.max(1)).map(<[f64]>::to_vec).collect()
}

pub(crate) fn scalar(t: Tensor<1>) -> f64 {
    to_vec(t)[0]
}

/// Column standardisation `(x - mean) / (std + 1e-8)`, as PyOD's `TorchDataset`.
#[derive(Debug, Clone, Default)]
pub struct Normalizer {
    mean: Vec<f64>,
    std: Vec<f64>,
}

impl Normalizer {
    pub fn fit(x: &Matrix, enabled: bool) -> Self {
        if !enabled {
            return Normalizer::default();
        }
        let n = x.len() as f64;
        let d = x[0].len();
        let mean: Vec<f64> = (0..d)
            .map(|j| x.iter().map(|r| r[j]).sum::<f64>() / n)
            .collect();
        let std = (0..d)
            .map(|j| (x.iter().map(|r| (r[j] - mean[j]).powi(2)).sum::<f64>() / n).sqrt())
            .collect();
        Normalizer { mean, std }
    }

    pub fn apply(&self, x: &Matrix) -> Vec<Vec<f64>> {
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
}

/// Shuffled mini-batch indices. With `drop_last`, a final partial batch is skipped
/// unless it is the only batch.
pub(crate) fn minibatches(
    n: usize,
    batch_size: usize,
    rng: &mut Rng,
    drop_last: bool,
) -> Vec<Vec<usize>> {
    let perm = rng.sample_indices(n, n);
    let bs = batch_size.max(1);
    let mut out: Vec<Vec<usize>> = perm.chunks(bs).map(<[usize]>::to_vec).collect();
    if drop_last && out.len() > 1 && out.last().is_some_and(|b| b.len() < bs) {
        out.pop();
    }
    out
}

pub(crate) fn pick(x: &Matrix, idx: &[usize]) -> Vec<Vec<f64>> {
    idx.iter().map(|&i| x[i].clone()).collect()
}

/// Hidden-layer activation functions (PyOD's `get_activation_by_name`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Activation {
    #[default]
    Relu,
    LeakyRelu,
    Tanh,
    Sigmoid,
    Elu,
    Softplus,
    Identity,
}

impl Activation {
    pub fn apply<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        match self {
            Activation::Relu => activation::relu(x),
            Activation::LeakyRelu => activation::leaky_relu(x, 0.01),
            Activation::Tanh => activation::tanh(x),
            Activation::Sigmoid => activation::sigmoid(x),
            Activation::Elu => {
                let neg = x.clone().clamp_max(0.0).exp() - 1.0;
                activation::relu(x) + neg
            }
            Activation::Softplus => activation::softplus(x, 1.0),
            Activation::Identity => x,
        }
    }
}

/// PyOD's `LinearBlock`: linear, then optional batch norm, activation and dropout.
#[derive(Module, Debug)]
pub struct Block {
    linear: Linear,
    bn: Option<BatchNorm>,
    dropout: Option<Dropout>,
    #[module(skip)]
    act: Option<Activation>,
}

impl Block {
    /// `kaiming` uses PyOD's `kaiming_uniform` (leaky-ReLU gain, a = 0); otherwise PyTorch's default init.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d_in: usize,
        d_out: usize,
        act: Option<Activation>,
        batch_norm: bool,
        dropout: f64,
        bias: bool,
        kaiming: bool,
        dev: &Device,
    ) -> Self {
        let mut cfg = LinearConfig::new(d_in, d_out).with_bias(bias);
        if kaiming {
            cfg = cfg.with_initializer(Initializer::KaimingUniform {
                gain: 2f64.sqrt(),
                fan_out_only: false,
            });
        }
        Block {
            linear: cfg.init(dev),
            bn: batch_norm.then(|| BatchNormConfig::new(d_out).init(dev)),
            dropout: (dropout > 0.0).then(|| DropoutConfig::new(dropout).init()),
            act,
        }
    }

    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let mut x = self.linear.forward(x);
        if let Some(bn) = &self.bn {
            x = bn.forward(x);
        }
        if let Some(a) = &self.act {
            x = a.apply(x);
        }
        match &self.dropout {
            Some(d) => d.forward(x),
            None => x,
        }
    }
}

/// A stack of [`Block`]s.
#[derive(Module, Debug)]
pub struct Mlp {
    blocks: Vec<Block>,
}

impl Mlp {
    pub fn new(blocks: Vec<Block>) -> Self {
        Mlp { blocks }
    }

    /// Hidden layers `sizes[0] -> sizes[1] -> ...`, each with the same activation, batch norm and dropout.
    pub fn hidden(
        sizes: &[usize],
        act: Activation,
        batch_norm: bool,
        dropout: f64,
        bias: bool,
        dev: &Device,
    ) -> Self {
        Mlp {
            blocks: sizes
                .windows(2)
                .map(|w| Block::new(w[0], w[1], Some(act), batch_norm, dropout, bias, true, dev))
                .collect(),
        }
    }

    pub fn push(mut self, b: Block) -> Self {
        self.blocks.push(b);
        self
    }

    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.blocks.iter().fold(x, |x, b| b.forward(x))
    }

    /// Frobenius norm of each layer's weight matrix, as 1-element tensors.
    pub fn weight_norms(&self) -> Vec<Tensor<1>> {
        self.blocks
            .iter()
            .map(|b| b.linear.weight.val().powf_scalar(2.0).sum().sqrt())
            .collect()
    }
}

/// Euclidean distance between matching rows (`pairwise_distances_no_broadcast`).
pub(crate) fn row_distances(a: Tensor<2>, b: Tensor<2>) -> Vec<f64> {
    to_vec((a - b).powf_scalar(2.0).sum_dim(1).sqrt())
}

/// Where a [`DenseNet`] applies dropout in each hidden layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropoutAt {
    /// dropout -> linear -> activation (AnoGAN).
    Input,
    /// linear -> dropout -> activation (ALAD).
    AfterLinear,
}

/// Plain dense network: hidden layers with one activation, then an output layer.
#[derive(Module, Debug)]
pub struct DenseNet {
    hidden: Vec<Linear>,
    out: Linear,
    dropout: Option<Dropout>,
    #[module(skip)]
    act: Activation,
    #[module(skip)]
    out_act: Option<Activation>,
    #[module(skip)]
    at: DropoutAt,
}

impl DenseNet {
    /// `sizes` = [input, hidden..., output].
    pub fn new(
        sizes: &[usize],
        act: Activation,
        out_act: Option<Activation>,
        dropout: f64,
        at: DropoutAt,
        dev: &Device,
    ) -> Self {
        let n = sizes.len();
        DenseNet {
            hidden: sizes[..n - 1]
                .windows(2)
                .map(|w| LinearConfig::new(w[0], w[1]).init(dev))
                .collect(),
            out: LinearConfig::new(sizes[n - 2], sizes[n - 1]).init(dev),
            dropout: (dropout > 0.0).then(|| DropoutConfig::new(dropout).init()),
            act,
            out_act,
            at,
        }
    }

    fn drop(&self, x: Tensor<2>) -> Tensor<2> {
        match &self.dropout {
            Some(d) => d.forward(x),
            None => x,
        }
    }

    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let mut x = x;
        for l in &self.hidden {
            x = match self.at {
                DropoutAt::Input => self.act.apply(l.forward(self.drop(x))),
                DropoutAt::AfterLinear => self.act.apply(self.drop(l.forward(x))),
            };
        }
        let x = self.out.forward(x);
        match &self.out_act {
            Some(a) => a.apply(x),
            None => x,
        }
    }
}

/// Mean binary cross-entropy of `sigmoid(z)` against `target`, computed stably
/// as `softplus(z) - target * z` (equal to PyTorch's `BCELoss` on probabilities).
pub(crate) fn bce_logits(z: Tensor<2>, target: Tensor<2>) -> Tensor<1> {
    (activation::softplus(z.clone(), 1.0) - target * z).mean()
}
