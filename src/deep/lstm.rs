//! LSTM-AD: LSTM forecasting with Mahalanobis-scored prediction errors
//! (Malhotra et al., ESANN 2015). Port of `pyod.models.ts_lstm.LSTMAD`.
//!
//! Input is a time series (one row per timestamp). The first `window_size`
//! timestamps have no forecast and get the threshold as their score.

use super::{minibatches, tensor2, to_rows, train_device};
use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};
use crate::linalg::{covariance, mahalanobis_sq, pinvh};
use crate::ts::ts_fit_state;
use crate::utils::Rng;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Lstm, LstmConfig};
use burn::optim::GradientsParams;
use burn::tensor::{Device, Tensor, TensorData};

#[derive(Module, Debug)]
struct Net {
    lstm: Vec<Lstm>,
    out: Linear,
}

impl Net {
    /// `[batch, time, channels]` -> forecast of the next step `[batch, channels]`.
    fn forward(&self, x: Tensor<3>) -> Tensor<2> {
        let h = self.lstm.iter().fold(x, |h, l| l.forward(h, None).0);
        let [b, t, d] = h.dims();
        self.out.forward(h.narrow(1, t - 1, 1).reshape([b, d]))
    }
}

#[derive(Debug, Clone)]
pub struct LstmAd {
    pub window_size: usize,
    pub hidden_size: usize,
    pub n_layers: usize,
    pub epochs: usize,
    pub lr: f64,
    pub batch_size: usize,
    pub random_state: u64,
    pub contamination: f64,
    net: Option<Net>,
    dev: Option<Device>,
    err_mu: Vec<f64>,
    err_prec: Vec<Vec<f64>>,
    n_channels: usize,
    state: Option<FitState>,
}

impl Default for LstmAd {
    fn default() -> Self {
        LstmAd {
            window_size: 50,
            hidden_size: 64,
            n_layers: 2,
            epochs: 50,
            lr: 1e-3,
            batch_size: 32,
            random_state: 42,
            contamination: 0.1,
            net: None,
            dev: None,
            err_mu: Vec::new(),
            err_prec: Vec::new(),
            n_channels: 0,
            state: None,
        }
    }
}

fn windows3(x: &Matrix, idx: &[usize], w: usize, dev: &Device) -> Tensor<3> {
    let c = x[0].len();
    let flat: Vec<f32> = idx
        .iter()
        .flat_map(|&i| x[i..i + w].iter().flatten().map(|&v| v as f32))
        .collect();
    Tensor::from_data(TensorData::new(flat, [idx.len(), w, c]), dev)
}

impl LstmAd {
    pub fn new(window_size: usize, contamination: f64) -> Self {
        LstmAd {
            window_size,
            contamination,
            ..Self::default()
        }
    }

    /// Forecast errors for every timestamp after the first window.
    fn errors(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let (net, dev) = (self.net.as_ref().unwrap(), self.dev.as_ref().unwrap());
        let w = self.window_size;
        let idx: Vec<usize> = (0..x.len() - w).collect();
        let mut out = Vec::with_capacity(idx.len());
        for chunk in idx.chunks(1024) {
            let pred = to_rows(net.forward(windows3(x, chunk, w, dev)));
            out.extend(chunk.iter().zip(pred).map(|(&i, p)| {
                x[i + w]
                    .iter()
                    .zip(p)
                    .map(|(t, y)| t - y)
                    .collect::<Vec<f64>>()
            }));
        }
        out
    }

    fn scores(&self, x: &Matrix) -> Vec<f64> {
        mahalanobis_sq(&self.errors(x), &self.err_mu, &self.err_prec)
    }
}

impl Detector for LstmAd {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        let c = check_array(x)?;
        self.n_channels = c;
        let w = self.window_size;
        if w == 0 || self.n_layers == 0 || x.len() < w + 10 {
            return Err(OdError::InvalidParameter(format!(
                "need at least window_size + 10 = {} timestamps",
                w + 10
            )));
        }
        let (dev, _lock) = train_device(self.random_state);
        let mut net = Net {
            lstm: (0..self.n_layers)
                .map(|i| {
                    LstmConfig::new(
                        if i == 0 { c } else { self.hidden_size },
                        self.hidden_size,
                        true,
                    )
                    .init(&dev)
                })
                .collect(),
            out: LinearConfig::new(self.hidden_size, c).init(&dev),
        };
        let mut opt = super::adam(0.0);
        let mut rng = Rng::new(self.random_state);
        let n_pairs = x.len() - w;
        for _ in 0..self.epochs {
            for idx in minibatches(n_pairs, self.batch_size, &mut rng, false) {
                let targets: Vec<Vec<f64>> = idx.iter().map(|&i| x[i + w].clone()).collect();
                let pred = net.forward(windows3(x, &idx, w, &dev));
                let loss = (pred - tensor2(&targets, &dev)).powf_scalar(2.0).mean();
                let grads = GradientsParams::from_grads(loss.backward(), &net);
                net = opt.step(self.lr, net, grads);
            }
        }
        self.net = Some(net.valid());
        self.dev = Some(dev.without_autodiff());
        // Mahalanobis distance of forecast errors (sample covariance + 1e-6 I).
        let errors = self.errors(x);
        self.err_mu = crate::linalg::col_mean(&errors);
        let mut cov = covariance(&errors, &self.err_mu, 1);
        for (i, row) in cov.iter_mut().enumerate() {
            row[i] += 1e-6;
        }
        self.err_prec = pinvh(&cov);
        let valid = mahalanobis_sq(&errors, &self.err_mu, &self.err_prec);
        let scores: Vec<Option<f64>> = std::iter::repeat_n(None, w)
            .chain(valid.into_iter().map(Some))
            .collect();
        self.state = Some(ts_fit_state(&scores, self.contamination));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        let state = self.fitted()?;
        check_features(x, self.n_channels)?;
        if x.len() <= self.window_size {
            return Err(OdError::InvalidParameter(
                "series must be longer than window_size".into(),
            ));
        }
        Ok(std::iter::repeat_n(state.threshold, self.window_size)
            .chain(self.scores(x))
            .collect())
    }

    crate::state_accessor!();
}
