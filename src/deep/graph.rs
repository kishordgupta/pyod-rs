//! Neural graph outlier detectors: DOMINANT, AnomalyDAE, CoLA, CONAD and GUIDE.
//! Ports of `pyod.models.pyg_*` (which use PyTorch Geometric); graph
//! convolutions are computed densely here, which suits graphs up to a few
//! thousand nodes. Like PyOD they are transductive: they score the nodes of the
//! fitted graph (see [`GraphDetector`]).

use super::{adam, bce_logits, tensor2, to_vec, train_device};
use crate::base::{check_array, check_contamination, FitState, Matrix, OdError, Result};
use crate::models::graph::{check_edges, GraphDetector};
use crate::utils::Rng;
use burn::module::{Module, Param};
use burn::nn::{Dropout, DropoutConfig, Initializer, Linear, LinearConfig};
use burn::optim::GradientsParams;
use burn::tensor::{activation, Device, Tensor, TensorData};

fn dense(rows: Vec<Vec<f64>>, dev: &Device) -> Tensor<2> {
    tensor2(&rows, dev)
}

/// PyG `gcn_norm` with self loops: `S[t][s] = 1 / sqrt(deg(s) deg(t))` per edge s -> t,
/// where deg counts incoming edges plus the self loop.
fn gcn_matrix(n: usize, edges: &[(usize, usize)], dev: &Device) -> Tensor<2> {
    let mut all: Vec<(usize, usize)> = edges.iter().copied().filter(|(s, t)| s != t).collect();
    all.extend((0..n).map(|i| (i, i)));
    let mut deg = vec![0.0f64; n];
    all.iter().for_each(|&(_, t)| deg[t] += 1.0);
    let mut s = vec![vec![0.0; n]; n];
    for &(a, b) in &all {
        s[b][a] += 1.0 / (deg[a] * deg[b]).sqrt();
    }
    dense(s, dev)
}

/// Binary adjacency `A[s][t] = 1` for each edge (PyTorch `adj[ei[0], ei[1]] = 1`).
fn adjacency(n: usize, edges: &[(usize, usize)], dev: &Device) -> Tensor<2> {
    let mut a = vec![vec![0.0; n]; n];
    edges.iter().for_each(|&(s, t)| a[s][t] = 1.0);
    dense(a, dev)
}

/// Graph convolution `S X W + b` (Glorot weights, zero bias, as `GCNConv`).
#[derive(Module, Debug)]
struct Gcn {
    lin: Linear,
    bias: Param<Tensor<1>>,
}

impl Gcn {
    fn new(d_in: usize, d_out: usize, dev: &Device) -> Self {
        Gcn {
            lin: LinearConfig::new(d_in, d_out)
                .with_bias(false)
                .with_initializer(Initializer::XavierUniform { gain: 1.0 })
                .init(dev),
            bias: Param::from_tensor(Tensor::zeros([d_out], dev)),
        }
    }

    fn forward(&self, s: Tensor<2>, x: Tensor<2>) -> Tensor<2> {
        s.matmul(self.lin.forward(x)) + self.bias.val().unsqueeze_dim(0)
    }
}

/// GCN encoder: ReLU and dropout between layers, none after the last.
#[derive(Module, Debug)]
struct GcnEncoder {
    convs: Vec<Gcn>,
    dropout: Dropout,
}

impl GcnEncoder {
    fn new(d_in: usize, hidden: usize, layers: usize, dropout: f64, dev: &Device) -> Self {
        GcnEncoder {
            convs: (0..layers.max(1))
                .map(|i| Gcn::new(if i == 0 { d_in } else { hidden }, hidden, dev))
                .collect(),
            dropout: DropoutConfig::new(dropout).init(),
        }
    }

    fn forward(&self, s: Tensor<2>, x: Tensor<2>) -> Tensor<2> {
        let last = self.convs.len() - 1;
        self.convs.iter().enumerate().fold(x, |z, (i, c)| {
            let z = c.forward(s.clone(), z);
            if i < last {
                self.dropout.forward(activation::relu(z))
            } else {
                z
            }
        })
    }
}

/// Multi-head graph attention with heads averaged (`GATConv(concat=False)`), self loops added.
#[derive(Module, Debug)]
struct Gat {
    lin: Linear,
    att_src: Param<Tensor<2>>,
    att_dst: Param<Tensor<2>>,
    bias: Param<Tensor<1>>,
    dropout: Dropout,
    #[module(skip)]
    heads: usize,
    #[module(skip)]
    out: usize,
}

impl Gat {
    fn new(d_in: usize, out: usize, heads: usize, dropout: f64, dev: &Device) -> Self {
        let glorot = |fan_in: usize, fan_out: usize, shape: [usize; 2]| {
            let a = (6.0 / (fan_in + fan_out) as f64).sqrt();
            Tensor::<2>::random(shape, burn::tensor::Distribution::Uniform(-a, a), dev)
        };
        Gat {
            lin: LinearConfig::new(d_in, heads * out)
                .with_bias(false)
                .with_initializer(Initializer::XavierUniform { gain: 1.0 })
                .init(dev),
            att_src: Param::from_tensor(glorot(out, heads, [heads, out])),
            att_dst: Param::from_tensor(glorot(out, heads, [heads, out])),
            bias: Param::from_tensor(Tensor::zeros([out], dev)),
            dropout: DropoutConfig::new(dropout).init(),
            heads,
            out,
        }
    }

    /// `mask[i][j]` = 0 where j -> i is an edge (or i == j), -inf otherwise.
    fn forward(&self, mask: Tensor<3>, x: Tensor<2>) -> Tensor<2> {
        let [n, _] = x.dims();
        let (h, c) = (self.heads, self.out);
        let xw = self.lin.forward(x).reshape([n, h, c]).swap_dims(0, 1); // [H, N, C]
        let a_src = (xw.clone() * self.att_src.val().unsqueeze_dim::<3>(1)).sum_dim(2); // [H, N, 1]
        let a_dst = (xw.clone() * self.att_dst.val().unsqueeze_dim::<3>(1)).sum_dim(2); // [H, N, 1]
        let e = activation::leaky_relu(a_dst + a_src.swap_dims(1, 2), 0.2) + mask; // [H, i, j]
        let alpha = self.dropout.forward(activation::softmax(e, 2));
        alpha.matmul(xw).mean_dim(0).reshape([n, c]) + self.bias.val().unsqueeze_dim(0)
    }
}

fn attention_mask(n: usize, edges: &[(usize, usize)], dev: &Device) -> Tensor<3> {
    let mut m = vec![vec![f32::NEG_INFINITY; n]; n];
    for i in 0..n {
        m[i][i] = 0.0;
    }
    edges.iter().for_each(|&(s, t)| m[t][s] = 0.0);
    Tensor::<2>::from_data(
        TensorData::new(m.into_iter().flatten().collect::<Vec<f32>>(), [n, n]),
        dev,
    )
    .unsqueeze_dim(0)
}

/// Per-node `alpha * ||A - sigmoid(Z Z^T)||^2 + (1 - alpha) * ||X - X_hat||^2`.
fn recon_errors(
    adj: Tensor<2>,
    z: Tensor<2>,
    x: Tensor<2>,
    x_hat: Tensor<2>,
    alpha: f64,
) -> Tensor<1> {
    let a_hat = activation::sigmoid(z.clone().matmul(z.transpose()));
    let s = (adj - a_hat)
        .powf_scalar(2.0)
        .sum_dim(1)
        .squeeze_dim::<1>(1);
    let a = (x - x_hat).powf_scalar(2.0).sum_dim(1).squeeze_dim::<1>(1);
    s * alpha + a * (1.0 - alpha)
}

fn prepare(x: &Matrix, edges: &[(usize, usize)], contamination: f64) -> Result<usize> {
    check_contamination(contamination)?;
    let d = check_array(x)?;
    check_edges(x.len(), edges)?;
    Ok(d)
}

/// Settings shared by the GCN-based detectors.
#[derive(Debug, Clone)]
pub struct GraphTrain {
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub dropout: f64,
    pub epochs: usize,
    pub lr: f64,
    pub random_state: u64,
}

impl GraphTrain {
    fn new(lr: f64) -> Self {
        GraphTrain {
            hidden_dim: 64,
            num_layers: 2,
            dropout: 0.3,
            epochs: 100,
            lr,
            random_state: 42,
        }
    }
}

#[derive(Module, Debug)]
struct AutoEnc {
    enc: GcnEncoder,
    dec: Linear,
}

/// DOMINANT: deep anomaly detection on attributed networks (Ding et al., SDM 2019).
/// A GCN autoencoder reconstructs both structure and attributes.
#[derive(Debug, Clone)]
pub struct Dominant {
    pub train: GraphTrain,
    /// Weight of structure vs attribute reconstruction error.
    pub alpha: f64,
    pub contamination: f64,
    state: Option<FitState>,
}

impl Default for Dominant {
    fn default() -> Self {
        Dominant {
            train: GraphTrain::new(5e-3),
            alpha: 0.5,
            contamination: 0.1,
            state: None,
        }
    }
}

impl GraphDetector for Dominant {
    fn fit_graph(&mut self, x: &Matrix, edges: &[(usize, usize)]) -> Result<()> {
        let d = prepare(x, edges, self.contamination)?;
        let (t, n) = (&self.train, x.len());
        let (dev, _lock) = train_device(t.random_state);
        let (s, adj, xt) = (
            gcn_matrix(n, edges, &dev),
            adjacency(n, edges, &dev),
            tensor2(x, &dev),
        );
        let mut m = AutoEnc {
            enc: GcnEncoder::new(d, t.hidden_dim, t.num_layers, t.dropout, &dev),
            dec: LinearConfig::new(t.hidden_dim, d).init(&dev),
        };
        let mut opt = adam(0.0);
        for _ in 0..t.epochs {
            let z = m.enc.forward(s.clone(), xt.clone());
            let loss = recon_errors(
                adj.clone(),
                z.clone(),
                xt.clone(),
                m.dec.forward(z),
                self.alpha,
            )
            .mean();
            let grads = GradientsParams::from_grads(loss.backward(), &m);
            m = opt.step(t.lr, m, grads);
        }
        let (m, ev) = (m.valid(), dev.without_autodiff());
        let (s, adj, xt) = (
            gcn_matrix(n, edges, &ev),
            adjacency(n, edges, &ev),
            tensor2(x, &ev),
        );
        let z = m.enc.forward(s, xt.clone());
        let scores = to_vec(recon_errors(
            adj,
            z.clone(),
            xt,
            m.dec.forward(z),
            self.alpha,
        ));
        self.state = Some(FitState::from_contamination(scores, self.contamination));
        Ok(())
    }

    fn state(&self) -> Option<&FitState> {
        self.state.as_ref()
    }
}

#[derive(Module, Debug)]
struct Dae {
    gat: Gat,
    attr_enc: Linear,
    attr_dec: Linear,
}

impl Dae {
    fn embed(&self, mask: Tensor<3>, x: Tensor<2>) -> Tensor<2> {
        (self.gat.forward(mask, x.clone()) + activation::relu(self.attr_enc.forward(x))) / 2.0
    }
}

/// AnomalyDAE: dual autoencoder (Fan et al., ICASSP 2020) — a graph-attention
/// structure encoder and an MLP attribute encoder share one embedding.
#[derive(Debug, Clone)]
pub struct AnomalyDae {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub alpha: f64,
    pub dropout: f64,
    pub epochs: usize,
    pub lr: f64,
    pub random_state: u64,
    pub contamination: f64,
    state: Option<FitState>,
}

impl Default for AnomalyDae {
    fn default() -> Self {
        AnomalyDae {
            embed_dim: 64,
            num_heads: 4,
            alpha: 0.5,
            dropout: 0.3,
            epochs: 100,
            lr: 5e-3,
            random_state: 42,
            contamination: 0.1,
            state: None,
        }
    }
}

impl GraphDetector for AnomalyDae {
    fn fit_graph(&mut self, x: &Matrix, edges: &[(usize, usize)]) -> Result<()> {
        let d = prepare(x, edges, self.contamination)?;
        let n = x.len();
        let (dev, _lock) = train_device(self.random_state);
        let (mask, adj, xt) = (
            attention_mask(n, edges, &dev),
            adjacency(n, edges, &dev),
            tensor2(x, &dev),
        );
        let mut m = Dae {
            gat: Gat::new(d, self.embed_dim, self.num_heads, self.dropout, &dev),
            attr_enc: LinearConfig::new(d, self.embed_dim).init(&dev),
            attr_dec: LinearConfig::new(self.embed_dim, d).init(&dev),
        };
        let mut opt = adam(0.0);
        for _ in 0..self.epochs {
            let z = m.embed(mask.clone(), xt.clone());
            let loss = recon_errors(
                adj.clone(),
                z.clone(),
                xt.clone(),
                m.attr_dec.forward(z),
                self.alpha,
            )
            .mean();
            let grads = GradientsParams::from_grads(loss.backward(), &m);
            m = opt.step(self.lr, m, grads);
        }
        let (m, ev) = (m.valid(), dev.without_autodiff());
        let (mask, adj, xt) = (
            attention_mask(n, edges, &ev),
            adjacency(n, edges, &ev),
            tensor2(x, &ev),
        );
        let z = m.embed(mask, xt.clone());
        let scores = to_vec(recon_errors(
            adj,
            z.clone(),
            xt,
            m.attr_dec.forward(z),
            self.alpha,
        ));
        self.state = Some(FitState::from_contamination(scores, self.contamination));
        Ok(())
    }

    fn state(&self) -> Option<&FitState> {
        self.state.as_ref()
    }
}

#[derive(Module, Debug)]
struct ColaNet {
    enc: GcnEncoder,
    /// Bilinear discriminator `z^T W c + b`.
    w: Linear,
}

impl ColaNet {
    fn discriminate(&self, z: Tensor<2>, ctx: Tensor<2>) -> Tensor<2> {
        (self.w.forward(z) * ctx).sum_dim(1)
    }
}

/// CoLA: contrastive self-supervised anomaly detection (Liu et al., TNNLS 2021).
/// A discriminator learns whether a node matches its neighbourhood context;
/// poorly matching nodes are anomalous.
#[derive(Debug, Clone)]
pub struct Cola {
    pub train: GraphTrain,
    pub contamination: f64,
    state: Option<FitState>,
}

impl Default for Cola {
    fn default() -> Self {
        Cola {
            train: GraphTrain::new(1e-3),
            contamination: 0.1,
            state: None,
        }
    }
}

/// Row-normalised adjacency: each node's context is the mean of its out-neighbours.
fn context_matrix(n: usize, edges: &[(usize, usize)], dev: &Device) -> Tensor<2> {
    let mut deg = vec![0.0f64; n];
    edges.iter().for_each(|&(s, _)| deg[s] += 1.0);
    let mut m = vec![vec![0.0; n]; n];
    edges
        .iter()
        .for_each(|&(s, t)| m[s][t] += 1.0 / deg[s].max(1.0));
    dense(m, dev)
}

impl GraphDetector for Cola {
    fn fit_graph(&mut self, x: &Matrix, edges: &[(usize, usize)]) -> Result<()> {
        let d = prepare(x, edges, self.contamination)?;
        let (t, n) = (&self.train, x.len());
        let (dev, _lock) = train_device(t.random_state);
        let (s, ctx_m, xt) = (
            gcn_matrix(n, edges, &dev),
            context_matrix(n, edges, &dev),
            tensor2(x, &dev),
        );
        let mut m = ColaNet {
            enc: GcnEncoder::new(d, t.hidden_dim, t.num_layers, 0.3, &dev),
            w: LinearConfig::new(t.hidden_dim, t.hidden_dim).init(&dev),
        };
        let mut opt = adam(0.0);
        let mut rng = Rng::new(t.random_state);
        let (ones, zeros) = (
            Tensor::<2>::ones([n, 1], &dev),
            Tensor::<2>::zeros([n, 1], &dev),
        );
        for _ in 0..t.epochs {
            let z = m.enc.forward(s.clone(), xt.clone());
            let ctx = ctx_m.clone().matmul(z.clone());
            let perm: Vec<usize> = rng.sample_indices(n, n);
            let x_neg = tensor2(
                &perm.iter().map(|&i| x[i].clone()).collect::<Vec<_>>(),
                &dev,
            );
            let z_neg = m.enc.forward(s.clone(), x_neg);
            let loss = bce_logits(m.discriminate(z, ctx.clone()), ones.clone())
                + bce_logits(m.discriminate(z_neg, ctx), zeros.clone());
            let grads = GradientsParams::from_grads(loss.backward(), &m);
            m = opt.step(t.lr, m, grads);
        }
        // As in PyOD, score with dropout active and average 5 passes.
        let mut scores = vec![0.0; n];
        for _ in 0..5 {
            let z = m.enc.forward(s.clone(), xt.clone()).detach();
            let ctx = ctx_m.clone().matmul(z.clone());
            for (acc, v) in scores.iter_mut().zip(to_vec(m.discriminate(z, ctx))) {
                *acc -= v / 5.0;
            }
        }
        self.state = Some(FitState::from_contamination(scores, self.contamination));
        Ok(())
    }

    fn state(&self) -> Option<&FitState> {
        self.state.as_ref()
    }
}

/// CONAD: contrastive attributed network anomaly detection with data
/// augmentation (Xu et al., PAKDD 2022).
#[derive(Debug, Clone)]
pub struct Conad {
    pub train: GraphTrain,
    /// Fraction of nodes/edges perturbed in the augmented view.
    pub aug_ratio: f64,
    pub alpha: f64,
    pub contamination: f64,
    state: Option<FitState>,
}

impl Default for Conad {
    fn default() -> Self {
        Conad {
            train: GraphTrain::new(1e-3),
            aug_ratio: 0.2,
            alpha: 0.5,
            contamination: 0.1,
            state: None,
        }
    }
}

/// Port of `_create_anomalous_view`: swap some nodes' features and add random edges.
fn anomalous_view(
    x: &Matrix,
    edges: &[(usize, usize)],
    ratio: f64,
    rng: &mut Rng,
) -> (Vec<Vec<f64>>, Vec<(usize, usize)>) {
    let n = x.len();
    let k = ((n as f64 * ratio) as usize).max(1).min(n);
    let mut xa = x.to_vec();
    let (targets, sources) = (rng.sample_indices(n, k), rng.sample_indices(n, k));
    for (t, s) in targets.into_iter().zip(sources) {
        xa[t] = x[s].clone();
    }
    let n_add = ((edges.len() as f64 * ratio) as usize).max(1);
    let mut ea = edges.to_vec();
    ea.extend((0..n_add).map(|_| (rng.below(n), rng.below(n))));
    (xa, ea)
}

fn cosine_rows(a: Tensor<2>, b: Tensor<2>) -> Tensor<1> {
    let norm = |t: Tensor<2>| t.clone() / t.powf_scalar(2.0).sum_dim(1).sqrt().clamp_min(1e-12);
    (norm(a) * norm(b)).sum_dim(1).squeeze_dim(1)
}

impl GraphDetector for Conad {
    fn fit_graph(&mut self, x: &Matrix, edges: &[(usize, usize)]) -> Result<()> {
        let d = prepare(x, edges, self.contamination)?;
        let (t, n) = (&self.train, x.len());
        let (dev, _lock) = train_device(t.random_state);
        let (s, adj, xt) = (
            gcn_matrix(n, edges, &dev),
            adjacency(n, edges, &dev),
            tensor2(x, &dev),
        );
        let mut m = AutoEnc {
            enc: GcnEncoder::new(d, t.hidden_dim, t.num_layers, t.dropout, &dev),
            dec: LinearConfig::new(t.hidden_dim, d).init(&dev),
        };
        let mut opt = adam(0.0);
        let mut rng = Rng::new(t.random_state);
        for _ in 0..t.epochs {
            let (xa, ea) = anomalous_view(x, edges, self.aug_ratio, &mut rng);
            let z = m.enc.forward(s.clone(), xt.clone());
            let za = m.enc.forward(gcn_matrix(n, &ea, &dev), tensor2(&xa, &dev));
            let contrast = cosine_rows(z.clone(), za).mean().neg();
            // Structure and attribute errors are averaged separately, then added.
            let recon = recon_errors(adj.clone(), z.clone(), xt.clone(), m.dec.forward(z), 0.5)
                .mean()
                * 2.0;
            let loss = contrast + recon * self.alpha;
            let grads = GradientsParams::from_grads(loss.backward(), &m);
            m = opt.step(t.lr, m, grads);
        }
        let (m, ev) = (m.valid(), dev.without_autodiff());
        let (xa, ea) = anomalous_view(x, edges, self.aug_ratio, &mut rng);
        let (s, adj, xt) = (
            gcn_matrix(n, edges, &ev),
            adjacency(n, edges, &ev),
            tensor2(x, &ev),
        );
        let z = m.enc.forward(s, xt.clone());
        let za = m.enc.forward(gcn_matrix(n, &ea, &ev), tensor2(&xa, &ev));
        let cos_dist = cosine_rows(z.clone(), za).neg() + 1.0;
        let recon = recon_errors(adj, z.clone(), xt, m.dec.forward(z), 0.5) * 2.0;
        let scores = to_vec(cos_dist + recon * self.alpha);
        self.state = Some(FitState::from_contamination(scores, self.contamination));
        Ok(())
    }

    fn state(&self) -> Option<&FitState> {
        self.state.as_ref()
    }
}

#[derive(Module, Debug)]
struct GuideNet {
    orig: AutoEnc,
    motif: AutoEnc,
}

/// GUIDE: higher-order structure based anomaly detection (Yuan et al., 2021).
/// Reconstructs the graph and its triangle-motif graph with two GCN autoencoders.
#[derive(Debug, Clone)]
pub struct Guide {
    pub train: GraphTrain,
    pub alpha: f64,
    pub contamination: f64,
    state: Option<FitState>,
}

impl Default for Guide {
    fn default() -> Self {
        Guide {
            train: GraphTrain::new(5e-3),
            alpha: 0.5,
            contamination: 0.1,
            state: None,
        }
    }
}

impl GraphDetector for Guide {
    fn fit_graph(&mut self, x: &Matrix, edges: &[(usize, usize)]) -> Result<()> {
        let d = prepare(x, edges, self.contamination)?;
        let (t, n) = (&self.train, x.len());
        // Motif edges: edges that close a triangle, ((A A) * A) != 0.
        let mut a = vec![vec![0.0f64; n]; n];
        edges.iter().for_each(|&(s, t)| a[s][t] += 1.0);
        let motif: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .filter(|&(i, j)| a[i][j] != 0.0 && (0..n).any(|k| a[i][k] != 0.0 && a[k][j] != 0.0))
            .collect();
        if motif.is_empty() {
            return Err(OdError::InvalidParameter(
                "GUIDE needs triangles in the graph; use DOMINANT or CoLA instead".into(),
            ));
        }
        let score = |m: &GuideNet, dev: &Device| {
            let xt = tensor2(x, dev);
            let zo = m.orig.enc.forward(gcn_matrix(n, edges, dev), xt.clone());
            let zm = m.motif.enc.forward(gcn_matrix(n, &motif, dev), xt.clone());
            let eo = recon_errors(
                adjacency(n, edges, dev),
                zo.clone(),
                xt.clone(),
                m.orig.dec.forward(zo),
                0.5,
            ) * 2.0;
            let em = recon_errors(
                adjacency(n, &motif, dev),
                zm.clone(),
                xt,
                m.motif.dec.forward(zm),
                0.5,
            ) * 2.0;
            eo * self.alpha + em * (1.0 - self.alpha)
        };
        let (dev, _lock) = train_device(t.random_state);
        let new_ae = || AutoEnc {
            enc: GcnEncoder::new(d, t.hidden_dim, t.num_layers, t.dropout, &dev),
            dec: LinearConfig::new(t.hidden_dim, d).init(&dev),
        };
        let mut m = GuideNet {
            orig: new_ae(),
            motif: new_ae(),
        };
        let mut opt = adam(0.0);
        for _ in 0..t.epochs {
            let loss = score(&m, &dev).mean();
            let grads = GradientsParams::from_grads(loss.backward(), &m);
            m = opt.step(t.lr, m, grads);
        }
        let scores = to_vec(score(&m.valid(), &dev.without_autodiff()));
        self.state = Some(FitState::from_contamination(scores, self.contamination));
        Ok(())
    }

    fn state(&self) -> Option<&FitState> {
        self.state.as_ref()
    }
}
