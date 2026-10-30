//! Graph outlier detection without neural networks: SCAN, ANOMALOUS and Radar.
//! Ports of `pyod.models.pyg_scan`, `pyg_anomalous` and `pyg_radar`.
//!
//! Graph detectors are transductive, as in PyOD: they score the nodes of the
//! graph they are fitted on. A graph is node features (one row per node) plus
//! directed edges `(source, target)`; list both directions for an undirected graph.

use crate::base::{check_array, check_contamination, FitState, Matrix, OdError, Result};
use crate::linalg::{identity, matmul, solve, transpose, Mat};

/// Common interface for the graph detectors.
pub trait GraphDetector {
    /// Fit on node features and directed edges; scores every node.
    fn fit_graph(&mut self, x: &Matrix, edges: &[(usize, usize)]) -> Result<()>;

    fn state(&self) -> Option<&FitState>;

    fn decision_scores(&self) -> Result<&[f64]> {
        Ok(&self.state().ok_or(OdError::NotFitted)?.decision_scores)
    }

    fn labels(&self) -> Result<&[u8]> {
        Ok(&self.state().ok_or(OdError::NotFitted)?.labels)
    }

    fn threshold(&self) -> Result<f64> {
        Ok(self.state().ok_or(OdError::NotFitted)?.threshold)
    }
}

pub(crate) fn check_edges(n: usize, edges: &[(usize, usize)]) -> Result<()> {
    match edges.iter().find(|&&(s, t)| s >= n || t >= n) {
        Some(e) => Err(OdError::InvalidParameter(format!(
            "edge {e:?} refers to a node >= {n}"
        ))),
        None => Ok(()),
    }
}

/// Dense adjacency with duplicate edges summed (scipy `csr_matrix` semantics).
pub(crate) fn summed_adjacency(n: usize, edges: &[(usize, usize)]) -> Mat {
    let mut a = vec![vec![0.0; n]; n];
    edges.iter().for_each(|&(s, t)| a[s][t] += 1.0);
    a
}

/// Graph Laplacian `D - A`, with D the row sums of A.
fn laplacian(a: &Mat) -> Mat {
    let n = a.len();
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    if i == j {
                        a[i].iter().sum::<f64>() - a[i][j]
                    } else {
                        -a[i][j]
                    }
                })
                .collect()
        })
        .collect()
}

/// `R = shrink(solve(I + gamma L, P))`: Laplacian-smoothed residual with row-wise
/// group-lasso shrinkage, shared by ANOMALOUS and Radar.
fn smoothed_residual(lap: &Mat, p: &Mat, gamma: f64, lambda: f64) -> Result<Mat> {
    let n = lap.len();
    let sys: Mat = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| (i == j) as u8 as f64 + gamma * lap[i][j])
                .collect()
        })
        .collect();
    let smooth = solve(&sys, p)
        .ok_or_else(|| OdError::InvalidParameter("singular smoothing system".into()))?;
    Ok(smooth
        .into_iter()
        .map(|row| {
            let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-10);
            let shrink = (1.0 - lambda / (2.0 * norm)).max(0.0);
            row.into_iter().map(|v| v * shrink).collect()
        })
        .collect())
}

fn row_norms(r: &Mat) -> Vec<f64> {
    r.iter()
        .map(|row| row.iter().map(|v| v * v).sum::<f64>().sqrt())
        .collect()
}

fn sub(a: &Mat, b: &Mat) -> Mat {
    a.iter()
        .zip(b)
        .map(|(x, y)| x.iter().zip(y).map(|(p, q)| p - q).collect())
        .collect()
}

/// SCAN: structural clustering (Xu et al., KDD 2007). Hubs between clusters and
/// outliers attached to at most one cluster score high; core nodes score low.
#[derive(Debug, Clone)]
pub struct Scan {
    /// Structural-similarity threshold for an edge to count as an ε-neighbour.
    pub epsilon: f64,
    /// Minimum ε-neighbours for a core node.
    pub mu: usize,
    pub contamination: f64,
    state: Option<FitState>,
}

impl Default for Scan {
    fn default() -> Self {
        Scan {
            epsilon: 0.5,
            mu: 2,
            contamination: 0.1,
            state: None,
        }
    }
}

impl Scan {
    fn scores(&self, n: usize, edges: &[(usize, usize)]) -> Vec<f64> {
        if edges.is_empty() {
            return vec![1.0; n];
        }
        let adj = summed_adjacency(n, edges);
        let adj_self: Mat = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| ((adj[i][j] + (i == j) as u8 as f64) > 0.0) as u8 as f64)
                    .collect()
            })
            .collect();
        let deg: Vec<f64> = adj_self.iter().map(|r| r.iter().sum()).collect();
        let inter = matmul(&adj_self, &transpose(&adj_self));
        let inv: Vec<f64> = deg
            .iter()
            .map(|d| if *d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
            .collect();
        let sim_edges: Mat = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| inv[i] * inter[i][j] * inv[j] * adj[i][j])
                    .collect()
            })
            .collect();
        let eps_adj: Mat = sim_edges
            .iter()
            .map(|r| {
                r.iter()
                    .map(|&v| if v < self.epsilon { 0.0 } else { v })
                    .collect()
            })
            .collect();
        let n_eps: Vec<usize> = eps_adj
            .iter()
            .map(|r| r.iter().filter(|&&v| v > 0.0).count())
            .collect();
        let core: Vec<bool> = n_eps.iter().map(|&c| c >= self.mu).collect();
        // Connected components of the core-to-core ε-graph (undirected).
        let mut comp: Vec<Option<usize>> = vec![None; n];
        let mut next = 0;
        for s in (0..n).filter(|&s| core[s]) {
            if comp[s].is_some() {
                continue;
            }
            let mut stack = vec![s];
            comp[s] = Some(next);
            while let Some(u) = stack.pop() {
                for v in 0..n {
                    if core[v] && comp[v].is_none() && (eps_adj[u][v] > 0.0 || eps_adj[v][u] > 0.0)
                    {
                        comp[v] = Some(next);
                        stack.push(v);
                    }
                }
            }
            next += 1;
        }
        (0..n)
            .map(|u| {
                let deg_u: f64 = adj[u].iter().sum();
                let mean_sim = if deg_u > 0.0 {
                    sim_edges[u].iter().sum::<f64>() / deg_u
                } else {
                    0.0
                };
                if core[u] {
                    (1.0 - n_eps[u] as f64 / (self.mu * 2).max(1) as f64).max(0.0)
                } else {
                    let mut clusters: Vec<usize> = (0..n)
                        .filter(|&v| adj[u][v] != 0.0)
                        .filter_map(|v| comp[v])
                        .collect();
                    clusters.sort_unstable();
                    clusters.dedup();
                    if clusters.len() >= 2 {
                        0.5 + 0.5 * (1.0 - mean_sim)
                    } else {
                        1.0 + (1.0 - mean_sim)
                    }
                }
            })
            .collect()
    }
}

impl GraphDetector for Scan {
    /// Uses only the graph structure; `x` just fixes the node count.
    fn fit_graph(&mut self, x: &Matrix, edges: &[(usize, usize)]) -> Result<()> {
        check_contamination(self.contamination)?;
        check_array(x)?;
        check_edges(x.len(), edges)?;
        self.state = Some(FitState::from_contamination(
            self.scores(x.len(), edges),
            self.contamination,
        ));
        Ok(())
    }

    fn state(&self) -> Option<&FitState> {
        self.state.as_ref()
    }
}

/// ANOMALOUS: joint attribute selection and residual analysis (Peng et al., IJCAI 2018).
#[derive(Debug, Clone)]
pub struct Anomalous {
    pub alpha: f64,
    pub gamma: f64,
    pub lambda_r: f64,
    pub max_iter: usize,
    pub contamination: f64,
    state: Option<FitState>,
}

impl Default for Anomalous {
    fn default() -> Self {
        Anomalous {
            alpha: 1.0,
            gamma: 1.0,
            lambda_r: 0.01,
            max_iter: 100,
            contamination: 0.1,
            state: None,
        }
    }
}

impl GraphDetector for Anomalous {
    fn fit_graph(&mut self, x: &Matrix, edges: &[(usize, usize)]) -> Result<()> {
        check_contamination(self.contamination)?;
        let d = check_array(x)?;
        let n = x.len();
        check_edges(n, edges)?;
        let lap = laplacian(&summed_adjacency(n, edges));
        let norms: Vec<f64> = (0..d)
            .map(|j| x.iter().map(|r| r[j] * r[j]).sum::<f64>().sqrt().max(1e-10))
            .collect();
        let xn: Mat = x
            .iter()
            .map(|r| r.iter().zip(&norms).map(|(v, s)| v / s).collect())
            .collect();
        let xt = transpose(&xn);
        let mut w = vec![vec![0.0; n]; d];
        let mut r = vec![vec![0.0; d]; n];
        let lr = 1e-3;
        for _ in 0..self.max_iter {
            let xwx = matmul(&matmul(&xn, &w), &xn);
            let residual = sub(&sub(&xn, &xwx), &r);
            let mut grad = matmul(&matmul(&xt, &residual), &xt);
            for (gi, wi) in grad.iter_mut().zip(&w) {
                for (g, wv) in gi.iter_mut().zip(wi) {
                    *g = -2.0 * *g + 2.0 * self.alpha * wv;
                }
            }
            let gn = grad.iter().flatten().map(|v| v * v).sum::<f64>().sqrt();
            let scale = if gn > 1.0 { 1.0 / gn } else { 1.0 };
            for (wi, gi) in w.iter_mut().zip(&grad) {
                for (wv, g) in wi.iter_mut().zip(gi) {
                    *wv -= lr * g * scale;
                }
            }
            let p = sub(&xn, &matmul(&matmul(&xn, &w), &xn));
            r = smoothed_residual(&lap, &p, self.gamma, self.lambda_r)?;
        }
        self.state = Some(FitState::from_contamination(
            row_norms(&r),
            self.contamination,
        ));
        Ok(())
    }

    fn state(&self) -> Option<&FitState> {
        self.state.as_ref()
    }
}

/// Radar: residual analysis for anomaly detection in attributed networks (Li et al., IJCAI 2017).
#[derive(Debug, Clone)]
pub struct Radar {
    pub alpha: f64,
    pub gamma: f64,
    pub max_iter: usize,
    pub contamination: f64,
    state: Option<FitState>,
}

impl Default for Radar {
    fn default() -> Self {
        Radar {
            alpha: 1.0,
            gamma: 0.01,
            max_iter: 100,
            contamination: 0.1,
            state: None,
        }
    }
}

impl GraphDetector for Radar {
    fn fit_graph(&mut self, x: &Matrix, edges: &[(usize, usize)]) -> Result<()> {
        check_contamination(self.contamination)?;
        check_array(x)?;
        let n = x.len();
        check_edges(n, edges)?;
        let lap = laplacian(&summed_adjacency(n, edges));
        let xt = transpose(x);
        let mut gram = matmul(x, &xt);
        for (i, row) in gram.iter_mut().enumerate() {
            row[i] += self.alpha;
        }
        // W = (X - R) X^T (X X^T + alpha I)^-1, computed as a solve against the symmetric system.
        let gram_inv = solve(&gram, &identity(n))
            .ok_or_else(|| OdError::InvalidParameter("singular Gram matrix".into()))?;
        let mut r = vec![vec![0.0; x[0].len()]; n];
        for _ in 0..self.max_iter {
            let w = matmul(&matmul(&sub(&x.to_vec(), &r), &xt), &gram_inv);
            let p = sub(&x.to_vec(), &matmul(&w, x));
            r = smoothed_residual(&lap, &p, self.gamma, self.gamma)?;
        }
        self.state = Some(FitState::from_contamination(
            row_norms(&r),
            self.contamination,
        ));
        Ok(())
    }

    fn state(&self) -> Option<&FitState> {
        self.state.as_ref()
    }
}
