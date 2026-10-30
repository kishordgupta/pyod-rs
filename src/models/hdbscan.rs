//! HDBSCAN-based outlier detection (Campello et al., 2013). Port of `pyod.models.hdbscan`,
//! including the parts of sklearn's `HDBSCAN` it relies on (Prim's MST over mutual
//! reachability, condensed tree, excess-of-mass cluster selection).
//!
//! Training scores are `1 - cluster membership probability`; new samples get the
//! inverse-distance-weighted mean score of their nearest training points.

use crate::base::{
    check_array, check_contamination, check_features, Detector, FitState, Matrix, OdError, Result,
};
use crate::utils::{euclidean, k_nearest};
use std::collections::{BTreeMap, HashMap, HashSet};

/// One merge in the single-linkage hierarchy (scipy `linkage` format).
#[derive(Debug, Clone, Copy)]
struct Merge {
    left: usize,
    right: usize,
    dist: f64,
    size: usize,
}

/// One edge of the condensed tree.
#[derive(Debug, Clone, Copy)]
struct Condensed {
    parent: usize,
    child: usize,
    lambda: f64,
    size: usize,
}

/// Port of sklearn's `mst_from_data_matrix` (Prim's algorithm, alpha = 1).
fn mst(x: &Matrix, core: &[f64]) -> Vec<(usize, usize, f64)> {
    let n = x.len();
    let mut in_tree = vec![false; n];
    let mut min_reach = vec![f64::INFINITY; n];
    let mut sources = vec![0usize; n];
    let mut out = Vec::with_capacity(n - 1);
    let mut current = 0;
    for _ in 0..n - 1 {
        in_tree[current] = true;
        let mut new_reach = f64::MAX;
        let (mut src, mut new_node) = (0, 0);
        for j in 0..n {
            if in_tree[j] {
                continue;
            }
            let mr = core[current]
                .max(core[j])
                .max(euclidean(&x[current], &x[j]));
            if mr < min_reach[j] {
                min_reach[j] = mr;
                sources[j] = current;
                if mr < new_reach {
                    new_reach = mr;
                    src = current;
                    new_node = j;
                }
            } else if min_reach[j] < new_reach {
                new_reach = min_reach[j];
                src = sources[j];
                new_node = j;
            }
        }
        out.push((src, new_node, new_reach));
        current = new_node;
    }
    out
}

/// Port of `make_single_linkage` over MST edges sorted by distance.
fn single_linkage(mut edges: Vec<(usize, usize, f64)>, n: usize) -> Vec<Merge> {
    edges.sort_by(|a, b| a.2.total_cmp(&b.2));
    let mut parent = vec![usize::MAX; 2 * n - 1];
    let mut size: Vec<usize> = (0..2 * n - 1).map(|i| (i < n) as usize).collect();
    let mut next = n;
    let find = |parent: &mut Vec<usize>, mut x: usize| {
        let start = x;
        while parent[x] != usize::MAX {
            x = parent[x];
        }
        let mut p = start;
        while parent[p] != usize::MAX && parent[p] != x {
            let up = parent[p];
            parent[p] = x;
            p = up;
        }
        x
    };
    edges
        .into_iter()
        .map(|(a, b, dist)| {
            let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
            let merged = size[ra] + size[rb];
            parent[ra] = next;
            parent[rb] = next;
            size[next] = merged;
            next += 1;
            Merge {
                left: ra,
                right: rb,
                dist,
                size: merged,
            }
        })
        .collect()
}

fn bfs_hierarchy(h: &[Merge], root: usize) -> Vec<usize> {
    let n = h.len() + 1;
    let mut result = Vec::new();
    let mut queue = vec![root];
    while !queue.is_empty() {
        result.extend(&queue);
        queue = queue
            .iter()
            .filter(|&&x| x >= n)
            .flat_map(|&x| [h[x - n].left, h[x - n].right])
            .collect();
    }
    result
}

/// Port of `_condense_tree`.
fn condense(h: &[Merge], min_cluster_size: usize) -> Vec<Condensed> {
    let n = h.len() + 1;
    let root = 2 * h.len();
    let mut next_label = n + 1;
    let mut relabel = vec![0usize; root + 1];
    relabel[root] = n;
    let mut ignore = vec![false; root + 1];
    let mut out = Vec::new();
    let count = |c: usize| if c >= n { h[c - n].size } else { 1 };
    for node in bfs_hierarchy(h, root) {
        if ignore[node] || node < n {
            continue;
        }
        let m = h[node - n];
        let lambda = if m.dist > 0.0 {
            1.0 / m.dist
        } else {
            f64::INFINITY
        };
        let (lc, rc) = (count(m.left), count(m.right));
        let drop_points =
            |sub: usize, out: &mut Vec<Condensed>, ignore: &mut Vec<bool>, parent: usize| {
                for s in bfs_hierarchy(h, sub) {
                    if s < n {
                        out.push(Condensed {
                            parent,
                            child: s,
                            lambda,
                            size: 1,
                        });
                    }
                    ignore[s] = true;
                }
            };
        let p = relabel[node];
        if lc >= min_cluster_size && rc >= min_cluster_size {
            relabel[m.left] = next_label;
            next_label += 1;
            out.push(Condensed {
                parent: p,
                child: relabel[m.left],
                lambda,
                size: lc,
            });
            relabel[m.right] = next_label;
            next_label += 1;
            out.push(Condensed {
                parent: p,
                child: relabel[m.right],
                lambda,
                size: rc,
            });
        } else if lc < min_cluster_size && rc < min_cluster_size {
            drop_points(m.left, &mut out, &mut ignore, p);
            drop_points(m.right, &mut out, &mut ignore, p);
        } else if lc < min_cluster_size {
            relabel[m.right] = p;
            drop_points(m.left, &mut out, &mut ignore, p);
        } else {
            relabel[m.left] = p;
            drop_points(m.right, &mut out, &mut ignore, p);
        }
    }
    out
}

/// Ports of `_compute_stability`, EOM `_get_clusters`, `_do_labelling` and
/// `get_probabilities`. Returns per-point membership probabilities.
fn probabilities(tree: &[Condensed], n: usize) -> Vec<f64> {
    let root = tree.iter().map(|c| c.parent).min().unwrap();
    let mut births: HashMap<usize, f64> = tree.iter().map(|c| (c.child, c.lambda)).collect();
    births.insert(root, 0.0);
    let mut stability: BTreeMap<usize, f64> = BTreeMap::new();
    for c in tree {
        *stability.entry(c.parent).or_insert(0.0) += (c.lambda - births[&c.parent]) * c.size as f64;
    }
    let cluster_tree: Vec<&Condensed> = tree.iter().filter(|c| c.size > 1).collect();
    // Excess of mass, visiting clusters from the leaves (largest ids) up, excluding the root.
    let nodes: Vec<usize> = stability
        .keys()
        .rev()
        .copied()
        .filter(|&c| c != root)
        .collect();
    let mut is_cluster: HashMap<usize, bool> = nodes.iter().map(|&c| (c, true)).collect();
    for &node in &nodes {
        let sub: f64 = cluster_tree
            .iter()
            .filter(|c| c.parent == node)
            .map(|c| stability.get(&c.child).copied().unwrap_or(0.0))
            .sum();
        if sub > stability[&node] {
            is_cluster.insert(node, false);
            stability.insert(node, sub);
        } else {
            let mut queue = vec![node];
            while !queue.is_empty() {
                for &q in &queue {
                    if q != node {
                        is_cluster.insert(q, false);
                    }
                }
                queue = cluster_tree
                    .iter()
                    .filter(|c| queue.contains(&c.parent))
                    .map(|c| c.child)
                    .collect();
            }
        }
    }
    let clusters: HashSet<usize> = is_cluster
        .iter()
        .filter(|(_, &v)| v)
        .map(|(&k, _)| k)
        .collect();

    // Label each point with its selected ancestor cluster (or noise).
    let mut parent_of: HashMap<usize, usize> = HashMap::new();
    for c in tree {
        if !clusters.contains(&c.child) {
            parent_of.insert(c.child, c.parent);
        }
    }
    let find = |mut x: usize| {
        while let Some(&p) = parent_of.get(&x) {
            x = p;
        }
        x
    };
    let labels: Vec<Option<usize>> = (0..n)
        .map(|i| Some(find(i)).filter(|&c| c != root))
        .collect();

    // Max lambda at which any child leaves each cluster.
    let mut deaths: HashMap<usize, f64> = HashMap::new();
    for c in tree {
        let e = deaths.entry(c.parent).or_insert(c.lambda);
        *e = e.max(c.lambda);
    }
    let mut probs = vec![0.0; n];
    for c in tree {
        if c.child >= root {
            continue;
        }
        if let Some(cluster) = labels[c.child] {
            let max_lambda = deaths.get(&cluster).copied().unwrap_or(0.0);
            probs[c.child] = if max_lambda == 0.0 || c.lambda.is_infinite() {
                1.0
            } else {
                c.lambda.min(max_lambda) / max_lambda
            };
        }
    }
    probs
}

#[derive(Debug, Clone)]
pub struct Hdbscan {
    pub min_cluster_size: usize,
    /// Neighbourhood size for core distances; `None` means `min_cluster_size`.
    pub min_samples: Option<usize>,
    pub contamination: f64,
    x_train: Vec<Vec<f64>>,
    state: Option<FitState>,
}

impl Default for Hdbscan {
    fn default() -> Self {
        Self::new(5, 0.1)
    }
}

impl Hdbscan {
    pub fn new(min_cluster_size: usize, contamination: f64) -> Self {
        Hdbscan {
            min_cluster_size,
            min_samples: None,
            contamination,
            x_train: Vec::new(),
            state: None,
        }
    }
}

impl Detector for Hdbscan {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        check_array(x)?;
        let n = x.len();
        let ms = self.min_samples.unwrap_or(self.min_cluster_size);
        if n < 2 || ms == 0 || ms > n || self.min_cluster_size < 2 {
            return Err(OdError::InvalidParameter(
                "need min_cluster_size >= 2 and 1 <= min_samples <= n_samples".into(),
            ));
        }
        // Core distance: distance to the min_samples-th neighbour, counting the point itself.
        let core: Vec<f64> = x
            .iter()
            .map(|r| k_nearest(x, r, ms, None).last().unwrap().1)
            .collect();
        let tree = condense(&single_linkage(mst(x, &core), n), self.min_cluster_size);
        let scores = probabilities(&tree, n).iter().map(|p| 1.0 - p).collect();
        self.x_train = x.to_vec();
        self.state = Some(FitState::from_contamination(scores, self.contamination));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        let state = self.fitted()?;
        check_features(x, self.x_train[0].len())?;
        let k = self.min_cluster_size.min(self.x_train.len());
        Ok(x.iter()
            .map(|r| {
                let nn = k_nearest(&self.x_train, r, k, None);
                let w: Vec<f64> = nn.iter().map(|&(_, d)| 1.0 / (d + 1e-10)).collect();
                let total: f64 = w.iter().sum();
                nn.iter()
                    .zip(&w)
                    .map(|(&(i, _), wi)| wi / total * state.decision_scores[i])
                    .sum()
            })
            .collect())
    }

    crate::state_accessor!();
}
