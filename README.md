# pyod-rs

A Rust port of [PyOD](https://github.com/yzhao062/pyod), covering every
detector in `pyod/models`.

- **Default build:** no dependencies. Includes the classical, ensemble, time-series
  and non-neural graph detectors.
- **`deep` feature:** adds the neural-network detectors, built on
  [burn](https://burn.dev) 0.22 (pure-Rust CPU backend).

```toml
[dependencies]
pyod-rs = { path = "pyod-rs" }                       # classical models
# pyod-rs = { path = "pyod-rs", features = ["deep"] } # + neural models
```

```rust
use pyod_rs::{Detector, ProbaMethod, Ecod};

let mut det = Ecod::new(0.1);              // contamination
det.fit(&x_train)?;                        // x: &[Vec<f64>], one row per sample
let train_labels = det.labels()?;          // labels_
let scores = det.decision_function(&x_test)?;
let labels = det.predict(&x_test)?;
let probs  = det.predict_proba(&x_test, ProbaMethod::Unify)?;
let conf   = det.predict_confidence(&x_test)?;
```

## Interfaces

| PyOD | Rust |
|---|---|
| `BaseDetector` | `Detector` trait: `fit`, `decision_function`, `predict`, `fit_predict`, `predict_proba` (linear/unify), `predict_confidence`, and the fitted `decision_scores` / `threshold` / `labels` |
| supervised `fit(X, y)` (XGBOD, DevNet) | `fit_supervised(x, y)`; plain `fit` returns an error |
| time-series detectors | `Detector` on a matrix with one row per timestamp and one column per channel |
| `pyg_*` graph detectors | `models::graph::GraphDetector::fit_graph(node_features, edges)`; transductive |
| `EmbeddingOD`, `MultiModalOD` | generic over the item type, with an `Encoder<T>` trait you implement |
| ensembles' base estimators | `DetectorFactory` closures or `Box<dyn Detector>` |
| PyThresh `contamination` objects | `Thresholder` trait + `apply_thresholder` |
| `combination` | `models::combination::{average, maximization, median, majority_vote, aom, moa}` |

## Models and how they compare with PyOD 3.6.7

Each model was run through PyOD and the port on the same data.
- **Deterministic models** are compared score by score.
- **Models that use randomness** (sampling, random init, SGD) can't match PyOD
  number for number, so they are compared by ROC AUC, and where the gap
  looked large, by AUC spread across several seeds on both sides.

### Same scores as PyOD (relative error ≤ 1e-7 unless noted)

ABOD, CD, COF, COPOD, ECOD, GMM (1 component), HBOS, HDBSCAN, KDE, KNN,
LOCI, LOF, MAD, PCA, QMCD, ROD, SOD, SOS, KPCA (test scores; see below),
OCSVM (abs. error 4e-4 = solver tolerance), R-Graph (2e-3, lasso solver
tolerance), Spectral Residual, Matrix Profile, TimeSeriesOD (+KNN), SCAN,
ANOMALOUS, Radar. The log-mel features behind AudioAE match librosa 0.11 to 2e-7 dB.

### Randomised: same algorithm, comparable AUC

| Group | Models |
|---|---|
| Classical | IForest, INNE, LODA, LMDD, CBLOF, Sampling, MCD (identical to PyOD with another seed), DIF |
| Ensembles | FeatureBagging, LSCP, SUOD, XGBOD |
| Time series | k-Shape, SAND, LSTM-AD, Anomaly Transformer |
| Neural (`deep`) | AutoEncoder, VAE, DeepSVDD, AE1SVM, LUNAR, ALAD, SO-GAAL, MO-GAAL, AnoGAN, DevNet, AudioAE |
| Graph (`deep`) | DOMINANT, AnomalyDAE, CoLA, CONAD, GUIDE |

Examples from the cross-checks:
- **Deep models:** AutoEncoder and VAE score 1.00 vs 1.00, DeepSVDD .987 vs .981, LUNAR .998 vs 1.00, and DIF and AE1SVM 1.00 vs 1.00.
- **Neural time series:** LSTM-AD .86 vs .89, Anomaly Transformer .80 vs .76.
- **Neural graph models:** CoLA .93 vs .95. DOMINANT and CONAD vary between about .6 and .8 with the seed in both libraries.

### Where the port deliberately differs

- **IForest** returns the paper's score `2^(-E[h]/c(ψ))`. PyOD's differs from it by a constant offset.
- **SO-/MO-GAAL** score `1 − D(x)`; PyOD returns `D(x)`. GAAL is sensitive to data scale, and on some datasets the ranking flips either way.
- **AnoGAN** runs the per-sample latent query on new data too. PyOD reuses the query state left by the last training sample, which makes its test scores distances to that one sample.
- **DevNet** uses the deviation loss from the paper. PyOD calls it with its arguments swapped.
- **ALAD** scores with dropout off, so scores are deterministic. PyOD leaves its networks in training mode while scoring.
- **LSCP and VAE** give the same scores when called twice on the same data. PyOD draws new randomness on each call.
- **PCA** floors zero explained-variance weights at machine epsilon, where PyOD would divide by zero.

### Kept as in PyOD, but worth knowing

- **Transductive methods** score a `decision_function` batch against itself, not the training data: COF, SOD, SOS, LOCI, LMDD and Spectral Residual. R-Graph scores it together with the training data. Matrix Profile and the graph models score only the fitted data.
- **KPCA** keeps every component by default, so training scores are rounding noise in both libraries. Set `n_components`.
- **The VAE's default `capacity = 0`** clamps the KL term to zero.
- **GUIDE** rejects triangle-free graphs.

## Not ported

- `so_gaal_new.py`: an unexported, unfinished refactor of SO-GAAL.
- PyThresh's ~35 threshold methods (`thresholds.py` wraps an external library). Plug your own in through `Thresholder`.
- File decoding and resampling for AudioAE: pass mono waveforms already at `sr`.
- Ready-made text and image encoders for the embedding models (sentence-transformers, OpenAI, DINOv2): implement `Encoder<T>`.

## Practical notes

- Neighbour searches compare every pair of points, and the graph models use dense
  adjacency, so these models are best for up to tens of thousands of samples or a few
  thousand nodes.
- burn's random seed is process-wide, so neural training is serialised across
  threads to keep results reproducible.

## Running

```bash
cargo test --release
cargo test --release --features deep
cargo run --release --example compare
```

`examples/score_csv.rs`, `deep_score_csv.rs`, `ts_score_csv.rs` and
`graph_score_csv.rs` print scores for CSV input. They are what the PyOD
cross-checks used.
