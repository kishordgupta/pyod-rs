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
- **DevNet** uses the deviation loss from the paper; PyOD calls it with its arguments swapped. When labelled outliers are present, each training batch alternates inliers and labelled outliers, as in the paper (`nb_batch` batches per epoch). Plain shuffled batches collapsed on some seeds.
- **ALAD** scores with dropout off, so scores are deterministic. PyOD leaves its networks in training mode while scoring.
- **LSCP and VAE** give the same scores when called twice on the same data. PyOD draws new randomness on each call.
- **PCA** floors zero explained-variance weights at machine epsilon, where PyOD would divide by zero.

### Kept as in PyOD, but worth knowing

- **Transductive methods** score a `decision_function` batch against itself, not the training data: COF, SOD, SOS, LOCI, LMDD and Spectral Residual. R-Graph scores it together with the training data. Matrix Profile and the graph models score only the fitted data.
- **KPCA** keeps every component by default, so training scores are rounding noise in both libraries. Set `n_components`.
- **The VAE's default `capacity = 0`** clamps the KL term to zero.
- **GUIDE** rejects triangle-free graphs.

## Speed

PyOD 3.6.7 vs the C port ([pyod-c](https://github.com/kishordgupta/pyod-c)) and the Rust port
([pyod-rs](https://github.com/kishordgupta/pyod-rs)) on an Apple M3 Max, 31 Oct 2026.
Times are `fit` + `decision_function` with default settings, median of 3 runs.
Data comes from PyOD's `generate_data`: 10 features, 10% contamination.

Speed-up over PyOD (geometric mean; above 1 is faster):

| Set | pyod-c | pyod-rs | Fastest (C / Rust / PyOD) |
|---|---:|---:|---|
| 22 classical detectors, 2,000 training rows | 2.8× | 2.9× | 11 / 11 / 0 |
| 5 neural detectors, 2,000 training rows | 1.2× | 1.4× | 2 / 1 / 2 |
| 10 classical detectors, 20,000 training rows | 1.3× | 1.5× | 4 / 5 / 1 |

- **Small data:** Python's fixed cost per call dominates. IForest and KDE run about 7× faster in both ports, and SUOD 3–5×.
- **Large data:** that overhead stops mattering. PyOD's KNN uses sklearn's tree search (1.4 s on 20,000 rows) against brute force in Rust (1.9 s) and C (1.3 s).
- **Rust vs C:** within about 10% of each other. Rust is about 2.5× faster on ECOD and COPOD; C is faster on CD, GMM and DIF.
- **Neural models:** small networks favour the ports (AutoEncoder on 20,000 rows: C 1.0 s, Rust 1.4 s, PyOD 10.9 s). LUNAR's wide full-batch layers favour PyOD's BLAS: PyOD 1.7 s, Rust 7.1 s, C 37 s.

<details>
<summary>All results (fastest in bold)</summary>

**2,000 training rows, 1,000 test rows**

| Model | pyod-c | pyod-rs | PyOD | PyOD, 1 thread |
|---|---:|---:|---:|---:|
| ABOD | **22 ms** | 28 ms | 145 ms | 144 ms |
| CBLOF | **1.6 ms** | 2.2 ms | 6.1 ms | 2.4 ms |
| CD | **2.0 ms** | 11 ms | 11 ms | 11 ms |
| COF | 248 ms | **91 ms** | 486 ms | 482 ms |
| COPOD | 7.4 ms | **3.2 ms** | 6.6 ms | 6.5 ms |
| ECOD | 7.2 ms | **3.1 ms** | 6.2 ms | 6.1 ms |
| FeatureBagging | 343 ms | **270 ms** | 373 ms | 369 ms |
| GMM | **1.0 ms** | 2.0 ms | 11 ms | 1.4 ms |
| HBOS | 0.4 ms | **0.2 ms** | 1.3 ms | 1.3 ms |
| HDBSCAN | **29 ms** | 36 ms | 44 ms | 45 ms |
| IForest | 10 ms | **9.9 ms** | 75 ms | 76 ms |
| INNE | **22 ms** | 26 ms | 133 ms | 131 ms |
| KDE | **32 ms** | 32 ms | 229 ms | 242 ms |
| KNN | **21 ms** | 27 ms | 38 ms | 38 ms |
| LODA | 5.5 ms | **3.9 ms** | 9.3 ms | 9.4 ms |
| LOF | 36 ms | **28 ms** | 48 ms | 48 ms |
| MCD | **114 ms** | 142 ms | 236 ms | 235 ms |
| OCSVM | 45 ms | **43 ms** | 120 ms | 118 ms |
| PCA | **0.3 ms** | 0.4 ms | 0.9 ms | 0.8 ms |
| Sampling | 0.2 ms | **0.2 ms** | 0.5 ms | 0.5 ms |
| SOD | **89 ms** | 117 ms | 328 ms | 1.94 s |
| SUOD | 915 ms | **573 ms** | 2.70 s | 2.74 s |
| AutoEncoder *(neural)* | **100 ms** | 135 ms | 1.11 s | 355 ms |
| DeepSVDD *(neural)* | 582 ms | **319 ms** | 1.50 s | 1.49 s |
| DIF *(neural)* | 2.72 s | 7.93 s | **1.79 s** | 2.09 s |
| LUNAR *(neural)* | 36.6 s | 7.08 s | **1.69 s** | 1.86 s |
| VAE *(neural)* | **646 ms** | 692 ms | 1.97 s | 1.47 s |

**20,000 training rows, 5,000 test rows**

| Model | pyod-c | pyod-rs | PyOD | PyOD, 1 thread |
|---|---:|---:|---:|---:|
| COPOD | 92 ms | **37 ms** | 64 ms | 64 ms |
| ECOD | 92 ms | **37 ms** | 63 ms | 64 ms |
| GMM | **9.9 ms** | 19 ms | 37 ms | 6.5 ms |
| HBOS | 4.1 ms | **1.8 ms** | 6.5 ms | 6.6 ms |
| IForest | 73 ms | **58 ms** | 170 ms | 172 ms |
| KNN | **1.33 s** | 1.92 s | 1.37 s | 1.36 s |
| LODA | 44 ms | **34 ms** | 60 ms | 58 ms |
| LOF | **1.54 s** | 2.30 s | 1.96 s | 1.95 s |
| MCD | **896 ms** | 1.00 s | 1.14 s | 1.14 s |
| PCA | 3.1 ms | 3.9 ms | **2.9 ms** | 2.9 ms |
| AutoEncoder *(neural)* | **997 ms** | 1.35 s | 10.9 s | 3.48 s |

Setup: PyOD 3.6.7, NumPy 2.0.2 (Accelerate BLAS), scikit-learn 1.6.1, PyTorch 2.8.0 (CPU), numba 0.60;
Rust 1.99.1 release build, burn 0.22; Apple clang 21 at `-O2`. Each run uses a fresh detector, and CSV loading is not timed.
</details>

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
