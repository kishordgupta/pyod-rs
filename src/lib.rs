//! # pyod-rs
//!
//! A Rust port of [PyOD](https://github.com/yzhao062/pyod). Every tabular
//! detector implements [`Detector`], mirroring PyOD's `BaseDetector`: `fit`,
//! `decision_function`, `predict`, `predict_proba`, `predict_confidence`, plus
//! the fitted `decision_scores`, `threshold` and `labels`.
//!
//! - [`models`]: classical, ensemble, time-series (`ts_*`) and non-neural graph
//!   detectors, with no dependencies.
//! - `deep` (with the `deep` feature): neural detectors built on burn.
//!
//! ```
//! use pyod_rs::{Detector, models::ecod::Ecod};
//!
//! let mut x: Vec<Vec<f64>> = (0..50).map(|i| vec![(i % 7) as f64, (i % 5) as f64]).collect();
//! x.push(vec![40.0, -30.0]);
//! let mut det = Ecod::new(0.05);
//! det.fit(&x).unwrap();
//! assert_eq!(det.labels().unwrap()[50], 1);
//! ```

// Index loops mirror the reference NumPy code and read better in numeric kernels.
#![allow(clippy::needless_range_loop)]

pub mod audio;
pub mod base;
pub mod data;
#[cfg(feature = "deep")]
pub mod deep;
pub mod linalg;
pub mod models;
pub mod trees;
pub mod ts;
pub mod utils;

pub use base::{
    apply_thresholder, Detector, DetectorFactory, FitState, Matrix, OdError, ProbaMethod, Result,
    Thresholder,
};
pub use models::{
    abod::Abod,
    copod::Copod,
    ecod::Ecod,
    gmm::Gmm,
    hbos::Hbos,
    iforest::IForest,
    kde::Kde,
    knn::{Knn, KnnMethod},
    lof::Lof,
    mad::Mad,
    mcd::Mcd,
    ocsvm::OcSvm,
    pca::Pca,
};
