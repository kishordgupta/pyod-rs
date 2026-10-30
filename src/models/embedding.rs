//! Embedding-based outlier detection for unstructured data. Port of
//! `pyod.models.embedding` (`EmbeddingOD` and `MultiModalOD`).
//!
//! An [`Encoder`] turns raw items (text, images, audio, …) into vectors, which
//! are optionally standardised and reduced with PCA before a regular detector
//! scores them. PyOD ships encoders backed by sentence-transformers, HuggingFace
//! and the OpenAI API; here you implement [`Encoder`] for whatever embedding
//! source you use.

use crate::base::{Detector, FitState, Matrix, OdError, Result};
use crate::linalg::{col_mean, covariance};
use crate::models::combination::{average, maximization};
use crate::models::feature_bagging::Combination;
use crate::models::knn::Knn;
use crate::utils::{symmetric_eigen, StandardScaler};

/// Turns raw items into fixed-length embedding vectors.
pub trait Encoder<T> {
    fn encode(&self, items: &[T]) -> Result<Vec<Vec<f64>>>;
}

impl<T, F: Fn(&[T]) -> Result<Vec<Vec<f64>>>> Encoder<T> for F {
    fn encode(&self, items: &[T]) -> Result<Vec<Vec<f64>>> {
        self(items)
    }
}

/// Mean-centred projection onto the top principal axes.
#[derive(Debug, Clone)]
struct Projection {
    mean: Vec<f64>,
    axes: Vec<Vec<f64>>,
}

impl Projection {
    fn fit(x: &Matrix, k: usize) -> Self {
        let mean = col_mean(x);
        let (_, vecs) = symmetric_eigen(&covariance(x, &mean, 1));
        Projection {
            mean,
            axes: vecs.into_iter().take(k).collect(),
        }
    }

    fn transform(&self, x: &Matrix) -> Vec<Vec<f64>> {
        x.iter()
            .map(|r| {
                let c: Vec<f64> = r.iter().zip(&self.mean).map(|(a, b)| a - b).collect();
                self.axes
                    .iter()
                    .map(|a| a.iter().zip(&c).map(|(p, q)| p * q).sum())
                    .collect()
            })
            .collect()
    }
}

/// Encoder + preprocessing + detector over raw items of type `T`.
pub struct EmbeddingOd<T> {
    pub encoder: Box<dyn Encoder<T>>,
    pub detector: Box<dyn Detector>,
    pub standardize: bool,
    /// Reduce embeddings to this many dimensions with PCA.
    pub reduce_dim: Option<usize>,
    scaler: Option<StandardScaler>,
    pca: Option<Projection>,
}

impl<T> EmbeddingOd<T> {
    /// Uses KNN as the detector (PyOD's "fast" preset).
    pub fn new(encoder: Box<dyn Encoder<T>>) -> Self {
        Self::with_detector(encoder, Box::new(Knn::default()))
    }

    pub fn with_detector(encoder: Box<dyn Encoder<T>>, detector: Box<dyn Detector>) -> Self {
        EmbeddingOd {
            encoder,
            detector,
            standardize: true,
            reduce_dim: None,
            scaler: None,
            pca: None,
        }
    }

    fn preprocess(&self, emb: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let emb = match &self.scaler {
            Some(s) => s.transform(&emb),
            None => emb,
        };
        match &self.pca {
            Some(p) => p.transform(&emb),
            None => emb,
        }
    }

    pub fn fit(&mut self, items: &[T]) -> Result<()> {
        let emb = self.encoder.encode(items)?;
        self.scaler = self.standardize.then(|| StandardScaler::fit(&emb));
        self.pca = None;
        let scaled = self.preprocess(emb);
        if let Some(k) = self.reduce_dim {
            if k == 0 || k > scaled[0].len() {
                return Err(OdError::InvalidParameter(
                    "reduce_dim must be in [1, embedding dimension]".into(),
                ));
            }
            self.pca = Some(Projection::fit(&scaled, k));
        }
        let x = match &self.pca {
            Some(p) => p.transform(&scaled),
            None => scaled,
        };
        self.detector.fit(&x)
    }

    pub fn decision_function(&self, items: &[T]) -> Result<Vec<f64>> {
        self.detector
            .decision_function(&self.preprocess(self.encoder.encode(items)?))
    }

    pub fn predict(&self, items: &[T]) -> Result<Vec<u8>> {
        self.detector
            .predict(&self.preprocess(self.encoder.encode(items)?))
    }

    pub fn decision_scores(&self) -> Result<&[f64]> {
        self.detector.decision_scores()
    }

    pub fn labels(&self) -> Result<&[u8]> {
        self.detector.labels()
    }
}

/// Several embedding detectors over different views of the same records,
/// with standardised scores combined into one (`MultiModalOD`).
pub struct MultiModalOd<T> {
    pub modalities: Vec<(String, EmbeddingOd<T>)>,
    pub combination: Combination,
    pub standardize_scores: bool,
    pub contamination: f64,
    scalers: Vec<Option<StandardScaler>>,
    state: Option<FitState>,
}

impl<T> MultiModalOd<T> {
    pub fn new(modalities: Vec<(String, EmbeddingOd<T>)>, contamination: f64) -> Self {
        MultiModalOd {
            modalities,
            combination: Combination::Average,
            standardize_scores: true,
            contamination,
            scalers: Vec::new(),
            state: None,
        }
    }

    fn combine(&self, cols: Vec<Vec<f64>>) -> Vec<f64> {
        let cols: Vec<Vec<f64>> = cols
            .into_iter()
            .zip(&self.scalers)
            .map(|(c, s)| match s {
                Some(s) => s
                    .transform(&c.iter().map(|v| vec![*v]).collect::<Vec<_>>())
                    .into_iter()
                    .map(|r| r[0])
                    .collect(),
                None => c,
            })
            .collect();
        let per_sample: Vec<Vec<f64>> = (0..cols[0].len())
            .map(|i| cols.iter().map(|c| c[i]).collect())
            .collect();
        match self.combination {
            Combination::Average => average(&per_sample, None),
            Combination::Max => maximization(&per_sample),
        }
    }

    pub fn fit(&mut self, items: &[T]) -> Result<()> {
        if self.modalities.is_empty() {
            return Err(OdError::InvalidParameter(
                "need at least one modality".into(),
            ));
        }
        let mut cols = Vec::new();
        for (_, m) in &mut self.modalities {
            m.fit(items)?;
            cols.push(m.decision_scores()?.to_vec());
        }
        self.scalers = cols
            .iter()
            .map(|c| {
                self.standardize_scores
                    .then(|| StandardScaler::fit(&c.iter().map(|v| vec![*v]).collect::<Vec<_>>()))
            })
            .collect();
        let scores = self.combine(cols);
        self.state = Some(FitState::from_contamination(scores, self.contamination));
        Ok(())
    }

    pub fn decision_function(&self, items: &[T]) -> Result<Vec<f64>> {
        self.state.as_ref().ok_or(OdError::NotFitted)?;
        let cols = self
            .modalities
            .iter()
            .map(|(_, m)| m.decision_function(items))
            .collect::<Result<Vec<_>>>()?;
        Ok(self.combine(cols))
    }

    pub fn predict(&self, items: &[T]) -> Result<Vec<u8>> {
        let threshold = self.state.as_ref().ok_or(OdError::NotFitted)?.threshold;
        Ok(self
            .decision_function(items)?
            .into_iter()
            .map(|s| (s > threshold) as u8)
            .collect())
    }

    pub fn state(&self) -> Option<&FitState> {
        self.state.as_ref()
    }
}
