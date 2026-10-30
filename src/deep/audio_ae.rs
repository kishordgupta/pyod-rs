//! AudioAE: AutoEncoder on log-mel spectrogram windows of audio clips.
//! Port of `pyod.models.audio_ae`; a clip's score is the mean reconstruction
//! error of its windows.
//!
//! Clips are mono waveforms already at `sr` (PyOD also accepts file paths and
//! resamples with librosa; decode and resample before calling here).

use super::autoencoder::AutoEncoder;
use crate::audio::logmel_windows;
use crate::base::{Detector, FitState, OdError, Result};

#[derive(Debug, Clone)]
pub struct AudioAe {
    pub n_mels: usize,
    /// Spectrogram frames per window.
    pub context: usize,
    pub hop_length: usize,
    pub sr: f64,
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    /// `None` uses [128, 32, 8] (layers not narrower than the input are dropped).
    pub hidden_neurons: Option<Vec<usize>>,
    pub random_state: u64,
    pub contamination: f64,
    ae: Option<AutoEncoder>,
    state: Option<FitState>,
}

impl Default for AudioAe {
    fn default() -> Self {
        AudioAe {
            n_mels: 64,
            context: 5,
            hop_length: 512,
            sr: 22050.0,
            epochs: 40,
            batch_size: 1024,
            lr: 1e-3,
            hidden_neurons: None,
            random_state: 42,
            contamination: 0.1,
            ae: None,
            state: None,
        }
    }
}

impl AudioAe {
    pub fn new(contamination: f64) -> Self {
        AudioAe {
            contamination,
            ..Self::default()
        }
    }

    /// All windows of all clips, and the clip each window came from.
    fn extract(&self, clips: &[Vec<f64>]) -> Result<(Vec<Vec<f64>>, Vec<usize>)> {
        if clips.is_empty() || clips.iter().any(Vec::is_empty) {
            return Err(OdError::EmptyInput);
        }
        let (mut frames, mut owner) = (Vec::new(), Vec::new());
        for (i, c) in clips.iter().enumerate() {
            let w = logmel_windows(c, self.sr, self.n_mels, self.context, self.hop_length);
            owner.extend(std::iter::repeat_n(i, w.len()));
            frames.extend(w);
        }
        Ok((frames, owner))
    }

    fn aggregate(scores: &[f64], owner: &[usize], n: usize) -> Vec<f64> {
        let (mut sum, mut cnt) = (vec![0.0; n], vec![0usize; n]);
        for (s, &o) in scores.iter().zip(owner) {
            sum[o] += s;
            cnt[o] += 1;
        }
        sum.iter()
            .zip(&cnt)
            .map(|(s, &c)| if c > 0 { s / c as f64 } else { 0.0 })
            .collect()
    }

    pub fn fit(&mut self, clips: &[Vec<f64>]) -> Result<()> {
        let (frames, owner) = self.extract(clips)?;
        let dim = frames[0].len();
        let hidden: Vec<usize> = self
            .hidden_neurons
            .clone()
            .unwrap_or(vec![128, 32, 8])
            .into_iter()
            .filter(|&h| h < dim)
            .collect();
        let mut ae = AutoEncoder::new(self.contamination);
        ae.hidden_neurons = if hidden.is_empty() {
            vec![(dim / 2).max(2)]
        } else {
            hidden
        };
        ae.train.epochs = self.epochs;
        ae.train.batch_size = self.batch_size.min(frames.len()).max(1);
        ae.train.lr = self.lr;
        ae.train.random_state = self.random_state;
        ae.fit(&frames)?;
        let scores = Self::aggregate(ae.decision_scores()?, &owner, clips.len());
        self.ae = Some(ae);
        self.state = Some(FitState::from_contamination(scores, self.contamination));
        Ok(())
    }

    pub fn decision_function(&self, clips: &[Vec<f64>]) -> Result<Vec<f64>> {
        let ae = self.ae.as_ref().ok_or(OdError::NotFitted)?;
        let (frames, owner) = self.extract(clips)?;
        Ok(Self::aggregate(
            &ae.decision_function(&frames)?,
            &owner,
            clips.len(),
        ))
    }

    pub fn predict(&self, clips: &[Vec<f64>]) -> Result<Vec<u8>> {
        let threshold = self.state.as_ref().ok_or(OdError::NotFitted)?.threshold;
        Ok(self
            .decision_function(clips)?
            .into_iter()
            .map(|s| (s > threshold) as u8)
            .collect())
    }

    pub fn state(&self) -> Option<&FitState> {
        self.state.as_ref()
    }
}
