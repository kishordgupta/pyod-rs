//! Spectral Residual saliency for time series (Ren et al., KDD 2019).
//! Port of `pyod.models.ts_spectral_residual`.

use crate::base::{check_array, check_contamination, Detector, FitState, Matrix, OdError, Result};
use crate::ts::{aggregate_channel_scores, dft, Aggregation};

#[derive(Debug, Clone)]
pub struct SpectralResidual {
    /// Width of the moving average applied to the log amplitude spectrum.
    pub score_window: usize,
    pub channel_aggregation: Aggregation,
    pub contamination: f64,
    state: Option<FitState>,
}

impl Default for SpectralResidual {
    fn default() -> Self {
        SpectralResidual {
            score_window: 3,
            channel_aggregation: Aggregation::Max,
            contamination: 0.1,
            state: None,
        }
    }
}

/// Saliency map of one channel.
fn saliency(x: &[f64], q: usize) -> Vec<f64> {
    let n = x.len();
    let f = dft(&x.iter().map(|&v| (v, 0.0)).collect::<Vec<_>>(), false);
    let amp: Vec<f64> = f.iter().map(|c| (c.0.hypot(c.1) + 1e-10).ln()).collect();
    let phase: Vec<f64> = f.iter().map(|c| c.1.atan2(c.0)).collect();
    // np.convolve(amp, ones(q) / q, mode='same')
    let offset = (q - 1) / 2;
    let smooth: Vec<f64> = (0..n)
        .map(|i| {
            let k = i + offset;
            (0..q)
                .filter(|&j| k >= j && k - j < n)
                .map(|j| amp[k - j])
                .sum::<f64>()
                / q as f64
        })
        .collect();
    let s: Vec<(f64, f64)> = (0..n)
        .map(|i| {
            let mag = (amp[i] - smooth[i]).exp();
            (mag * phase[i].cos(), mag * phase[i].sin())
        })
        .collect();
    dft(&s, true).iter().map(|c| c.0.hypot(c.1)).collect()
}

impl SpectralResidual {
    fn scores(&self, x: &Matrix) -> Vec<f64> {
        let d = x[0].len();
        let per: Vec<Vec<f64>> = (0..d)
            .map(|c| {
                saliency(
                    &x.iter().map(|r| r[c]).collect::<Vec<_>>(),
                    self.score_window,
                )
            })
            .collect();
        if d == 1 {
            per.into_iter().next().unwrap()
        } else {
            aggregate_channel_scores(&per, self.channel_aggregation)
        }
    }
}

impl Detector for SpectralResidual {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        check_array(x)?;
        if self.score_window == 0 || x.len() < self.score_window.max(2) {
            return Err(OdError::InvalidParameter(
                "series is shorter than score_window".into(),
            ));
        }
        self.state = Some(FitState::from_contamination(
            self.scores(x),
            self.contamination,
        ));
        Ok(())
    }

    /// Scores the given series on its own spectrum (the method has no trained state).
    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.fitted()?;
        check_array(x)?;
        Ok(self.scores(x))
    }

    crate::state_accessor!();
}
