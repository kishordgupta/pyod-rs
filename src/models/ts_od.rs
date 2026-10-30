//! Sliding-window wrapper that applies any tabular detector to a time series.
//! Port of `pyod.models.ts_od.TimeSeriesOD`.

use crate::base::{
    check_array, check_contamination, check_features, Detector, DetectorFactory, FitState, Matrix,
    OdError, Result,
};
use crate::models::iforest::IForest;
use crate::ts::{fill_with, map_scores_to_timestamps, sliding_windows, ts_fit_state, Aggregation};

pub struct TimeSeriesOd {
    pub detector: DetectorFactory,
    pub window_size: usize,
    pub step: usize,
    pub score_aggregation: Aggregation,
    pub contamination: f64,
    fitted: Option<Box<dyn Detector>>,
    n_channels: usize,
    state: Option<FitState>,
}

impl Default for TimeSeriesOd {
    fn default() -> Self {
        Self::new(Box::new(|seed| Box::new(IForest::new(100, 0.1, seed))), 50)
    }
}

impl TimeSeriesOd {
    pub fn new(detector: DetectorFactory, window_size: usize) -> Self {
        TimeSeriesOd {
            detector,
            window_size,
            step: 1,
            score_aggregation: Aggregation::Max,
            contamination: 0.1,
            fitted: None,
            n_channels: 0,
            state: None,
        }
    }
}

impl Detector for TimeSeriesOd {
    fn fit(&mut self, x: &Matrix) -> Result<()> {
        check_contamination(self.contamination)?;
        self.n_channels = check_array(x)?;
        if self.window_size == 0 || self.step == 0 || x.len() < self.window_size {
            return Err(OdError::InvalidParameter(format!(
                "series of length {} is shorter than window_size {}",
                x.len(),
                self.window_size
            )));
        }
        let mut det = (self.detector)(0);
        det.fit(&sliding_windows(x, self.window_size, self.step))?;
        let scores = map_scores_to_timestamps(
            det.decision_scores()?,
            self.window_size,
            self.step,
            x.len(),
            self.score_aggregation,
        );
        self.fitted = Some(det);
        self.state = Some(ts_fit_state(&scores, self.contamination));
        Ok(())
    }

    fn decision_function(&self, x: &Matrix) -> Result<Vec<f64>> {
        let state = self.fitted()?;
        check_features(x, self.n_channels)?;
        if x.len() < self.window_size {
            return Err(OdError::InvalidParameter(
                "series is shorter than window_size".into(),
            ));
        }
        let ws = self
            .fitted
            .as_ref()
            .unwrap()
            .decision_function(&sliding_windows(x, self.window_size, self.step))?;
        let scores = map_scores_to_timestamps(
            &ws,
            self.window_size,
            self.step,
            x.len(),
            self.score_aggregation,
        );
        Ok(fill_with(scores, state.threshold))
    }

    crate::state_accessor!();
}
