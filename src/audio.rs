//! Log-mel spectrogram features, matching librosa's defaults as used by
//! PyOD's `AudioAE`: `melspectrogram` (n_fft 2048, periodic Hann window,
//! centred zero padding, power 2, Slaney mel filters) followed by
//! `power_to_db` (ref 1, amin 1e-10, top_db 80).

use crate::ts::dft;

fn hz_to_mel(f: f64) -> f64 {
    let (f_sp, min_log_hz, logstep) = (200.0 / 3.0, 1000.0, 6.4f64.ln() / 27.0);
    if f >= min_log_hz {
        min_log_hz / f_sp + (f / min_log_hz).ln() / logstep
    } else {
        f / f_sp
    }
}

fn mel_to_hz(m: f64) -> f64 {
    let (f_sp, min_log_hz, logstep) = (200.0 / 3.0, 1000.0, 6.4f64.ln() / 27.0);
    let min_log_mel = min_log_hz / f_sp;
    if m >= min_log_mel {
        min_log_hz * (logstep * (m - min_log_mel)).exp()
    } else {
        f_sp * m
    }
}

/// Slaney-normalised mel filterbank, `n_mels x (1 + n_fft / 2)` (librosa.filters.mel).
pub fn mel_filters(sr: f64, n_fft: usize, n_mels: usize) -> Vec<Vec<f64>> {
    let n_bins = 1 + n_fft / 2;
    let fft_freqs: Vec<f64> = (0..n_bins)
        .map(|i| i as f64 * (sr / 2.0) / (n_bins - 1) as f64)
        .collect();
    let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(sr / 2.0));
    let mel_f: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (n_mels + 1) as f64))
        .collect();
    (0..n_mels)
        .map(|i| {
            let enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
            fft_freqs
                .iter()
                .map(|&f| {
                    let lower = (f - mel_f[i]) / (mel_f[i + 1] - mel_f[i]);
                    let upper = (mel_f[i + 2] - f) / (mel_f[i + 2] - mel_f[i + 1]);
                    lower.min(upper).max(0.0) * enorm
                })
                .collect()
        })
        .collect()
}

/// Log-mel spectrogram in dB, `n_mels x n_frames`.
pub fn log_mel_spectrogram(y: &[f64], sr: f64, n_mels: usize, hop_length: usize) -> Vec<Vec<f64>> {
    let n_fft = 2048;
    let pad = n_fft / 2;
    let mut padded = vec![0.0; pad];
    padded.extend_from_slice(y);
    padded.extend(std::iter::repeat_n(0.0, pad));
    let window: Vec<f64> = (0..n_fft)
        .map(|n| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / n_fft as f64).cos())
        .collect();
    let n_frames = 1 + (padded.len() - n_fft) / hop_length;
    let filters = mel_filters(sr, n_fft, n_mels);
    let mut spec = vec![vec![0.0; n_frames]; n_mels];
    for t in 0..n_frames {
        let frame: Vec<(f64, f64)> = (0..n_fft)
            .map(|k| (padded[t * hop_length + k] * window[k], 0.0))
            .collect();
        let power: Vec<f64> = dft(&frame, false)[..=n_fft / 2]
            .iter()
            .map(|c| c.0 * c.0 + c.1 * c.1)
            .collect();
        for (m, filt) in filters.iter().enumerate() {
            spec[m][t] = filt.iter().zip(&power).map(|(a, b)| a * b).sum();
        }
    }
    let mut max_db = f64::NEG_INFINITY;
    for row in spec.iter_mut() {
        for v in row.iter_mut() {
            *v = 10.0 * v.max(1e-10).log10();
            max_db = max_db.max(*v);
        }
    }
    spec.iter_mut()
        .flatten()
        .for_each(|v| *v = v.max(max_db - 80.0));
    spec
}

/// Overlapping context windows of `context` frames, each flattened frame-major
/// (`n_mels` values per frame). Short spectrograms are zero-padded to one window.
pub fn logmel_windows(
    y: &[f64],
    sr: f64,
    n_mels: usize,
    context: usize,
    hop_length: usize,
) -> Vec<Vec<f64>> {
    let mut spec = log_mel_spectrogram(y, sr, n_mels, hop_length);
    let n_frames = spec[0].len();
    if n_frames < context {
        spec.iter_mut().for_each(|r| r.resize(context, 0.0));
    }
    let n_frames = n_frames.max(context);
    (0..=n_frames - context)
        .map(|t| {
            (t..t + context)
                .flat_map(|f| spec.iter().map(move |row| row[f]))
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_scale_round_trips() {
        for f in [0.0, 440.0, 1000.0, 8000.0] {
            assert!((mel_to_hz(hz_to_mel(f)) - f).abs() < 1e-9);
        }
    }

    #[test]
    fn tone_energy_lands_in_matching_band() {
        let sr = 22050.0;
        let y: Vec<f64> = (0..22050)
            .map(|i| (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / sr).sin())
            .collect();
        let spec = log_mel_spectrogram(&y, sr, 64, 512);
        let mid = spec[0].len() / 2;
        let peak = (0..64)
            .max_by(|&a, &b| spec[a][mid].total_cmp(&spec[b][mid]))
            .unwrap();
        let filters = mel_filters(sr, 2048, 64);
        let bin = (1000.0 / (sr / 2048.0)).round() as usize;
        let best = (0..64)
            .max_by(|&a, &b| filters[a][bin].total_cmp(&filters[b][bin]))
            .unwrap();
        assert!(
            (peak as i64 - best as i64).abs() <= 1,
            "peak {peak} vs filter {best}"
        );
    }
}
