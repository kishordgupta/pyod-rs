//! Print each time-series detector's training scores for a CSV series (rows = timestamps).
//! Used to cross-check against PyOD: `cargo run --example ts_score_csv series.csv`.

use pyod_rs::models::{
    ts_kshape::KShape, ts_matrix_profile::MatrixProfile, ts_od::TimeSeriesOd, ts_sand::Sand,
    ts_spectral_residual::SpectralResidual,
};
use pyod_rs::*;
use std::{env, fs};

fn main() -> Result<()> {
    let path = env::args().nth(1).expect("csv path");
    let x: Vec<Vec<f64>> = fs::read_to_string(path)
        .expect("readable csv")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split(',')
                .map(|v| v.trim().parse().expect("number"))
                .collect()
        })
        .collect();
    let detectors: Vec<(&str, Box<dyn Detector>)> = vec![
        ("SpectralResidual", Box::new(SpectralResidual::default())),
        ("MatrixProfile", Box::new(MatrixProfile::new(30, 0.1))),
        (
            "TimeSeriesOD_KNN",
            Box::new(TimeSeriesOd::new(
                Box::new(|_| Box::new(Knn::default())),
                20,
            )),
        ),
        ("KShape", Box::new(KShape::new(30, 0.1))),
        ("SAND", Box::new(Sand::new(30, 0.1))),
    ];
    for (name, mut det) in detectors {
        det.fit(&x)?;
        let s: Vec<String> = det
            .decision_scores()?
            .iter()
            .map(|v| format!("{v:.12e}"))
            .collect();
        println!("{name}|{}", s.join(","));
    }
    Ok(())
}
