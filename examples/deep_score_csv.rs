//! Print each neural detector's test scores for CSV train/test files, one line per model.
//! Used to compare against PyOD: `cargo run --release --features deep --example deep_score_csv train.csv test.csv [names]`.

use pyod_rs::deep::{
    ae1svm::Ae1Svm,
    alad::Alad,
    anogan::AnoGan,
    autoencoder::AutoEncoder,
    deep_svdd::DeepSvdd,
    gaal::{MoGaal, SoGaal},
    lunar::Lunar,
    vae::Vae,
};
use pyod_rs::*;
use std::{env, fs};

fn read_csv(path: &str) -> Vec<Vec<f64>> {
    fs::read_to_string(path)
        .expect("readable csv")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split(',')
                .map(|v| v.trim().parse().expect("number"))
                .collect()
        })
        .collect()
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let (train, test) = (read_csv(&args[1]), read_csv(&args[2]));
    let only: Option<Vec<String>> = args
        .get(3)
        .map(|s| s.split(',').map(str::to_string).collect());
    let detectors: Vec<(&str, Box<dyn Detector>)> = vec![
        ("AutoEncoder", Box::new(AutoEncoder::default())),
        ("VAE", Box::new(Vae::default())),
        ("DeepSVDD", Box::new(DeepSvdd::default())),
        ("SO_GAAL", Box::new(SoGaal::default())),
        ("MO_GAAL", Box::new(MoGaal::default())),
        ("ALAD", Box::new(Alad::default())),
        ("AnoGAN", Box::new(AnoGan::default())),
        ("AE1SVM", Box::new(Ae1Svm::default())),
        ("LUNAR", Box::new(Lunar::default())),
        ("DIF", Box::new(models::dif::Dif::default())),
    ];
    for (name, mut det) in detectors {
        if only.as_ref().is_some_and(|o| !o.iter().any(|n| n == name)) {
            continue;
        }
        det.fit(&train)?;
        let s: Vec<String> = det
            .decision_function(&test)?
            .iter()
            .map(|v| format!("{v:.9e}"))
            .collect();
        println!("{name}|{}", s.join(","));
    }
    Ok(())
}
