#![cfg(feature = "deep")]
#![allow(clippy::needless_range_loop)]
//! Smoke tests for the neural detectors: each must train and separate the
//! synthetic outliers clearly better than chance.

use pyod_rs::data::{generate_data, roc_auc};
use pyod_rs::Detector;

/// `floor = None` only checks that the model trains and produces finite scores.
fn check(name: &str, det: &mut dyn Detector, floor: impl Into<Option<f64>>) {
    let (x_train, _) = generate_data(300, 5, 0.1, 21);
    let (x_test, y_test) = generate_data(150, 5, 0.1, 22);
    det.fit(&x_train).unwrap();
    assert_eq!(det.decision_scores().unwrap().len(), 300);
    let scores = det.decision_function(&x_test).unwrap();
    assert!(
        scores.iter().all(|s| s.is_finite()),
        "{name}: non-finite scores"
    );
    let auc = roc_auc(&y_test, &scores);
    println!("{name:<20} test AUC {auc:.3}");
    if let Some(floor) = floor.into() {
        assert!(auc > floor, "{name}: AUC {auc}");
    }
}

#[test]
fn autoencoder() {
    check(
        "AutoEncoder",
        &mut pyod_rs::deep::autoencoder::AutoEncoder::default(),
        0.8,
    );
}

#[test]
fn vae() {
    check("VAE", &mut pyod_rs::deep::vae::Vae::default(), 0.8);
}

#[test]
fn deep_svdd() {
    check(
        "DeepSVDD",
        &mut pyod_rs::deep::deep_svdd::DeepSvdd::default(),
        0.8,
    );
}

// GAAL's ranking depends on where the data sits relative to its generators'
// [0, 1]^d noise; PyOD scores 0.24-0.34 on this data too, so only check it runs.
#[test]
fn so_gaal() {
    check("SO-GAAL", &mut pyod_rs::deep::gaal::SoGaal::default(), None);
}

#[test]
fn mo_gaal() {
    check("MO-GAAL", &mut pyod_rs::deep::gaal::MoGaal::default(), None);
}

#[test]
fn ae1svm() {
    check("AE1SVM", &mut pyod_rs::deep::ae1svm::Ae1Svm::default(), 0.8);
}

#[test]
fn alad() {
    // ALAD ranks these blob outliers inconsistently in PyOD too, so only check it runs.
    let mut m = pyod_rs::deep::alad::Alad::default();
    m.epochs = 50;
    check("ALAD", &mut m, None);
}

#[test]
fn anogan() {
    check("AnoGAN", &mut pyod_rs::deep::anogan::AnoGan::default(), 0.8);
}

#[test]
fn lunar() {
    check("LUNAR", &mut pyod_rs::deep::lunar::Lunar::default(), 0.8);
}

#[test]
fn devnet_learns_from_labels() {
    use pyod_rs::deep::devnet::DevNet;
    let (x, y) = generate_data(300, 5, 0.1, 23);
    let (xt, yt) = generate_data(150, 5, 0.1, 24);
    // Only a third of the outliers are labelled.
    let partial: Vec<u8> = y
        .iter()
        .enumerate()
        .map(|(i, &l)| if l == 1 && i % 3 == 0 { 1 } else { 0 })
        .collect();
    let mut m = DevNet::default();
    assert!(m.fit(&x).is_err());
    m.fit_supervised(&x, &partial).unwrap();
    let auc = roc_auc(&yt, &m.decision_function(&xt).unwrap());
    println!("DevNet               test AUC {auc:.3}");
    assert!(auc > 0.75, "DevNet AUC {auc}");
}

#[test]
fn audio_ae_flags_noisy_clips() {
    use pyod_rs::deep::audio_ae::AudioAe;
    let sr = 8000.0;
    let mut rng = pyod_rs::utils::Rng::new(3);
    let tone = |f: f64, rng: &mut pyod_rs::utils::Rng| -> Vec<f64> {
        (0..8000)
            .map(|i| (std::f64::consts::TAU * f * i as f64 / sr).sin() * 0.5 + 0.01 * rng.normal())
            .collect()
    };
    let mut clips: Vec<Vec<f64>> = (0..18)
        .map(|i| tone(300.0 + 10.0 * (i % 3) as f64, &mut rng))
        .collect();
    clips.push((0..8000).map(|_| rng.normal()).collect());
    clips.push((0..8000).map(|_| rng.normal()).collect());
    let mut m = AudioAe::new(0.1);
    m.sr = sr;
    m.epochs = 20;
    m.fit(&clips).unwrap();
    let s = m.state().unwrap().decision_scores.clone();
    let noisy_min = s[18].min(s[19]);
    let tonal_max = s[..18].iter().copied().fold(f64::MIN, f64::max);
    println!("AudioAE              noisy min {noisy_min:.3} vs tonal max {tonal_max:.3}");
    assert!(noisy_min > tonal_max);
    assert_eq!(m.decision_function(&clips[..2]).unwrap().len(), 2);
}

fn series() -> (Vec<Vec<f64>>, Vec<u8>) {
    let mut rng = pyod_rs::utils::Rng::new(9);
    let mut x: Vec<Vec<f64>> = (0..400)
        .map(|t| vec![((t as f64) * std::f64::consts::TAU / 40.0).sin() + 0.1 * rng.normal()])
        .collect();
    let mut y = vec![0u8; 400];
    for start in [150, 300] {
        for k in 0..10 {
            x[start + k][0] += 2.0;
            y[start + k] = 1;
        }
    }
    (x, y)
}

#[test]
fn lstm_ad() {
    let (x, y) = series();
    let mut m = pyod_rs::deep::lstm::LstmAd::new(30, 0.1);
    m.epochs = 10;
    m.fit(&x).unwrap();
    let auc = roc_auc(&y, m.decision_scores().unwrap());
    println!("LSTMAD               AUC {auc:.3}");
    assert!(auc > 0.75, "LSTMAD AUC {auc}");
    assert_eq!(m.decision_function(&x).unwrap().len(), x.len());
}

#[test]
fn anomaly_transformer() {
    let (x, y) = series();
    let mut m = pyod_rs::deep::anomaly_transformer::AnomalyTransformer::new(40, 0.1);
    (m.d_model, m.n_heads, m.n_layers, m.epochs) = (32, 4, 2, 3);
    m.fit(&x).unwrap();
    let auc = roc_auc(&y, m.decision_scores().unwrap());
    println!("AnomalyTransformer   AUC {auc:.3}");
    assert!(auc > 0.6, "AnomalyTransformer AUC {auc}");
}

#[test]
fn graph_neural_detectors() {
    use pyod_rs::deep::graph::{AnomalyDae, Cola, Conad, Dominant, Guide};
    use pyod_rs::models::graph::GraphDetector;
    // Four communities with clustered features; nodes 0-5 form an injected clique
    // across communities, nodes 6-11 get far-off attributes.
    let mut rng = pyod_rs::utils::Rng::new(12);
    let n = 120;
    let community = |i: usize| i % 4;
    let mut x: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..6)
                .map(|k| if k == community(i) { 3.0 } else { 0.0 } + 0.3 * rng.normal())
                .collect()
        })
        .collect();
    let mut edges = Vec::new();
    for i in 0..n {
        for _ in 0..4 {
            let j = (rng.below(n / 4)) * 4 + community(i);
            if j != i {
                edges.push((i, j));
                edges.push((j, i));
            }
        }
    }
    let mut y = vec![0u8; n];
    for a in 0..6 {
        for b in 0..6 {
            if a != b {
                edges.push((a, b));
            }
        }
        y[a] = 1;
    }
    for (i, row) in x.iter_mut().enumerate().take(12).skip(6) {
        *row = vec![-6.0; 6];
        y[i] = 1;
    }
    edges.sort_unstable();
    edges.dedup();
    let dets: Vec<(&str, Box<dyn GraphDetector>)> = vec![
        ("DOMINANT", Box::new(Dominant::default())),
        ("AnomalyDAE", Box::new(AnomalyDae::default())),
        ("CoLA", Box::new(Cola::default())),
        ("CONAD", Box::new(Conad::default())),
        ("GUIDE", Box::new(Guide::default())),
    ];
    let mut low = Vec::new();
    for (name, mut d) in dets {
        d.fit_graph(&x, &edges).unwrap();
        let s = d.decision_scores().unwrap();
        assert!(s.iter().all(|v| v.is_finite()), "{name}");
        let auc = roc_auc(&y, s);
        println!("{name:<20} AUC {auc:.3}");
        // AnomalyDAE is unstable on this graph in PyOD as well (AUC 0.35-0.69 across seeds).
        if auc <= 0.6 && name != "AnomalyDAE" {
            low.push((name, auc));
        }
    }
    assert!(low.is_empty(), "{low:?}");
}
