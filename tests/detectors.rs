use pyod_rs::data::{generate_data, roc_auc};
use pyod_rs::models::combination::{aom, average, majority_vote, maximization, moa};
use pyod_rs::utils::StandardScaler;
use pyod_rs::*;

fn all_detectors() -> Vec<(&'static str, Box<dyn Detector>)> {
    vec![
        ("ABOD", Box::new(Abod::new(10, 0.1))),
        ("COPOD", Box::new(Copod::new(0.1))),
        ("ECOD", Box::new(Ecod::new(0.1))),
        ("HBOS", Box::new(Hbos::new(10, 0.1))),
        ("IForest", Box::new(IForest::new(100, 0.1, 7))),
        ("KNN", Box::new(Knn::new(5, KnnMethod::Largest, 0.1))),
        ("KNN-mean", Box::new(Knn::new(5, KnnMethod::Mean, 0.1))),
        ("LOF", Box::new(Lof::new(20, 0.1))),
        ("PCA", Box::new(Pca::new(0.1))),
        ("KDE", Box::new(Kde::new(1.0, 0.1))),
        ("GMM", Box::new(Gmm::new(2, 0.1))),
        ("MCD", Box::new(Mcd::new(0.1))),
        ("OCSVM", Box::new(OcSvm::new(0.5, 0.1))),
        ("CBLOF", Box::new(models::cblof::Cblof::default())),
        ("COF", Box::new(models::cof::Cof::default())),
        ("INNE", Box::new(models::inne::Inne::default())),
        ("LODA", Box::new(models::loda::Loda::default())),
        ("SOD", Box::new(models::sod::Sod::default())),
        ("SOS", Box::new(models::sos::Sos::default())),
        (
            "LMDD",
            Box::new(models::lmdd::Lmdd::new(
                10,
                models::lmdd::Dissimilarity::Aad,
                0.1,
            )),
        ),
        ("LOCI", Box::new(models::loci::Loci::default())),
        ("QMCD", Box::new(models::qmcd::Qmcd::default())),
        ("ROD", Box::new(models::rod::Rod::default())),
        ("KPCA", Box::new(models::kpca::Kpca::default())),
        ("CD", Box::new(models::cd::Cd::default())),
        ("Sampling", Box::new(models::sampling::Sampling::default())),
        ("HDBSCAN", Box::new(models::hdbscan::Hdbscan::default())),
        ("RGraph", Box::new(models::rgraph::RGraph::default())),
        ("DIF", Box::new(models::dif::Dif::default())),
        (
            "FeatureBagging",
            Box::new(models::feature_bagging::FeatureBagging::default()),
        ),
        ("SUOD", Box::new(models::suod::Suod::default())),
        (
            "LSCP",
            Box::new(models::lscp::Lscp::new(
                vec![
                    Box::new(Knn::default()),
                    Box::new(Lof::new(10, 0.1)),
                    Box::new(Hbos::default()),
                ],
                0.1,
            )),
        ),
    ]
}

#[test]
fn every_detector_separates_outliers() {
    let (x_train, y_train) = generate_data(300, 2, 0.1, 1);
    let (x_test, y_test) = generate_data(100, 2, 0.1, 2);
    for (name, mut det) in all_detectors() {
        det.fit(&x_train).unwrap();
        let train_scores = det.decision_scores().unwrap();
        assert_eq!(train_scores.len(), 300, "{name}");
        assert_eq!(det.labels().unwrap().len(), 300, "{name}");

        let test_scores = det.decision_function(&x_test).unwrap();
        let auc_train = roc_auc(&y_train, train_scores);
        let auc_test = roc_auc(&y_test, &test_scores);
        println!("{name:<14} train AUC {auc_train:.3}  test AUC {auc_test:.3}");
        // Methods that assume one Gaussian cluster, or score data transductively
        // in small batches, are weak on this bimodal data, as in PyOD.
        let floor = match name {
            "PCA" | "MCD" | "CD" | "LMDD" => 0.4,
            "SOS" | "SOD" | "LOCI" | "HDBSCAN" => 0.5,
            // R-Graph ranks blob outliers poorly in PyOD as well (AUC 0.25 there on the cross-check data).
            "RGraph" => 0.2,
            _ => 0.8,
        };
        assert!(auc_test > floor, "{name}: test AUC {auc_test}");

        // About `contamination` of the training data is flagged (ties and
        // fixed-threshold methods may deviate a little).
        // LODA and R-Graph are exempt: many of their scores tie at the threshold
        // (LODA's 2-D projections, R-Graph's never-visited nodes), as in PyOD.
        let flagged = det.labels().unwrap().iter().filter(|&&l| l == 1).count();
        assert!(
            matches!(name, "LODA" | "RGraph") || (15..=45).contains(&flagged),
            "{name}: flagged {flagged}"
        );

        let preds = det.predict(&x_test).unwrap();
        let threshold = det.threshold().unwrap();
        for (p, s) in preds.iter().zip(&test_scores) {
            assert_eq!(*p, (*s > threshold) as u8, "{name}");
        }
        for method in [ProbaMethod::Linear, ProbaMethod::Unify] {
            let probs = det.predict_proba(&x_test, method).unwrap();
            assert!(probs.iter().all(|p| (0.0..=1.0).contains(p)), "{name}");
        }
        let conf = det.predict_confidence(&x_test).unwrap();
        assert!(conf.iter().all(|c| (0.0..=1.0).contains(c)), "{name}");
    }
}

#[test]
fn unfitted_and_bad_input_errors() {
    let det = Ecod::default();
    assert_eq!(det.predict(&[vec![1.0]]), Err(OdError::NotFitted));
    let mut det = Knn::default();
    assert_eq!(det.fit(&[]), Err(OdError::EmptyInput));
    assert_eq!(
        det.fit(&[vec![1.0, 2.0], vec![1.0]]),
        Err(OdError::RaggedInput)
    );
    let (x, _) = generate_data(50, 3, 0.1, 3);
    det.fit(&x).unwrap();
    assert_eq!(
        det.decision_function(&[vec![0.0; 2]]),
        Err(OdError::DimensionMismatch {
            expected: 3,
            got: 2
        })
    );
    assert!(matches!(
        Ecod::new(0.9).fit(&x),
        Err(OdError::InvalidParameter(_))
    ));
}

#[test]
fn mad_uses_fixed_threshold() {
    let mut x: Vec<Vec<f64>> = (0..20).map(|i| vec![10.0 + (i % 5) as f64 * 0.1]).collect();
    x.push(vec![100.0]);
    let mut det = Mad::default();
    det.fit(&x).unwrap();
    assert_eq!(det.threshold().unwrap(), 3.5);
    assert_eq!(det.labels().unwrap().iter().filter(|&&l| l == 1).count(), 1);
    assert_eq!(det.labels().unwrap()[20], 1);
    assert_eq!(det.predict(&[vec![10.2], vec![-50.0]]).unwrap(), vec![0, 1]);
    assert!(det.fit(&[vec![1.0, 2.0]]).is_err());
}

#[test]
fn ecod_matches_pyod_reference() {
    // Reference values from PyOD 3.6.7: ECOD(contamination=0.2).fit(X).decision_scores_
    let x = vec![
        vec![1.0, 2.0],
        vec![2.0, 1.0],
        vec![3.0, 4.0],
        vec![4.0, 3.0],
        vec![10.0, -5.0],
    ];
    let mut det = Ecod::new(0.2);
    det.fit(&x).unwrap();
    let s = det.decision_scores().unwrap();
    let expected = [
        2.120_263_54,
        1.832_581_46,
        2.120_263_54,
        1.832_581_46,
        3.218_875_82,
    ];
    for (a, b) in s.iter().zip(expected) {
        assert!((a - b).abs() < 1e-8, "{s:?}");
    }
    assert_eq!(det.labels().unwrap(), &[0, 0, 0, 0, 1]);
}

#[test]
fn iforest_is_deterministic_per_seed() {
    let (x, _) = generate_data(200, 4, 0.1, 5);
    let mut a = IForest::new(50, 0.1, 9);
    let mut b = IForest::new(50, 0.1, 9);
    a.fit(&x).unwrap();
    b.fit(&x).unwrap();
    assert_eq!(a.decision_scores().unwrap(), b.decision_scores().unwrap());
}

#[test]
fn combination_functions() {
    let s = vec![vec![1.0, 3.0, 2.0, 0.0], vec![0.0, 0.0, 4.0, 2.0]];
    assert_eq!(average(&s, None), vec![1.5, 1.5]);
    assert_eq!(maximization(&s), vec![3.0, 4.0]);
    assert_eq!(aom(&s, 2), vec![2.5, 2.0]);
    assert_eq!(moa(&s, 2), vec![2.0, 3.0]);
    assert_eq!(majority_vote(&[vec![1, 1, 0], vec![1, 0, 0]]), vec![1, 0]);

    // Ensemble of standardized detector scores still ranks outliers first.
    let (x, y) = generate_data(200, 2, 0.1, 11);
    let mut cols = Vec::new();
    for (_, mut det) in all_detectors() {
        det.fit(&x).unwrap();
        cols.push(det.decision_scores().unwrap().to_vec());
    }
    let per_sample: Vec<Vec<f64>> = (0..x.len())
        .map(|i| cols.iter().map(|c| c[i]).collect())
        .collect();
    let z = StandardScaler::fit(&per_sample).transform(&per_sample);
    assert!(roc_auc(&y, &average(&z, None)) > 0.9);
}

#[test]
fn thresholder_hook() {
    let (x, _) = generate_data(100, 2, 0.1, 4);
    let mut det = Knn::default();
    det.fit(&x).unwrap();
    let mean_plus_2sd = |s: &[f64]| {
        let m = s.iter().sum::<f64>() / s.len() as f64;
        m + 2.0 * (s.iter().map(|v| (v - m).powi(2)).sum::<f64>() / s.len() as f64).sqrt()
    };
    let (th, labels) = apply_thresholder(&det, &mean_plus_2sd).unwrap();
    assert!(th > 0.0);
    assert_eq!(labels.len(), 100);
}

#[test]
fn xgbod_uses_labels() {
    let (x, y) = generate_data(200, 3, 0.1, 31);
    let (xt, yt) = generate_data(100, 3, 0.1, 32);
    let mut det = models::xgbod::Xgbod::default();
    assert!(det.fit(&x).is_err());
    det.fit_supervised(&x, &y).unwrap();
    assert!(roc_auc(&yt, &det.decision_function(&xt).unwrap()) > 0.9);
}

#[test]
fn embedding_detector_with_custom_encoder() {
    use models::embedding::{EmbeddingOd, MultiModalOd};
    // "Documents" are strings; the encoder counts characters per class.
    let encode = |items: &[String]| -> Result<Vec<Vec<f64>>> {
        Ok(items
            .iter()
            .map(|s| {
                vec![
                    s.len() as f64,
                    s.chars().filter(|c| c.is_ascii_digit()).count() as f64,
                    s.matches(' ').count() as f64,
                ]
            })
            .collect())
    };
    let mut docs: Vec<String> = (0..60)
        .map(|i| format!("plain words sentence number {}", i % 7))
        .collect();
    docs.push("4815162342 0000000000000000000000000000".to_string());
    let mut det = EmbeddingOd::new(Box::new(encode));
    det.fit(&docs).unwrap();
    assert_eq!(det.labels().unwrap()[60], 1);
    let mut mm = MultiModalOd::new(
        vec![("chars".into(), EmbeddingOd::new(Box::new(encode)))],
        0.05,
    );
    mm.fit(&docs).unwrap();
    assert_eq!(mm.predict(&docs[59..]).unwrap(), vec![0, 1]);
}

fn sine_series(n: usize, channels: usize) -> (Vec<Vec<f64>>, Vec<u8>) {
    let mut rng = pyod_rs::utils::Rng::new(5);
    let mut x: Vec<Vec<f64>> = (0..n)
        .map(|t| {
            (0..channels)
                .map(|c| {
                    ((t as f64) * std::f64::consts::TAU / 40.0 + c as f64).sin()
                        + 0.1 * rng.normal()
                })
                .collect()
        })
        .collect();
    let mut y = vec![0u8; n];
    for start in [150, 330, 480] {
        for k in 0..12 {
            x[start + k][0] += 2.5 * k as f64 / 11.0;
            y[start + k] = 1;
        }
    }
    (x, y)
}

#[test]
fn time_series_detectors() {
    use models::{
        ts_kshape::KShape, ts_matrix_profile::MatrixProfile, ts_od::TimeSeriesOd, ts_sand::Sand,
        ts_spectral_residual::SpectralResidual,
    };
    for channels in [1, 2] {
        let (x, y) = sine_series(600, channels);
        let dets: Vec<(&str, Box<dyn Detector>, f64)> = vec![
            (
                "SpectralResidual",
                Box::new(SpectralResidual::default()),
                0.6,
            ),
            ("MatrixProfile", Box::new(MatrixProfile::new(30, 0.1)), 0.7),
            (
                "TimeSeriesOD",
                Box::new(TimeSeriesOd::new(
                    Box::new(|_| Box::new(Knn::default())),
                    20,
                )),
                0.8,
            ),
            ("KShape", Box::new(KShape::new(30, 0.1)), 0.5),
            ("SAND", Box::new(Sand::new(30, 0.1)), 0.4),
        ];
        for (name, mut det, floor) in dets {
            det.fit(&x).unwrap();
            let scores = det.decision_scores().unwrap();
            assert_eq!(scores.len(), x.len(), "{name}");
            let auc = roc_auc(&y, scores);
            println!("{name:<16} channels={channels} AUC {auc:.3}");
            assert!(auc > floor, "{name}: AUC {auc}");
            if name != "MatrixProfile" {
                assert_eq!(
                    det.decision_function(&x[..200]).unwrap().len(),
                    200,
                    "{name}"
                );
            } else {
                assert!(matches!(
                    det.decision_function(&x),
                    Err(OdError::Unsupported(_))
                ));
            }
        }
    }
}

#[test]
fn graph_detectors_without_nn() {
    use models::graph::{Anomalous, GraphDetector, Radar, Scan};
    // Two dense communities plus a node bridging them and an isolated pair.
    let n = 22;
    let mut edges = Vec::new();
    for c in 0..2 {
        for i in 0..10 {
            for j in 0..10 {
                if i != j {
                    edges.push((c * 10 + i, c * 10 + j));
                }
            }
        }
    }
    edges.extend([(20, 0), (0, 20), (20, 10), (10, 20), (21, 5), (5, 21)]);
    let x: Vec<Vec<f64>> = (0..n).map(|i| vec![(i / 10) as f64, 1.0]).collect();
    let mut scan = Scan::default();
    scan.fit_graph(&x, &edges).unwrap();
    let s = scan.decision_scores().unwrap();
    assert!(s[20] > s[3] && s[21] > s[3], "{s:?}");
    for det in [
        &mut Anomalous::default() as &mut dyn GraphDetector,
        &mut Radar::default(),
    ] {
        det.fit_graph(&x, &edges).unwrap();
        assert_eq!(det.decision_scores().unwrap().len(), n);
    }
    assert!(Scan::default().fit_graph(&x, &[(0, 99)]).is_err());
}
