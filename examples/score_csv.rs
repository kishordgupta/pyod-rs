//! Print each detector's train and test scores for CSV files, one line per model.
//! Used to cross-check against PyOD: `cargo run --example score_csv train.csv test.csv`.

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
    let train = read_csv(&args[1]);
    let test = read_csv(&args[2]);
    let first_col = |m: &Matrix| m.iter().map(|r| vec![r[0]]).collect::<Vec<_>>();

    let detectors: Vec<(&str, Box<dyn Detector>, bool)> = vec![
        ("ABOD", Box::new(Abod::default()), false),
        ("COPOD", Box::new(Copod::default()), false),
        ("ECOD", Box::new(Ecod::default()), false),
        ("HBOS", Box::new(Hbos::default()), false),
        ("KNN", Box::new(Knn::default()), false),
        (
            "KNN_mean",
            Box::new(Knn::new(5, KnnMethod::Mean, 0.1)),
            false,
        ),
        ("LOF", Box::new(Lof::default()), false),
        ("PCA", Box::new(Pca::default()), false),
        ("MAD", Box::new(Mad::default()), true),
        ("KDE", Box::new(Kde::default()), false),
        ("GMM", Box::new(Gmm::default()), false),
        ("MCD", Box::new(Mcd::default()), false),
        ("OCSVM", Box::new(OcSvm::default()), false),
        ("CBLOF", Box::new(models::cblof::Cblof::default()), false),
        ("COF", Box::new(models::cof::Cof::default()), false),
        ("INNE", Box::new(models::inne::Inne::default()), false),
        ("LODA", Box::new(models::loda::Loda::default()), false),
        ("SOD", Box::new(models::sod::Sod::default()), false),
        ("SOS", Box::new(models::sos::Sos::default()), false),
        ("LMDD", Box::new(models::lmdd::Lmdd::default()), false),
        ("LOCI", Box::new(models::loci::Loci::default()), false),
        ("QMCD", Box::new(models::qmcd::Qmcd::default()), false),
        ("ROD", Box::new(models::rod::Rod::default()), false),
        ("KPCA", Box::new(models::kpca::Kpca::default()), false),
        ("CD", Box::new(models::cd::Cd::default()), false),
        (
            "Sampling",
            Box::new(models::sampling::Sampling::default()),
            false,
        ),
        (
            "HDBSCAN",
            Box::new(models::hdbscan::Hdbscan::default()),
            false,
        ),
        (
            "FeatureBagging",
            Box::new(models::feature_bagging::FeatureBagging::default()),
            false,
        ),
        (
            "LSCP",
            Box::new(models::lscp::Lscp::new(
                vec![
                    Box::new(Knn::new(5, KnnMethod::Largest, 0.1)),
                    Box::new(Knn::new(10, KnnMethod::Largest, 0.1)),
                    Box::new(Lof::new(10, 0.1)),
                    Box::new(Lof::new(20, 0.1)),
                ],
                0.1,
            )),
            false,
        ),
        ("SUOD", Box::new(models::suod::Suod::default()), false),
        ("RGraph", Box::new(models::rgraph::RGraph::default()), false),
    ];
    for (name, mut det, univariate) in detectors {
        let (tr, te) = if univariate {
            (first_col(&train), first_col(&test))
        } else {
            (train.clone(), test.clone())
        };
        det.fit(&tr)?;
        let fmt = |v: &[f64]| {
            v.iter()
                .map(|s| format!("{s:.12e}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        let labels: Vec<String> = det.labels()?.iter().map(|l| l.to_string()).collect();
        println!(
            "{name}|{}|{}|{}|{:.12e}",
            fmt(det.decision_scores()?),
            fmt(&det.decision_function(&te)?),
            labels.join(","),
            det.threshold()?
        );
    }
    Ok(())
}
