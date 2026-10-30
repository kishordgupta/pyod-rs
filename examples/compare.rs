//! Benchmark every detector on synthetic data, like PyOD's `compare_all_models.py`.
//! Run with `cargo run --release --example compare`.

use pyod_rs::data::{generate_data, precision_n_scores, roc_auc};
use pyod_rs::*;
use std::time::Instant;

fn main() -> Result<()> {
    let (x_train, _) = generate_data(2000, 5, 0.1, 42);
    let (x_test, y_test) = generate_data(1000, 5, 0.1, 43);

    let detectors: Vec<(&str, Box<dyn Detector>)> = vec![
        ("ABOD", Box::new(Abod::default())),
        ("COPOD", Box::new(Copod::default())),
        ("ECOD", Box::new(Ecod::default())),
        ("HBOS", Box::new(Hbos::default())),
        ("IForest", Box::new(IForest::default())),
        ("KNN", Box::new(Knn::default())),
        ("LOF", Box::new(Lof::default())),
        ("PCA", Box::new(Pca::default())),
    ];

    println!(
        "{:<8} {:>8} {:>8} {:>10}",
        "model", "ROC", "P@n", "time (ms)"
    );
    for (name, mut det) in detectors {
        let t = Instant::now();
        det.fit(&x_train)?;
        let scores = det.decision_function(&x_test)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        println!(
            "{name:<8} {:>8.4} {:>8.4} {:>10.1}",
            roc_auc(&y_test, &scores),
            precision_n_scores(&y_test, &scores),
            ms
        );
    }
    Ok(())
}
