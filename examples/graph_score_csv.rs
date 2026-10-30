//! Print each graph detector's node scores for CSV node features and edges.
//! `cargo run --release --features deep --example graph_score_csv nodes.csv edges.csv`

use pyod_rs::deep::graph::{AnomalyDae, Cola, Conad, Dominant, Guide};
use pyod_rs::models::graph::{Anomalous, GraphDetector, Radar, Scan};
use std::{env, fs};

fn read(p: &str) -> Vec<Vec<f64>> {
    fs::read_to_string(p)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split(',').map(|v| v.trim().parse().unwrap()).collect())
        .collect()
}

fn main() {
    let a: Vec<String> = env::args().collect();
    let x = read(&a[1]);
    let edges: Vec<(usize, usize)> = read(&a[2])
        .iter()
        .map(|r| (r[0] as usize, r[1] as usize))
        .collect();
    let detectors: Vec<(&str, Box<dyn GraphDetector>)> = vec![
        ("SCAN", Box::new(Scan::default())),
        ("ANOMALOUS", Box::new(Anomalous::default())),
        ("Radar", Box::new(Radar::default())),
        ("DOMINANT", Box::new(Dominant::default())),
        ("AnomalyDAE", Box::new(AnomalyDae::default())),
        ("CoLA", Box::new(Cola::default())),
        ("CONAD", Box::new(Conad::default())),
        ("GUIDE", Box::new(Guide::default())),
    ];
    for (name, mut det) in detectors {
        det.fit_graph(&x, &edges).unwrap();
        let s: Vec<String> = det
            .decision_scores()
            .unwrap()
            .iter()
            .map(|v| format!("{v:.12e}"))
            .collect();
        println!("{name}|{}", s.join(","));
    }
}
