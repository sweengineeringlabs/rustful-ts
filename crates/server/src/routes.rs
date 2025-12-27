//! API route handlers

use axum::Json;
use serde::{Deserialize, Serialize};
use algorithm::{Predictor, regression::Arima};
use anomaly::{ZScoreDetector, AnomalyDetector};

#[derive(Debug, Deserialize)]
pub struct ForecastRequest {
    pub data: Vec<f64>,
    pub steps: usize,
    pub model: String,
}

#[derive(Debug, Serialize)]
pub struct ForecastResponse {
    pub predictions: Vec<f64>,
    pub model: String,
}

#[derive(Debug, Serialize)]
#[allow(dead_code)]
pub struct ErrorResponse {
    pub error: String,
}

pub async fn forecast(Json(req): Json<ForecastRequest>) -> Json<ForecastResponse> {
    // Simple ARIMA forecast for now
    let mut model = Arima::new(1, 1, 0).unwrap();
    model.fit(&req.data).unwrap();
    let predictions = model.predict(req.steps).unwrap();

    Json(ForecastResponse {
        predictions,
        model: req.model,
    })
}

#[derive(Debug, Deserialize)]
pub struct AnomalyRequest {
    pub data: Vec<f64>,
    pub threshold: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct AnomalyResponse {
    pub is_anomaly: Vec<bool>,
    pub scores: Vec<f64>,
}

pub async fn detect_anomalies(Json(req): Json<AnomalyRequest>) -> Json<AnomalyResponse> {
    let threshold = req.threshold.unwrap_or(3.0);
    let mut detector = ZScoreDetector::new(threshold);
    detector.fit(&req.data).unwrap();
    let result = detector.detect(&req.data).unwrap();

    Json(AnomalyResponse {
        is_anomaly: result.is_anomaly,
        scores: result.scores,
    })
}
