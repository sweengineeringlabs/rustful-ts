//! # rustful-server
//!
//! REST API server for the rustful-ts time series library.

use axum::{
    routing::{get, post},
    Router, Json,
};
use serde::Serialize;
use std::net::SocketAddr;

mod routes;

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    version: String,
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/health", get(health))
        .route("/api/v1/forecast", post(routes::forecast))
        .route("/api/v1/anomaly/detect", post(routes::detect_anomalies));

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    println!("rustful-server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
