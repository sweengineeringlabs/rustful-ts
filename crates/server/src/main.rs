//! # rustful-server
//!
//! REST API server for the rustful-ts time series library.
//! Built on rustboot infrastructure for production-ready features.

use axum::{
    routing::{get, post},
    Json, Router,
};
use rustboot_health::{AlwaysHealthyCheck, CheckResult, FunctionCheck, HealthAggregator};
use std::env;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod routes;

/// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    health: Arc<HealthAggregator>,
}

/// Liveness probe - is the server running?
async fn liveness() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "alive",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

/// Readiness probe - is the server ready to handle requests?
async fn readiness(
    axum::extract::State(state): axum::extract::State<AppState>,
) -> Json<serde_json::Value> {
    let report = state.health.check().await;
    Json(serde_json::json!({
        "status": report.status.to_string(),
        "version": env!("CARGO_PKG_VERSION"),
        "timestamp": report.timestamp,
        "duration_ms": report.duration_ms,
        "checks": report.checks.iter().map(|(name, result)| {
            serde_json::json!({
                "name": name,
                "status": format!("{:?}", result.status),
                "message": result.message
            })
        }).collect::<Vec<_>>()
    }))
}

fn create_health_aggregator() -> HealthAggregator {
    HealthAggregator::new()
        .add_check(Box::new(AlwaysHealthyCheck::new("server")))
        .add_check(Box::new(FunctionCheck::new("algorithms", || async {
            // Verify algorithm crate is functional
            use algorithm::Predictor;
            let mut model = algorithm::exponential::SimpleExponentialSmoothing::new(0.3).unwrap();
            model.fit(&[1.0, 2.0, 3.0]).unwrap();
            CheckResult::healthy("algorithms")
        })))
        .with_version(env!("CARGO_PKG_VERSION"))
}

#[tokio::main]
async fn main() {
    // Load .env file (optional - won't fail if missing)
    dotenvy::dotenv().ok();

    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "server=info,tower_http=info".into()),
        )
        .init();

    // Create application state
    let state = AppState {
        health: Arc::new(create_health_aggregator()),
    };

    // CORS configuration
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Build router with middleware
    let app = Router::new()
        // Health endpoints (Kubernetes-compatible)
        .route("/health/live", get(liveness))
        .route("/health/ready", get(readiness))
        // Legacy health endpoint
        .route("/health", get(liveness))
        // API endpoints
        .route("/api/v1/forecast", post(routes::forecast))
        .route("/api/v1/anomaly/detect", post(routes::detect_anomalies))
        // Middleware layers
        .layer(TraceLayer::new_for_http())
        .layer(cors)
        .with_state(state);

    // Server configuration from environment
    let host = env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port: u16 = env::var("PORT")
        .unwrap_or_else(|_| "8080".to_string())
        .parse()
        .expect("PORT must be a valid number");
    let addr: SocketAddr = format!("{}:{}", host, port)
        .parse()
        .expect("Invalid HOST:PORT configuration");

    tracing::info!("rustful-server v{} listening on {}", env!("CARGO_PKG_VERSION"), addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
