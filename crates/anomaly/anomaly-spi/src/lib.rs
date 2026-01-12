//! Anomaly Detection Service Provider Interface
//!
//! Defines traits and types for anomaly detection.

pub mod contract;
pub mod error;
pub mod model;

// Re-export all public items at crate root for convenience
pub use contract::{AnomalyDetector, MonitoringStream};
pub use error::{AnomalyError, Result};
pub use model::{Alert, AlertSeverity, AnomalyResult};
