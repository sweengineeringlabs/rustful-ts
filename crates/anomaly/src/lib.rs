//! # rustful-anomaly
//!
//! Anomaly detection module for rustful-ts.
//! Provides various detectors, real-time monitoring, and alerting.

mod detectors;
mod monitoring;
mod alerting;

pub use detectors::*;
pub use monitoring::*;
pub use alerting::*;
