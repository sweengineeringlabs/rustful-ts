//! Detector Facade
//!
//! High-level, simplified API for anomaly detection.
//!
//! This crate provides a unified entry point for all anomaly detection
//! functionality, re-exporting the most commonly used types and traits.

// Re-export everything from detector-api
pub use detector_api::*;

// Re-export prelude for convenience
pub use detector_api::prelude;

// Re-export utilities from core
pub use detector_core::Monitor;
