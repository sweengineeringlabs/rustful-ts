//! Algorithm Facade
//!
//! High-level API for time series algorithms. Re-exports all public types
//! from the algorithm stack for convenient usage.
//!
//! # Example
//!
//! ```rust
//! use algorithm_facade::prelude::*;
//!
//! let data: Vec<f64> = (1..=20).map(|x| x as f64).collect();
//! let mut model = Arima::new(1, 1, 0).unwrap();
//! model.fit(&data).unwrap();
//! let forecast = model.predict(3).unwrap();
//! ```

// Re-export everything from core (includes implementations)
pub use algorithm_core::*;

// Re-export from API for completeness (mostly overlaps with core re-exports)
#[allow(unused_imports)]
pub use algorithm_api::*;

// Explicit re-exports for documentation
pub use algorithm_core::prelude;
pub use algorithm_core::utils;
