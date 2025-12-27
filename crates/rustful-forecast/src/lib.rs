//! # rustful-forecast
//!
//! Forecasting pipeline infrastructure for rustful-ts.
//! Provides composable preprocessing steps, decomposition, and seasonality detection.

mod pipeline;
mod decomposition;
mod seasonality;
mod confidence;

pub use pipeline::*;
pub use decomposition::*;
pub use seasonality::*;
pub use confidence::*;
