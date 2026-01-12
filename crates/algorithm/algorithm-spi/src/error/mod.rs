//! Error types for algorithm operations
//!
//! This module provides the [`TsError`] enum and [`Result`] type alias
//! for standardized error handling across all algorithm implementations.

mod ts_error;

pub use ts_error::{Result, TsError};
