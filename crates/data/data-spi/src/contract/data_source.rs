//! Data source trait definition.

use crate::error::Result;
use crate::model::{Interval, Quote};

/// Trait for data sources that can fetch historical price data.
///
/// Implementations provide access to financial data from various providers.
pub trait DataSource: Send + Sync {
    /// Data source name.
    fn name(&self) -> &str;

    /// Fetch historical data synchronously.
    fn fetch_sync(
        &self,
        symbol: &str,
        start_date: &str,
        end_date: &str,
        interval: Interval,
    ) -> Result<Vec<Quote>>;
}

// Note: AsyncDataSource trait would be defined here if async-trait is added as a dependency
// For now, async support is handled directly in the implementations (data-core)
