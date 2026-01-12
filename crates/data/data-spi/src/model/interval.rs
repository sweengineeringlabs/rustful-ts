//! Time interval types for historical data.

use serde::{Deserialize, Serialize};

/// Time interval for historical data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Interval {
    /// 1 minute bars
    Minute1,
    /// 5 minute bars
    Minute5,
    /// 15 minute bars
    Minute15,
    /// 30 minute bars
    Minute30,
    /// 1 hour bars
    Hour1,
    /// Daily bars
    Daily,
    /// Weekly bars
    Weekly,
    /// Monthly bars
    Monthly,
}

impl Interval {
    /// Convert to Yahoo Finance API string representation.
    pub fn as_yahoo_str(&self) -> &'static str {
        match self {
            Interval::Minute1 => "1m",
            Interval::Minute5 => "5m",
            Interval::Minute15 => "15m",
            Interval::Minute30 => "30m",
            Interval::Hour1 => "1h",
            Interval::Daily => "1d",
            Interval::Weekly => "1wk",
            Interval::Monthly => "1mo",
        }
    }
}
