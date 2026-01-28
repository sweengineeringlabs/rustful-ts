//! Intermarket Analysis Indicators
//!
//! Indicators for analyzing relationships between multiple markets,
//! pairs trading, currency strength, and sector rotation.

pub mod cointegration;
pub mod currency_strength;
pub mod relative_strength;
pub mod sector_rotation;
pub mod commodity;
pub mod fixed_income;
pub mod forex;
pub mod extended;

// Re-exports
pub use cointegration::{Cointegration, CointegrationOutput, CointegrationSignal};
pub use currency_strength::{CurrencyPair, CurrencyStrength, CurrencyStrengthOutput};
pub use relative_strength::{RelativeStrength, RelativeStrengthOutput, RelativeStrengthSignal};
pub use sector_rotation::{SectorRank, SectorRotation, SectorRotationOutput};
pub use commodity::{
    ContangoBackwardation, RollYield, Basis, ConvenienceYield,
    InventorySurprise, CrackSpread,
};
pub use fixed_income::{
    YieldCurveShape, ButterflySpread, BreakevenInflation,
    RealRate, EffectiveDuration, KeyRateDuration,
};
pub use forex::{
    CarryTradeIndex, FXVolatilityTerm, RiskReversal25D, Butterfly25D,
    FXPositioning, DollarSmile, PPPDeviation, BEER,
};
pub use extended::{
    CrossMarketMomentum, BetaCoefficient, MarketRegimeIndicator,
    SectorRelativePerformance, CorrelationMomentum, RiskAppetiteIndex, DivergenceIndex,
};

// ============================================================================
// Dual Series Support
// ============================================================================

/// Dual series data for intermarket analysis.
///
/// Used for indicators that require two price series (e.g., pairs trading).
#[derive(Debug, Clone)]
pub struct DualSeries {
    /// First series (e.g., asset prices).
    pub series1: Vec<f64>,
    /// Second series (e.g., benchmark or pair prices).
    pub series2: Vec<f64>,
}

impl DualSeries {
    /// Create a new DualSeries from two price vectors.
    pub fn new(series1: Vec<f64>, series2: Vec<f64>) -> Self {
        Self { series1, series2 }
    }

    /// Create from slices (copies the data).
    pub fn from_slices(series1: &[f64], series2: &[f64]) -> Self {
        Self {
            series1: series1.to_vec(),
            series2: series2.to_vec(),
        }
    }

    /// Get the length of the shorter series.
    pub fn len(&self) -> usize {
        self.series1.len().min(self.series2.len())
    }

    /// Check if either series is empty.
    pub fn is_empty(&self) -> bool {
        self.series1.is_empty() || self.series2.is_empty()
    }

    /// Validate that both series have the same length.
    pub fn validate_equal_length(&self) -> bool {
        self.series1.len() == self.series2.len()
    }
}

/// Multi-series data for intermarket analysis with multiple assets.
///
/// Used for indicators like currency strength that require multiple series.
#[derive(Debug, Clone)]
pub struct MultiSeries {
    /// Vector of named series.
    pub series: Vec<(String, Vec<f64>)>,
}

impl MultiSeries {
    /// Create a new empty MultiSeries.
    pub fn new() -> Self {
        Self { series: Vec::new() }
    }

    /// Add a named series.
    pub fn add(&mut self, name: &str, data: Vec<f64>) {
        self.series.push((name.to_string(), data));
    }

    /// Get the number of series.
    pub fn count(&self) -> usize {
        self.series.len()
    }

    /// Get the minimum length across all series.
    pub fn min_len(&self) -> usize {
        self.series.iter().map(|(_, d)| d.len()).min().unwrap_or(0)
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.series.is_empty()
    }
}

impl Default for MultiSeries {
    fn default() -> Self {
        Self::new()
    }
}
