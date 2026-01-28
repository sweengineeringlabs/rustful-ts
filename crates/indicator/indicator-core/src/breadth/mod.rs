//! Market Breadth Indicators
//!
//! Indicators that measure the overall health and participation of the market
//! by analyzing advance/decline data, volume breadth, and market-wide statistics.

pub mod advance_decline;
pub mod breadth_thrust;
pub mod bullish_percent;
pub mod cumulative_volume;
pub mod high_low_index;
pub mod mcclellan;
pub mod mcclellan_sum;
pub mod new_highs_lows;
pub mod percent_above_ma;
pub mod put_call;
pub mod tick_index;
pub mod trin;
pub mod extended;
pub mod advanced;

// Re-exports - Primary indicators
pub use advance_decline::AdvanceDeclineLine;
pub use breadth_thrust::BreadthThrust;
pub use bullish_percent::{BPISeries, BPISignal, BPIStatus, BullishPercent};
pub use cumulative_volume::{CumulativeVolumeIndex, UpDownVolume};
pub use high_low_index::{HighLowData, HighLowIndex, HighLowMethod, HighLowSeries};
pub use mcclellan::McClellanOscillator;
pub use new_highs_lows::{NewHighsLows, NewHighsLowsMode};
pub use mcclellan_sum::McClellanSummationIndex;
pub use percent_above_ma::{MarketCondition, PercentAboveMA, PercentAboveMASeries};
pub use put_call::{ContrarianSignal, PutCallRatio, PutCallSeries, PutCallSignal};
pub use tick_index::{TickBias, TickIndex, TickSeries, TickSignal, TickStats};
pub use trin::{TRINSignal, TRIN};
pub use extended::{
    ADThrust, ZweigBreadthThrust, TRINSmoothed, BreadthMomentum,
    VolumeBreadth, PercentageBreadth,
};
pub use advanced::{
    MarketMomentumBreadth, BreadthOscillator, CumulativeBreadthIndex,
    VolumeBreadthRatio, BreadthDivergence, ParticipationRate,
    BreadthMomentumAdvanced, BreadthStrength, BreadthOverbought,
    BreadthOversold, BreadthTrend, BreadthConfirmation,
    // New breadth indicators
    BreadthMomentumIndex, CumulativeBreadthMomentum, BreadthVolatility,
    BreadthTrendStrength, BreadthExtremeDetector, BreadthDivergenceIndex,
    // 6 NEW breadth indicators (requested additions)
    BreadthTrustThrust, AdvanceDeclineOscillator, BreadthStrengthIndex,
    MarketInternalsScore, BreadthPersistence, BreadthAcceleration,
    // 6 NEWEST breadth indicators (AdvanceDeclineRatio, BreadthMomentumIndicator, etc.)
    AdvanceDeclineRatio, BreadthMomentumIndicator, CumulativeBreadthLine,
    HighLowIndex as AdvancedHighLowIndex, PercentAboveMA as AdvancedPercentAboveMA, BreadthDiffusion,
    // 6 NEW breadth indicators (BreadthRatio, BreadthScore, MarketParticipation, etc.)
    BreadthRatio, BreadthScore, MarketParticipation, TrendBreadth, BreadthSignal,
    // 5 NEW breadth indicators (SectorBreadthIndex, ParticipationOscillator, BreadthMomentumOscillator, etc.)
    SectorBreadthIndex, ParticipationOscillator, BreadthRegimeClassifier, CumulativeParticipation,
    BreadthMomentumOscillator,
};

// Re-export SPI types from crate root
pub use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator, OHLCV,
};

/// Market breadth data for a single period.
///
/// Unlike OHLCV data which represents a single asset, breadth data
/// represents market-wide statistics like advances, declines, and volume.
#[derive(Debug, Clone)]
pub struct BreadthData {
    /// Number of advancing issues
    pub advances: f64,
    /// Number of declining issues
    pub declines: f64,
    /// Number of unchanged issues
    pub unchanged: f64,
    /// Volume of advancing issues
    pub advance_volume: f64,
    /// Volume of declining issues
    pub decline_volume: f64,
    /// Volume of unchanged issues
    pub unchanged_volume: f64,
}

impl BreadthData {
    pub fn new(
        advances: f64,
        declines: f64,
        unchanged: f64,
        advance_volume: f64,
        decline_volume: f64,
        unchanged_volume: f64,
    ) -> Self {
        Self {
            advances,
            declines,
            unchanged,
            advance_volume,
            decline_volume,
            unchanged_volume,
        }
    }

    /// Create breadth data from just advance/decline counts (no volume)
    pub fn from_ad(advances: f64, declines: f64) -> Self {
        Self {
            advances,
            declines,
            unchanged: 0.0,
            advance_volume: 0.0,
            decline_volume: 0.0,
            unchanged_volume: 0.0,
        }
    }

    /// Create breadth data with advance/decline counts and volumes
    pub fn from_ad_volume(
        advances: f64,
        declines: f64,
        advance_volume: f64,
        decline_volume: f64,
    ) -> Self {
        Self {
            advances,
            declines,
            unchanged: 0.0,
            advance_volume,
            decline_volume,
            unchanged_volume: 0.0,
        }
    }

    /// Total issues traded
    pub fn total_issues(&self) -> f64 {
        self.advances + self.declines + self.unchanged
    }

    /// Total volume
    pub fn total_volume(&self) -> f64 {
        self.advance_volume + self.decline_volume + self.unchanged_volume
    }

    /// Net advances (advances - declines)
    pub fn net_advances(&self) -> f64 {
        self.advances - self.declines
    }

    /// Net advancing volume (advance_volume - decline_volume)
    pub fn net_advance_volume(&self) -> f64 {
        self.advance_volume - self.decline_volume
    }

    /// Advance/decline ratio
    pub fn ad_ratio(&self) -> f64 {
        if self.declines == 0.0 {
            f64::INFINITY
        } else {
            self.advances / self.declines
        }
    }
}

/// Series of market breadth data.
#[derive(Debug, Clone, Default)]
pub struct BreadthSeries {
    pub advances: Vec<f64>,
    pub declines: Vec<f64>,
    pub unchanged: Vec<f64>,
    pub advance_volume: Vec<f64>,
    pub decline_volume: Vec<f64>,
    pub unchanged_volume: Vec<f64>,
}

impl BreadthSeries {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            advances: Vec::with_capacity(capacity),
            declines: Vec::with_capacity(capacity),
            unchanged: Vec::with_capacity(capacity),
            advance_volume: Vec::with_capacity(capacity),
            decline_volume: Vec::with_capacity(capacity),
            unchanged_volume: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, data: BreadthData) {
        self.advances.push(data.advances);
        self.declines.push(data.declines);
        self.unchanged.push(data.unchanged);
        self.advance_volume.push(data.advance_volume);
        self.decline_volume.push(data.decline_volume);
        self.unchanged_volume.push(data.unchanged_volume);
    }

    pub fn len(&self) -> usize {
        self.advances.len()
    }

    pub fn is_empty(&self) -> bool {
        self.advances.is_empty()
    }

    /// Compute net advances series (advances - declines)
    pub fn net_advances(&self) -> Vec<f64> {
        self.advances
            .iter()
            .zip(self.declines.iter())
            .map(|(a, d)| a - d)
            .collect()
    }

    /// Compute net advancing volume series
    pub fn net_advance_volume(&self) -> Vec<f64> {
        self.advance_volume
            .iter()
            .zip(self.decline_volume.iter())
            .map(|(av, dv)| av - dv)
            .collect()
    }

    /// Total issues for each period
    pub fn total_issues(&self) -> Vec<f64> {
        self.advances
            .iter()
            .zip(self.declines.iter())
            .zip(self.unchanged.iter())
            .map(|((a, d), u)| a + d + u)
            .collect()
    }

    /// Convert to OHLCVSeries using net advances as the "close" price.
    /// This allows reusing indicators designed for OHLCV data.
    pub fn to_ohlcv_net_advances(&self) -> OHLCVSeries {
        let net = self.net_advances();
        let total_volume: Vec<f64> = self
            .advance_volume
            .iter()
            .zip(self.decline_volume.iter())
            .zip(self.unchanged_volume.iter())
            .map(|((av, dv), uv)| av + dv + uv)
            .collect();

        OHLCVSeries {
            open: net.clone(),
            high: net.clone(),
            low: net.clone(),
            close: net,
            volume: total_volume,
        }
    }
}

/// Trait for market breadth indicators.
///
/// Similar to TechnicalIndicator but operates on BreadthSeries data
/// instead of OHLCVSeries.
pub trait BreadthIndicator: Send + Sync {
    /// Indicator name.
    fn name(&self) -> &str;

    /// Compute indicator values from breadth data.
    fn compute_breadth(&self, data: &BreadthSeries) -> Result<IndicatorOutput>;

    /// Minimum periods required for valid output.
    fn min_periods(&self) -> usize;

    /// Number of output features.
    fn output_features(&self) -> usize {
        1
    }
}
