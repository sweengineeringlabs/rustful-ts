//! Technical Indicator Composite - Multi-Indicator Systems
//!
//! This crate provides composite indicators that combine multiple technical
//! indicators to generate more robust trading signals.
//!
//! # Composite Indicators
//!
//! - [`TTMSqueeze`]: TTM Squeeze (Bollinger inside Keltner + momentum)
//! - [`ElderImpulse`]: Elder Impulse System (EMA + MACD histogram)
//! - [`SchaffTrendCycle`]: Schaff Trend Cycle (MACD + Stochastic smoothing)
//! - [`ElderTripleScreen`]: Elder Triple Screen trading system
//! - [`CommoditySelectionIndex`]: Commodity Selection Index (ADXR + ATR)
//! - [`SqueezeMomentum`]: LazyBear's Squeeze Momentum
//! - [`TrendStrengthIndex`]: Composite trend strength measurement
//! - [`RegimeDetector`]: Market regime detection (trending/ranging)
//!
//! # Example
//!
//! ```rust,ignore
//! use indicator_composite::{TTMSqueeze, TTMSqueezeConfig};
//! use indicator_spi::OHLCVSeries;
//!
//! let squeeze = TTMSqueeze::new(TTMSqueezeConfig::default());
//! let data = OHLCVSeries::from_close(vec![100.0; 50]);
//! let result = squeeze.calculate(&data.high, &data.low, &data.close);
//! ```

// ============================================================================
// Module Declarations
// ============================================================================

pub mod ttm_squeeze;
pub mod elder_impulse;
pub mod schaff;
pub mod elder_triple_screen;
pub mod commodity_selection;
pub mod squeeze_momentum;
pub mod trend_strength;
pub mod regime_detector;

// ============================================================================
// Re-exports
// ============================================================================

// TTM Squeeze
pub use ttm_squeeze::{TTMSqueeze, TTMSqueezeConfig, TTMSqueezeOutput};

// Elder Impulse System
pub use elder_impulse::{ElderImpulse, ElderImpulseConfig, ElderImpulseOutput};

// Schaff Trend Cycle
pub use schaff::{SchaffTrendCycle, SchaffConfig, SchaffOutput};

// Elder Triple Screen
pub use elder_triple_screen::{ElderTripleScreen, ElderTripleScreenConfig, ElderTripleScreenOutput};

// Commodity Selection Index
pub use commodity_selection::{CommoditySelectionIndex, CommoditySelectionConfig, CommoditySelectionOutput};

// Squeeze Momentum (LazyBear)
pub use squeeze_momentum::{SqueezeMomentum, SqueezeMomentumConfig, SqueezeMomentumOutput};

// Trend Strength Index
pub use trend_strength::{TrendStrengthIndex, TrendStrengthConfig, TrendStrengthOutput, TrendComponents};

// Regime Detector
pub use regime_detector::{RegimeDetector, RegimeDetectorConfig, RegimeDetectorOutput, MarketRegime};

// Re-export SPI types for convenience
pub use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCV, OHLCVSeries,
};
