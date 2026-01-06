//! Technical Indicator Service Provider Interface
//!
//! Defines traits and types for technical analysis indicators.

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

/// Indicator errors.
#[derive(Debug, Error)]
pub enum IndicatorError {
    #[error("Insufficient data: required {required}, got {got}")]
    InsufficientData { required: usize, got: usize },

    #[error("Invalid parameter: {name} - {reason}")]
    InvalidParameter { name: String, reason: String },

    #[error("Computation error: {0}")]
    ComputationError(String),
}

pub type Result<T> = std::result::Result<T, IndicatorError>;

// ============================================================================
// OHLCV Types
// ============================================================================

/// OHLCV (Open, High, Low, Close, Volume) bar data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCV {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl OHLCV {
    pub fn new(open: f64, high: f64, low: f64, close: f64, volume: f64) -> Self {
        Self { open, high, low, close, volume }
    }

    /// Typical price: (High + Low + Close) / 3
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// True range for ATR calculation.
    pub fn true_range(&self, prev_close: f64) -> f64 {
        let hl = self.high - self.low;
        let hc = (self.high - prev_close).abs();
        let lc = (self.low - prev_close).abs();
        hl.max(hc).max(lc)
    }
}

/// Series of OHLCV data for indicator computation.
#[derive(Debug, Clone)]
pub struct OHLCVSeries {
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
}

impl OHLCVSeries {
    pub fn new() -> Self {
        Self {
            open: Vec::new(),
            high: Vec::new(),
            low: Vec::new(),
            close: Vec::new(),
            volume: Vec::new(),
        }
    }

    pub fn from_close(close: Vec<f64>) -> Self {
        let len = close.len();
        Self {
            open: close.clone(),
            high: close.clone(),
            low: close.clone(),
            close,
            volume: vec![0.0; len],
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            open: Vec::with_capacity(capacity),
            high: Vec::with_capacity(capacity),
            low: Vec::with_capacity(capacity),
            close: Vec::with_capacity(capacity),
            volume: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, bar: OHLCV) {
        self.open.push(bar.open);
        self.high.push(bar.high);
        self.low.push(bar.low);
        self.close.push(bar.close);
        self.volume.push(bar.volume);
    }

    pub fn len(&self) -> usize {
        self.close.len()
    }

    pub fn is_empty(&self) -> bool {
        self.close.is_empty()
    }
}

impl Default for OHLCVSeries {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Indicator Output
// ============================================================================

/// Output from an indicator computation.
#[derive(Debug, Clone)]
pub struct IndicatorOutput {
    /// Primary output series (required).
    pub primary: Vec<f64>,
    /// Optional secondary output (e.g., upper band, signal line).
    pub secondary: Option<Vec<f64>>,
    /// Optional tertiary output (e.g., lower band, histogram).
    pub tertiary: Option<Vec<f64>>,
}

impl IndicatorOutput {
    /// Single-output indicator.
    pub fn single(values: Vec<f64>) -> Self {
        Self {
            primary: values,
            secondary: None,
            tertiary: None,
        }
    }

    /// Dual-output indicator (e.g., MACD with signal).
    pub fn dual(primary: Vec<f64>, secondary: Vec<f64>) -> Self {
        Self {
            primary,
            secondary: Some(secondary),
            tertiary: None,
        }
    }

    /// Triple-output indicator (e.g., Bollinger Bands).
    pub fn triple(primary: Vec<f64>, secondary: Vec<f64>, tertiary: Vec<f64>) -> Self {
        Self {
            primary,
            secondary: Some(secondary),
            tertiary: Some(tertiary),
        }
    }
}

// ============================================================================
// Core Traits
// ============================================================================

/// Technical indicator trait.
///
/// Implementations compute indicator values from OHLCV data.
pub trait TechnicalIndicator: Send + Sync {
    /// Indicator name.
    fn name(&self) -> &str;

    /// Compute indicator values.
    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput>;

    /// Minimum periods required for valid output.
    fn min_periods(&self) -> usize;

    /// Number of output features.
    fn output_features(&self) -> usize {
        1
    }

    /// Compute from close prices only.
    fn compute_close(&self, close: &[f64]) -> Result<IndicatorOutput> {
        self.compute(&OHLCVSeries::from_close(close.to_vec()))
    }
}

/// Streaming indicator trait for real-time updates.
pub trait StreamingIndicator: TechnicalIndicator {
    /// Update with new bar and return latest value.
    fn update(&mut self, bar: &OHLCV) -> Result<f64>;

    /// Reset internal state.
    fn reset(&mut self);

    /// Current value (if enough data).
    fn current(&self) -> Option<f64>;
}

// ============================================================================
// Signal Integration
// ============================================================================

/// Signal direction from indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndicatorSignal {
    Bullish,
    Bearish,
    Neutral,
}

impl IndicatorSignal {
    /// Convert to numeric signal: Bullish = 1, Bearish = -1, Neutral = 0.
    pub fn to_numeric(&self) -> f64 {
        match self {
            IndicatorSignal::Bullish => 1.0,
            IndicatorSignal::Bearish => -1.0,
            IndicatorSignal::Neutral => 0.0,
        }
    }
}

/// Trait for indicators that generate trading signals.
pub trait SignalIndicator: TechnicalIndicator {
    /// Generate signal from current indicator state.
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal>;

    /// Generate signal series.
    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>>;
}
