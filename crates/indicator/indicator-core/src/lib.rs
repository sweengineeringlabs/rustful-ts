//! Technical Indicator Core Implementations
//!
//! SIMD-optimized technical analysis indicators.

// ============================================================================
// Moving Averages
// ============================================================================
pub mod sma;
pub mod ema;
pub mod wma;
pub mod dema;
pub mod tema;
pub mod hma;
pub mod kama;
pub mod zlema;
pub mod smma;
pub mod alma;
pub mod frama;
pub mod vidya;
pub mod t3;
pub mod triangular;
pub mod gmma;
pub mod sine_wma;

// ============================================================================
// Filters
// ============================================================================
pub mod kalman;
pub mod median;
pub mod gaussian;

// ============================================================================
// Momentum/Oscillators
// ============================================================================
pub mod rsi;
pub mod stochastic;
pub mod williams_r;
pub mod cci;
pub mod trix;
pub mod ultimate_oscillator;

// ============================================================================
// Trend Indicators
// ============================================================================
pub mod macd;
pub mod adx;
pub mod ichimoku;
pub mod supertrend;
pub mod parabolic_sar;

// ============================================================================
// Volatility Indicators
// ============================================================================
pub mod bollinger;
pub mod atr;
pub mod donchian;
pub mod keltner;

// ============================================================================
// Volume Indicators
// ============================================================================
pub mod vwap;
pub mod obv;
pub mod mfi;
pub mod cmf;

// ============================================================================
// Support/Resistance
// ============================================================================
pub mod pivot_points;
pub mod fibonacci;

// ============================================================================
// Other
// ============================================================================
pub mod elder_ray;

#[cfg(feature = "simd")]
pub mod simd;

// ============================================================================
// Moving Average Exports
// ============================================================================
pub use sma::SMA;
pub use ema::EMA;
pub use wma::WMA;
pub use dema::DEMA;
pub use tema::TEMA;
pub use hma::HMA;
pub use kama::KAMA;
pub use zlema::ZLEMA;
pub use smma::SMMA;
pub use alma::ALMA;
pub use frama::FRAMA;
pub use vidya::VIDYA;
pub use t3::T3;
pub use triangular::TRIMA;
pub use gmma::{GMMA, GMMAOutput};
pub use sine_wma::SineWMA;

// ============================================================================
// Filter Exports
// ============================================================================
pub use kalman::KalmanFilter;
pub use median::MedianFilter;
pub use gaussian::GaussianFilter;

// ============================================================================
// Oscillator Exports
// ============================================================================
pub use rsi::RSI;
pub use stochastic::Stochastic;
pub use williams_r::WilliamsR;
pub use cci::CCI;
pub use trix::TRIX;
pub use ultimate_oscillator::UltimateOscillator;

// ============================================================================
// Trend Exports
// ============================================================================
pub use macd::MACD;
pub use adx::{ADX, ADXOutput};
pub use ichimoku::{Ichimoku, IchimokuOutput};
pub use supertrend::{SuperTrend, SuperTrendOutput};
pub use parabolic_sar::{ParabolicSAR, ParabolicSAROutput};

// ============================================================================
// Volatility Exports
// ============================================================================
pub use bollinger::BollingerBands;
pub use atr::ATR;
pub use donchian::DonchianChannels;
pub use keltner::KeltnerChannels;

// ============================================================================
// Volume Exports
// ============================================================================
pub use vwap::VWAP;
pub use obv::OBV;
pub use mfi::MFI;
pub use cmf::CMF;

// ============================================================================
// Support/Resistance Exports
// ============================================================================
pub use pivot_points::{PivotPoints, PivotPointsResult};
pub use fibonacci::{Fibonacci, FibonacciLevels};

// ============================================================================
// Other Exports
// ============================================================================
pub use elder_ray::{ElderRay, ElderRayOutput};

// Re-export SPI types
pub use indicator_spi::{
    TechnicalIndicator, StreamingIndicator, SignalIndicator,
    IndicatorOutput, IndicatorSignal, IndicatorError, Result,
    OHLCV, OHLCVSeries,
};

// Re-export API configs
pub use indicator_api::{
    // Moving Averages
    SMAConfig, EMAConfig, WMAConfig, DEMAConfig, TEMAConfig,
    HMAConfig, KAMAConfig, ZLEMAConfig,
    SMMAConfig, ALMAConfig, FRAMAConfig, VIDYAConfig,
    T3Config, TRIMAConfig, GMMAConfig, SineWMAConfig,
    // Filters
    KalmanConfig, MedianConfig, GaussianConfig,
    // Oscillators
    RSIConfig, StochasticConfig, WilliamsRConfig, CCIConfig,
    TRIXConfig, UltimateOscillatorConfig,
    // Trend
    MACDConfig, ADXConfig, IchimokuConfig, SuperTrendConfig, ParabolicSARConfig,
    // Volatility
    BollingerConfig, ATRConfig, DonchianConfig, KeltnerConfig,
    // Volume
    VWAPConfig, OBVConfig, MFIConfig, CMFConfig,
    // Other
    ElderRayConfig, PivotType,
};
