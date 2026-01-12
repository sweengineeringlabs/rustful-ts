//! Technical Indicator Facade
//!
//! Unified re-exports for the indicator module.
//!
//! All indicators are now consolidated in indicator-core under category submodules:
//! - `oscillators` - Momentum oscillators (ROC, TSI, Stochastic RSI, etc.)
//! - `trend` - Trend indicators (Aroon, Vortex, Alligator, etc.)
//! - `volatility` - Volatility measures (Historical Vol, Parkinson, etc.)
//! - `volume` - Volume indicators (VWMA, A/D Line, Klinger, etc.)
//! - `statistical` - Statistical tools (StdDev, Z-Score, Correlation, etc.)
//! - `pattern` - Pattern recognition (ZigZag, Candlestick patterns, etc.)
//! - `risk` - Risk metrics (Sharpe, Sortino, VaR, etc.)
//! - `bands` - Band indicators (STARC, Envelope, Chandelier, etc.)
//! - `dsp` - Ehlers DSP indicators (MESA, MAMA, Hilbert, etc.)
//! - `composite` - Multi-indicator systems (TTM Squeeze, Elder Impulse, etc.)
//! - `breadth` - Market breadth (McClellan, TRIN, A/D Line, etc.)
//! - `swing` - Swing trading tools (Order Blocks, FVG, Market Structure, etc.)
//! - `demark` - DeMark indicators (TD Sequential, TD Combo, etc.)

// Re-export everything from SPI
pub use indicator_spi::*;

// Re-export everything from API
pub use indicator_api::*;

// Re-export everything from Core (includes all categories)
pub use indicator_core::*;
